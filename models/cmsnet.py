"""CMSNet model and fusion modules.

The implementation supports the full dual-branch model and the single-branch
ablations used in the experiments.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b
from .vmamba import Backbone_VSSM


class LDFM_V2(nn.Module):
    """Local detail fusion for the first two encoder stages."""

    def __init__(self, cnn_dim: int, vssm_dim: int, out_dim: int):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False)
        self.vssm_proj = nn.Conv2d(vssm_dim, out_dim, kernel_size=1, bias=False)

        hidden_dim = max(out_dim // 4, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_cnn: torch.Tensor, x_mamba: torch.Tensor) -> torch.Tensor:
        if x_mamba.shape[2:] != x_cnn.shape[2:]:
            x_mamba = F.interpolate(
                x_mamba, size=x_cnn.shape[2:], mode="bilinear", align_corners=False
            )

        cnn_feat = self.cnn_proj(x_cnn)
        mamba_feat = self.vssm_proj(x_mamba)

        mamba_max = torch.max(mamba_feat, dim=1, keepdim=True).values
        mamba_avg = torch.mean(mamba_feat, dim=1, keepdim=True)
        spatial_gate = self.spatial_attention(torch.cat((mamba_max, mamba_avg), dim=1))
        cnn_enhanced = cnn_feat * spatial_gate

        channel_gate = self.channel_attention(cnn_feat)
        mamba_enhanced = mamba_feat * channel_gate

        fused = self.fuse(torch.cat((cnn_enhanced, mamba_enhanced), dim=1))
        return fused + cnn_feat


class DCCA(nn.Module):
    """Dual-branch cross-covariance attention for deep encoder stages."""

    def __init__(self, cnn_dim: int, vssm_dim: int, out_dim: int):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False)
        self.mamba_proj = nn.Conv2d(vssm_dim, out_dim, kernel_size=1, bias=False)

        self.q_conv = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False
        )
        self.k_conv = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False
        )
        self.v_conv = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x_cnn: torch.Tensor, x_mamba: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x_cnn.shape
        cnn_feat = self.cnn_proj(x_cnn)
        mamba_feat = self.mamba_proj(x_mamba)

        q_cnn = self.q_conv(cnn_feat).flatten(2)
        k_mamba = self.k_conv(mamba_feat).flatten(2)
        v_mamba = self.v_conv(mamba_feat).flatten(2)
        q_cnn = F.normalize(q_cnn, dim=-1)
        k_mamba = F.normalize(k_mamba, dim=-1)
        attn_cnn_to_mamba = torch.bmm(q_cnn, k_mamba.transpose(1, 2))
        attn_cnn_to_mamba = F.softmax(
            attn_cnn_to_mamba * self.temperature, dim=-1
        )
        out_cnn = torch.bmm(attn_cnn_to_mamba, v_mamba).view(
            batch_size, -1, height, width
        )

        q_mamba = self.q_conv(mamba_feat).flatten(2)
        k_cnn = self.k_conv(cnn_feat).flatten(2)
        v_cnn = self.v_conv(cnn_feat).flatten(2)
        q_mamba = F.normalize(q_mamba, dim=-1)
        k_cnn = F.normalize(k_cnn, dim=-1)
        attn_mamba_to_cnn = torch.bmm(q_mamba, k_cnn.transpose(1, 2))
        attn_mamba_to_cnn = F.softmax(attn_mamba_to_cnn * self.temperature, dim=-1)
        out_mamba = torch.bmm(attn_mamba_to_cnn, v_cnn).view(
            batch_size, -1, height, width
        )

        return self.fuse(torch.cat((out_cnn + cnn_feat, out_mamba + mamba_feat), dim=1))


class BaselineFuse(nn.Module):
    """Projection-and-add fusion used by the simple-fusion ablation."""

    def __init__(self, cnn_dim: int, vssm_dim: int, out_dim: int):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False)
        self.vssm_proj = nn.Conv2d(vssm_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x_cnn: torch.Tensor, x_mamba: torch.Tensor) -> torch.Tensor:
        if x_mamba.shape[2:] != x_cnn.shape[2:]:
            x_mamba = F.interpolate(
                x_mamba, size=x_cnn.shape[2:], mode="bilinear", align_corners=False
            )
        return self.cnn_proj(x_cnn) + self.vssm_proj(x_mamba)


class LightweightSegHead(nn.Module):
    """Multi-scale projection head used by the main and auxiliary decoders."""

    def __init__(self, in_channels, embed_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                )
                for channels in in_channels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),
        )

    def forward(self, features) -> torch.Tensor:
        target_size = features[0].shape[2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(feature)
        return self.fuse(torch.cat(projected, dim=1))


class CMSNet(nn.Module):
    """CNN-VMamba multi-level synergy network."""

    def __init__(
        self,
        cnn_name: str = "resnet50",
        vssm_name: str = "vssm_tiny",
        num_classes: int = 4,
        input_size: int = 512,
        fuse_config=None,
        mode: str = "dual",
        pretrained: bool = True,
    ):
        super().__init__()
        if fuse_config is None:
            fuse_config = ("LDFM", "LDFM", "DCCA", "DCCA")
        if len(fuse_config) != 4:
            raise ValueError("fuse_config must contain four stage names")

        self.num_classes = num_classes
        self.input_size = input_size
        self.fuse_config = tuple(fuse_config)
        self.mode = mode
        self.cnn_name = cnn_name
        self.vssm_name = vssm_name

        if cnn_name in {"resnet18", "resnet34"}:
            self.cnn_dims = [64, 128, 256, 512]
        elif cnn_name in {"resnet50", "resnet101"}:
            self.cnn_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_name}")

        if vssm_name in {"vssm_tiny", "vssm_small"}:
            self.vssm_dims = [96, 192, 384, 768]
        elif vssm_name == "vssm_base":
            self.vssm_dims = [128, 256, 512, 1024]
        else:
            raise ValueError(f"Unsupported VMamba backbone: {vssm_name}")

        self.upsample_final = nn.Upsample(
            size=(input_size, input_size), mode="bilinear", align_corners=False
        )

        if mode == "dual":
            self._create_cnn(cnn_name, pretrained)
            self.vmamba = self._create_vmamba(vssm_name, pretrained)
            fuse_dims = [256] * 4
            self.fuse_modules = nn.ModuleList(
                [
                    self._build_fuse_module(
                        self.fuse_config[index],
                        self.cnn_dims[index],
                        self.vssm_dims[index],
                        fuse_dims[index],
                    )
                    for index in range(4)
                ]
            )
            self.main_head = LightweightSegHead(fuse_dims, 256, num_classes)
            self.aux_head_cnn = LightweightSegHead(self.cnn_dims, 128, num_classes)
            self.aux_head_mamba = LightweightSegHead(self.vssm_dims, 128, num_classes)
        elif mode == "cnn_only":
            self._create_cnn(cnn_name, pretrained)
            self.main_head = LightweightSegHead(self.cnn_dims, 256, num_classes)
        elif mode == "mamba_only":
            self.vmamba = self._create_vmamba(vssm_name, pretrained)
            self.main_head = LightweightSegHead(self.vssm_dims, 256, num_classes)
        else:
            raise ValueError("mode must be 'dual', 'cnn_only', or 'mamba_only'")

    def _build_fuse_module(self, fuse_type, cnn_dim, vssm_dim, out_dim):
        if fuse_type == "LDFM":
            return LDFM_V2(cnn_dim, vssm_dim, out_dim)
        if fuse_type == "DCCA":
            return DCCA(cnn_dim, vssm_dim, out_dim)
        if fuse_type == "None":
            return BaselineFuse(cnn_dim, vssm_dim, out_dim)
        raise ValueError(f"Unsupported fusion module: {fuse_type}")

    def _create_cnn(self, cnn_name: str, pretrained: bool) -> None:
        constructors = {
            "resnet18": resnet18_v1b,
            "resnet34": resnet34_v1b,
            "resnet50": resnet50_v1b,
            "resnet101": resnet101_v1b,
        }
        backbone = constructors[cnn_name](pretrained=pretrained, dilated=False)
        self.resnet_stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.res_l1 = backbone.layer1
        self.res_l2 = backbone.layer2
        self.res_l3 = backbone.layer3
        self.res_l4 = backbone.layer4
        print(f"CNN backbone initialized: {cnn_name}")

    def _create_vmamba(self, vssm_name: str, pretrained: bool):
        configs = {
            "vssm_tiny": {
                "depths": [2, 2, 9, 2],
                "dims": 96,
                "drop_path_rate": 0.2,
                "checkpoint": "pretrained_weights/vssmtiny_dp01_ckpt_epoch_292.pth",
            },
            "vssm_small": {
                "depths": [2, 2, 27, 2],
                "dims": 96,
                "drop_path_rate": 0.3,
                "checkpoint": "pretrained_weights/vssm_small_0229_ckpt_epoch_222.pth",
            },
            "vssm_base": {
                "depths": [2, 2, 27, 2],
                "dims": 128,
                "drop_path_rate": 0.5,
                "checkpoint": "pretrained_weights/vssmbase_dp05_ckpt_epoch_260.pth",
            },
        }
        config = configs[vssm_name]
        vmamba = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=None,
            norm_layer="ln",
            depths=config["depths"],
            dims=config["dims"],
            drop_path_rate=config["drop_path_rate"],
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v0",
            mlp_ratio=0.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            downsample_version="v1",
            patchembed_version="v1",
            use_checkpoint=False,
            posembed=False,
            imgsize=self.input_size,
        )

        if pretrained:
            checkpoint_path = Path(config["checkpoint"])
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                state_dict = checkpoint.get("model", checkpoint)
                state_dict = {
                    key.replace("backbone.", ""): value
                    for key, value in state_dict.items()
                }
                vmamba.load_state_dict(state_dict, strict=False)
                print(f"VMamba weights loaded: {checkpoint_path}")
            except Exception as exc:
                print(
                    f"VMamba weights unavailable ({checkpoint_path}); "
                    f"using random initialization. Details: {exc}"
                )
        return vmamba

    def forward(self, x: torch.Tensor):
        if x.shape[2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        if self.mode == "dual":
            stem = self.resnet_stem(x)
            r1 = self.res_l1(stem)
            r2 = self.res_l2(r1)
            r3 = self.res_l3(r2)
            r4 = self.res_l4(r3)
            v1, v2, v3, v4 = self.vmamba(x)

            fused = [
                self.fuse_modules[0](r1, v1),
                self.fuse_modules[1](r2, v2),
                self.fuse_modules[2](r3, v3),
                self.fuse_modules[3](r4, v4),
            ]
            main_output = self.upsample_final(self.main_head(fused))
            if self.training:
                cnn_output = self.upsample_final(self.aux_head_cnn([r1, r2, r3, r4]))
                mamba_output = self.upsample_final(
                    self.aux_head_mamba([v1, v2, v3, v4])
                )
                return main_output, cnn_output, mamba_output
            return main_output

        if self.mode == "cnn_only":
            stem = self.resnet_stem(x)
            r1 = self.res_l1(stem)
            r2 = self.res_l2(r1)
            r3 = self.res_l3(r2)
            r4 = self.res_l4(r3)
            return self.upsample_final(self.main_head([r1, r2, r3, r4]))

        v1, v2, v3, v4 = self.vmamba(x)
        return self.upsample_final(self.main_head([v1, v2, v3, v4]))


def create_model(
    cnn_name: str = "resnet50",
    vssm_name: str = "vssm_tiny",
    num_classes: int = 4,
    input_size: int = 512,
    fuse_config=None,
    mode: str = "dual",
    pretrained: bool = True,
):
    """Build and summarize a CMSNet instance."""
    model = CMSNet(
        cnn_name=cnn_name,
        vssm_name=vssm_name,
        num_classes=num_classes,
        input_size=input_size,
        fuse_config=fuse_config,
        mode=mode,
        pretrained=pretrained,
    )
    print("=" * 50)
    print(f"CMSNet initialized (mode: {mode})")
    if mode in {"dual", "cnn_only"}:
        print(f"CNN backbone: {cnn_name}")
    if mode in {"dual", "mamba_only"}:
        print(f"VMamba backbone: {vssm_name}")
    if mode == "dual":
        print(f"Fusion stages: {model.fuse_config}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("=" * 50)
    return model


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the standalone model smoke test.")

    device = torch.device("cuda")
    model = create_model(mode="dual").to(device)
    dummy_input = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    if isinstance(output, tuple):
        print("Training outputs:", [tuple(item.shape) for item in output])
    else:
        print("Inference output:", tuple(output.shape))
