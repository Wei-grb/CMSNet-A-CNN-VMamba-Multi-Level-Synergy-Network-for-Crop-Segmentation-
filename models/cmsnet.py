import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保这里的导入路径与你项目中的文件位置一致
from models.vmamba import Backbone_VSSM  
from models.resnet import resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b 

# ==========================================
# 1. 浅层模块: LDFM_V2 (跨模态注意力双向引导)
# ==========================================
class LDFM_V2(nn.Module):
    def __init__(self, cnn_dim, vssm_dim, out_dim):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.vssm_proj = nn.Conv2d(vssm_dim, out_dim, 1, bias=False)
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 4, out_dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_cnn, x_mamba):
        if x_mamba.shape[2:] != x_cnn.shape[2:]:
            x_mamba = F.interpolate(x_mamba, size=x_cnn.shape[2:], mode='bilinear', align_corners=False)
        
        c = self.cnn_proj(x_cnn)
        m = self.vssm_proj(x_mamba)
        
        m_max, _ = torch.max(m, dim=1, keepdim=True)
        m_avg = torch.mean(m, dim=1, keepdim=True)
        m_spatial = self.sa(torch.cat([m_max, m_avg], dim=1))
        c_out = c * m_spatial
        
        c_channel = self.ca(c)
        m_out = m * c_channel
        
        feat = self.fuse(torch.cat([c_out, m_out], dim=1))
        return feat + c 
    
# ==========================================
# 2. Deep-stage module: DCCA (Dual-branch Cross-Covariance Attention)
# ==========================================
class DCCA(nn.Module):
    def __init__(self, cnn_dim, vssm_dim, out_dim):
        super().__init__()
        self.conv_cnn = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.conv_mamba = nn.Conv2d(vssm_dim, out_dim, 1, bias=False)
        
        self.q_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim, bias=False)
        self.k_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim, bias=False)
        self.v_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim, bias=False)
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x_cnn, x_mamba):
        B, _, H, W = x_cnn.shape
        
        c = self.conv_cnn(x_cnn)
        m = self.conv_mamba(x_mamba)
        
        q_c = self.q_conv(c).view(B, -1, H * W)
        k_m = self.k_conv(m).view(B, -1, H * W)
        v_m = self.v_conv(m).view(B, -1, H * W)
        
        q_c = F.normalize(q_c, dim=-1)
        k_m = F.normalize(k_m, dim=-1)
        
        attn_c2m = torch.bmm(q_c, k_m.transpose(1, 2)) * self.temperature
        attn_c2m = F.softmax(attn_c2m, dim=-1)
        out_c = torch.bmm(attn_c2m, v_m).view(B, -1, H, W)
        
        q_m = self.q_conv(m).view(B, -1, H * W)
        k_c = self.k_conv(c).view(B, -1, H * W)
        v_c = self.v_conv(c).view(B, -1, H * W)
        
        q_m = F.normalize(q_m, dim=-1)
        k_c = F.normalize(k_c, dim=-1)
        
        attn_m2c = torch.bmm(q_m, k_c.transpose(1, 2)) * self.temperature
        attn_m2c = F.softmax(attn_m2c, dim=-1)
        out_m = torch.bmm(attn_m2c, v_c).view(B, -1, H, W)
        
        fused = self.fuse_conv(torch.cat([out_c + c, out_m + m], dim=1))
        return fused

# ==========================================
# 3. 基线融合模块 (用于消融实验的无注意力基准)
# ==========================================
class BaselineFuse(nn.Module):
    def __init__(self, cnn_dim, vssm_dim, out_dim):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.vssm_proj = nn.Conv2d(vssm_dim, out_dim, 1, bias=False)
        
    def forward(self, x_cnn, x_mamba):
        if x_mamba.shape[2:] != x_cnn.shape[2:]:
            x_mamba = F.interpolate(x_mamba, size=x_cnn.shape[2:], mode='bilinear', align_corners=False)
        return self.cnn_proj(x_cnn) + self.vssm_proj(x_mamba)

# ==========================================
# 4. 统一轻量化解码头 (SegFormer 风格 MLP)
# ==========================================
class LightweightSegHead(nn.Module):
    def __init__(self, in_channels, embed_dim=256, num_classes=4):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for c in in_channels
        ])
        
        # 核心优化：全 MLP 架构，使用 1x1 卷积做纯通道级降维融合，大幅削减计算量
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False), 
            nn.BatchNorm2d(embed_dim), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(embed_dim, num_classes, 1) 
        )

    def forward(self, features):
        target_size = features[0].shape[2:]
        out = []
        for i, feat in enumerate(features):
            x = self.projs[i](feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            out.append(x)
        return self.fuse(torch.cat(out, dim=1))

# ==========================================
# 5. CMSNet architecture (supporting dual- and single-branch ablations)
# ==========================================
class CMSNet(nn.Module):
    def __init__(self, 
                 cnn_name='resnet50', 
                 vssm_name='vssm_tiny', 
                 num_classes=4, 
                 input_size=512, 
                 fuse_config=['LDFM', 'LDFM', 'DCCA', 'DCCA'],
                 mode='dual',
                 pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.fuse_config = fuse_config
        self.mode = mode 
        self.cnn_name = cnn_name
        self.vssm_name = vssm_name
        self.pretrained = pretrained
        
        # --- 1. 确定骨干网络通道维度 ---
        if cnn_name in ['resnet18', 'resnet34']:
            self.cnn_dims = [64, 128, 256, 512]
        elif cnn_name in ['resnet50', 'resnet101']:
            self.cnn_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"不支持的 CNN 名称: {cnn_name}")

        if vssm_name in ['vssm_tiny', 'vssm_small']:
            self.vssm_dims = [96, 192, 384, 768]
        elif vssm_name == 'vssm_base':
            self.vssm_dims = [128, 256, 512, 1024]
        else:
            raise ValueError(f"不支持的 VMamba 名称: {vssm_name}")

        self.upsample_final = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)

        # ==========================================
        # 2. 根据 Mode 动态构建网络拓扑
        # ==========================================
        if self.mode == 'dual':
            # 双分支模式: 加载两个 Backbone，构建融合模块，使用 256 维的头
            self._create_cnn(cnn_name, pretrained)
            self.vmamba = self._create_vmamba(vssm_name, pretrained)
            fuse_dims = [256, 256, 256, 256] 
            
            self.fuse_modules = nn.ModuleList()
            for i in range(4):
                self.fuse_modules.append(
                    self._build_fuse_module(fuse_config[i], self.cnn_dims[i], self.vssm_dims[i], fuse_dims[i])
                )
            
            self.main_head = LightweightSegHead(in_channels=fuse_dims, embed_dim=256, num_classes=num_classes)
            self.aux_head_cnn = LightweightSegHead(in_channels=self.cnn_dims, embed_dim=128, num_classes=num_classes)
            self.aux_head_mamba = LightweightSegHead(in_channels=self.vssm_dims, embed_dim=128, num_classes=num_classes)

        elif self.mode == 'cnn_only':
            # 纯 CNN 模式: 只加载 ResNet，直接用 cnn_dims 构建解码头
            self._create_cnn(cnn_name, pretrained)
            self.main_head = LightweightSegHead(in_channels=self.cnn_dims, embed_dim=256, num_classes=num_classes)
            
        elif self.mode == 'mamba_only':
            # 纯 Mamba 模式: 只加载 VMamba，直接用 vssm_dims 构建解码头
            self.vmamba = self._create_vmamba(vssm_name, pretrained)
            self.main_head = LightweightSegHead(in_channels=self.vssm_dims, embed_dim=256, num_classes=num_classes)
            
        else:
            raise ValueError("mode 必须是 'dual', 'cnn_only' 或 'mamba_only'")

    def _build_fuse_module(self, fuse_type, cnn_dim, vssm_dim, out_dim):
        if fuse_type == 'LDFM':
            return LDFM_V2(cnn_dim, vssm_dim, out_dim)
        elif fuse_type == 'DCCA':
            return DCCA(cnn_dim, vssm_dim, out_dim)
        elif fuse_type == 'None':
            return BaselineFuse(cnn_dim, vssm_dim, out_dim)
        else:
            raise ValueError(f"未知的融合模块类型: {fuse_type}")

    def _create_cnn(self, cnn_name, pretrained):
        cnn_dict = {
            'resnet18': 'resnet18_v1b',
            'resnet34': 'resnet34_v1b',
            'resnet50': 'resnet50_v1b',
            'resnet101': 'resnet101_v1b',
        }
        res_func = eval(cnn_dict[cnn_name])
        res = res_func(pretrained=pretrained, dilated=False)
        
        self.resnet_stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.res_l1, self.res_l2, self.res_l3, self.res_l4 = res.layer1, res.layer2, res.layer3, res.layer4
        print(f"CNN 分支: {cnn_dict[cnn_name]} 初始化成功!")

    def _create_vmamba(self, vssm_name, pretrained):
        vssm_configs = {
            'vssm_tiny':  {'depths': [2, 2, 9, 2],  'dims': 96,  'dp_rate': 0.2, 'ckpt': 'pretrained_weights/vssmtiny_dp01_ckpt_epoch_292.pth'},
            'vssm_small': {'depths': [2, 2, 27, 2], 'dims': 96,  'dp_rate': 0.3, 'ckpt': 'pretrained_weights/vssm_small_0229_ckpt_epoch_222.pth'},
            'vssm_base':  {'depths': [2, 2, 27, 2], 'dims': 128, 'dp_rate': 0.5, 'ckpt': 'pretrained_weights/vssmbase_dp05_ckpt_epoch_260.pth'},
        }
        config = vssm_configs[vssm_name]
        
        vmamba = Backbone_VSSM(
            out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln",
            depths=config['depths'], dims=config['dims'], drop_path_rate=config['dp_rate'], patch_size=4, in_chans=3,
            num_classes=1000, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
            ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
            ssm_init="v0", forward_type="v0", mlp_ratio=0.0, mlp_act_layer="gelu",
            mlp_drop_rate=0.0, gmlp=False, patch_norm=True, downsample_version="v1",
            patchembed_version="v1", use_checkpoint=False, posembed=False, imgsize=self.input_size
        )
        if pretrained:
            try:
                checkpoint = torch.load(config['ckpt'], map_location='cpu')
                new_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint.get('model', checkpoint).items()}
                vmamba.load_state_dict(new_state_dict, strict=False)
                print(f"VMamba branch: {vssm_name} pretrained weights loaded.")
            except Exception as e:
                print(f"VMamba branch: failed to load {config['ckpt']}; using random initialization. Error: {e}")
            
        return vmamba

    def forward(self, x):
        size = x.shape[2:]
        if size != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        if self.mode == 'dual':
            r0 = self.resnet_stem(x)
            r1 = self.res_l1(r0); r2 = self.res_l2(r1); r3 = self.res_l3(r2); r4 = self.res_l4(r3)
            v1, v2, v3, v4 = self.vmamba(x)
            
            f1 = self.fuse_modules[0](r1, v1)
            f2 = self.fuse_modules[1](r2, v2)
            f3 = self.fuse_modules[2](r3, v3)
            f4 = self.fuse_modules[3](r4, v4)
            
            out_main = self.upsample_final(self.main_head([f1, f2, f3, f4]))
            
            if self.training:
                out_cnn = self.upsample_final(self.aux_head_cnn([r1, r2, r3, r4]))
                out_mamba = self.upsample_final(self.aux_head_mamba([v1, v2, v3, v4]))
                return out_main, out_cnn, out_mamba
            return out_main

        elif self.mode == 'cnn_only':
            r0 = self.resnet_stem(x)
            r1 = self.res_l1(r0); r2 = self.res_l2(r1); r3 = self.res_l3(r2); r4 = self.res_l4(r3)
            out_main = self.upsample_final(self.main_head([r1, r2, r3, r4]))
            return out_main 

        elif self.mode == 'mamba_only':
            v1, v2, v3, v4 = self.vmamba(x)
            out_main = self.upsample_final(self.main_head([v1, v2, v3, v4]))
            return out_main


def create_model(cnn_name='resnet50', 
                 vssm_name='vssm_tiny', 
                 num_classes=4, 
                 input_size=512, 
                 fuse_config=['LDFM', 'LDFM', 'DCCA', 'DCCA'],
                 mode='dual',
                 pretrained=True):  
    model = CMSNet(
        cnn_name=cnn_name, vssm_name=vssm_name, num_classes=num_classes, 
        input_size=input_size, fuse_config=fuse_config, mode=mode,
        pretrained=pretrained
    )
    print("="*50)
    print(f"CMSNet initialized. Mode: [{mode.upper()}]")
    if mode in ['dual', 'cnn_only']:
        print(f" - CNN Backbone: {cnn_name}")
    if mode in ['dual', 'mamba_only']:
        print(f" - VMamba Backbone: {vssm_name}")
    if mode == 'dual':
        print(f" - 融合配置: {fuse_config}")
    print(f" - 总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("="*50)
    return model

if __name__ == '__main__':
    import torch
    
    if not torch.cuda.is_available():
        print("错误: 缺少 GPU 无法运行底层算子。")
        exit()
        
    device = torch.device('cuda')
    
    # 简单的实例化测试
    model = create_model(mode='dual').to(device)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        out = model(dummy_input)
        if isinstance(out, tuple):
            print(f"双分支训练模式输出 Shape: {out[0].shape}, {out[1].shape}, {out[2].shape}")
        else:
            print(f"推理模式输出 Shape: {out.shape}")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from models.vmamba import Backbone_VSSM  # 确保路径正确

# # ==========================================
# # 1. 浅层局部细节融合模块 (LDFM) - 对应 Stage 1 & 2
# # ==========================================
# class LDFM(nn.Module):
#     """
#     Local Detail Fusion Module: 专注于保留 CNN 的高分辨率边界信息，
#     利用空间掩码引导 Mamba 特征在边缘处的表达。
#     """
#     def __init__(self, cnn_dim, vssm_dim, out_dim):
#         super().__init__()
#         self.fuse = nn.Sequential(
#             nn.Conv2d(cnn_dim + vssm_dim, out_dim, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(inplace=True)
#         )
#         # 空间注意力掩码：学习从 CNN 取“边”，从 Mamba 取“面”
#         self.spatial_gate = nn.Sequential(
#             nn.Conv2d(out_dim, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x_cnn, x_mamba):
#         if x_mamba.shape[2:] != x_cnn.shape[2:]:
#             x_mamba = F.interpolate(x_mamba, size=x_cnn.shape[2:], mode='bilinear', align_corners=False)
        
#         feat = self.fuse(torch.cat([x_cnn, x_mamba], dim=1))
#         gate = self.spatial_gate(feat)
#         # 残差连接：强化 CNN 提供的原始细节
#         return feat + (x_cnn * gate if x_cnn.shape[1] == feat.shape[1] else feat * gate)

# # ==========================================
# # 2. 深层全局语义对齐模块 (GACF) - 对应 Stage 3 & 4
# # ==========================================
# class GACF(nn.Module):
#     """
#     Global-Aware Cross Fusion: 利用通道注意力耦合两个分支的高层语义，
#     解决 VMamba 全局扫描与 CNN 深度特征之间的认知冲突。
#     """
#     def __init__(self, cnn_dim, vssm_dim, out_dim):
#         super().__init__()
#         self.cnn_proj = nn.Conv2d(cnn_dim, out_dim, 1)
#         self.vssm_proj = nn.Conv2d(vssm_dim, out_dim, 1)
        
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.coupling = nn.Sequential(
#             nn.Linear(out_dim * 2, out_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_dim, out_dim),
#             nn.Sigmoid()
#         )
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x_cnn, x_mamba):
#         c = self.cnn_proj(x_cnn)
#         m = self.vssm_proj(x_mamba)
        
#         # 全局上下文耦合权重
#         c_gap = self.avg_pool(c).view(c.size(0), -1)
#         m_gap = self.avg_pool(m).view(m.size(0), -1)
        
#         # 计算两个分支的互补重要性
#         weight = self.coupling(torch.cat([c_gap, m_gap], dim=1)).view(c.size(0), c.size(1), 1, 1)
        
#         fused = c * weight + m * (1 - weight)
#         return self.out_conv(fused)

# # ==========================================
# # 3. 轻量化通用解码头 (SegHead)
# # ==========================================
# class LightweightSegHead(nn.Module):
#     def __init__(self, in_channels, embed_dim=256, num_classes=4):
#         super().__init__()
#         self.projs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(c, embed_dim, 1, bias=False),
#                 nn.BatchNorm2d(embed_dim),
#                 nn.ReLU(inplace=True)
#             ) for c in in_channels
#         ])
#         self.fuse = nn.Sequential(
#             nn.Conv2d(embed_dim * len(in_channels), embed_dim, 3, padding=1, bias=False),
#             nn.BatchNorm2d(embed_dim), 
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(embed_dim, num_classes, 1)
#         )

#     def forward(self, features):
#         target_size = features[0].shape[2:]
#         out = []
#         for i, feat in enumerate(features):
#             x = self.projs[i](feat)
#             if x.shape[2:] != target_size:
#                 x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
#             out.append(x)
#         return self.fuse(torch.cat(out, dim=1))

# # ==========================================
# # 4. RMDNet 完全体: RMDNet-V2 (双轨融合 + 三头监督版)
# # ==========================================
# class RMDNet_V2(nn.Module):
#     def __init__(self, num_classes=4, input_size=512):
#         super().__init__()
#         self.num_classes = num_classes
#         self.input_size = input_size
        
#         # --- Backbone 加载 ---
#         self.resnet = self._create_resnet50()
#         self.vmamba = self._create_vmamba()
        
#         cnn_dims = [256, 512, 1024, 2048]
#         vssm_dims = [96, 192, 384, 768]
#         fuse_dims = [256, 512, 512, 512] # 降低高层维度以减少计算量并统一特征强度
        
#         # --- 双轨融合系统 ---
#         # 1-2层使用 LDFM (局部细节驱动)
#         self.fuse1 = LDFM(cnn_dims[0], vssm_dims[0], fuse_dims[0])
#         self.fuse2 = LDFM(cnn_dims[1], vssm_dims[1], fuse_dims[1])
#         # 3-4层使用 GACF (全局语义耦合)
#         self.fuse3 = GACF(cnn_dims[2], vssm_dims[2], fuse_dims[2])
#         self.fuse4 = GACF(cnn_dims[3], vssm_dims[3], fuse_dims[3])
        
#         # --- 多头监督系统 ---
#         # 主头：融合特征
#         self.main_head = LightweightSegHead(in_channels=fuse_dims, embed_dim=256, num_classes=num_classes)
#         # 辅助头 1: 监督 CNN 分支 (强化边界)
#         self.aux_head_cnn = LightweightSegHead(in_channels=cnn_dims, embed_dim=128, num_classes=num_classes)
#         # 辅助头 2: 监督 Mamba 分支 (强化全局语义)
#         self.aux_head_mamba = LightweightSegHead(in_channels=vssm_dims, embed_dim=128, num_classes=num_classes)
        
#         self.upsample_final = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)

#     def _create_resnet50(self):
#         res = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
#         self.resnet_stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
#         self.res_l1, self.res_l2, self.res_l3, self.res_l4 = res.layer1, res.layer2, res.layer3, res.layer4
#         return res

#     def _create_vmamba(self):

#         vmamba = Backbone_VSSM(

#             out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln",

#             depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2, patch_size=4, in_chans=3,

#             num_classes=1000, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 

#             ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,

#             ssm_init="v0", forward_type="v0", mlp_ratio=0.0, mlp_act_layer="gelu",

#             mlp_drop_rate=0.0, gmlp=False, patch_norm=True, downsample_version="v1",

#             patchembed_version="v1", use_checkpoint=False, posembed=False, imgsize=self.input_size

#         )

#         try:

#             checkpoint = torch.load('pretrained_weights/vssmtiny_dp01_ckpt_epoch_292.pth', map_location='cpu')

#             new_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint.get('model', checkpoint).items()}

#             vmamba.load_state_dict(new_state_dict, strict=False)

#             print("双分支: VMamba权重加载成功!")

#         except:

#             print("VMamba权重加载失败")

#         return vmamba

#     def forward(self, x):
#         size = x.shape[2:]
#         if size != (self.input_size, self.input_size):
#             x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
#         # 分支提取
#         r0 = self.resnet_stem(x)
#         r1 = self.res_l1(r0); r2 = self.res_l2(r1); r3 = self.res_l3(r2); r4 = self.res_l4(r3)
#         v1, v2, v3, v4 = self.vmamba(x)
        
#         # 分阶段融合
#         f1 = self.fuse1(r1, v1)
#         f2 = self.fuse2(r2, v2)
#         f3 = self.fuse3(r3, v3)
#         f4 = self.fuse4(r4, v4)
        
#         # 主输出
#         out_main = self.upsample_final(self.main_head([f1, f2, f3, f4]))
        
#         if self.training:
#             # 辅助输出
#             out_cnn = self.upsample_final(self.aux_head_cnn([r1, r2, r3, r4]))
#             out_mamba = self.upsample_final(self.aux_head_mamba([v1, v2, v3, v4]))
#             return out_main, out_cnn, out_mamba
            
#         return out_main

# def create_model(num_classes=4, input_size=512):
#     model = RMDNet_V2(num_classes=num_classes, input_size=input_size)
#     print(f"RMDNet-V2 (双轨融合+多头监督)创建成功! 参数量: {sum(p.numel() for p in model.parameters()):,}")
#     return model
