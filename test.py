"""Evaluate a trained CMSNet checkpoint on the independent test split.

The loader supplies fixed-size overlapping patches. This script pools valid
pixels from all test patches into one confusion matrix, matching the metric
aggregation used by the training-time validation routine.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RemoteData
from models.cmsnet import create_model
from seg_metric import SegmentationMetric


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CMSNet")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--dataset", default="barley")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--crop_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--seed", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RemoteData(
        base_dir=args.data_dir,
        split="test",
        dataset=args.dataset,
        crop_size=args.crop_size,
    )
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    model = create_model(
        num_classes=4, input_size=args.crop_size[0], mode="dual", pretrained=False
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Checkpoints created by the release training script contain the bare
    # CMSNet state dict; accept a FullModel-prefixed checkpoint as well.
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    metric = SegmentationMetric(4)
    with torch.no_grad():
        for sample in loader:
            image = sample["image"].to(device, non_blocking=True)
            label = sample["label"].long().squeeze(1)
            valid = sample.get("alpha")
            prediction = model(image).argmax(dim=1).cpu().numpy()
            target = label.numpy()
            if valid is None:
                valid = np.ones_like(target, dtype=bool)
            else:
                valid = valid.numpy() > 0
            metric.addBatch(prediction[valid], target[valid])

    iou = metric.IntersectionOverUnion()
    precision = metric.Precision()
    recall = metric.Recall()
    f1 = 2 * precision * recall / (precision + recall)
    print("class IoU:", iou)
    print("class F1:", f1)
    print("mIoU: {:.6f}".format(np.nanmean(iou)))
    print("mF1: {:.6f}".format(np.nanmean(f1)))
    print("OA: {:.6f}".format(metric.Accuracy()))


if __name__ == "__main__":
    main()
