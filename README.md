# CMSNet: A CNN-VMamba Multi-Level Synergy Network for Crop Segmentation

This repository provides the PyTorch implementation of CMSNet for crop segmentation from ultra-high-resolution UAV remote-sensing imagery.

## Overview

CMSNet combines a ResNet-50 CNN branch with a VMamba-Tiny branch. The two branches exchange complementary information at four encoder stages:

- **LDFM** is applied at Stages 1--2 to improve local-detail representations through bidirectional spatial and channel attention.
- **DCCA** is applied at Stages 3--4 to perform dual-branch cross-covariance attention in the channel domain.
- A lightweight multi-scale segmentation head produces the main prediction. Two branch-specific auxiliary heads are used only during training.

For the default configuration and a $512\times512$ input, the CNN feature dimensions are $[256,512,1024,2048]$, the VMamba feature dimensions are $[96,192,384,768]$, and all fusion outputs have 256 channels.

## Environment

Create a Python environment with Python 3.10 or later, install a PyTorch build compatible with your CUDA driver, and then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The VMamba implementation uses optional CUDA selective-scan extensions when they are available and otherwise falls back to its PyTorch implementation. The reported training results were obtained with a CUDA-enabled PyTorch environment.

## Data preparation

The Barley dataset is not distributed with this repository. Prepare the data as follows:

```text
data/
└── barley/
    ├── images/
    │   ├── <subregion_name>.png
    │   └── ...
    ├── labels/
    │   ├── <subregion_name>.png
    │   └── ...
    └── annotations/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

Each annotation file contains one sub-region name per line, without the `.png` suffix. The training, validation, and test lists must be spatially disjoint at the sub-region level. During loading, each sub-region is divided into $512\times512$ patches with a stride of 341 pixels (approximately one-third overlap). Patches with no valid alpha-channel pixel are discarded.

For an RGBA image, the RGB channels are used as input and the alpha channel identifies valid pixels. Images without an alpha channel are also supported; all pixels are then treated as valid.

## Pretrained weights

The ResNet backbone requests ImageNet-pretrained weights through PyTorch when `pretrained=True`. To initialize the VMamba-Tiny branch with the configuration used in the paper, place the compatible checkpoint at:

```text
pretrained_weights/vssmtiny_dp01_ckpt_epoch_292.pth
```

If this file is unavailable, the implementation reports the missing checkpoint and initializes the VMamba branch randomly. Pretrained weights are intentionally not included in this repository.

## Training

The default setting trains CMSNet with ResNet-50 and VMamba-Tiny:

```bash
python train.py \
  --models cmsnet \
  --data_dir ./data \
  --save_dir ./work_dirs \
  --end_epoch 50 \
  --lr 0.0001 \
  --train_batchsize 4 \
  --val_batchsize 4 \
  --crop_size 512 512 \
  --seed 6
```

The main head is optimized with $\mathcal{L}_{\mathrm{CE}}+\mathcal{L}_{\mathrm{Dice}}$. The CNN and VMamba auxiliary heads use cross-entropy only, each with a coefficient of 0.4:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{CE}}^{\mathrm{main}}
+\mathcal{L}_{\mathrm{Dice}}^{\mathrm{main}}
+0.4\left(\mathcal{L}_{\mathrm{CE}}^{\mathrm{cnn}}
+\mathcal{L}_{\mathrm{CE}}^{\mathrm{vmamba}}\right).
$$

The validation split is used to select the best checkpoint. The seed is fixed to 6 for Python, NumPy, and PyTorch; cuDNN deterministic mode is enabled in the training script.

## Test-time evaluation

Use the independent test split only after model selection:

```bash
python test.py \
  --data_dir ./data \
  --checkpoint ./work_dirs/cmsnet_lr0.0001_epoch50_batchsize4_RS/weights/best_weight.pkl \
  --crop_size 512 512
```

The supplied `test.py` evaluates the valid pixels from all overlapping test
patches and computes OA, mIoU, and mF1 from one pooled confusion matrix. If
full-orthomosaic reconstruction is required, retain the source-patch
coordinates during preprocessing and average the corresponding softmax
probabilities before computing the final map.

## Acknowledgements

The data-loading and training utilities build on the open-source CCTNet code base. The bundled Apache-2.0 license is retained for the derived components. The VMamba backbone and its selective-scan support files follow the corresponding upstream implementation.

## Citation

If you use this code, please cite the accompanying CMSNet manuscript. The bibliographic entry will be added after publication.
