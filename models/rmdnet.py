"""Backward-compatible imports for earlier experimental checkpoints.

CMSNet is implemented in :mod:`models.cmsnet`. This module is retained only
so that code referring to the former experimental module name continues to
load the same architecture.
"""

from .cmsnet import BaselineFuse, CMSNet, DCCA, LDFM_V2, LightweightSegHead, create_model

RMDNet_V2 = CMSNet

__all__ = [
    "BaselineFuse",
    "CMSNet",
    "DCCA",
    "LDFM_V2",
    "LightweightSegHead",
    "RMDNet_V2",
    "create_model",
]
