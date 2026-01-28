"""
Dataset loaders for MixConfig experiments.

Supports:
- Tabular: OpenML-CC18 benchmark suite
- Vision: CIFAR-100, ImageNet-1K
- Molecular: MolHIV, QM9, BBBP, BACE
- Text: SST-2, AG News
"""

from .tabular import OpenMLCC18Loader, load_openml_dataset
from .vision import CIFAR100Loader, ImageNet1KLoader
from .molecular import MolHIVLoader, QM9Loader, BBBPLoader, BACELoader
from .text import SST2Loader, AGNewsLoader

__all__ = [
    # Tabular
    "OpenMLCC18Loader",
    "load_openml_dataset",
    # Vision
    "CIFAR100Loader",
    "ImageNet1KLoader",
    # Molecular
    "MolHIVLoader",
    "QM9Loader",
    "BBBPLoader",
    "BACELoader",
    # Text
    "SST2Loader",
    "AGNewsLoader",
]
