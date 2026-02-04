"""
Training module for SatFormerNet.
"""

from .losses import DiceLoss, FocalLoss, BoundaryLoss, LovaszSoftmax, CompoundLoss
from .augmentations import (
    get_train_transforms, get_val_transforms,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90,
    RandomBrightnessContrast, Normalize, ToTensor, Compose
)
from .dataset import SatelliteDataset, TileDataset, create_dataloader
from .india_dataset import (
    BhuvanDataset, IndiaSatDataset, 
    create_india_dataloader, download_bhuvan_sample,
    BHUVAN_CLASSES, INDIA_CLASS_WEIGHTS
)
from .metrics import SegmentationMetrics, InferenceTimer
from .trainer import Trainer, EMA, get_cosine_schedule_with_warmup

__all__ = [
    # Losses
    'DiceLoss', 'FocalLoss', 'BoundaryLoss', 'LovaszSoftmax', 'CompoundLoss',
    # Augmentations
    'get_train_transforms', 'get_val_transforms',
    'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomRotate90',
    'RandomBrightnessContrast', 'Normalize', 'ToTensor', 'Compose',
    # Data - General
    'SatelliteDataset', 'TileDataset', 'create_dataloader',
    # Data - India Specific
    'BhuvanDataset', 'IndiaSatDataset', 'create_india_dataloader',
    'download_bhuvan_sample', 'BHUVAN_CLASSES', 'INDIA_CLASS_WEIGHTS',
    # Metrics
    'SegmentationMetrics', 'InferenceTimer',
    # Training
    'Trainer', 'EMA', 'get_cosine_schedule_with_warmup'
]
