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
    DubaiAerialDataset, LandCoverAIDataset,
    create_satellite_dataloader, print_dataset_instructions,
    SATELLITE_CLASSES, SATELLITE_CLASS_WEIGHTS
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
    # Data - Satellite Datasets (verified available)
    'DubaiAerialDataset', 'LandCoverAIDataset', 'create_satellite_dataloader',
    'print_dataset_instructions', 'SATELLITE_CLASSES', 'SATELLITE_CLASS_WEIGHTS',
    # Metrics
    'SegmentationMetrics', 'InferenceTimer',
    # Training
    'Trainer', 'EMA', 'get_cosine_schedule_with_warmup'
]
