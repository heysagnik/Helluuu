"""
Configuration for SatFormerNet
==============================

Centralized configuration with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Number of classes (including background)
    num_classes: int = 5
    
    # Class names
    class_names: List[str] = field(default_factory=lambda: [
        'background', 'roads', 'water', 'trees', 'buildings'
    ])
    
    # Class weights for loss balancing
    class_weights: List[float] = field(default_factory=lambda: [
        1.0, 1.2, 1.0, 0.8, 1.0
    ])
    
    # Input channels (3 for RGB, 4 for RGBNIR, etc.)
    in_channels: int = 3
    
    # Backbone configuration
    backbone_channels: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    backbone_depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    backbone_num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    backbone_window_size: int = 7
    backbone_mlp_ratio: float = 4.0
    
    # Decoder configuration
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    
    # Feature flags
    use_pretrained: bool = True
    use_adapters: bool = True
    use_msda: bool = False  # Multi-Scale Deformable Attention
    use_baam: bool = True   # Boundary-Aware Attention Module
    deep_supervision: bool = True
    
    # Dropout
    dropout: float = 0.1
    drop_path_rate: float = 0.2


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Dataset paths
    train_data_path: str = './data/train'
    val_data_path: str = './data/val'
    test_data_path: str = './data/test'
    
    # Image configuration
    image_size: Tuple[int, int] = (512, 512)
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation
    scale_range: Tuple[float, float] = (0.5, 2.0)
    
    # Normalization (ImageNet stats)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Tile-based loading for large images
    use_tiles: bool = False
    tile_size: int = 512
    tile_overlap: int = 128


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training
    num_epochs: int = 100
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    
    # Learning rate schedule
    warmup_epochs: int = 5
    scheduler: str = 'cosine'  # Options: 'cosine', 'step', 'plateau'
    
    # Gradient
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    
    # Logging
    log_every: int = 50
    val_every_epoch: int = 1
    
    # Checkpointing
    output_dir: str = './outputs'
    save_every_epoch: int = 5


@dataclass
class LossConfig:
    """Loss function configuration."""
    
    # Loss weights
    dice_weight: float = 0.5
    focal_weight: float = 0.3
    boundary_weight: float = 0.15
    lovasz_weight: float = 0.05
    
    # Focal loss parameters
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # Boundary loss parameters
    boundary_theta: float = 5.0
    
    # Deep supervision weights
    aux_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2])
    
    # BAAM boundary loss weight
    baam_weight: float = 0.1
    
    # Ignore index
    ignore_index: int = -100


@dataclass
class Config:
    """Complete configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Device
    device: str = 'cuda'
    seed: int = 42
    
    def save(self, path: str):
        """Save configuration to JSON."""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'loss': self.loss.__dict__,
            'device': self.device,
            'seed': self.seed
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        if 'model' in config_dict:
            for k, v in config_dict['model'].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        
        if 'data' in config_dict:
            for k, v in config_dict['data'].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)
        
        if 'training' in config_dict:
            for k, v in config_dict['training'].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        
        if 'loss' in config_dict:
            for k, v in config_dict['loss'].items():
                if hasattr(config.loss, k):
                    setattr(config.loss, k, v)
        
        config.device = config_dict.get('device', 'cuda')
        config.seed = config_dict.get('seed', 42)
        
        return config
    
    def __repr__(self) -> str:
        lines = ["Config("]
        lines.append(f"  model={self.model},")
        lines.append(f"  data={self.data},")
        lines.append(f"  training={self.training},")
        lines.append(f"  loss={self.loss},")
        lines.append(f"  device='{self.device}',")
        lines.append(f"  seed={self.seed}")
        lines.append(")")
        return "\n".join(lines)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_fast_dev_config() -> Config:
    """Get configuration for fast development/debugging."""
    config = Config()
    config.training.num_epochs = 5
    config.data.batch_size = 2
    config.data.image_size = (256, 256)
    config.training.log_every = 10
    config.model.use_msda = False
    config.model.deep_supervision = False
    return config


def get_high_performance_config() -> Config:
    """Get configuration for maximum performance."""
    config = Config()
    config.data.batch_size = 16
    config.data.image_size = (512, 512)
    config.model.use_msda = True
    config.model.use_baam = True
    config.model.deep_supervision = True
    config.training.use_ema = True
    config.training.num_epochs = 200
    return config


if __name__ == "__main__":
    # Print default config
    print("Default Configuration")
    print("=" * 60)
    
    config = get_default_config()
    print(config)
    
    # Save example config
    config.save('example_config.json')
    print(f"\nSaved to example_config.json")
    
    # Load it back
    loaded = Config.load('example_config.json')
    print(f"\nLoaded config: {loaded.model.num_classes} classes")
    
    # Clean up
    import os
    os.remove('example_config.json')
