#!/usr/bin/env python
"""
SatFormerNet Training Script
============================

Main entry point for training SatFormerNet on satellite imagery.

Usage:
    python train.py --data_path ./data --output_dir ./outputs --epochs 100
    
    python train.py --config config.json
"""

import argparse
import os
import sys
import random
import numpy as np
import torch

from config import Config, get_default_config
from model import SatFormerNet
from train import (
    Trainer,
    CompoundLoss,
    SatelliteDataset,
    create_dataloader,
    get_train_transforms,
    get_val_transforms
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SatFormerNet on satellite imagery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of segmentation classes')
    parser.add_argument('--use_msda', action='store_true',
                        help='Use Multi-Scale Deformable Attention')
    parser.add_argument('--use_baam', action='store_true', default=True,
                        help='Use Boundary-Aware Attention Module')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Disable pretrained backbone weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training features
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable EMA')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Misc
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load or create config
    if args.config is not None:
        config = Config.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = get_default_config()
        
        # Override with command line args
        config.data.train_data_path = args.data_path
        config.data.val_data_path = args.data_path
        config.data.batch_size = args.batch_size
        config.data.image_size = (args.image_size, args.image_size)
        config.data.num_workers = args.num_workers
        
        config.model.num_classes = args.num_classes
        config.model.use_msda = args.use_msda
        config.model.use_baam = args.use_baam
        config.model.use_pretrained = not args.no_pretrained
        
        config.training.num_epochs = args.epochs
        config.training.lr = args.lr
        config.training.weight_decay = args.weight_decay
        config.training.use_amp = not args.no_amp
        config.training.use_ema = not args.no_ema
        config.training.gradient_accumulation_steps = args.gradient_accumulation
        config.training.output_dir = args.output_dir
        
        config.device = args.device
        config.seed = args.seed
    
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Save config
    config.save(os.path.join(config.training.output_dir, 'config.json'))
    
    # Print configuration
    print("=" * 60)
    print("SatFormerNet Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Classes: {config.model.num_classes}")
    print(f"  Image size: {config.data.image_size}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Learning rate: {config.training.lr}")
    print(f"  Device: {config.device}")
    print(f"  AMP: {config.training.use_amp}")
    print(f"  EMA: {config.training.use_ema}")
    print(f"  BAAM: {config.model.use_baam}")
    print(f"  MSDA: {config.model.use_msda}")
    
    # Check device
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("\nWarning: CUDA not available, falling back to CPU")
        config.device = 'cpu'
    
    if config.device == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create model
    print("\nCreating model...")
    model = SatFormerNet(
        num_classes=config.model.num_classes,
        in_channels=config.model.in_channels,
        backbone_channels=config.model.backbone_channels,
        decoder_channels=config.model.decoder_channels,
        use_pretrained=config.model.use_pretrained,
        use_adapters=config.model.use_adapters,
        use_msda=config.model.use_msda,
        use_baam=config.model.use_baam,
        deep_supervision=config.model.deep_supervision,
        dropout=config.model.dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    
    try:
        train_loader = create_dataloader(
            root=config.data.train_data_path,
            split='train',
            batch_size=config.data.batch_size,
            image_size=config.data.image_size,
            num_workers=config.data.num_workers
        )
        print(f"Train samples: {len(train_loader.dataset)}")
        
        val_loader = create_dataloader(
            root=config.data.val_data_path,
            split='val',
            batch_size=config.data.batch_size,
            image_size=config.data.image_size,
            num_workers=config.data.num_workers
        )
        print(f"Val samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("\nPlease ensure your data is organized as:")
        print("  data/")
        print("    images/")
        print("      img_001.png")
        print("    masks/")
        print("      img_001.png")
        print("\nOr use --data_path to specify the correct location.")
        
        # Demo mode with random data
        print("\nRunning in demo mode with random data...")
        
        from torch.utils.data import TensorDataset, DataLoader
        
        demo_images = torch.randn(32, 3, config.data.image_size[0], config.data.image_size[1])
        demo_masks = torch.randint(0, config.model.num_classes, 
                                  (32, config.data.image_size[0], config.data.image_size[1]))
        
        demo_dataset = TensorDataset(demo_images, demo_masks)
        train_loader = DataLoader(demo_dataset, batch_size=config.data.batch_size, shuffle=True)
        val_loader = DataLoader(demo_dataset, batch_size=config.data.batch_size, shuffle=False)
    
    # Create loss function
    loss_fn = CompoundLoss(
        num_classes=config.model.num_classes,
        class_weights=config.model.class_weights,
        dice_weight=config.loss.dice_weight,
        focal_weight=config.loss.focal_weight,
        boundary_weight=config.loss.boundary_weight,
        lovasz_weight=config.loss.lovasz_weight,
        ignore_index=config.loss.ignore_index
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=config.device,
        output_dir=config.training.output_dir,
        num_classes=config.model.num_classes,
        class_names=config.model.class_names,
        num_epochs=config.training.num_epochs,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        warmup_epochs=config.training.warmup_epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        use_amp=config.training.use_amp,
        use_ema=config.training.use_ema,
        ema_decay=config.training.ema_decay,
        early_stopping_patience=config.training.patience,
        log_every=config.training.log_every,
        val_every_epoch=config.training.val_every_epoch
    )
    
    # Resume from checkpoint
    if args.resume is not None:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n")
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Best mIoU: {trainer.best_miou:.4f}")
    print(f"Outputs saved to: {config.training.output_dir}")


if __name__ == '__main__':
    main()
