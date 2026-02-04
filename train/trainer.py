"""
Trainer Module for SatFormerNet
===============================

Complete training pipeline with:
- Mixed precision training (FP16)
- AdamW optimizer with cosine annealing
- Gradient accumulation
- EMA (Exponential Moving Average)
- Early stopping
- Checkpointing
- Logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Callable, List
import math
from copy import deepcopy

from .losses import CompoundLoss
from .metrics import SegmentationMetrics, InferenceTimer


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a moving average of model weights for more stable inference.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        self._register()
    
    def _register(self):
        """Register model parameters for EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with current model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
):
    """
    Create learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum learning rate as ratio of initial
    
    Returns:
        LR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Trainer for SatFormerNet segmentation model.
    
    Args:
        model: SatFormerNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer (default: AdamW)
        loss_fn: Loss function (default: CompoundLoss)
        device: Training device
        output_dir: Directory for checkpoints and logs
        num_classes: Number of segmentation classes
        class_names: Names for each class
        
        # Training hyperparameters
        num_epochs: Number of training epochs
        lr: Initial learning rate
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of warmup epochs
        gradient_accumulation_steps: Accumulation steps for effective batch size
        max_grad_norm: Gradient clipping norm
        
        # Features
        use_amp: Use mixed precision training
        use_ema: Use EMA for inference
        ema_decay: EMA decay rate
        early_stopping_patience: Patience for early stopping
        
        # Logging
        log_every: Log every N steps
        val_every_epoch: Validate every N epochs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = 'cuda',
        output_dir: str = './outputs',
        num_classes: int = 5,
        class_names: Optional[List[str]] = None,
        
        # Hyperparameters
        num_epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        
        # Features
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        early_stopping_patience: int = 15,
        
        # Logging
        log_every: int = 50,
        val_every_epoch: int = 1
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_classes = num_classes
        self.class_names = class_names or ['background', 'roads', 'water', 'trees', 'buildings']
        
        # Hyperparameters
        self.num_epochs = num_epochs
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Features
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_ema = use_ema
        self.early_stopping_patience = early_stopping_patience
        
        # Logging
        self.log_every = log_every
        self.val_every_epoch = val_every_epoch
        
        # Setup optimizer
        if optimizer is None:
            if hasattr(model, 'get_param_groups'):
                param_groups = model.get_param_groups(lr, weight_decay)
            else:
                param_groups = model.parameters()
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer
        
        # Setup loss
        self.loss_fn = loss_fn or CompoundLoss(num_classes=num_classes)
        self.loss_fn = self.loss_fn.to(device)
        
        # Setup scheduler
        steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        
        # Setup AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup EMA
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes=num_classes, class_names=self.class_names)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_miou = 0.0
        self.patience_counter = 0
        self.history = {'train': [], 'val': []}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0
        loss_components = {}
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                
                # Get auxiliary outputs if available
                aux_preds = outputs.get('aux', None)
                boundary_pred = outputs.get('boundary', None)
                
                losses = self.loss_fn(
                    outputs['out'], masks,
                    aux_preds=aux_preds,
                    boundary_pred=boundary_pred
                )
                loss = losses['total'] / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # EMA update
                if self.ema is not None:
                    self.ema.update()
                
                self.global_step += 1
            
            # Accumulate losses
            epoch_loss += losses['total'].item()
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0
                loss_components[k] += v.item()
            
            # Logging
            if (batch_idx + 1) % self.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx + 1}/{num_batches} | "
                      f"Loss: {losses['total'].item():.4f} | "
                      f"LR: {lr:.6f}")
        
        # Average losses
        epoch_loss /= num_batches
        for k in loss_components:
            loss_components[k] /= num_batches
        
        return {'loss': epoch_loss, **loss_components}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.metrics.reset()
        
        # Apply EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow()
        
        val_loss = 0
        num_batches = len(self.val_loader)
        
        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                losses = self.loss_fn(outputs['out'], masks)
            
            val_loss += losses['total'].item()
            
            # Update metrics
            self.metrics.update(outputs['out'], masks)
            self.metrics.update_boundary(outputs['out'], masks)
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        # Compute metrics
        metrics = self.metrics.compute()
        metrics['loss'] = val_loss / num_batches
        
        return metrics
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'history': self.history
        }
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (mIoU: {self.best_miou:.4f})")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_miou = checkpoint['best_miou']
        self.history = checkpoint['history']
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, List]:
        """
        Main training loop.
        
        Returns:
            Training history
        """
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.lr}")
        print(f"  AMP: {self.use_amp}")
        print(f"  EMA: {self.use_ema}")
        print("")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            # Training
            train_metrics = self.train_epoch()
            self.history['train'].append(train_metrics)
            
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            
            # Validation
            if (epoch + 1) % self.val_every_epoch == 0:
                val_metrics = self.validate()
                self.history['val'].append(val_metrics)
                
                if val_metrics:
                    print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                          f"mIoU: {val_metrics['mIoU']:.4f} | "
                          f"Boundary F1: {val_metrics['boundary_f1']:.4f}")
                    
                    # Check for best model
                    is_best = val_metrics['mIoU'] > self.best_miou
                    if is_best:
                        self.best_miou = val_metrics['mIoU']
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Save checkpoint
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best=is_best)
                    
                    # Early stopping
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
            
            # Timing
            epoch_time = time.time() - epoch_start
            print(f"  Time: {epoch_time / 60:.2f} min")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training Complete!")
        print(f"  Total time: {total_time / 3600:.2f} hours")
        print(f"  Best mIoU: {self.best_miou:.4f}")
        print("=" * 60)
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Save history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


if __name__ == "__main__":
    print("Trainer module for SatFormerNet")
    print("=" * 60)
    
    print("\nUsage:")
    print("  from model import SatFormerNet")
    print("  from train import Trainer, create_dataloader")
    print("")
    print("  model = SatFormerNet(num_classes=5)")
    print("  train_loader = create_dataloader('path/to/data', split='train')")
    print("  val_loader = create_dataloader('path/to/data', split='val')")
    print("")
    print("  trainer = Trainer(")
    print("      model=model,")
    print("      train_loader=train_loader,")
    print("      val_loader=val_loader,")
    print("      num_epochs=100,")
    print("      lr=1e-4")
    print("  )")
    print("")
    print("  history = trainer.train()")
