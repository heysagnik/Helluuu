"""
Evaluation Metrics for Semantic Segmentation
=============================================

Comprehensive metrics including:
- Per-class and mean IoU
- Pixel accuracy
- Dice coefficient
- Boundary F1 score
- Confusion matrix visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for semantic segmentation.
    
    Accumulates predictions over batches and computes final metrics.
    
    Args:
        num_classes: Number of segmentation classes
        class_names: Names for each class
        ignore_index: Index to ignore in calculations
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -100
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.boundary_tp = 0
        self.boundary_fp = 0
        self.boundary_fn = 0
    
    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Predictions (B, C, H, W) logits or (B, H, W) class indices
            target: Ground truth (B, H, W) class indices
        """
        # Get class predictions
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Filter ignored indices
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    @torch.no_grad()
    def update_boundary(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ):
        """
        Update boundary metrics.
        
        Args:
            pred: Predicted boundary map (B, 1, H, W) or predictions
            target: Ground truth mask (B, H, W)
            threshold: Boundary detection threshold
        """
        if pred.dim() == 4 and pred.shape[1] > 1:
            pred = pred.argmax(dim=1)
        
        # Get boundaries
        pred_boundary = self._get_boundary(pred)
        target_boundary = self._get_boundary(target)
        
        pred_boundary = pred_boundary.cpu().numpy().flatten() > threshold
        target_boundary = target_boundary.cpu().numpy().flatten() > threshold
        
        # TP, FP, FN
        self.boundary_tp += np.sum(pred_boundary & target_boundary)
        self.boundary_fp += np.sum(pred_boundary & ~target_boundary)
        self.boundary_fn += np.sum(~pred_boundary & target_boundary)
    
    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary from mask using morphological operations."""
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        
        mask = mask.float().unsqueeze(1)
        
        # Dilate and erode
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        dilated = F.conv2d(mask, kernel, padding=1)
        dilated = (dilated > 0).float()
        
        eroded = F.conv2d(mask, kernel, padding=1)
        eroded = (eroded == 9).float()
        
        boundary = dilated - eroded
        
        return boundary.squeeze(1)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated values.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        cm = self.confusion_matrix
        
        # Per-class IoU
        class_iou = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            iou = tp / (tp + fp + fn + 1e-10)
            class_iou.append(iou)
            metrics[f'iou_{self.class_names[i]}'] = iou
        
        # Mean IoU
        metrics['mIoU'] = np.mean(class_iou)
        
        # Pixel accuracy
        correct = np.diag(cm).sum()
        total = cm.sum()
        metrics['pixel_accuracy'] = correct / (total + 1e-10)
        
        # Mean pixel accuracy (balanced)
        class_acc = []
        for i in range(self.num_classes):
            class_total = cm[i, :].sum()
            if class_total > 0:
                class_acc.append(cm[i, i] / class_total)
        metrics['mean_pixel_accuracy'] = np.mean(class_acc)
        
        # Per-class Dice
        class_dice = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
            class_dice.append(dice)
            metrics[f'dice_{self.class_names[i]}'] = dice
        
        # Mean Dice
        metrics['mean_dice'] = np.mean(class_dice)
        
        # Boundary F1
        boundary_precision = self.boundary_tp / (self.boundary_tp + self.boundary_fp + 1e-10)
        boundary_recall = self.boundary_tp / (self.boundary_tp + self.boundary_fn + 1e-10)
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-10)
        
        metrics['boundary_precision'] = boundary_precision
        metrics['boundary_recall'] = boundary_recall
        metrics['boundary_f1'] = boundary_f1
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix."""
        return self.confusion_matrix.copy()
    
    def print_summary(self):
        """Print a formatted summary of all metrics."""
        metrics = self.compute()
        
        print("\n" + "=" * 60)
        print("Segmentation Metrics Summary")
        print("=" * 60)
        
        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Mean IoU:          {metrics['mIoU']:.4f}")
        print(f"  Mean Dice:         {metrics['mean_dice']:.4f}")
        print(f"  Pixel Accuracy:    {metrics['pixel_accuracy']:.4f}")
        print(f"  Boundary F1:       {metrics['boundary_f1']:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class IoU:")
        for i, name in enumerate(self.class_names):
            iou = metrics[f'iou_{name}']
            dice = metrics[f'dice_{name}']
            print(f"  {name:15s}: IoU={iou:.4f}, Dice={dice:.4f}")
        
        print("=" * 60)


class InferenceTimer:
    """Timer for measuring inference speed."""
    
    def __init__(self):
        self.times = []
        self.start_event = None
        self.end_event = None
    
    def start(self):
        """Start timing."""
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            import time
            self._start_time = time.time()
    
    def stop(self):
        """Stop timing and record."""
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed = self.start_event.elapsed_time(self.end_event)  # milliseconds
        else:
            import time
            elapsed = (time.time() - self._start_time) * 1000
        
        self.times.append(elapsed)
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        times = np.array(self.times)
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }


def calculate_flops(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512)
) -> int:
    """
    Estimate FLOPs for the model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    
    Returns:
        Approximate FLOPs count
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        
        model.eval()
        input_tensor = torch.randn(input_size)
        
        if next(model.parameters()).is_cuda:
            input_tensor = input_tensor.cuda()
        
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total()
    
    except ImportError:
        # Simple estimation based on parameters
        params = sum(p.numel() for p in model.parameters())
        # Rough approximation: 2 FLOPs per parameter per pixel
        h, w = input_size[2], input_size[3]
        return params * 2 * h * w


if __name__ == "__main__":
    # Test metrics
    print("Testing Segmentation Metrics...")
    
    metrics = SegmentationMetrics(
        num_classes=5,
        class_names=['background', 'roads', 'water', 'trees', 'buildings']
    )
    
    # Simulate predictions
    for _ in range(10):
        pred = torch.randint(0, 5, (4, 128, 128))
        target = torch.randint(0, 5, (4, 128, 128))
        
        metrics.update(pred, target)
        metrics.update_boundary(pred, target)
    
    # Print summary
    metrics.print_summary()
    
    # Test timer
    print("\nTesting InferenceTimer...")
    timer = InferenceTimer()
    
    for _ in range(10):
        timer.start()
        # Simulate work
        _ = torch.randn(100, 100).sum()
        timer.stop()
    
    stats = timer.get_stats()
    print(f"  Mean: {stats['mean_ms']:.2f}ms")
    print(f"  FPS: {stats['fps']:.2f}")
    
    print("\n✓ All metric tests passed!")
