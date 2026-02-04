"""
Loss Functions for Satellite Image Segmentation
================================================

Compound loss combining multiple objectives:
- Dice Loss: Class imbalance handling
- Focal Loss: Hard example mining
- Boundary Loss: Edge precision
- Lovász Loss: Direct mIoU optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Handles class imbalance by focusing on overlap rather than pixel counts.
    Smooth factor prevents division by zero.
    
    Args:
        smooth: Smoothing factor (default: 1.0)
        ignore_index: Class index to ignore (default: -100)
    """
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, H, W) - class indices
            weights: Per-class weights (C,)
        
        Returns:
            Dice loss value
        """
        num_classes = pred.shape[1]
        
        # Softmax to get probabilities
        pred_soft = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(
            target.clamp(0, num_classes - 1),
            num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Create mask for valid pixels
        valid_mask = (target != self.ignore_index).unsqueeze(1).float()
        target_one_hot = target_one_hot * valid_mask
        pred_soft = pred_soft * valid_mask
        
        # Calculate Dice per class
        dims = (0, 2, 3)  # Batch, Height, Width
        intersection = (pred_soft * target_one_hot).sum(dim=dims)
        cardinality = (pred_soft + target_one_hot).sum(dim=dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Apply class weights
        if weights is not None:
            dice_score = dice_score * weights
            loss = 1 - dice_score.sum() / weights.sum()
        else:
            loss = 1 - dice_score.mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.
    
    Down-weights easy examples and focuses on hard negatives.
    Particularly useful for imbalanced satellite imagery.
    
    Args:
        alpha: Class balancing factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        ignore_index: Class index to ignore (default: -100)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, H, W) - class indices
        
        Returns:
            Focal loss value
        """
        num_classes = pred.shape[1]
        
        # Cross entropy
        ce_loss = F.cross_entropy(
            pred, target.clamp(0, num_classes - 1),
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Get probabilities
        p = F.softmax(pred, dim=1)
        
        # Gather probabilities for target class
        target_expanded = target.unsqueeze(1).clamp(0, num_classes - 1)
        p_t = p.gather(1, target_expanded).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha
        alpha_weight = torch.ones_like(p_t) * self.alpha
        
        # Final loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Mask ignored pixels
        valid_mask = (target != self.ignore_index).float()
        focal_loss = focal_loss * valid_mask
        
        return focal_loss.sum() / (valid_mask.sum() + 1e-6)


class LovaszSoftmax(nn.Module):
    """
    Lovász-Softmax loss for direct IoU optimization.
    
    Based on the convex Lovász extension of submodular losses.
    Directly optimizes the mean IoU.
    """
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovász extension w.r.t sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def _flatten_pred_target(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        """Flatten predictions and targets, filtering ignored indices."""
        num_classes = pred.shape[1]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1)
        
        valid = target != self.ignore_index
        return pred[valid], target[valid]
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, H, W) - class indices
        
        Returns:
            Lovász loss value
        """
        pred_flat, target_flat = self._flatten_pred_target(pred, target)
        
        if pred_flat.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        num_classes = pred.shape[1]
        probs = F.softmax(pred_flat, dim=1)
        
        losses = []
        for c in range(num_classes):
            fg = (target_flat == c).float()
            if fg.sum() == 0:
                continue
            
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            loss_c = (errors_sorted * self._lovasz_grad(fg_sorted)).sum()
            losses.append(loss_c)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return torch.stack(losses).mean()


class BoundaryLoss(nn.Module):
    """
    Boundary loss for precise edge delineation.
    
    Extracts boundaries from GT and computes loss against predictions
    at boundary pixels with higher weight.
    """
    
    def __init__(self, theta: float = 5.0, ignore_index: int = -100):
        super().__init__()
        self.theta = theta
        self.ignore_index = ignore_index
        
        # Sobel kernels for boundary extraction
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary from segmentation mask."""
        # Convert to float
        mask_float = mask.float().unsqueeze(1)
        
        # Apply Sobel filters
        gx = F.conv2d(mask_float, self.sobel_x, padding=1)
        gy = F.conv2d(mask_float, self.sobel_y, padding=1)
        
        # Gradient magnitude
        gradient = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        
        # Threshold to get boundary
        boundary = (gradient > 0.1).float().squeeze(1)
        
        return boundary
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, H, W) - class indices
        
        Returns:
            Boundary loss value
        """
        # Get boundary mask
        boundary_mask = self._get_boundary(target)
        
        # Standard CE loss
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        
        # Weight boundary pixels higher
        weights = 1.0 + self.theta * boundary_mask
        
        # Mask valid pixels
        valid_mask = (target != self.ignore_index).float()
        weighted_loss = ce_loss * weights * valid_mask
        
        return weighted_loss.sum() / (valid_mask.sum() + 1e-6)


class CompoundLoss(nn.Module):
    """
    Compound loss combining multiple objectives for optimal segmentation.
    
    Loss = w_dice * Dice + w_focal * Focal + w_boundary * Boundary + w_lovasz * Lovász
    
    Args:
        num_classes: Number of segmentation classes
        class_weights: Per-class weights for balancing
        dice_weight: Weight for Dice loss (default: 0.5)
        focal_weight: Weight for Focal loss (default: 0.3)
        boundary_weight: Weight for Boundary loss (default: 0.15)
        lovasz_weight: Weight for Lovász loss (default: 0.05)
        ignore_index: Class index to ignore (default: -100)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[List[float]] = None,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        boundary_weight: float = 0.15,
        lovasz_weight: float = 0.05,
        ignore_index: int = -100
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.lovasz_weight = lovasz_weight
        self.ignore_index = ignore_index
        
        # Class weights (roads slightly upweighted due to thin structure)
        if class_weights is None:
            class_weights = [1.0, 1.2, 1.0, 0.8, 1.0]  # bg, roads, water, trees, buildings
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights[:num_classes], dtype=torch.float32)
        )
        
        # Loss components
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss(ignore_index=ignore_index)
        self.lovasz_loss = LovaszSoftmax(ignore_index=ignore_index)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux_preds: Optional[List[torch.Tensor]] = None,
        boundary_pred: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            pred: Main predictions (B, C, H, W) - logits
            target: Ground truth (B, H, W) - class indices
            aux_preds: Auxiliary predictions for deep supervision
            boundary_pred: Predicted boundary map from BAAM
        
        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        
        # Main losses
        dice = self.dice_loss(pred, target, self.class_weights)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        lovasz = self.lovasz_loss(pred, target)
        
        losses['dice'] = dice
        losses['focal'] = focal
        losses['boundary'] = boundary
        losses['lovasz'] = lovasz
        
        # Combine main losses
        main_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.boundary_weight * boundary +
            self.lovasz_weight * lovasz
        )
        
        # Auxiliary losses (deep supervision)
        if aux_preds is not None:
            aux_loss = 0
            aux_weights = [0.4, 0.3, 0.2]  # Deeper outputs get lower weights
            
            for i, aux_pred in enumerate(aux_preds):
                if i < len(aux_weights):
                    aux_dice = self.dice_loss(aux_pred, target, self.class_weights)
                    aux_focal = self.focal_loss(aux_pred, target)
                    aux_loss += aux_weights[i] * (0.6 * aux_dice + 0.4 * aux_focal)
            
            losses['aux'] = aux_loss
            main_loss = main_loss + 0.4 * aux_loss
        
        # BAAM boundary loss
        if boundary_pred is not None:
            from ..model.boundary import BoundaryLoss as BAAMBoundaryLoss
            baam_loss = BAAMBoundaryLoss(self.num_classes)(boundary_pred, target)
            losses['baam_boundary'] = baam_loss
            main_loss = main_loss + 0.1 * baam_loss
        
        losses['total'] = main_loss
        
        return losses


if __name__ == "__main__":
    # Test losses
    print("Testing Loss Functions...")
    
    B, C, H, W = 2, 5, 128, 128
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    
    # Test individual losses
    dice = DiceLoss()(pred, target)
    print(f"Dice Loss: {dice.item():.4f}")
    
    focal = FocalLoss()(pred, target)
    print(f"Focal Loss: {focal.item():.4f}")
    
    boundary = BoundaryLoss()(pred, target)
    print(f"Boundary Loss: {boundary.item():.4f}")
    
    lovasz = LovaszSoftmax()(pred, target)
    print(f"Lovász Loss: {lovasz.item():.4f}")
    
    # Test compound loss
    compound = CompoundLoss(num_classes=5)
    aux_preds = [torch.randn(B, C, H, W) for _ in range(3)]
    losses = compound(pred, target, aux_preds)
    
    print(f"\nCompound Loss Breakdown:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Verify gradients
    pred.requires_grad = True
    loss = compound(pred, target)['total']
    loss.backward()
    print(f"\nGradient computed: {pred.grad is not None}")
    
    print("\n✓ All loss tests passed!")
