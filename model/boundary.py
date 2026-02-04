"""
Boundary-Aware Attention Module (BAAM)
======================================

Critical for satellite imagery where precise boundary delineation matters
for roads, buildings, and water bodies.

This module:
1. Detects edges using learnable Sobel-like filters
2. Refines predictions at class boundaries through cross-attention
3. Models class-boundary correlations for improved accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LearnableEdgeDetector(nn.Module):
    """
    Learnable edge detection using gradient-based filters.
    
    Initializes with Sobel filters but allows learning task-specific
    edge detection patterns.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # Initialize with Sobel filters
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
        
        # Learnable edge filters per channel
        self.conv_x = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.conv_y = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        
        # Initialize with Sobel
        self.conv_x.weight.data = sobel_x.repeat(in_channels, 1, 1, 1)
        self.conv_y.weight.data = sobel_y.repeat(in_channels, 1, 1, 1)
        
        # Reduce to single edge map
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns edge probability map.
        
        Args:
            x: Feature map (B, C, H, W)
        
        Returns:
            Edge probability map (B, 1, H, W)
        """
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        
        # Combine gradients
        edges = torch.cat([edge_x, edge_y], dim=1)
        edge_map = self.reduce(edges)
        
        return edge_map


class EdgeGuidedAttention(nn.Module):
    """
    Cross-attention between edge features and semantic features.
    
    Edge features guide where to pay attention in semantic features,
    improving boundary precision.
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Edge -> Query
        self.edge_proj = nn.Linear(channels, channels)
        
        # Semantic -> Key, Value
        self.semantic_k = nn.Linear(channels, channels)
        self.semantic_v = nn.Linear(channels, channels)
        
        # Output projection
        self.out_proj = nn.Linear(channels, channels)
        
        # Edge-semantic fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        edge_features: torch.Tensor,
        semantic_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            edge_features: Edge-enhanced features (B, C, H, W)
            semantic_features: Semantic features (B, C, H, W)
        
        Returns:
            Boundary-refined features (B, C, H, W)
        """
        B, C, H, W = semantic_features.shape
        
        # Flatten spatial dimensions
        edge_flat = edge_features.flatten(2).transpose(1, 2)  # B, HW, C
        semantic_flat = semantic_features.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Projections
        q = self.edge_proj(edge_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.semantic_k(semantic_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.semantic_v(semantic_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # Fuse with original semantic features
        fused = self.fusion(torch.cat([out, semantic_features], dim=1))
        
        return fused + semantic_features


class ClassBoundaryCorrelation(nn.Module):
    """
    Models correlations between class predictions and boundaries.
    
    Different classes have different boundary characteristics:
    - Roads: thin, elongated boundaries
    - Buildings: rectangular, sharp boundaries
    - Water: smooth, irregular boundaries
    - Trees: complex, fractal-like boundaries
    """
    
    def __init__(self, num_classes: int, channels: int):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Per-class boundary attention
        self.class_boundary_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 3, padding=1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])
        
        # Combine class-specific boundaries
        self.boundary_fusion = nn.Sequential(
            nn.Conv2d(num_classes, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        features: torch.Tensor,
        edge_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns refined boundary map considering class-specific patterns.
        
        Args:
            features: Semantic features (B, C, H, W)
            edge_map: Initial edge map (B, 1, H, W)
        
        Returns:
            Refined boundary map (B, 1, H, W)
        """
        class_boundaries = []
        
        for class_conv in self.class_boundary_conv:
            class_boundary = class_conv(features)
            # Combine with general edge map
            class_boundary = class_boundary * edge_map
            class_boundaries.append(class_boundary)
        
        # Stack class boundaries
        stacked = torch.cat(class_boundaries, dim=1)  # B, num_classes, H, W
        
        # Fuse into final boundary map
        refined_boundary = self.boundary_fusion(stacked)
        
        return refined_boundary


class BoundaryAwareAttention(nn.Module):
    """
    Boundary-Aware Attention Module (BAAM).
    
    Complete module for boundary-aware feature refinement:
    1. Detects edges using learnable filters
    2. Cross-attention between edge and semantic features
    3. Class-specific boundary refinement
    4. Boundary-guided feature enhancement
    
    Args:
        in_channels: Input feature channels
        num_classes: Number of segmentation classes
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 5,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Edge detection
        self.edge_detector = LearnableEdgeDetector(in_channels)
        
        # Edge feature extraction
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Edge-guided attention
        self.edge_attention = EdgeGuidedAttention(in_channels, num_heads)
        
        # Class-boundary correlation
        self.class_boundary = ClassBoundaryCorrelation(num_classes, in_channels)
        
        # Boundary-guided refinement
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_boundary: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input features (B, C, H, W)
            return_boundary: Whether to return boundary map for loss
        
        Returns:
            Refined features (B, C, H, W)
            Boundary map (B, 1, H, W) if return_boundary=True
        """
        # Detect edges
        edge_map = self.edge_detector(x)
        
        # Create edge-enhanced features
        edge_features = self.edge_conv(torch.cat([x, edge_map], dim=1))
        
        # Edge-guided attention
        attended = self.edge_attention(edge_features, x)
        
        # Class-specific boundary refinement
        refined_boundary = self.class_boundary(attended, edge_map)
        
        # Boundary-guided refinement
        boundary_features = self.boundary_refine(
            torch.cat([attended, refined_boundary], dim=1)
        )
        
        # Fuse with original features
        output = self.output_conv(torch.cat([x, boundary_features], dim=1))
        
        if return_boundary:
            return output, refined_boundary
        return output, None


class BoundaryLoss(nn.Module):
    """
    Boundary loss for training the boundary-aware module.
    
    Combines:
    1. BCE loss between predicted and GT boundary
    2. Dice loss for boundary overlap
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    @staticmethod
    def get_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Extract boundary from segmentation mask."""
        # Dilate - Erode to get boundary
        padding = kernel_size // 2
        
        # One-hot encode if needed
        if mask.dim() == 3:
            mask = mask.unsqueeze(1).float()
        
        # Apply morphological operations
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
        
        boundary = dilated - eroded
        boundary = (boundary > 0).float()
        
        return boundary
    
    def forward(
        self,
        pred_boundary: torch.Tensor,
        gt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_boundary: Predicted boundary (B, 1, H, W)
            gt_mask: Ground truth segmentation (B, H, W) or (B, num_classes, H, W)
        
        Returns:
            Boundary loss
        """
        # Get GT boundary
        if gt_mask.dim() == 4:
            # Multi-class, get union of all boundaries
            gt_boundary = torch.zeros_like(pred_boundary)
            for c in range(gt_mask.shape[1]):
                class_boundary = self.get_boundary(gt_mask[:, c:c+1])
                gt_boundary = torch.maximum(gt_boundary, class_boundary)
        else:
            gt_boundary = self.get_boundary(gt_mask.unsqueeze(1).float())
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(pred_boundary, gt_boundary)
        
        # Dice loss
        intersection = (pred_boundary * gt_boundary).sum()
        union = pred_boundary.sum() + gt_boundary.sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        
        return bce_loss + dice_loss


if __name__ == "__main__":
    # Test BAAM
    baam = BoundaryAwareAttention(in_channels=64, num_classes=5, num_heads=8)
    
    x = torch.randn(2, 64, 128, 128)
    out, boundary = baam(x, return_boundary=True)
    
    print("BoundaryAwareAttention Output:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Boundary: {boundary.shape}")
    
    # Test boundary loss
    loss_fn = BoundaryLoss(num_classes=5)
    gt_mask = torch.randint(0, 5, (2, 128, 128))
    loss = loss_fn(boundary, gt_mask)
    print(f"  Boundary Loss: {loss.item():.4f}")
    
    params = sum(p.numel() for p in baam.parameters())
    print(f"\nBAAM parameters: {params:,}")
