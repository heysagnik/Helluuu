"""
Attention Modules for SatFormerNet
==================================

Contains:
- Multi-Scale Deformable Attention (MSDA): Adaptive sampling across scales
- Semantic Context Aggregation Module (SCAM): Global-local context fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module.
    
    Unlike fixed atrous convolutions in DeepLabV3+, this module:
    - Learns adaptive sampling locations per pixel
    - Aggregates features from multiple scales
    - Handles varying object sizes (small roads vs large buildings)
    
    Args:
        embed_dim: Input embedding dimension
        num_heads: Number of attention heads
        num_levels: Number of feature map scales
        num_points: Number of sampling points per head per level
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4
    ):
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        
        # Sampling offsets: predict 2D offset for each head, level, point
        self.sampling_offsets = nn.Linear(
            embed_dim, 
            num_heads * num_levels * num_points * 2
        )
        
        # Attention weights: predict weight for each head, level, point
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points
        )
        
        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize sampling offsets to spread around reference point
        nn.init.zeros_(self.sampling_offsets.weight)
        
        # Initialize grid-based sampling pattern
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: (B, N, C) query features
            reference_points: (B, N, num_levels, 2) normalized reference points
            input_flatten: (B, sum(H*W), C) flattened multi-scale features
            spatial_shapes: (num_levels, 2) spatial shape of each level
            level_start_index: (num_levels,) start index of each level
        
        Returns:
            Output features (B, N, C)
        """
        B, N, C = query.shape
        _, L, _ = input_flatten.shape
        
        # Project values
        value = self.value_proj(input_flatten)
        value = value.view(B, L, self.num_heads, self.head_dim)
        
        # Predict sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            B, N, self.num_heads, self.num_levels, self.num_points, 2
        )
        
        # Predict attention weights and normalize
        attention_weights = self.attention_weights(query).view(
            B, N, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            B, N, self.num_heads, self.num_levels, self.num_points
        )
        
        # Sample and aggregate
        output = self._sample_and_aggregate(
            value, spatial_shapes, level_start_index,
            sampling_offsets, attention_weights, reference_points
        )
        
        output = self.output_proj(output)
        return output
    
    def _sample_and_aggregate(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        sampling_offsets: torch.Tensor,
        attention_weights: torch.Tensor,
        reference_points: torch.Tensor
    ) -> torch.Tensor:
        """Sample features at offset locations and aggregate."""
        B, N, num_heads, num_levels, num_points, _ = sampling_offsets.shape
        
        # Compute sampling locations
        offset_normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + \
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
        # Sample from each level
        output = torch.zeros(B, N, num_heads, self.head_dim, device=value.device)
        
        for level_idx in range(num_levels):
            H_i, W_i = spatial_shapes[level_idx]
            start_idx = level_start_index[level_idx]
            end_idx = start_idx + H_i * W_i
            
            # Get value for this level
            value_l = value[:, start_idx:end_idx, :, :].view(B, H_i, W_i, num_heads, self.head_dim)
            value_l = value_l.permute(0, 3, 4, 1, 2)  # B, heads, head_dim, H, W
            
            # Get sampling locations for this level
            sampling_locations_l = sampling_locations[:, :, :, level_idx, :, :]  # B, N, heads, points, 2
            
            # Normalize to [-1, 1] for grid_sample
            sampling_locations_l = 2 * sampling_locations_l - 1
            
            # Sample
            for point_idx in range(num_points):
                grid = sampling_locations_l[:, :, :, point_idx, :].view(B * num_heads, N, 1, 2)
                value_l_flat = value_l.reshape(B * num_heads, self.head_dim, H_i, W_i)
                
                sampled = F.grid_sample(
                    value_l_flat, grid,
                    mode='bilinear', padding_mode='zeros', align_corners=False
                )
                sampled = sampled.view(B, num_heads, self.head_dim, N).permute(0, 3, 1, 2)
                
                # Weight and accumulate
                weight = attention_weights[:, :, :, level_idx, point_idx:point_idx+1]
                output = output + sampled * weight
        
        output = output.view(B, N, -1)
        return output


class StripPooling(nn.Module):
    """
    Strip Pooling for capturing long-range dependencies.
    
    Particularly effective for elongated structures like roads and rivers.
    """
    
    def __init__(self, in_channels: int, pool_size: int = 20):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((pool_size, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, pool_size))
        
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_w = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Horizontal strip pooling
        x_h = self.pool_h(x)
        x_h = self.conv_h(x_h)
        x_h = F.interpolate(x_h, size=(H, W), mode='bilinear', align_corners=False)
        
        # Vertical strip pooling
        x_w = self.pool_w(x)
        x_w = self.conv_w(x_w)
        x_w = F.interpolate(x_w, size=(H, W), mode='bilinear', align_corners=False)
        
        return x_h + x_w


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    Multi-scale context extraction with dilated convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 256, rates: List[int] = [6, 12, 18]):
        super().__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolutions
        self.dilated_convs = nn.ModuleList()
        for rate in rates:
            self.dilated_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        num_branches = 2 + len(rates)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # All branches
        features = [self.conv1(x)]
        
        for conv in self.dilated_convs:
            features.append(conv(x))
        
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        features.append(global_feat)
        
        # Concatenate and fuse
        x = torch.cat(features, dim=1)
        x = self.fusion(x)
        
        return x


class SemanticContextAggregation(nn.Module):
    """
    Semantic Context Aggregation Module (SCAM).
    
    Combines multiple context extraction methods:
    1. Global Average Pooling (scene-level)
    2. Strip Pooling (elongated structures)
    3. ASPP (multi-scale local)
    4. Learnable fusion with attention
    
    Args:
        in_channels: Input channels
        out_channels: Output channels (default: same as input)
        reduction: Channel reduction ratio for efficiency
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        reduction: int = 4
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = max(in_channels // reduction, 64)
        
        # Branch 1: Global context
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1)
        )
        
        # Branch 2: Strip pooling for elongated structures
        self.strip_branch = StripPooling(in_channels)
        self.strip_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # Branch 3: ASPP for multi-scale local context
        self.aspp_branch = ASPPModule(in_channels, out_channels, rates=[3, 6, 12, 18])
        
        # Branch 4: Local refinement
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention for adaptive weighting
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Global branch
        global_out = self.global_branch(x)
        global_out = F.interpolate(global_out, size=(H, W), mode='bilinear', align_corners=False)
        
        # Strip pooling branch
        strip_out = self.strip_branch(x)
        strip_out = self.strip_conv(strip_out)
        
        # ASPP branch
        aspp_out = self.aspp_branch(x)
        
        # Local branch
        local_out = self.local_branch(x)
        
        # Weighted combination
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted_features = [
            weights[0] * global_out,
            weights[1] * strip_out,
            weights[2] * aspp_out,
            weights[3] * local_out
        ]
        
        # Concatenate and fuse
        combined = torch.cat(weighted_features, dim=1)
        out = self.fusion(combined)
        
        # Apply channel attention
        attention = self.channel_attention(out)
        out = out * attention
        
        return out


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding."""
    
    def __init__(self, channels: int, height: int = 128, width: int = 128):
        super().__init__()
        
        if channels % 4 != 0:
            raise ValueError(f"channels ({channels}) must be divisible by 4")
        
        pe = torch.zeros(channels, height, width)
        
        d_model = channels // 2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_h = torch.arange(height).unsqueeze(1)
        pos_w = torch.arange(width).unsqueeze(1)
        
        pe[0:d_model:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[1:d_model:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pe = F.interpolate(self.pe, size=(H, W), mode='bilinear', align_corners=False)
        return x + pe.expand(B, -1, -1, -1)


if __name__ == "__main__":
    # Test SCAM
    scam = SemanticContextAggregation(768, 256)
    x = torch.randn(2, 768, 16, 16)
    out = scam(x)
    print(f"SCAM Input: {x.shape}, Output: {out.shape}")
    
    # Test MSDA (simplified test)
    msda = MultiScaleDeformableAttention(embed_dim=256, num_heads=8, num_levels=4, num_points=4)
    query = torch.randn(2, 100, 256)
    
    # Create dummy multi-scale inputs
    spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]])
    level_start_index = torch.tensor([0, 1024, 1024 + 256, 1024 + 256 + 64])
    reference_points = torch.rand(2, 100, 4, 2)
    input_flatten = torch.randn(2, 1024 + 256 + 64 + 16, 256)
    
    out = msda(query, reference_points, input_flatten, spatial_shapes, level_start_index)
    print(f"MSDA Input: {query.shape}, Output: {out.shape}")
    
    params = sum(p.numel() for p in scam.parameters()) + sum(p.numel() for p in msda.parameters())
    print(f"\nTotal attention module parameters: {params:,}")
