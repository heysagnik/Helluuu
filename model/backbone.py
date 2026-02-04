"""
Hierarchical Vision Backbone (HViT-Conv)
========================================

A hybrid encoder combining:
- ConvNeXt-style stem for initial feature extraction
- Swin Transformer blocks with window attention
- Depthwise separable convolutions for spatial inductive bias
- Remote Sensing Adapters for domain adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps (B, C, H, W)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtStem(nn.Module):
    """
    ConvNeXt-style stem for initial feature extraction.
    Better than patch embedding for preserving fine details.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4),
            LayerNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class RSAdapter(nn.Module):
    """
    Remote Sensing Adapter - Lightweight domain adaptation module.
    
    Adapts pre-trained features to satellite imagery domain with
    minimal additional parameters (~2-5% overhead).
    """
    
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(dim // reduction, 16)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        # Learnable scale initialized near zero for stable training
        self.scale = nn.Parameter(torch.zeros(1) + 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.adapter(x)


class DropPath(nn.Module):
    """Stochastic depth for regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention.
    
    Computes attention within non-overlapping windows for efficiency.
    Supports both regular and shifted window partitioning.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution with SE attention.
    
    Provides local spatial processing and inductive bias for edge detection.
    """
    
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()
        
        # Squeeze-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pwconv(x)
        x = x * self.se(x)
        return x + residual


class HybridTransformerBlock(nn.Module):
    """
    Hybrid Transformer Block combining:
    - Window Self-Attention (global context)
    - Depthwise Conv (local spatial processing)
    - RS Adapter (domain adaptation)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_adapter: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Depthwise conv for local processing
        self.local_conv = DepthwiseSeparableConv(dim, kernel_size=7)
        
        # RS Adapter for domain adaptation
        self.adapter = RSAdapter(dim) if use_adapter else nn.Identity()
    
    def _window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, H, W, C = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // self.window_size, self.window_size, 
                   Wp // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size * self.window_size, C)
        return windows, Hp, Wp
    
    def _window_reverse(self, windows: torch.Tensor, Hp: int, Wp: int, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (Hp * Wp / self.window_size / self.window_size))
        x = windows.view(B, Hp // self.window_size, Wp // self.window_size,
                         self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Local conv branch (operates on B, C, H, W)
        local_out = self.local_conv(x)
        
        # Attention branch
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Window partition
        x_windows, Hp, Wp = self._window_partition(shifted_x)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Reverse window partition
        shifted_x = self._window_reverse(attn_windows, Hp, Wp, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = shortcut + self.drop_path(x)
        
        # MLP with adapter
        x_mlp = self.mlp(self.norm2(x))
        x_mlp = self.adapter(x_mlp)
        x = x + self.drop_path(x_mlp)
        
        # Combine attention and local conv
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = x + local_out
        
        return x


class PatchMerging(nn.Module):
    """Downsampling layer that merges 2x2 patches."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Pad if needed
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        return x


class HierarchicalBackbone(nn.Module):
    """
    Hierarchical Vision Backbone (HViT-Conv).
    
    A 4-stage encoder producing multi-scale features:
    - Stage 1: H/4 x W/4, 96 channels
    - Stage 2: H/8 x W/8, 192 channels
    - Stage 3: H/16 x W/16, 384 channels
    - Stage 4: H/32 x W/32, 768 channels
    
    Args:
        in_channels: Input image channels (3 for RGB)
        embed_dim: Base embedding dimension (96)
        depths: Number of blocks in each stage [2, 2, 6, 2]
        num_heads: Number of attention heads per stage [3, 6, 12, 24]
        window_size: Window size for attention
        mlp_ratio: MLP expansion ratio
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        use_adapters: Whether to use RS Adapters
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        use_adapters: bool = True
    ):
        super().__init__()
        self.num_stages = 4
        self.embed_dim = embed_dim
        self.depths = depths
        self.out_channels = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        
        # Stem
        self.stem = ConvNeXtStem(in_channels, embed_dim)
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(self.num_stages):
            dim = embed_dim * (2 ** i)
            
            # Transformer blocks for this stage
            stage_blocks = []
            for j in range(depths[i]):
                block_idx = sum(depths[:i]) + j
                shift_size = 0 if (j % 2 == 0) else window_size // 2
                
                stage_blocks.append(
                    HybridTransformerBlock(
                        dim=dim,
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[block_idx],
                        use_adapter=use_adapters
                    )
                )
            
            self.stages.append(nn.Sequential(*stage_blocks))
            
            # Downsample (except for last stage)
            if i < self.num_stages - 1:
                self.downsamples.append(PatchMerging(dim))
            else:
                self.downsamples.append(nn.Identity())
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            List of feature maps at each scale
        """
        features = []
        
        # Stem
        x = self.stem(x)
        
        # Stages
        for i in range(self.num_stages):
            x = self.stages[i](x)
            features.append(x)
            x = self.downsamples[i](x)
        
        return features
    
    def get_output_channels(self) -> List[int]:
        """Return output channels for each stage."""
        return self.out_channels


if __name__ == "__main__":
    # Test the backbone
    model = HierarchicalBackbone()
    x = torch.randn(2, 3, 512, 512)
    features = model(x)
    
    print("HierarchicalBackbone Output:")
    for i, f in enumerate(features):
        print(f"  Stage {i+1}: {f.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")
