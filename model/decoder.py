"""
Decoder Modules for SatFormerNet
================================

Contains:
- Dense Nested Skip Connections (Enhanced UNet++)
- Adaptive Feature Fusion (AFF) Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBNReLU(nn.Module):
    """Standard Conv + BatchNorm + ReLU block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """Channel attention module using squeeze-and-excitation."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class DenseSkipConnection(nn.Module):
    """
    Dense Nested Skip Connection Block.
    
    Implements enhanced UNet++ connections where each decoder node
    receives features from all preceding nodes at the same level.
    
    X^{i,j} = H([X^{i,0}, X^{i,1}, ..., X^{i,j-1}, Up(X^{i+1,j-1})])
    
    where H is a convolution with channel attention.
    
    Args:
        in_channels: List of input channel counts from each connection
        out_channels: Output channels
        use_attention: Whether to use CBAM attention
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        use_attention: bool = True
    ):
        super().__init__()
        
        total_in_channels = sum(in_channels)
        
        # Channel alignment convolutions for each input
        self.align_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        
        # Aggregation convolution
        self.agg_conv = nn.Sequential(
            ConvBNReLU(out_channels * len(in_channels), out_channels, 3, 1, 1),
            ConvBNReLU(out_channels, out_channels, 3, 1, 1)
        )
        
        # Attention
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        
        # Residual projection if needed
        self.residual = nn.Conv2d(in_channels[0], out_channels, 1) if in_channels[0] != out_channels else nn.Identity()
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps to combine
                     First feature is from the encoder (skip connection)
                     Others are from previous decoder columns
        """
        target_size = features[0].shape[-2:]
        
        # Align channels and sizes
        aligned = []
        for feat, conv in zip(features, self.align_convs):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(conv(feat))
        
        # Concatenate and aggregate
        combined = torch.cat(aligned, dim=1)
        out = self.agg_conv(combined)
        
        # Attention
        out = self.attention(out)
        
        # Residual connection
        residual = self.residual(features[0])
        if residual.shape[-2:] != out.shape[-2:]:
            residual = F.interpolate(residual, size=out.shape[-2:], mode='bilinear', align_corners=False)
        
        return out + residual


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion Module.
    
    Instead of simple concatenation, this module:
    1. Aligns features from different scales
    2. Learns semantic-aware fusion weights
    3. Applies residual refinement
    
    Args:
        channels_list: List of input channel counts
        out_channels: Output channels
    """
    
    def __init__(self, channels_list: List[int], out_channels: int):
        super().__init__()
        
        self.num_inputs = len(channels_list)
        
        # Channel alignment
        self.align_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) for c in channels_list
        ])
        
        # Learnable fusion weights
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * self.num_inputs, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, self.num_inputs, 1),
            nn.Softmax(dim=1)
        )
        
        # Refinement
        self.refine = nn.Sequential(
            ConvBNReLU(out_channels, out_channels, 3, 1, 1),
            ConvBNReLU(out_channels, out_channels, 3, 1, 1)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        
        # Align all features
        aligned = []
        for feat, conv in zip(features, self.align_convs):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(conv(feat))
        
        # Stack for weighted sum
        stacked = torch.stack(aligned, dim=1)  # B, N, C, H, W
        
        # Compute fusion weights
        concat = torch.cat(aligned, dim=1)  # B, N*C, H, W
        weights = self.weight_net(concat)  # B, N, 1, 1
        
        # Weighted fusion
        weights = weights.unsqueeze(2)  # B, N, 1, 1, 1
        fused = (stacked * weights).sum(dim=1)  # B, C, H, W
        
        # Refine
        out = self.refine(fused) + fused
        
        return out


class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling and feature fusion.
    
    Args:
        in_channels: Input channels from previous decoder stage
        skip_channels: Channels from encoder skip connection
        out_channels: Output channels
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, 
            kernel_size=2, stride=2
        )
        
        # Feature fusion
        self.fusion = AdaptiveFeatureFusion(
            [in_channels, skip_channels],
            out_channels
        )
        
        # Refinement with attention
        self.refine = nn.Sequential(
            ConvBNReLU(out_channels, out_channels, 3, 1, 1),
            ConvBNReLU(out_channels, out_channels, 3, 1, 1),
            CBAM(out_channels)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Ensure size matches
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        # Fuse
        fused = self.fusion([x, skip])
        
        # Refine
        out = self.refine(fused)
        
        return out


class AdaptiveFusionDecoder(nn.Module):
    """
    Full decoder with dense nested skip connections and adaptive fusion.
    
    Features a UNet++-like structure with:
    - Dense connections between all decoder nodes
    - Adaptive feature fusion instead of concatenation
    - Optional deep supervision outputs
    
    Args:
        encoder_channels: List of encoder output channels [96, 192, 384, 768]
        decoder_channels: List of decoder output channels [256, 128, 64, 32]
        num_classes: Number of segmentation classes
        deep_supervision: Whether to output intermediate predictions
    """
    
    def __init__(
        self,
        encoder_channels: List[int] = [96, 192, 384, 768],
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 5,
        deep_supervision: bool = True
    ):
        super().__init__()
        
        self.num_stages = len(encoder_channels)
        self.deep_supervision = deep_supervision
        
        # Bridge from encoder
        self.bridge = nn.Sequential(
            ConvBNReLU(encoder_channels[-1], decoder_channels[0], 3, 1, 1),
            ConvBNReLU(decoder_channels[0], decoder_channels[0], 3, 1, 1)
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(self.num_stages - 1):
            in_ch = decoder_channels[i]
            skip_ch = encoder_channels[-(i+2)]
            out_ch = decoder_channels[i+1]
            
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch)
            )
        
        # Dense skip connection storage for UNet++ style
        # X^{0,0} -> X^{0,1} -> X^{0,2} -> X^{0,3}
        # Each gets input from all previous columns
        self.dense_connections = nn.ModuleList()
        
        for j in range(1, self.num_stages):  # Decoder columns 1, 2, 3
            column_connections = nn.ModuleList()
            
            for i in range(self.num_stages - j):  # Rows
                # Input channels: encoder skip + all previous decoder outputs at this row
                in_channels_list = [encoder_channels[i]]  # Encoder skip
                for k in range(j):
                    in_channels_list.append(decoder_channels[min(k, len(decoder_channels)-1)])
                
                out_ch = decoder_channels[min(j, len(decoder_channels)-1)]
                
                column_connections.append(
                    DenseSkipConnection(in_channels_list, out_ch)
                )
            
            self.dense_connections.append(column_connections)
        
        # Final output
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        
        # Deep supervision heads
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_channels[i], num_classes, 1)
                for i in range(len(decoder_channels) - 1)
            ])
    
    def forward(
        self,
        encoder_features: List[torch.Tensor],
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: List of encoder features [E1, E2, E3, E4]
                             from shallow to deep
            return_features: If True, return (features, prediction) tuple
        
        Returns:
            If return_features=False: Segmentation prediction (and aux outputs if deep_supervision)
            If return_features=True: (features, prediction, aux_outputs) or (features, prediction)
        """
        # Reverse to process from deep to shallow
        encoder_features = encoder_features[::-1]
        
        # Bridge
        x = self.bridge(encoder_features[0])
        
        # Decoder
        decoder_outputs = [x]
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = encoder_features[i + 1]
            x = decoder_block(x, skip)
            decoder_outputs.append(x)
        
        # Get final features (before classification)
        target_size = encoder_features[-1].shape[-2:]
        features = decoder_outputs[-1]
        
        if features.shape[-2:] != target_size:
            features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
        
        # Final prediction
        out = self.final_conv(features)
        
        # Deep supervision
        if self.deep_supervision and self.training:
            aux_outputs = []
            for i, (feat, head) in enumerate(zip(decoder_outputs[:-1], self.aux_heads)):
                aux = head(feat)
                aux = F.interpolate(aux, size=target_size, mode='bilinear', align_corners=False)
                aux_outputs.append(aux)
            
            if return_features:
                return features, out, aux_outputs
            return out, aux_outputs
        
        if return_features:
            return features, out
        return out


if __name__ == "__main__":
    # Test decoder
    decoder = AdaptiveFusionDecoder(
        encoder_channels=[96, 192, 384, 768],
        decoder_channels=[256, 128, 64, 32],
        num_classes=5,
        deep_supervision=True
    )
    
    # Mock encoder features
    encoder_features = [
        torch.randn(2, 96, 128, 128),
        torch.randn(2, 192, 64, 64),
        torch.randn(2, 384, 32, 32),
        torch.randn(2, 768, 16, 16)
    ]
    
    decoder.train()
    out, aux = decoder(encoder_features)
    
    print("AdaptiveFusionDecoder Output:")
    print(f"  Main output: {out.shape}")
    for i, a in enumerate(aux):
        print(f"  Aux {i}: {a.shape}")
    
    decoder.eval()
    out = decoder(encoder_features)
    print(f"  Eval output: {out.shape}")
    
    params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder parameters: {params:,}")
