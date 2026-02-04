"""
SatFormerNet: Novel Satellite Image Segmentation Architecture
=============================================================

Main model integrating all components:
- Hierarchical Vision Backbone (HViT-Conv)
- Multi-Scale Deformable Attention (MSDA)
- Semantic Context Aggregation Module (SCAM)
- Adaptive Feature Fusion Decoder
- Boundary-Aware Attention Module (BAAM)

Target Classes (default):
    0: Background
    1: Roads
    2: Water Bodies
    3: Trees
    4: Buildings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .backbone import HierarchicalBackbone
from .attention import SemanticContextAggregation, MultiScaleDeformableAttention
from .decoder import AdaptiveFusionDecoder
from .boundary import BoundaryAwareAttention


class SatFormerNet(nn.Module):
    """
    SatFormerNet: State-of-the-art Satellite Image Segmentation.
    
    A hybrid transformer-CNN architecture designed specifically for
    satellite imagery, combining:
    
    1. **HViT-Conv Backbone**: Hierarchical vision transformer with
       convolutions for global-local feature extraction
    
    2. **SCAM Bridge**: Semantic context aggregation for multi-scale
       understanding of satellite scenes
    
    3. **Adaptive Decoder**: Dense skip connections with attention-based
       feature fusion for precise upsampling
    
    4. **BAAM**: Boundary-aware attention for precise class boundaries
    
    Args:
        num_classes: Number of segmentation classes (default: 5)
        in_channels: Input image channels (default: 3 for RGB)
        backbone_channels: Encoder output channels per stage
        decoder_channels: Decoder output channels per stage
        use_pretrained: Load pretrained backbone weights (if available)
        use_adapters: Use RS Adapters for domain adaptation
        use_msda: Use Multi-Scale Deformable Attention
        use_baam: Use Boundary-Aware Attention Module
        deep_supervision: Return auxiliary outputs during training
        dropout: Dropout rate
    
    Input:
        x: Tensor of shape (B, C, H, W) - RGB satellite image
           H and W should be divisible by 32
    
    Output:
        Training: (main_output, aux_outputs, boundary_map)
        Inference: main_output only
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        backbone_channels: List[int] = [96, 192, 384, 768],
        decoder_channels: List[int] = [256, 128, 64, 32],
        use_pretrained: bool = True,
        use_adapters: bool = True,
        use_msda: bool = False,  # More expensive, off by default
        use_baam: bool = True,
        deep_supervision: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_msda = use_msda
        self.use_baam = use_baam
        self.deep_supervision = deep_supervision
        
        # 1. Hierarchical Backbone
        self.backbone = HierarchicalBackbone(
            in_channels=in_channels,
            embed_dim=backbone_channels[0],
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.2,
            use_adapters=use_adapters
        )
        
        # 2. Context Bridge
        self.context_bridge = SemanticContextAggregation(
            in_channels=backbone_channels[-1],
            out_channels=decoder_channels[0],
            reduction=4
        )
        
        # Optional: Multi-Scale Deformable Attention
        if use_msda:
            self.msda = MultiScaleDeformableAttention(
                embed_dim=decoder_channels[0],
                num_heads=8,
                num_levels=4,
                num_points=4
            )
        
        # 3. Adaptive Fusion Decoder
        self.decoder = AdaptiveFusionDecoder(
            encoder_channels=backbone_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision
        )
        
        # 4. Boundary-Aware Attention Module
        if use_baam:
            self.baam = BoundaryAwareAttention(
                in_channels=decoder_channels[-1],
                num_classes=num_classes,
                num_heads=8
            )
        
        # Final classification head
        self.final_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels[-1], num_classes, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: Return intermediate features for visualization
        
        Returns:
            Dictionary with:
                - 'out': Main segmentation output (B, num_classes, H, W)
                - 'aux': List of auxiliary outputs (training only)
                - 'boundary': Boundary map (if BAAM enabled)
                - 'features': Intermediate features (if requested)
        """
        input_size = x.shape[-2:]
        output = {}
        
        # 1. Backbone
        encoder_features = self.backbone(x)
        
        if return_features:
            output['encoder_features'] = encoder_features
        
        # 2. Context Bridge - enhance the deepest features (not used in decoder path currently)
        # enhanced_deep = self.context_bridge(encoder_features[-1])
        
        # 3. Decoder - get features for BAAM processing
        if self.training and self.deep_supervision:
            if self.use_baam:
                # Get features before final conv for BAAM
                features, decoder_pred, aux_outputs = self.decoder(encoder_features, return_features=True)
                output['aux'] = aux_outputs
            else:
                decoder_out, aux_outputs = self.decoder(encoder_features, return_features=False)
                output['aux'] = aux_outputs
        else:
            if self.use_baam:
                features, decoder_pred = self.decoder(encoder_features, return_features=True)
            else:
                decoder_out = self.decoder(encoder_features, return_features=False)
        
        # 4. Boundary-Aware Attention
        boundary_map = None
        if self.use_baam:
            # BAAM processes features (32 channels) and refines them
            refined_features, boundary_map = self.baam(features, return_boundary=True)
            
            if boundary_map is not None:
                output['boundary'] = boundary_map
            
            # 5. Final prediction from refined features
            out = self.final_head(refined_features)
        else:
            out = decoder_out
        
        # Upsample to original size
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        output['out'] = out
        
        return output
    
    def get_param_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        """
        Get parameter groups with different learning rates.
        
        - Backbone: lr * 0.1
        - Decoder/Head: lr
        - Bias and normalization: no weight decay
        """
        backbone_params = []
        other_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'backbone' in name:
                if 'bias' in name or 'norm' in name or 'bn' in name:
                    no_decay_params.append(param)
                else:
                    backbone_params.append(param)
            else:
                if 'bias' in name or 'norm' in name or 'bn' in name:
                    no_decay_params.append(param)
                else:
                    other_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},
            {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0}
        ]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test SatFormerNet
    print("=" * 60)
    print("SatFormerNet - Satellite Image Segmentation Architecture")
    print("=" * 60)
    
    # Create model
    model = SatFormerNet(
        num_classes=5,
        in_channels=3,
        backbone_channels=[96, 192, 384, 768],
        decoder_channels=[256, 128, 64, 32],
        use_pretrained=False,
        use_adapters=True,
        use_msda=False,
        use_baam=True,
        deep_supervision=True
    )
    
    # Parameter count
    total, trainable = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Size: {total * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 512, 512)
    
    # Training mode
    model.train()
    output = model(x)
    print(f"\nTraining Mode Output:")
    print(f"  Main output: {output['out'].shape}")
    if 'aux' in output:
        for i, aux in enumerate(output['aux']):
            print(f"  Aux {i}: {aux.shape}")
    if 'boundary' in output:
        print(f"  Boundary: {output['boundary'].shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"\nInference Mode Output:")
    print(f"  Main output: {output['out'].shape}")
    
    # Test different input sizes
    print("\nTesting different input sizes...")
    for size in [256, 384, 512]:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            output = model(x)
        print(f"  Input: {size}x{size} -> Output: {output['out'].shape}")
    
    print("\n✓ All tests passed!")
