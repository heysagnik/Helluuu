"""
SatFormerNet - Novel Satellite Image Segmentation Architecture
===============================================================

A state-of-the-art architecture designed for satellite imagery segmentation,
combining hierarchical vision transformers with convolutional networks,
multi-scale deformable attention, and boundary-aware refinement.

Target Classes:
    0: Background
    1: Roads
    2: Water Bodies
    3: Trees
    4: Buildings
"""

from .satformernet import SatFormerNet
from .backbone import HierarchicalBackbone
from .attention import MultiScaleDeformableAttention, SemanticContextAggregation
from .decoder import AdaptiveFusionDecoder, DenseSkipConnection
from .boundary import BoundaryAwareAttention

__all__ = [
    'SatFormerNet',
    'HierarchicalBackbone',
    'MultiScaleDeformableAttention',
    'SemanticContextAggregation',
    'AdaptiveFusionDecoder',
    'DenseSkipConnection',
    'BoundaryAwareAttention',
]

__version__ = '1.0.0'
