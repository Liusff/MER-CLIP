# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseHead
from .feature_head import FeatureHead
from .transformer_head import TransfClsHead
from .transformer_head_vit import TransfClsHead_Vit

__all__ = [
    'BaseHead', 'FeatureHead', 'TransfClsHead', 'TransfClsHead_Vit'
]
