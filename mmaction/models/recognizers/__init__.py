# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseRecognizer
from .recognizer2d import Recognizer2D
from .recognizer3d import Recognizer3D

__all__ = [
    'BaseRecognizer', 'Recognizer2D', 'Recognizer3D'
]
