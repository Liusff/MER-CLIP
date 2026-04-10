# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.model import BaseDataPreprocessor, stack_batch

from mmaction.registry import MODELS
from mmaction.utils import SampleList
import numpy as np
import random

def blend_region(img1, img2, region_mask, alpha=0.5):
    """Blends two images together in specific regions based on the mask."""
    blended_image = img1.clone()
    blended_image[region_mask] = img1[region_mask] * alpha + img2[region_mask] * (1 - alpha)
    return blended_image

def generate_region_mask(height, width, region="forehead"):
    """Generates a mask for a specific region of the face."""
    mask = torch.zeros((height, width), dtype=torch.bool)
    
    # Example for generating region masks
    if region == "forehead":
        mask[:height//4, :] = True  # Top 1/4 of the image (forehead area)
    elif region == "chin":
        mask[3*height//4:, :] = True  # Bottom 1/4 of the image (chin area)
    elif region == "left_cheek":
        mask[height//4:3*height//4, :width//2] = True  # Left half of the middle region (left cheek)
    elif region == "right_cheek":
        mask[height//4:3*height//4, width//2:] = True  # Right half of the middle region (right cheek)
    
    return mask

def augment_sequence_with_local_static_face(batch, alpha=0.5, mix_prob=0.5, mix_type=0, regions = ['forehead', 'chin', 'left_cheek', 'right_cheek']):
    """
    Augments a batch of image sequences by blending frames in specific regions with other sequence's first frames.
    
    Args:
        batch: Tensor of shape (batch_size, sequence_length, channels, height, width).
        alpha: Blending factor.
        mix_prob: Probability of mixing other sequence's first frame.
        region: The region of the face to blend (e.g., 'forehead', 'chin', 'left_cheek', 'right_cheek').
        
    Returns:
        Augmented batch.
    """

    batch_size, sequence_length, channels, height, width = batch.shape
    augmented_batch = batch.clone()

    # Iterate over each sequence in the batch
    for i in range(batch_size):
        if np.random.rand() < mix_prob:
            # Randomly select another sequence's first frame to blend
            j = np.random.randint(0, batch_size)
            #if j == i:
            #    j = (i+1) % batch_size
            static_face = batch[j, 0]  # First frame of sequence j
            selected_region = random.choice(regions)
            region_mask = generate_region_mask(height, width, selected_region)
            if mix_type == 1:
                region_mask = (region_mask == False)
            
            # Blend the static face with each frame of sequence i in the specified region
            for k in range(sequence_length):
                for c in range(channels):  # Blend each channel separately
                    augmented_batch[i, k, c] = blend_region(batch[i, k, c], static_face[c], region_mask, alpha)
    
    return augmented_batch


@MODELS.register_module()
class ActionDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for action recognition tasks.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        to_float32 (bool): Whether to convert data to float32.
            Defaults to True.
        blending (dict, optional): Config for batch blending.
            Defaults to None.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 to_rgb: bool = False,
                 to_float32: bool = True,
                 blending: Optional[dict] = None,
                 mix: int = -1,
                 alpha: float = 0.5,
                 mix_prob: float = 0.5,
                 format_shape: str = 'NCHW') -> None:
        super().__init__()
        self.to_rgb = to_rgb
        self.to_float32 = to_float32
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape in ['NCTHW', 'MIX2d3d']:
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(normalizer_shape),
                False)
        else:
            self._enable_normalize = False

        if blending is not None:
            self.blending = MODELS.build(blending)
        else:
            self.blending = None
        
        self.mix = mix
        self.alpha = alpha
        self.mix_prob = mix_prob

    def forward(self,
                data: Union[dict, Tuple[dict]],
                training: bool = False) -> Union[dict, Tuple[dict]]:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict or Tuple[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or Tuple[dict]: Data in the same format as the model input.
        """
        
        data = self.cast_data(data)
        if isinstance(data, dict):
            return self.forward_onesample(data, training=training)
        elif isinstance(data, (tuple, list)):
            outputs = []
            for data_sample in data:
                output = self.forward_onesample(data_sample, training=training)
                outputs.append(output)
            return tuple(outputs)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}!')

    def forward_onesample(self, data, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation on one data sample.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs, data_samples = data['inputs'], data['data_samples']
        inputs, data_samples = self.preprocess(inputs, data_samples, training)
        data['inputs'] = inputs
        data['data_samples'] = data_samples
        #print(data_samples)
        return data

    def preprocess(self,
                   inputs: List[torch.Tensor],
                   data_samples: SampleList,
                   training: bool = False) -> Tuple:
        # --- Pad and stack --
        batch_inputs = stack_batch(inputs)

        if self.format_shape == 'MIX2d3d':
            if batch_inputs.ndim == 4:
                format_shape, view_shape = 'NCHW', (-1, 1, 1)
            else:
                format_shape, view_shape = 'NCTHW', None
        else:
            format_shape, view_shape = self.format_shape, None

        # ------ To RGB ------
        if self.to_rgb:
            if format_shape == 'NCHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
            elif format_shape == 'NCTHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

        # -- Normalization ---
        if self._enable_normalize:
            if view_shape is None:
                batch_inputs = (batch_inputs - self.mean) / self.std
            else:
                mean = self.mean.view(view_shape)
                std = self.std.view(view_shape)
                batch_inputs = (batch_inputs - mean) / std
        elif self.to_float32:
            batch_inputs = batch_inputs.to(torch.float32)
        
        # ----- LocalStaticFaceMixing -----
        if training and self.mix >= 0:
            #print(batch_inputs.shape)
            #batch_inputs = batch_inputs.to(torch.float32)
            batch_inputs = batch_inputs.squeeze(1).permute(0,2,1,3,4)  #N T C H W
            batch_inputs = augment_sequence_with_local_static_face(batch_inputs, alpha=self.alpha, mix_prob=self.mix_prob, mix_type=self.mix)
            batch_inputs = batch_inputs.permute(0,2,1,3,4).unsqueeze(1)  #N C T H W
            #print("after: ", batch_inputs.shape)
        
        # ----- Blending -----
        if training and self.blending is not None:
            batch_inputs, data_samples = self.blending(batch_inputs,
                                                       data_samples)

        return batch_inputs, data_samples
