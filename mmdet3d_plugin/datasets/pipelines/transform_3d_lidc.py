# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
import copy
import inspect
from operator import methodcaller
import torch
import cv2
from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter, ObjectNameFilter

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None
from PIL import Image


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImageXray:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. random Gamma Correction
    4. randon Gaussian noise
    5. random Gaussian blur
    6. random Salt and pepper noise
    7. random vignetting effect
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        
    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        scale_factors = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,' \
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta,
                                          self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # Gamma correction
            # if np.random.randint(2):
            #     gamma = np.random.uniform(0.8, 1.2)
            #     img = img ** gamma

            # Gaussian noise
            if np.random.randint(2):
                noise_std = np.random.uniform(2, 10)
                noise = np.random.normal(0, noise_std, img.shape)
                img = np.clip(img + noise, 0, 255)

            #Gaussian blur
            if np.random.randint(2):
                kernel_size = np.random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

            #Salt and pepper noise
            if np.random.randint(2):
                salt_pepper_ratio = 0.02
                num_salt = int(salt_pepper_ratio * img.size)
                coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
                img[coords[0], coords[1]] = 255  # Salt noise (white pixels)
    
                num_pepper = int(salt_pepper_ratio * img.size)
                coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
                img[coords[0], coords[1]] = 0  # Pepper noise (black pixels)

            # Vignetting effect
        
            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            img32 = img.astype(np.float32)
            new_imgs.append(img32)
            scale_factors.append(1.0)
        results['img'] = new_imgs
        results['scale_factor'] = scale_factors
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        return repr_str
