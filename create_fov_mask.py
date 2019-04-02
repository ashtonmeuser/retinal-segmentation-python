"""
Get boolean FOV mask
"""

from math import floor
import numpy as np
from image_utils import read_image, pad_image

def create_fov_mask(mask_path, k_size):
    """
    Return padded boolean mask
    """
    pad = floor(k_size / 2)
    fov_mask = read_image(mask_path, greyscale=True)
    fov_mask = pad_image(fov_mask, pad, 0)

    return fov_mask.astype(np.bool)
