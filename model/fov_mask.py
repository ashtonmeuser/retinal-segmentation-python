"""
Get boolean FOV mask
"""

from math import floor
import numpy as np
from image_utils import read_image, pad_image

class FovMask: # pylint: disable=R0903
    """
    Boolean mask of defining whether pixels are within FOV
    """
    def __init__(self, mask_path, k_size):
        pad = floor(k_size / 2)
        fov_mask = read_image(mask_path, greyscale=True)
        fov_mask = pad_image(fov_mask, pad, 0)
        self.mask = fov_mask.astype(np.bool)
