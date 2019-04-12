"""
Defines class with binary line masks
"""

import logging
import numpy as np
import image_utils

class ImageCollection: # pylint: disable=R0903
    """
    Reads images from DRIVE database
    Contains full-color image, boolean FOV mask, and boolean ground truth
    """
    def __init__(self, image_number):
        if isinstance(image_number, int):
            image_number = '{:02d}'.format(image_number) # Leading zeros used in DRIVE database

        logging.info('Reading image, mask, truth %s from database', image_number)

        self.image = image_utils.read_image(f'DRIVE/image/{image_number}_test.tif')
        self.truth = image_utils.read_image(f'DRIVE/truth/{image_number}_test_truth.tif',
                                            greyscale=True).astype(np.bool)
        self.fov_mask = image_utils.read_image(f'DRIVE/mask/{image_number}_test_mask.tif',
                                               greyscale=True)
