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
    def __init__(self, image_number, database):
        if isinstance(image_number, int):
            image_number = '{:02d}'.format(image_number) # Leading zeros used in DRIVE database

        logging.debug('Reading image, mask, truth %s from database', image_number)

        self.image = image_utils.read_image('{}/image/{}.tif'.format(database, image_number))
        self.truth = image_utils.read_image('{}/truth/{}.tif'.format(database, image_number),
                                            greyscale=True).astype(np.bool)
        self.fov_mask = image_utils.read_image('{}/mask/{}.tif'.format(database, image_number),
                                               greyscale=True).astype(np.bool)
