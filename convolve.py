"""
Convolve image with supplied function
"""

from math import floor
import numpy as np
from matrix_utils import pad_matrix
from log_execution import log_execution

@log_execution
def convolve(image, k_size, function, mask, features): # pylint: disable=R0914
    """
    Apply function to each pixel in image with a neighborhood size of k_size
    Returns numpy array of floats of depth features
    """
    if k_size % 2 != 1 or k_size < 1:
        raise ValueError('Neighborhod size must be a positive odd number')
    if not isinstance(1, int) or features < 1:
        raise ValueError('Must extract at least one feature')

    pad = floor(k_size / 2)
    (image_height, image_width) = image.shape
    result = np.full((image_height, image_width, features), None, dtype=np.float) # Store result
    image_padded = pad_matrix(image, pad, 0) # Pad value will be masked out
    mask_padded = pad_matrix(mask, pad, False) # Pad value will be masked out

    def get_neighborhood(image, x_index, y_index, pad):
        """
        Return neighborhood given x and y of center point, pad
        """
        return image[y_index - pad: y_index + 1 + pad, x_index - pad: x_index + 1 + pad]

    for y_index in np.arange(pad, image_height + pad):
        for x_index in np.arange(pad, image_width + pad):
            neighborhood = get_neighborhood(image_padded, x_index, y_index, pad)
            mask_neighborhood = get_neighborhood(mask_padded, x_index, y_index, pad)
            result[y_index - pad, x_index - pad] = function(neighborhood, mask_neighborhood)

    return result
