"""
Convolve image with supplied function
"""

from math import floor
import numpy as np
from image_utils import pad_image

def convolve(image, k_size, function, verbose=False):
    """
    Apply function to each pixel in image with a neighborhood size of k_size
    Returns numpy array of objects to be cast by caller
    """
    if k_size % 2 != 1 or k_size < 1:
        raise ValueError('Neighborhod size must be a positive odd number')

    pad = floor(k_size / 2)
    (image_height, image_width) = image.shape[:2]
    result = np.full((image_height, image_width), None, dtype=object) # To store resultant image
    # image_padded = pad_image(image, pad, 0) # Pad value will be masked out
    image_padded = image

    def get_neighborhood(image, x_index, y_index, pad):
        """
        Return neighborhood given x and y of center point, pad
        """
        return image[y_index - pad: y_index + 1 + pad, x_index - pad: x_index + 1 + pad]

    for y_index in np.arange(pad, image_height + pad):
        if verbose:
            print('progress: {:05.2f}%'.format((y_index - pad) / image_height * 100),
                  end='\r', flush=True)
        for x_index in np.arange(pad, image_width + pad):
            neighborhood = get_neighborhood(image_padded, x_index, y_index, pad)
            k = function(neighborhood)
            result[y_index - pad, x_index - pad] = k[0]

    return result
