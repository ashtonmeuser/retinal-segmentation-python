"""
Convolve image with supplied function
"""

from math import floor
import cv2
import numpy as np

def convolve(img, k_size, function, verbose=False):
    """
    Apply function to each pixel in image with a neighborhood size of k_size
    Returns numpy array of objects to be cast by caller
    """
    if k_size % 2 != 1 or k_size < 1:
        raise ValueError('Neighborhod size must be a positive odd number')

    pad = floor(k_size / 2)
    (img_height, img_width) = img.shape[:2]
    result = np.full((img_height, img_width), None, dtype=object) # To store resultant image
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    def get_neighborhood(img, x_index, y_index, pad):
        """
        Return neighborhood given x and y of center point, pad
        """
        return img[y_index - pad: y_index + 1 + pad, x_index - pad: x_index + 1 + pad]

    for y_index in np.arange(pad, img_height + pad):
        if verbose:
            print('progress: {:05.2f}%'.format((y_index - pad) / img_height * 100),
                  end='\r', flush=True)
        for x_index in np.arange(pad, img_width + pad):
            neighborhood = get_neighborhood(img_padded, x_index, y_index, pad)
            k = function(neighborhood)
            result[y_index - pad, x_index - pad] = k[0]

    return result
