"""
Convolve image with supplied function
"""

from math import floor
import cv2
import numpy as np

def convolve(img, k_size, function):
    """
    Apply function to each pixel in image with a neighborhood size of k_size
    """
    if k_size % 2 != 1 or k_size < 1:
        raise ValueError('Neighborhod size must be a positive odd number')

    pad = floor(k_size / 2)
    (img_height, img_width) = img.shape[:2]
    result = np.zeros((img_height, img_width), dtype=np.uint8) # To store resultant image
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    def get_neighborhood(img, p_x, p_y, pad):
        """
        Return neighborhood given x and y of center point, pad
        """
        return img[p_y - pad: p_y + 1 + pad, p_x - pad: p_x + 1 + pad]

    for q_y in np.arange(pad, img_height + pad):
        for q_x in np.arange(pad, img_width + pad):
            neighborhood = get_neighborhood(img_padded, q_x, q_y, pad)
            k = function(neighborhood)
            result[q_y - pad, q_x - pad] = k

    return result
