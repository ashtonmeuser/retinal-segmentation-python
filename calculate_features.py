"""
Calculate feature vectors for an image
"""

import numpy as np
from convolve import convolve
from image_utils import as_inverse_green
from line_score import line_score
from normalize_features import normalize_features

def calculate_features(image, fov_mask, mask_list, k_size):
    """
    Calculate line score, orthogonal line score, and inverse green-channel intensity vectors
    """
    inverse_green = as_inverse_green(image) # Line detector applied to inverse green
    function = lambda x, y: line_score(x, y, mask_list) # Function to apply to each neighborhood
    result = convolve(inverse_green, k_size, function, fov_mask, 2)
    vectors = np.dstack((result, inverse_green)) # Union of all feature vectors
    vectors = normalize_features(vectors)

    return vectors
