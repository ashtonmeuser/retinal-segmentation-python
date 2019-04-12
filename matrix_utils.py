"""
Common functions perfromed on matrices
"""

import cv2
import numpy as np

def pad_matrix(matrix, pad, value):
    """
    Apply a border of value and width pad around matrix
    """
    def pad_integers(matrix, pad, value):
        """
        Common integer matrix padding
        """
        return cv2.copyMakeBorder(matrix, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)

    if isinstance(value, np.bool):
        matrix = matrix.astype(np.uint8)
        padded = pad_integers(matrix.astype(np.uint8), pad, 0)
        return padded.astype(np.bool)

    return pad_integers(matrix, pad, value)
