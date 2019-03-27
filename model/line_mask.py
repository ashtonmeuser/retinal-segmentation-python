"""
Defines class with binary line masks
"""

import math
import numpy as np

def radians_to_degrees(rads):
    """
    Convert radians to degrees
    """
    return rads * math.pi / 180.0

class LineMask: # pylint: disable=R0903
    """
    Binary mask of line at angle around center
    """
    def __init__(self, k_size, angle, orthogonal_length=3):
        if orthogonal_length % 2 != 1 or orthogonal_length < 1:
            raise ValueError('Orthogonal line mask length must be a positive odd number')

        angle = angle % 180.0 # Repeats after 180 degrees
        acute = angle % 90.0
        quarter_size = math.ceil(k_size / 2)
        quarter = np.zeros((quarter_size, quarter_size), dtype=np.bool)
        diagonal_difference = abs(45.0 - acute)
        rise = math.tan(radians_to_degrees(45.0 - diagonal_difference))

        # Only captures 0 - 45 degrees as x must be linear
        for i in range(0, quarter_size):
            quarter[quarter_size - round(rise * i) - 1, i] = 1

        mask = np.zeros((k_size, k_size), dtype=np.bool)
        mask[:quarter_size, quarter_size - 1:] = quarter # Q1
        mask[quarter_size - 1:, :quarter_size] = np.rot90(quarter, 2) # Q3

        # Account for angles greater than 45 degrees
        if 45.0 < angle <= 135.0:
            mask = np.rot90(np.fliplr(mask), -1)
        if angle > 90.0:
            mask = np.fliplr(mask)

        orthogonal_radius = math.floor(orthogonal_length / 2)
        center = mask[quarter_size - 1 - orthogonal_radius: quarter_size + orthogonal_radius,
                      quarter_size - 1 - orthogonal_radius: quarter_size + orthogonal_radius]

        self.mask = mask
        self.orthogonal_mask = np.rot90(center) # Orthogonal line of length
