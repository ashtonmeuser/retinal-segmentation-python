"""
Line score feature and accompanying orthogonal line score
"""

from math import floor
import warnings
import numpy as np

def line_score(neighborhood, fov_mask, mask_list):
    """
    Line and orthogonal scores
    """
    center = floor(fov_mask.shape[0] / 2)

    if not mask_list:
        raise ValueError('Must supply at least one line mask to calculate score')
    if not fov_mask[center][center]:
        return [0.0, 0.0] # Center pixel outside of mask

    scores = list()
    with warnings.catch_warnings(): # Expect warnings for mean of empty slice
        warnings.filterwarnings('error')
        try:
            neighborhood_average = np.mean(neighborhood[fov_mask])
            neighborhood[fov_mask is False] = neighborhood_average
        except RuntimeWarning:
            return (0.0, 0.0) # Entire neighborhood outside of mask

    def score(average):
        """
        Calculate score from line average, prevent negative scores
        """
        return max(average - neighborhood_average, 0.0) # Prevent negative scores

    for line_mask in mask_list:
        line_average = np.mean(neighborhood[line_mask.mask])
        orthogonal_average = np.mean(neighborhood[line_mask.orthogonal_mask])
        scores.append((score(line_average), score(orthogonal_average)))

    return max(scores, key=lambda x: x[0])
