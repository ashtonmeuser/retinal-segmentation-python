"""
Line score feature and accompanying orthogonal line score
"""

from math import floor
import numpy as np

def line_score(neighborhood, fov_mask, mask_list):
    """
    Line and orthogonal scores
    """
    center = floor(fov_mask.shape[0] / 2)

    if not mask_list:
        raise ValueError('Must supply at least one line mask to calculate score')
    if not fov_mask[center][center]:
        return np.array([0.0, 0.0]) # Center pixel outside of mask

    scores = list()
    neighborhood_average = np.mean(neighborhood[fov_mask])
    neighborhood[fov_mask is False] = neighborhood_average

    def score_array(line_average, orthogonal_average):
        """
        Numpy array of line and orthogonal scores, prevent negative scores
        """
        return np.array([
            max(line_average - neighborhood_average, 0.0),
            max(orthogonal_average - neighborhood_average, 0.0)
        ])

    for line_mask in mask_list:
        line_average = np.mean(neighborhood[line_mask.mask])
        orthogonal_average = np.mean(neighborhood[line_mask.orthogonal_mask])
        scores.append(score_array(line_average, orthogonal_average))

    return max(scores, key=lambda x: x[0])
