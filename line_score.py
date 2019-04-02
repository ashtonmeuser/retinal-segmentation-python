"""
Line score feature and accompanying orthogonal line score
"""

import numpy as np

def line_score(neighborhood, mask_list):
    """
    Line and orthogonal scores
    """
    if not mask_list:
        raise ValueError('Must supply at least one line mask to calculate score')

    neighborhood = neighborhood[:, :, 0]
    scores = list()
    neighborhood_average = np.mean(neighborhood)

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
