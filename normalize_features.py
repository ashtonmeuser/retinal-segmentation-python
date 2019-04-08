"""
Normalize feature vectors
"""

import numpy as np

def normalize_features(vectors):
    """
    Take image features, normalize
    """
    means = np.mean(vectors, axis=(0, 1))
    deviations = np.std(vectors, axis=(0, 1))

    return (vectors - means) / deviations
