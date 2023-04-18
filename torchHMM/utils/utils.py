import numpy as np


def total_variance_dist(array1, array2):
    return (0.5 * np.abs(array1 - array2).sum(axis=0)).mean()
