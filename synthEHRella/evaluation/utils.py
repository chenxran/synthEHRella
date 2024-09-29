import numpy as np


def mean_ignore_nan_inf(arr):
    # Replace Inf and -Inf with NaN
    arr = np.where(np.isinf(arr), np.nan, arr)
    # Calculate the mean, ignoring NaN
    return np.nanmean(arr)