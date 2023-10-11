from rich import print

import numpy as np
from scipy.stats import ks_2samp, ttest_ind, chisquare

def to_cdf(data):
    """
    Converts a list of data to a CDF.

    Args:
        data (list): Data to convert.

    Returns:
        list: The CDF of the data.
    """
    data = sorted(data)
    cdf = np.cumsum(data)
    cdf = cdf / cdf[-1]

    return cdf

def similarity_count(splinters1, splinters2):
    max = np.max([len(splinters1), len(splinters2)])
    min = np.min([len(splinters1), len(splinters2)])

    return 100.0 * min / max

def similarity_ks(reference, measure, binrange):
    """
    Calculates the matching percentage between two sets of splinters based on their area using the KS test.

    Args:
        reference (list): Reference data.
        measure (list): Measured data.

    Returns:
        float: The matching percentage between the two sets of splinters.
    """


    area1 = sorted(reference)
    area2 = sorted(measure)

    area1 = np.histogram(area1, bins=binrange)[0]
    area2 = np.histogram(area2, bins=binrange)[0]

    _, p_value = ks_2samp(area1, area2, mode='asymp')
    matching_percentage = 100 * p_value


    return matching_percentage

def similarity_mse(
    reference,
    measure,
    binrange
):
    """
    Calculates the matching percentage between two sets of splinters based on their area.

    Args:
        reference (list): Reference data.
        measure (list): Measured data.
        binrange (int): Number of bins to use for the histogram.

    Returns:
        float: The matching percentage between the two sets of splinters.
    """
    reference = np.asarray(reference)
    measure = np.asarray(measure)

    # mean square error
    mse = np.mean((reference - measure) ** 2)
    max_error1 = np.mean((reference - np.mean(reference)) ** 2)
    max_error2 = np.mean((measure - np.mean(measure)) ** 2)
    max_mse = max_error1 + max_error2
    similarity = (1 - mse / max_mse) * 100
    return similarity

def pearson_correlation(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_sq = np.sum(x**2)
    sum_y_sq = np.sum(y**2)
    sum_xy = np.sum(x * y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))

    return 100 * (numerator / denominator)

def spearman_correlation(x, y):
    n = len(x)
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))

    d = rank_x - rank_y
    d_squared = d**2
    sum_d_squared = np.sum(d_squared)

    return 100 * (1 - (6 * sum_d_squared) / (n * (n**2 - 1)))

def similarity(a, b, binrange=100):
    count = similarity_count(a, b)

    a = np.histogram(a, bins=binrange)[0]
    b = np.histogram(b, bins=binrange)[0]

    r_pearson = pearson_correlation(a, b)
    r_spearman = spearman_correlation(a, b)
    mean_se = similarity_mse(a, b, binrange)
    ks = similarity_ks(a, b, binrange)

    print(f"Pearson:            {r_pearson:>7.2f}%")
    print(f"Spearman:           {r_spearman:>7.2f}%")
    print(f"Mean squared error: {mean_se:>7.2f}%")
    print(f"KS:                 {ks:>7.2f}%")
    print(f"Count:              {count:>7.2f}%")

    return r_pearson, r_spearman, mean_se, ks, count