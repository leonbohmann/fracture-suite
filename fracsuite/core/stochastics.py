import numpy as np
from scipy.stats import ks_2samp

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

def calculate_match_count(splinters1, splinters2):
    max = np.max([len(splinters1), len(splinters2)])
    min = np.min([len(splinters1), len(splinters2)])

    return 100.0 * min / max

def calculate_match_hist_ks(reference, measure, binrange):
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

def calculate_similarity_hist(
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
    reference, _ = np.histogram(reference, bins=binrange)
    measure, _ = np.histogram(measure, bins=binrange)

    # mean square error
    mse = np.mean((reference - measure) ** 2)
    max_error1 = np.mean((reference - np.mean(reference)) ** 2)
    max_error2 = np.mean((measure - np.mean(measure)) ** 2)
    max_mse = max_error1 + max_error2
    similarity = (1 - mse / max_mse) * 100
    return similarity

    # max_error = np.sum((np.maximum(hist1, hist2) - np.minimum(hist1, hist2)) ** 2)
    # mean_sqr_err = np.mean((hist2 - hist1) ** 2)
    # matching_percentage = 100 * (1 - mean_sqr_err / max_error)

    # return mean_sqr_err