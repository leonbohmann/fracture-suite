from rich import print

import numpy as np
from scipy.stats import ks_2samp, ttest_ind, chisquare, gaussian_kde

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

def similarity_lberror(
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
    """
    Correlation coefficient between two variables, between -1 and 1,
    where 1 is total positive linear correlation, 0 is no linear correlation.

    Remarks:
        - This is used to compare linear relationships.
    """
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
    """
    Assesses how well the relationship between two variables can be described
    using a monotonic function.

    Value between -1 and 1, where 1 is total positive monotonic correlation,
    0 is no monotonic correlation, and -1 is total negative monotonic correlation.

    Remarks:
        - This is used to compare monotonic relationships, not linear relationships.
    """
    n = len(x)
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))

    d = rank_x - rank_y
    d_squared = d**2
    sum_d_squared = np.sum(d_squared)

    return 100 * (1 - (6 * sum_d_squared) / (n * (n**2 - 1)))
def mse(reference, measure):
    return np.mean((reference - measure) ** 2)

def nrmse(reference, measure):
    return np.sqrt(mse(reference, measure)) / np.mean(reference)

def detercoeff(reference, measure):
    return (1 - mse(reference, measure) / np.var(reference)) * 100

def smape(reference, measure):
    return 100 * np.mean(np.abs(reference - measure) / (np.abs(reference) + np.abs(measure))/2)
def mape(reference, measure):
    ref = reference.copy()
    ref[reference == 0] = 1
    return 100 * (1-np.mean(np.abs(reference - measure) / ref))

def mae(reference, measure):
    return np.mean(np.abs(reference - measure))



def jaccard(x, y):
    jaccard_index = np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))
    return jaccard_index * 100

def similarity(reference, measure, binrange=100, no_print=False):
    reference = np.histogram(reference, bins=binrange)[0]
    measure = np.histogram(measure, bins=binrange)[0]

    r_pearson = pearson_correlation(reference, measure)
    r_spearman = spearman_correlation(reference, measure)
    lb_error = similarity_lberror(reference, measure, binrange)
    ks = similarity_ks(reference, measure, binrange)
    jac = jaccard(reference,measure)
    mse_val = mse(reference, measure)
    nrmse_val = nrmse(reference, measure)
    detercoeff_val = detercoeff(reference, measure)
    smape_val = smape(reference, measure)
    mape_val = mape(reference, measure)
    count = similarity_count(reference, measure)
    mae_val = mae(reference, measure)
    if not no_print:
        print(f"Pearson:            {r_pearson:>7.2f} %")
        print(f"Spearman:           {r_spearman:>7.2f} %")
        print(f"LB Error:           {lb_error:>7.2f} %")
        print(f"MSE:                {mse_val:>7.2f} #²")
        print(f"NRMSE:              {nrmse_val:>7.2f} %")
        print(f"Deter-Coeff:        {detercoeff_val:>7.2f} %")
        print(f"SMAPE:              {smape_val:>7.2f} %")
        print(f"MAPE:               {mape_val:>7.2f} %")
        print(f"Jaccard:            {jac:>7.2f} %")
        print(f"KS:                 {ks:>7.2f} %")
        print(f"Count:              {count:>7.2f} %")
        print(f"MAE:                {mae_val:>7.2f} #²")

    return r_pearson, r_spearman, lb_error, ks, count

def similarity_count(reference,measure,binrange=100) -> float:
    reference = np.histogram(reference, bins=binrange)[0]
    measure = np.histogram(measure, bins=binrange)[0]
    ref = reference.copy()
    ref[reference == 0] = 1

    # delta
    d = np.abs(reference - measure)
    # relative error to reference
    err = d / ref
    # mean error
    err = np.mean(err)

    return 100 * (1 - err)


def calculate_dmode(data, bins: int = 1000):
    """
    Finds the most probable value of a distribution by calculating the mode.
    Applies a Gaussian KDE to the data and finds the maximum value of the KDE.

    Args:
        data (list): Data to find the most probable value of.
        bins (int, optional): Number of bins to use for the histogram. Defaults to 100.

    Returns:
        float: The most probable value of the distribution.
    """
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), bins)
    y_vals = kde(x_vals)
    return x_vals[np.argmax(y_vals)]