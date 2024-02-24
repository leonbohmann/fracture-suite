from matplotlib import pyplot as plt
from rich import print

import numpy as np
from scipy.stats import ks_2samp, gaussian_kde
from scipy.spatial.distance import pdist, squareform
import tempfile

from spazial import khat_test, lhat_test, lhatc_test, poisson
from scipy.stats import chi2
from scipy.signal import argrelextrema
from fracsuite.core.logging import debug

from fracsuite.core.signal import smooth_hanning
from fracsuite.state import State

def r_squared_f(x, y_real, func, popt):
    y_fit = func(x, *popt)
    return r_squared(y_real, y_fit)

def r_squared(y_real, y_fit):
    residuals = y_real - y_fit
    ss_res = np.nansum(residuals**2)
    ss_tot = np.nansum((y_real - np.mean(y_real))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

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

    abserr = np.sum(np.absolute(reference - measure))

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

    return r_pearson, r_spearman, lb_error, ks, count, abserr

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


def calculate_kde(data, bins=1000):
    #filter nan values
    data = data[~np.isnan(data)]

    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), bins)
    y_vals = kde(x_vals)

    return x_vals, y_vals

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
    x_vals, y_vals = calculate_kde(data, bins)

    # calculate most probable value (mode)
    mpv = x_vals[np.argmax(y_vals)]

    # calculate probability of mode
    y_vals = y_vals / np.sum(y_vals)

    mpv_prob = y_vals[np.argmax(y_vals)]

    return mpv, mpv_prob


def distances(events: list[tuple[float,float]]):
    """
    Calculates the distances between events.
    """
    # create array for events
    events = np.asarray(events)

    # calculate distances between all events
    distances = pdist(events, 'euclidean')

    distances = squareform(distances)

    return distances

def moving_average(x, y, rng) -> tuple[np.ndarray, list[np.ndarray] | np.ndarray]:
    """
    Berechnet den gleitenden Durchschnitt eines Arrays. Werte, deren x-Wert NaN ist, werden ignoriert.

    Parameter:
        x : ndarray
            Stützwerte der y-Arrays.
        y : ndarray | list[ndarray]
            Ein oder mehrere y-Arrays.
        rng : linspace
            Stützstellen für Auswertung

    Rückgabe:
        tuple(ndarray, list[ndarray], list[ndarray]):
            Stützstellen des Durchschnitts, der Durchschnitt selbst und seine Standardabweichung.
    """
    if not isinstance(y, list) or not isinstance(y, np.ndarray):
        y = [y]
    x = np.asarray(x)
    # x_raw = x[np.isfinite(x)]

    # dr = np.max(x_raw) / w
    # r0 = np.min(x_raw)
    # n = int(np.ceil((np.max(x_raw) - np.min(x_raw)) / dr))
    # rs = [r0 + i * dr for i in range(n + 1)]

    # rs = np.linspace(np.min(x_raw), np.max(x_raw), rng)

    # create result structure for every input y
    n = len(rng)
    R = []
    DEV = []
    for i in range(len(y)):
        Ri = np.zeros(n)
        dev = np.zeros(n)
        R.append(Ri)
        DEV.append(dev)

    # calculate moving average
    for i in range(n):
        if i == n - 1:
            mask = (x >= rng[i]) & ~np.isnan(x)
        else:
            mask = (x >= rng[i]) & (x < rng[i + 1]) & ~np.isnan(x)

        # calculate average for every input y
        for ii, yi in enumerate(y):
            y_r = yi[mask]

            if mask.sum() > 0:
                R[ii][i] = np.mean(y_r)
                DEV[ii][i] = np.std(y_r)
            else:
                R[ii][i] = np.nan
                DEV[ii][i] = np.nan

    return rng, *R, *DEV

def quadrat_count(points, size, d, alpha=0.05):
    """
    Counts the number of points in a grid of quadrats.

    Args:
        points (list[tuple[float,float]]): List of points to count.
        size(tuple[int,int] | tuple[int,int,int,int]): Size of the grid (w,h) or (x1,x2,y1,y2).
        d (float): Size of the quadrats.

    Returns:
        tuple[float,float,float]: X2, degrees of freedom, critical value.
    """
    if len(size) == 2:
        x1, y1 = 0, 0
        x2, y2 = size
    elif len(size) == 4:
        x1, y1, x2, y2 = size
    else:
        raise Exception("Invalid size parameter. Must be a tuple of 2 (w,h) or 4 (x1,x2,y1,y2) elements.")


    # create array for events
    events = np.asarray(points)

    # calculate quadrat counts
    x = np.arange(x1, x2, d)
    y = np.arange(y1, y2, d)
    counts = np.zeros((len(y), len(x)))
    for point in events:
        x_index = int((point[0] - x1) // d)
        y_index = int((point[1] - y1) // d)

        counts[y_index, x_index] += 1

    ## from Gelfand
    # yhat = np.mean(counts)
    # yhatv = np.var(counts)
    # I = yhatv**2 / yhat

    n = len(points)
    area = (x2 - x1) * (y2 - y1)
    debug(f'Area: {area}')
    expected = n * (d**2) / area
    debug(f'Expected amount per Window: {expected}')
    debug(f'Mean actual amount per Window: {np.mean(counts)}')
    # X2 after pearson (papula)
    X2 = np.sum((counts - expected) ** 2 / expected)
    # dof
    dof = (len(x) - 1) * (len(y) - 1)

    ## Auswertung
    alpha = 0.05
    gamma = 1 - alpha

    # get critical value
    c = chi2.ppf(gamma, dof)

    return X2, dof, c

def square(x, a, b, c):
    return a * x**2 + b*x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b*x**2 + c*x + d

def rhc_minimum(data, x):
    """
    Find the smallest local minimum in a dataset.

    Remarks:
        The assumption is, that the centered L-Function is strictly decreasing until the
        minium indicating rhc is reached. After that, if there are second local minima, the
        first and smallest minimum is returned.

        Data will be smoothed to get rid of local noise.

    Arguments:
        data (ndarray): The data to search for a minimum.
        x (ndarray, optional): The x-axis values of the data. Defaults to None, only used when debugging with State.debug.

    Returns:
        int: The index of the first smallest local minimum. If no minimum is found, -1 is returned.
    """
    # smooth data to get rid of local noise
    data_s = smooth_hanning(data, window_len=len(data) // 3)


    # info(f"Data length: {len(data)}, Order: {len(data)//5}")
    # find local minima
    mins = argrelextrema(data_s, np.less, order=len(data)//5)
    mins = mins[0]
    if len(mins) == 1:
        first_min = mins[0]
    if len(mins) > 1:
        if data_s[mins[0]] > data_s[mins[1]]:
            first_min = mins[1]
        else:
            first_min = mins[0]
    # return -1 if nothing found
    elif len(mins) == 0:
        first_min = -1

    # generate debug output if enabled
    if 'show_rhc_minimum' in State.kwargs:
        from fracsuite.core.plotting import FigureSize, get_fig_width

        fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW2))
        axs.plot(x, data, label="Original")
        axs.plot(x, data_s, label="Gleit. Durchschnitt")
        axs.scatter(x[mins], data_s[mins], c='gray')
        axs.scatter(x[first_min], data_s[first_min], c='red', marker='o', facecolors='none', label="Minimum")
        axs.legend()
        file = tempfile.mktemp(".png", "rhcmin")
        fig.savefig(file)
        plt.show()
        plt.close(fig)

    return first_min

def khat(points, width, height, d, use_weights=True):
    return np.array(khat_test(points, width, height, d, use_weights))

def lhat(points, width, height, d, use_weights=True):
    return np.array(lhat_test(points, width, height, d, use_weights))

def lhatc(points, width, height, d, use_weights=True):
    return np.array(lhatc_test(points, width, height, d, use_weights))

def khat_xy(points, width, height, d, use_weights=True):
    p = khat(points, width, height, d, use_weights=use_weights)
    x = p[:,0]
    y = p[:,1]
    return x,y

def lhat_xy(points, width, height, d, use_weights=True):
    p = lhat(points, width, height, d, use_weights=use_weights)
    x = p[:,0]
    y = p[:,1]
    return x,y

def lhatc_xy(points, width, height, d, use_weights=True):
    p = lhatc(points, width, height, d, use_weights)
    x = p[:,0]
    y = p[:,1]
    return x,y

def pois(w,h,n):
    return np.array(poisson(w,h,n))