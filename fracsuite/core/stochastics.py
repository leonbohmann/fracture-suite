import numpy as np

def calculate_match_hist(splinters1, splinters2, binrange):
    """
    Calculates the matching percentage between two sets of splinters based on their area.

    Args:
        splinters1 (list): List of Splinter objects from the first set.
        splinters2 (list): List of Splinter objects from the second set.
        binrange (int): Number of bins to use for the histogram.

    Returns:
        float: The matching percentage between the two sets of splinters.
    """

    area1 = [x.area for x in splinters1]
    area2 = [x.area for x in splinters2]
    hist1, _ = np.histogram(area1, bins=binrange)
    hist2, _ = np.histogram(area2, bins=binrange)

    max_error = np.sum((np.maximum(hist1, hist2) - np.minimum(hist1, hist2)) ** 2)
    mean_sqr_err = np.mean((hist1 - hist2) ** 2)
    matching_percentage = 100 * (1 - mean_sqr_err / max_error)

    return matching_percentage