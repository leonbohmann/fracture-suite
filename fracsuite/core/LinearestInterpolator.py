from typing import Union

import numpy as np
import scipy.interpolate


class LinearestInterpolator:

    def __init__(self, points: np.ndarray, values: np.ndarray):
        """ Use ND-linear interpolation over the convex hull of points, and nearest neighbor outside (for
            extrapolation)

            Idea taken from https://stackoverflow.com/questions/20516762/extrapolate-with-linearndinterpolator
        """
        self.linear_interpolator = scipy.interpolate.LinearNDInterpolator(points, values)
        self.nearest_neighbor_interpolator = scipy.interpolate.NearestNDInterpolator(points, values)

    def __call__(self, *args) -> Union[float, np.ndarray]:
        t = self.linear_interpolator(*args)

        if np.isnan(t).any():
            t[np.isnan(t)] = self.nearest_neighbor_interpolator(*args)[np.isnan(t)]

        if t.size == 1:
            return t.item(0)
        return t
