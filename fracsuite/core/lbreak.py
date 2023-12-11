from enum import Enum
import os
from typing import Callable
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from fracsuite.core.plotting import FigureSize, get_fig_width, renew_ticks_cb
from fracsuite.general import GeneralSettings
from scipy.interpolate import interp2d, griddata


general = GeneralSettings.get()

class ModelBoundary(str, Enum):
    A = "A"
    B = "B"
    Z = "Z"

class ModeChoices(str,Enum):
    AREA = 'area'
    ORIENTATION = 'orientation'
    ROUNDNESS = 'roundness'
    ROUGHNESS = 'roughness'
    ASP = 'asp'
    ASP0 = 'asp0'
    L1 = 'l1'
    L2 = 'l2'

def get_model_folder():
    return os.path.join(general.out_path, "model")

def get_model_file(file_name):
    return os.path.join(get_model_folder(), file_name)

def load_model(file_name):
    """
    Loads a model from the model folder.

    Parameters
    ----------
        file_name : str
            The name of the model file.

    Returns
    -------
        R : ndarray
            The radius range.
        U : ndarray
            The elastic strain energy range.
        V : ndarray
            The interpolated values.
    """
    file_path = get_model_file(file_name)
    data = np.load(get_model_file(file_path))
    R = data[0,1:]
    U = data[1:,0]
    V = data[1:,1:]

    return R,U,V

def plt_model(R,U,V,filter_nan=False, xlabel="Radius", ylabel="Energy", clabel="~") -> Figure:
    """
    Plots the aspect ratio.

    Args:
        R (array): Radius range.
        U (array): Energy range.
        V (n-d-array): Values matching R[i] and U[j].
    """

    # create interpolatable data
    p = np.meshgrid(R, U, indexing='xy')
    points = np.vstack([p[0].ravel(), p[1].ravel()]).T

    # coordinates for interpolation
    X = np.linspace(np.min(R), np.max(R), 100)
    Y = np.linspace(np.min(U), np.max(U), 100)
    X,Y = np.meshgrid(X,Y)

    # Filter both points and nr_r arrays to exclude NaNs
    if filter_nan:
        raveled_nr = V.ravel()
        non_nan_mask = ~np.isnan(raveled_nr)
        filtered_points = points[non_nan_mask]
        filtered_nr_r = raveled_nr[non_nan_mask]
    else:
        filtered_points = points
        filtered_nr_r = V.ravel()

    Z = griddata(filtered_points, filtered_nr_r, (X, Y), method='linear')

    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    cmesh = axs.pcolormesh(X,Y,Z,shading='auto',cmap='turbo')
    cbar = fig.colorbar(cmesh, label=clabel, ax=axs)
    renew_ticks_cb(cbar)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.autoscale()
    axs.set_xlim((0, R[-3]))
    return fig

def get_asp(U: float, boundary: ModelBoundary) -> Callable[[float], float]:
    """
    Returns the aspect ratio function for a given elastic strain energy and a boundary mode.

    Args:
        U (float): Elastic strain energy.
        boundary (ModelBoundary): Boundary mode.

    Returns:
        Callable[[float], float]: A function that takes the radius and returns the aspect ratio.
    """
    R,U,V = load_model(f'interpolate_asp_{boundary}_corner.npy')

    # create an interpolation function for the given U
    p = np.meshgrid(R, U, indexing='xy')
    points = np.vstack([p[0].ravel(), p[1].ravel()]).T
    values = V.ravel()
    f = interp2d(*points.T, values, kind='linear')

    def asp(r: float) -> float:
        return f(r, U)[0]

    return asp

def get_l1(U: float, boundary: ModelBoundary) -> Callable[[float], float]:
    """
    Returns the aspect ratio function for a given elastic strain energy and a boundary mode.

    Args:
        U (float): Elastic strain energy.
        boundary (ModelBoundary): Boundary mode.

    Returns:
        Callable[[float], float]: A function that takes the radius and returns the aspect ratio.
    """
    R,U,V = load_model(f'interpolate_l1_{boundary}_corner.npy')

    # create an interpolation function for the given U
    p = np.meshgrid(R, U, indexing='xy')
    points = np.vstack([p[0].ravel(), p[1].ravel()]).T
    values = V.ravel()
    f = interp2d(*points.T, values, kind='linear')

    def l1(r: float) -> float:
        return f(r, U)[0]

    return l1
