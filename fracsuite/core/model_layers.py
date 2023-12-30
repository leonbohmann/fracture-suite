from enum import Enum
import os
from typing import Callable
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from fracsuite.core.plotting import FigureSize, get_fig_width, renew_ticks_cb
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.general import GeneralSettings
from scipy.interpolate import interp2d, griddata
from fracsuite.core.specimen import SpecimenBoundary, SpecimenBreakPosition

general = GeneralSettings.get()

class ModelLayer(str, Enum):
    IMPACT = "impact-layer"
    BASE = "base-layer"

def get_layer_folder():
    return os.path.join(general.out_path, "layer")

def get_layer_file(file_name):
    return os.path.join(get_layer_folder(), file_name)

def load_layer(file_name):
    """
    Loads a layer from the layer folder.

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
    file_path = get_layer_file(file_name)
    return load_layer_file(file_path)

def load_layer_file(file_path):
    """Loads a layer from a file path."""
    data = np.load(file_path)
    R = data[0,1:]
    U = data[1:,0]
    V = data[1:,1:]

    return R,U,V

def interp_layer(
    layer_name: ModelLayer,
    mode: SplinterProp,
    boundary: SpecimenBoundary,
    break_pos: SpecimenBreakPosition,
    U: float
):
    # X: Distance from Impact
    # Y: Energy
    # V: Layer value
    # Layout:
    #     X1    X2      X3      X4
    # Y1  V11   V12     V13     V14
    # Y2  V21   V22     V23     V24
    # Y3  ...
    # Y4  ...
    X,Y,V = load_layer(f'{layer_name}_{mode}_{boundary}_{break_pos}.npy')
    print(f'{layer_name}_{mode}_{boundary}_{break_pos}.npy')

    # fill nans with linear interpolation
    nans = np.isnan(V)
    non_nans = ~nans
    interpolated_V = np.interp(np.flatnonzero(nans), np.flatnonzero(non_nans), V[non_nans])
    V[nans] = interpolated_V

    # print(X)
    # print(Y)
    # print(V)
    f = interp2d(X, Y, V, kind='linear')

    def r_func(r: float) -> float:
        return f(r, U)

    Xs,Ys,Vs = load_layer(f'{layer_name}-stddev_{mode}_{boundary}_{break_pos}.npy')

    # fill nans with linear interpolation
    nans_s = np.isnan(Vs)
    non_nans_s = ~nans_s
    interpolated_Vs = np.interp(np.flatnonzero(nans_s), np.flatnonzero(non_nans_s), Vs[non_nans_s])
    Vs[nans_s] = interpolated_Vs

    f_s = interp2d(Xs, Ys, Vs, kind='linear')

    def r_func_s(r: float) -> float:
        return f_s(r, U)

    return r_func, r_func_s

def interp_layer_stddev(
    layer_name: ModelLayer,
    mode: SplinterProp,
    boundary: SpecimenBoundary,
    break_pos: SpecimenBreakPosition,
    U: float):
    X,Y,V = load_layer(f'{layer_name}-stddev_{mode}_{boundary}_{break_pos}.npy')

    # create an interpolation function for the given U
    p = np.meshgrid(X, Y, indexing='xy')
    points = np.vstack([p[0].ravel(), p[1].ravel()]).T
    values = V.ravel()
    f = interp2d(*points.T, values, kind='linear')

    def r_func(r: float) -> float:
        return f(r, U)[0]

    return r_func

def interp_impact_layer(model_path, U):
    """
    Interpolates the impact layer for a given U.

    Args:
        model_path (str): Path to the model file.
        U (float): Elastic strain energy.

    Returns:
        Callable: A function that takes the radius and returns the model value.
    """
    X,Y,V = load_layer(model_path)

    # create an interpolation function for the given U
    p = np.meshgrid(X, Y, indexing='xy')
    points = np.vstack([p[0].ravel(), p[1].ravel()]).T
    values = V.ravel()
    f = interp2d(*points.T, values, kind='linear')

    def r_func(r: float) -> float:
        return f(r, U)[0]

    return r_func

def plt_layer(R,U,V,ignore_nan=False, xlabel="Radius", ylabel="Energy", clabel="~",interpolate=True) -> Figure:
    """
    Plots a given layer on a 2d contour plot.

    Args:
        R (array): Radius range.
        U (array): Energy range.
        V (n-d-array): Values matching R[i] and U[j].
        xlabel (str, optional): x-axis label. Defaults to "Radius".
        ylabel (str, optional): y-axis label. Defaults to "Energy".
        clabel (str, optional): Colorbar label. Defaults to "~".
        interpolate (bool, optional): Interpolate the data. Defaults to True.
        ignore_nan (bool, optional): Ignore NaN values. Defaults to False. If set to true,
            the NaN values are filtered out of the data and the plot is continuous.
    """

    # create interpolatable data
    p = np.meshgrid(R, U, indexing='xy')
    points = np.vstack([p[0].ravel(), p[1].ravel()]).T

    # coordinates for interpolation
    X = np.linspace(np.min(R), np.max(R), 100)
    Y = np.linspace(np.min(U), np.max(U), 100)
    X,Y = np.meshgrid(X,Y)

    # Filter both points and nr_r arrays to exclude NaNs
    if ignore_nan:
        raveled_nr = V.ravel()
        non_nan_mask = ~np.isnan(raveled_nr)
        filtered_points = points[non_nan_mask]
        filtered_nr_r = raveled_nr[non_nan_mask]
    else:
        filtered_points = points
        filtered_nr_r = V.ravel()

    Z = griddata(filtered_points, filtered_nr_r, (X, Y), method='nearest' if not interpolate else 'linear')


    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    cmesh = axs.pcolormesh(X,Y,Z,shading='auto',cmap='turbo')
    cbar = fig.colorbar(cmesh, label=clabel, ax=axs)
    renew_ticks_cb(cbar)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.autoscale()
    axs.set_xlim((0, R[-3]))
    return fig

def get_asp(U: float, boundary: SpecimenBoundary) -> Callable[[float], float]:
    """
    Returns the aspect ratio function for a given elastic strain energy and a boundary mode.

    Args:
        U (float): Elastic strain energy.
        boundary (ModelBoundary): Boundary mode.

    Returns:
        Callable[[float], float]: A function that takes the radius and returns the aspect ratio.
    """
    return interp_impact_layer(f'impact-layer_asp_{boundary}_corner.npy', U)

def get_l1(U: float, boundary: SpecimenBoundary) -> Callable[[float], float]:
    """
    Returns the aspect ratio function for a given elastic strain energy and a boundary mode.

    Args:
        U (float): Elastic strain energy.
        boundary (ModelBoundary): Boundary mode.

    Returns:
        Callable[[float], float]: A function that takes the radius and returns the aspect ratio.
    """
    return interp_impact_layer(f'impact-layer_l1_{boundary}_corner.npy', U)