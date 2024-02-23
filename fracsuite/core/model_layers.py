from enum import Enum
import os
from pyexpat.errors import XML_ERROR_SUSPEND_PE
from typing import Callable
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from fracsuite.core.LinearestInterpolator import LinearestInterpolator
from fracsuite.core.logging import info
from fracsuite.core.plotting import FigureSize, get_fig_width, renew_ticks_cb
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.general import GeneralSettings
from scipy.interpolate import interp2d, griddata, bisplrep, LinearNDInterpolator
from fracsuite.core.specimenprops import SpecimenBreakPosition, SpecimenBoundary

general = GeneralSettings.get()

DEFAULT_RADIUS_DELTA = 25
DEFAULT_ANGLE_DELTA = 360


class ModelLayer(str, Enum):
    IMPACT = "impact-layer"

    @staticmethod
    def get_name(
        layer_name: str,
        mode: SplinterProp,
        boundary: SpecimenBoundary,
        thickness: int,
        break_pos: SpecimenBreakPosition,
        is_stddev: bool
    ):
        stddev = "-stddev" if is_stddev else ""
        return f'{layer_name}{stddev}_{thickness:.0f}_{boundary}_{mode}_{break_pos}.npy'



def get_layer_folder():
    return os.path.join(general.out_path, "layer")

def get_layer_filepath(file_name):
    return os.path.join(get_layer_folder(), file_name)

def save_layer(
    mode: SplinterProp,
    boundary: SpecimenBoundary,
    thickness: int,
    break_pos: SpecimenBreakPosition,
    is_stddev: bool,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    layer_name: str = ModelLayer.IMPACT
):
    # assert, that Z is has the same x-length as x and y-length as y
    assert Z.shape == (len(Y), len(X)), f"Z has shape {Z.shape}, but should have shape ({len(Y)}, {len(X)})"
    # assert, that X and Y are 1-dimensional
    assert X.ndim == 1, f"X has {X.ndim} dimensions, but should have 1"
    assert Y.ndim == 1, f"Y has {Y.ndim} dimensions, but should have 1"

    layer_name = ModelLayer.get_name(layer_name, mode, boundary, thickness, break_pos, is_stddev)
    file_path = get_layer_filepath(layer_name)

    # show error if any in Z is nan
    assert not np.isnan(Z).any(), "Z contains NaN values"

    # create 2d array with shape (len(Y)+1, len(X)+1)
    # the first row and column are the x and y values
    # the rest is the z values
    data = np.zeros((len(Y)+1, len(X)+1))
    data[0,1:] = X.flatten()
    data[1:,0] = Y.flatten()
    data[1:,1:] = Z

    info(f"Saving layer to {file_path}")
    np.save(file_path, data)


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
    file_path = get_layer_filepath(file_name)
    return load_layer_file(file_path)

def load_layer_file(file_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads a layer from a file path."""
    assert os.path.exists(file_path), f"File {file_path} does not exist"

    data = np.load(file_path)
    R = data[0,1:]
    U = data[1:,0]
    V = data[1:,1:]

    return R,U,V

def has_layer(
    mode: SplinterProp,
    boundary: SpecimenBoundary,
    thickness: int,
    break_pos: SpecimenBreakPosition,
    is_stddev: bool,
    layer_name: str = ModelLayer.IMPACT
):
    layer_name = ModelLayer.get_name(layer_name, mode, boundary, thickness, break_pos, is_stddev)
    file_path = get_layer_filepath(layer_name)
    return os.path.exists(file_path)


def interp_layer(
    mode: SplinterProp,
    boundary: SpecimenBoundary,
    thickness: int,
    break_pos: SpecimenBreakPosition,
    U: float,
    layer: ModelLayer = ModelLayer.IMPACT,
):
    # X: Distance from Impact
    # Y: Energy
    # V: Layer value
    # Layout:
    # nan   X1    X2      X3      X4
    # Y1    V11   V12     V13     V14
    # Y2    V21   V22     V23     V24
    # Y3    ...
    # Y4    ...
    layer_name = ModelLayer.get_name(layer, mode, boundary, thickness, break_pos, False)

    assert has_layer(mode, boundary, thickness, break_pos, False, layer), f"Layer {layer_name} does not exist"

    X,Y,V = load_layer(layer_name)
    info('Loading layer: ', layer_name)

    # print('Radii', X)
    # print('Energies', Y)
    # print('Values', V)
    X,Y = np.meshgrid(X,Y)
    X = X.flatten()
    Y = Y.flatten()
    V = V.flatten()

    interpolator = LinearestInterpolator(np.vstack([X,Y]).T, V)

    # f = interp2d(X, Y, V, kind='linear')
    # f = bisplrep(X, Y, V,  s=0.1)
    def r_func(r: float) -> float:
        """Calculate the value of the layer at a given radius. The energy was given to interp_layer."""
        return interpolator(r, U)

    layer_name = ModelLayer.get_name(layer, mode, boundary, thickness, break_pos, True)
    Xs,Ys,Vs = load_layer(layer_name)

    Xs,Ys = np.meshgrid(Xs,Ys)
    Xs = Xs.flatten()
    Ys = Ys.flatten()
    Vs = Vs.flatten()
    # print(Xs)
    # print(Ys)
    interpolator_std = LinearestInterpolator(np.vstack([Xs,Ys]).T, Vs)

    def r_func_std(r: float) -> float:
        return interpolator_std(r, U)

    return r_func, r_func_std

def plt_layer(R,U,V,ignore_nan=False, xlabel="Radius", ylabel="Energy", clabel="~",interpolate=True,figwidth=FigureSize.ROW1) -> Figure:
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


    fig,axs = plt.subplots(figsize=get_fig_width(figwidth))
    cmesh = axs.pcolormesh(X,Y,Z,shading='auto',cmap='turbo')
    cbar = fig.colorbar(cmesh, label=clabel, ax=axs)
    renew_ticks_cb(cbar)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.autoscale()
    axs.set_xlim((0, R[-3]))
    return fig

def get_asp(U: float, boundary: SpecimenBoundary, thickness: int, break_pos: SpecimenBreakPosition) -> Callable[[float], float]:
    """
    Returns the aspect ratio function for a given elastic strain energy and a boundary mode.

    Args:
        U (float): Elastic strain energy.
        boundary (ModelBoundary): Boundary mode.

    Returns:
        Callable[[float], float]: A function that takes the radius and returns the aspect ratio.
    """
    return interp_layer(SplinterProp.ASP, boundary, thickness, break_pos, U)

def get_l1(U: float, boundary: SpecimenBoundary, thickness: int, break_pos: SpecimenBreakPosition) -> Callable[[float], float]:
    """
    Returns the aspect ratio function for a given elastic strain energy and a boundary mode.

    Args:
        U (float): Elastic strain energy.
        boundary (ModelBoundary): Boundary mode.

    Returns:
        Callable[[float], float]: A function that takes the radius and returns the aspect ratio.
    """
    return interp_layer(SplinterProp.L1, boundary, thickness, break_pos, U)


def arrange_regions(
    d_r_mm: int = DEFAULT_RADIUS_DELTA,
    d_t_deg: int = DEFAULT_ANGLE_DELTA,
    break_pos: SpecimenBreakPosition | tuple[float,float] = SpecimenBreakPosition.CORNER,
    w_mm: int = 500,
    h_mm: int = 500,
    cx_mm: int = None,
    cy_mm: int = None,
    **kwargs
):
    """
    Arranges polar regions for model layers.

    Returns:
        (r_range, t_range): The radius and angle ranges.
    """
    # get break position
    if isinstance(break_pos, SpecimenBreakPosition):
        ip_x, ip_y = break_pos.default_position()
    else:
        ip_x, ip_y = break_pos

    if cx_mm is not None and cy_mm is not None:
        ip_x, ip_y = cx_mm, cy_mm

    # maximum radius
    r_max = np.sqrt((w_mm-ip_x)**2 + (h_mm-ip_y)**2)
    r_min = 10 # 1cm around impact has no splinters

    # calculate angle and radius steps
    n_t = int(360 / d_t_deg)

    # radius range
    r_range = np.arange(r_min, r_max, d_r_mm, dtype=np.float32)
    # for i in range(len(r_range)):
    #     xi = -((i / len(r_range))**1.2)+1
    #     r_range[i] = r_range[i] / xi
    # angle range
    t_range = np.linspace(-180, 180, n_t+1, endpoint=True)

    return r_range,t_range

def arrange_regions_px(
    px_per_mm: float,
    d_r_mm: int = 25,
    d_t_deg: int = 360,
    break_pos: SpecimenBreakPosition = SpecimenBreakPosition.CORNER,
    w_mm: int = 500,
    h_mm: int = 500,
    **kwargs
):
    r_range,t_range = arrange_regions(
        d_r_mm=d_r_mm,
        d_t_deg=d_t_deg,
        break_pos=break_pos,
        w_mm=w_mm,
        h_mm=h_mm,
        **kwargs
    )

    return r_range*px_per_mm,t_range
