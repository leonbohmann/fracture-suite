"""
Plotting helper functions
"""
from __future__ import annotations

import tempfile
from enum import Enum
from typing import Any, Callable
from matplotlib.collections import PolyCollection

from rich import print

import cv2
from deprecated import deprecated
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import Divider, Size

from fracsuite.core.coloring import get_color, norm_color, rand_col
from fracsuite.core.image import to_rgb
from fracsuite.core.imageprocessing import modify_border
from fracsuite.core.kernels import ImageKerneler, ObjectKerneler
from fracsuite.core.logging import debug, info, warning
from fracsuite.core.splinter import Splinter
from fracsuite.general import GeneralSettings
from fracsuite.state import State, StateOutput

general = GeneralSettings.get()

CONTOUR_ALPHA = 0.8

new_colormap  = mpl.colormaps['turbo'].resampled(7)
new_colormap.colors[0] = (1, 1, 1, 0)  # (R, G, B, Alpha)
modified_turbo = mpl.colors.LinearSegmentedColormap.from_list('modified_turbo', new_colormap.colors, 256,)
"Turbo but with starting color white."


class FigureSize(str, Enum):
    """ Figwidth factor for different figure configurations! """
    ROW1 = 'row1'
    "The width of a figure in a row with one figure."
    ROW2 = 'row2'
    "The width of a figure in a row with two figures."
    ROW3 = 'row3'
    "The width of a figure in a row with three figures."
    ROW3W = 'row3w'
    "The width of a figure in a row with three figures but wider."
    ROW1H = 'row1h'
    "The width of a figure in one row in landscape."
    ROW1HL = 'row1h_l'
    "Large landscape."

    WIDE = 'wide'

    IMG = 'img'
    "Indifferent for images"

    @staticmethod
    def has_value(value):
        return value in FigureSize.values()

    @classmethod
    def values(cls):
        return set(item.value for item in cls)


class KernelContourMode(str, Enum):
    FILLED = 'filled'
    CONTOURS = 'contours'
    IMAGE_ONLY = 'image_only'

    @staticmethod
    def has_value(value):
        return value in KernelContourMode._value2member_map_

    @staticmethod
    def values():
        return list(map(lambda c: c.value, KernelContourMode))

class AxLabels:
    SPLINTER_AREA_SHORT = "A_\\text{S} (mm²)"
    SPLINTER_AREA = "Bruchstückflächeninhalt $A_\\text{S}$ (mm²)"

def get_legend(*axs: Axes):
    handles = []
    labels = []
    for ax in axs:
        if ax is None:
            continue
        h,l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    return handles, labels



def get_fig_width(w: FigureSize, hf=None, dimf=1.0) -> float:
    """
    Calculates the figure width and height in inches based on the given width factor, height factor and dimension factor.

    The dimension factor dimf is used so that the figures are rendered slightly larger than the actual size,
    because in Latex the textwidth is not the same as the actual width of the environment.

    Args:
        w (FigWidth): The width factor of the figure.
        hf (Optional[float]): The height factor of the figure. Defaults to None.
        dimf (float): The dimension factor of the figure. Defaults to 1.1.

    Returns:
        Tuple[float, float]: The figure width and height in inches.
    """
    golden_ratio = (5**.5 - 1) / 2

    is_landscape = False
    if w.endswith('h'):
        is_landscape = True

    if 'override_figwidth' in State.kwargs:
        w = State.kwargs['override_figwidth']

    if w in general.figure_sizes_mm:
        w_mm,h_mm = general.figure_sizes_mm[w]
    elif isinstance(w, float):
        w_mm = float(w)
        h_mm = w * golden_ratio
    elif isinstance(w, tuple):
        w_mm,h_mm = w


    mm_per_inch = 1 / 25.4
    w_inch = mm_per_inch * w_mm
    h_inch = mm_per_inch * h_mm

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    wfac = 1 if not is_landscape else golden_ratio

    fig_width_in = w_inch * dimf
    fig_height_in = h_inch * dimf

    return (fig_width_in, fig_height_in)

@deprecated(action='error')
def get_figure_size_fraction(wf, hf=None, dimf=1.0) -> tuple[tuple[float,float],float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
        wf: float
            Width factor in fraction of the textwidth.
        hf: float, optional
            Height factor. The default is None.
        dimf: float, optional
            Dimension factor. The default is 1.0.

    Returns
    -------
        fig_dim: tuple
            Dimensions of figure in inches
        fraction: float
            The fraction, so it can be appended to the filename.
    """
    assert False, "This function is deprecated. Use get_fig_width instead."

def to_img(fig):
    temp_file = tempfile.mkstemp("TEMP_FIG_TO_IMG.png")[1]
    fig.savefig(temp_file, dpi=300, bbox_inches=0, pad_inches=0)
    plt.close(fig)
    return to_rgb(cv2.imread(temp_file))


def get_log_range(data, bins=30):
    data = np.log10(data)
    return np.linspace(np.min(data), np.max(data), bins)



def plot_image_movavg(
    image: np.ndarray,
    kw_px: int,
    n_points: int,
    z_action: Callable[[npt.NDArray], float] = None,  # noqa: F821
    clr_label=None,
    plot_vertices: bool = False,
    plot_kernel: bool = False,
    skip_edge: bool = False,
    mode: KernelContourMode = KernelContourMode.CONTOURS,
    exclude_points: list[tuple[int,int]] = None,
    no_ticks = True,
    figwidth = FigureSize.ROW2,
    clr_format: str = None,
    normalize: bool = True,
    crange: tuple[float,float] = None,
    fill_skipped_with_mean: bool = True,
    transparent_border: bool = False,
) -> StateOutput:
    """Create an intensity plot of the fracture.

    Args:
        image (np.ndarray): The input image as a 2D numpy array.
        kernel_width (float): The width of the kernel.
        z_action (Callable[[list[Splinter]], float], optional): A function that takes a list of Splinter objects and returns a single value to be plotted as the Z coordinate. Defaults to None.
        clr_label (str, optional): The label of the colorbar. Defaults to None.
        plot_vertices (bool, optional): Whether to plot the vertices of the individual kernels. Defaults to False.
        skip_edge (bool, optional): Whether to skip the edges of the image. Defaults to False.
        mode (KernelContourMode, optional): The mode used for plotting the kernel contours. Defaults to KernelContourMode.CONTOURS.
        exclude_points (list[tuple[int,int]], optional): A list of points to exclude from the plot. Defaults to None.
        no_ticks (bool, optional): Whether to show the ticks on the plot. Defaults to True.
        figwidth (FigWidth, optional): The width of the figure. Defaults to FigWidth.ROW2.
        clr_format (str, optional): The format of the colorbar labels. Defaults to None.
        crange (tuple[float,float], optional): The range of values to be plotted. Defaults to None.

    Returns:
        Figure: A matplotlib figure object showing the intensity plot.
    """

    # print(f'Creating intensity plot with region={region}...')
    kernel = ImageKerneler(image, skip_edge=skip_edge)
    X, Y, Z = kernel.run(
        n_points,
        kw_px,
        z_action,
        exclude_points=exclude_points,
        fill_skipped_with_mean=fill_skipped_with_mean
    )

    if normalize:
        Z = Z / np.max(Z)


    return plot_kernel_results(
        original_image=image,
        clr_label=clr_label,
        no_ticks=no_ticks,
        plot_vertices=plot_vertices,
        mode=mode,
        X=X,
        Y=Y,
        results=Z,
        kw_px=kw_px,
        figwidth=figwidth,
        clr_format=clr_format,
        crange=crange,
        plot_kernel=plot_kernel,
        fill_skipped_with_mean=fill_skipped_with_mean,
        make_border_transparent=transparent_border

    )

def plot_splinter_movavg(
    original_image: np.ndarray,
    splinters: list[Splinter],
    kw_px: int,
    n_points: int,
    z_action: Callable[[list[Splinter]], float] = None,
    clr_label: str = None,
    no_ticks: bool = True,
    plot_vertices: bool = False,
    plot_kernel: bool = False,
    skip_edge: bool = False,
    mode: KernelContourMode = KernelContourMode.CONTOURS,
    exclude_points: list[tuple[int,int]] = None,
    figwidth: FigureSize = FigureSize.ROW1,
    clr_format=None,
    normalize: bool = False,
    crange: tuple[float,float] = None,
    fill_skipped_with_mean: bool = True,
    transparent_border: bool = False,
) -> StateOutput:
    """
    Plot the results of a kernel operation on a list of objects, using a moving average
    filter to smooth the kernel contours.

    Parameters:
    -----------
        original_image : np.ndarray
            The original image to plot.
        splinters : list[Splinter]
            A list of Splinter objects representing the regions of interest in the image.
        kernel_width : float
            The width of the kernel to use in the operation in px.
        z_action : Callable[[list[Splinter]], float], optional
            A function that takes a list of Splinter objects and returns a scalar value
            to use as the z-coordinate of the kernel results. If None, the default
            z-coordinate is the amount of splinters in the kernel.
        clr_label : str, optional
            The label to use for the colorbar.
        no_ticks : bool, optional
            Whether to show ticks on the plot axes.
        plot_vertices : bool, optional
            Whether to plot the vertices of the Splinter objects.
        mode : KernelContourMode, optional
            The mode to use for the kernel contours. Must be either 'contours' or 'rect'.
        figwidth : FigWidth, optional
            The width of the figure. Must be one of the values defined in the FigWidth enum.
        clr_format : str, optional
            The format string to use for the colorbar labels.
        normalize : bool, optional
            Whether to normalize the kernel results to the range [0, 1].
        crange : tuple[float,float], optional
            The range of values to use for the colorbar. If None, the range is determined
            automatically from the kernel results.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The resulting figure object.
    """
    assert KernelContourMode.has_value(mode), f"Contour mode must be one of {KernelContourMode.values()}. plot_splinter_movavg."
    assert FigureSize.has_value(figwidth), f"FigWidth must be one of {FigureSize.values()}. plot_splinter_movavg."
    assert kw_px > 0, "kernel_width must be greater than 0."
    assert kw_px < np.min(original_image.shape[:2]), "kernel_width must be smaller than the image size."
    assert figwidth in general.figure_sizes_mm, f"figwidth {figwidth} not found."

    region = (original_image.shape[0], original_image.shape[1])
    print(f'Creating intensity plot with region={region}...')

    kernel = ObjectKerneler(
        region,
        splinters,
        collector=lambda x,r: x.in_region_px(r),
        skip_edge=skip_edge,
    )

    X, Y, Z = kernel.run(
        z_action,
        kw_px,
        n_points,
        mode="area",
        exclude_points=exclude_points,
        fill_skipped_with_mean=fill_skipped_with_mean
    )

    if normalize:
        Z = Z / np.max(Z)

    return plot_kernel_results(
        original_image=original_image,
        clr_label=clr_label,
        no_ticks=no_ticks,
        plot_vertices=plot_vertices,
        mode=mode,
        X=X,
        Y=Y,
        results=Z,
        kw_px=kw_px,
        figwidth=figwidth,
        clr_format=clr_format,
        crange=crange,
        plot_kernel=plot_kernel,
        fill_skipped_with_mean=fill_skipped_with_mean,
        make_border_transparent=transparent_border,
        smooth=True
    )

def plot_kernel_results(
    original_image,
    clr_label: str,
    no_ticks: bool,
    plot_vertices: bool,
    mode: str,
    X,
    Y,
    results,
    kw_px,
    figwidth: FigureSize,
    clr_format: str = None,
    crange: tuple[float,float] = None,
    plot_kernel: bool = False,
    fill_skipped_with_mean: bool = True,
    make_border_transparent: bool = False,
    overwrite_alpha: float = None,
    smooth: bool = False
) -> StateOutput:
    """
    Plot the results of a kernel operation on an image.

    Args:
        original_image (np.ndarray): The original image to plot.
        clr_label (str): The label for the colorbar.
        no_ticks (bool): Whether to hide the tick marks on the plot.
        plot_vertices (bool): Whether to plot the vertices.
        mode (KernelContourMode): The mode for plotting the kernel contours.
        X (np.ndarray): The X coordinates of the results.
        Y (np.ndarray): The Y coordinates of the results.
        Z (np.ndarray): Kerneled results.
        figwidth (FigWidth): FigWidth of the figure. Choose from enum!
        clr_format (str, optional): The format string for the colorbar labels. Defaults to None.
        crange (tuple[float,float], optional): The range of values to display on the colorbar. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects for the plot.
    """
    figsize = get_fig_width(figwidth)
    fig,axs = plt.subplots(figsize=figsize)

    if crange is None:
        crange = (np.nanmin(results), np.nanmax(results))
        debug(f"Using crange: {crange}")

    def show_img():
        axs.imshow(original_image, interpolation='bilinear' if smooth else 'nearest')

    def show_vertices():
        if plot_vertices:
            axs.scatter(X, Y, marker='x', c='white', s=3, linewidth=0.5)
        if plot_kernel:
            axs.add_patch(mpatches.Rectangle((kw_px, kw_px), kw_px, kw_px,
                edgecolor = 'pink',
                fill=False,
                lw=1))

    if mode == KernelContourMode.CONTOURS:
        axim = axs.contourf(X, Y, results, cmap='turbo', alpha=CONTOUR_ALPHA)
        show_vertices()
        show_img()
    elif mode == KernelContourMode.FILLED:
        show_vertices()
        show_img()


        # scale the results up to get a smooth image
        results = cv2.resize(results, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT if smooth else cv2.INTER_NEAREST)

        # results = results / np.max(results)
        # make the outer edge of 5% of the image transparent
        if make_border_transparent:
            mask, results = modify_border(results, 5, 0.85*CONTOUR_ALPHA, fill_skipped_with_mean)
        else:
            mask = overwrite_alpha or CONTOUR_ALPHA

        axim = axs.imshow(results, cmap='turbo', vmin=crange[0], vmax=crange[1], alpha=mask) # alpha=mask,
    elif mode == KernelContourMode.IMAGE_ONLY:
        axim = axs.imshow(original_image, interpolation='bilinear' if smooth else 'nearest')




    if clr_label is not None:
        # box = axs.get_position()
        # axs.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # divider = make_axes_locatable(axs)
        # cax = divider.append_axes("right", size="8%", pad=0.1)
        # cax.grid(False)
        # cbar_ax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.03, axs.get_position().height])
        if mode == KernelContourMode.IMAGE_ONLY:
            cbar = fig.colorbar(axim, label=clr_label, ax=axs)
        cbar = fig.colorbar(axim, label=clr_label)

        renew_ticks_cb(cbar)


        if clr_format is not None:
            debug(f"Using colorbar format: {clr_format}")
            formatter = FuncFormatter(lambda x, p: f"{{0:{clr_format}}}".format(x))
            cbar.ax.yaxis.set_major_formatter(formatter)



        # ticks = np.arange(cbar.vmin, cbar.vmax, (cbar.vmax - cbar.vmin) / 5)
        # cbar.set_ticks(ticks)
        # labels = [f"{x:.2f}" for x in ticks]
        # cbar.set_ticklabels(labels)
        # tick_list = cbar.get_ticks()
        # tick_list = [cbar.vmin] + list(tick_list) + [cbar.vmax]
        # cbar.set_ticks(tick_list)

    if no_ticks:
        axs.set_xticks([])
        axs.set_yticks([])

    # height_desired = figsize[1]  # For example, 6 inches. Adjust as needed.
    # current_size = fig.get_size_inches()
    # new_width = current_size[0] * (height_desired / current_size[1])
    # fig.set_size_inches(new_width, height_desired)
    return StateOutput(fig, figwidth, ax=axs)

def renew_ticks_cb(cbar):
    manual_ticks = [cbar.vmin, cbar.vmin + (cbar.vmax - cbar.vmin) / 2, cbar.vmax]
    # print(manual_ticks)
    cbar.ax.set_yticks(manual_ticks)
    labels = cbar.ax.get_yticklabels()

    labels[0].set_verticalalignment('bottom')
    labels[1].set_verticalalignment('center')
    labels[-1].set_verticalalignment('top')

def renew_ticks_ax(ax: Axes, yticks = None, xticks = None, padding = 0):
    if yticks is None:
        yticks = ax.get_yticks()
        yticks = [yticks[0], yticks[-1]]
    ax.set_yticks(yticks)
    ax.set_ylim(yticks[0] - padding, yticks[-1] + padding)

    if xticks is None:
        xticks = ax.get_xticks()
        xticks = [xticks[0], xticks[-1]]
    ax.set_xticks(xticks)
    ax.set_xlim(xticks[0] - padding, xticks[-1] + padding)

def voronoi_to_image(img, voronoi, color=255, thickness=1):
    for i, r in enumerate(voronoi.regions):
        if -1 not in r and len(r) > 0:
            polygon = [voronoi.vertices[i] for i in r]
            polygon = np.array(polygon, dtype=int)
            cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=thickness)

def fixed_axes(size):
    fig = plt.figure(figsize=size)
    w,h = size
    u = [Size.Fixed(w * 0.05), Size.Fixed(w * 0.95)]
    v = [Size.Fixed(h * 0.05), Size.Fixed(h * 0.95)]

    divider = Divider(fig, (0, 0, 1, 1), u, v, aspect=False)

    ax = fig.add_axes(divider.get_position(),
                axes_locator=divider.new_locator(nx=1, ny=1))

    return fig, ax

# T2 = TypeVar('T2')
# def plot_values(values: list[T2], values_func: Callable[[T2, Axes], Any]) -> tuple[Figure, Axes]:
#     """Plot the values of a list of objects.

#     Args:
#         values (list[T2]): The values to plot.
#         values_func (Callable[[T2], Any]): The function that returns the value to plot.
#     """
#     fig, axs = plt.subplots(1, len(values))
#     for i,x in enumerate(values):
#         values_func(x, axs[i])
#     return fig,axs

def create_splinter_sizes_image(
    splinters: list[Splinter],
    shape: tuple[int,int, int],
    annotate: bool = True,
    annotate_title: str = "",
    with_contours: bool = False,
    crange: tuple[float,float] = None,
    s_value: float = None
) -> np.ndarray:
    """Create an image with the splinter sizes colored in a colormap."""
    img = np.zeros(shape, dtype=np.uint8)

    if s_value is None:
        def s_value(s: Splinter):
            return s.area

    areas = [s_value(x) for x in splinters]

    min_area = np.min(areas)
    max_area = np.max(areas)

    if crange is not None:
        min_area = crange[0]
        max_area = crange[1]

    for s in splinters:
        clr = get_color(s_value(s), min_value=min_area, max_value=max_area)
        cv2.drawContours(img, [s.contour], -1, clr, -1)

    if with_contours:
        cv2.drawContours(img, [s.contour for s in splinters], -1, (255,255,255), 1)

    if annotate:
        img = annotate_image(
            img,
            title=annotate_title,
            min_value=min_area,
            max_value=max_area,
            return_fig=False)

    return img

def create_splinter_colored_image(
    splinters: list[Splinter],
    shape: tuple[int,int, int],
    out_file: str = None
):
    """Create an image with the splinters colored randomly."""
    img = np.zeros(shape, dtype=np.uint8)


    for s in splinters:
        clr = rand_col()
        cv2.drawContours(img, [s.contour], -1, clr, -1)

    if out_file is not None:
        cv2.imwrite(out_file, img)

    return img

def hist_abs(data1, data2, binrange):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    data1 = data1[data1 > 0]
    data2 = data2[data2 > 0]

    data1.sort()
    data2.sort()

    data1, _ = np.histogram(data1, binrange, density=True)
    data2, _ = np.histogram(data2, binrange, density=True)

    return np.abs(data1-data2)


class DataHistMode(str, Enum):
    PDF = 'pdf'
    "Probability density function."
    CDF = 'cdf'
    "Cumulative density function."

class DataHistPlotMode(str, Enum):
    KDE = 'kde'
    "Kernel density estimation."
    KDE2 = 'kde2'
    "Kernel density estimation without fill"
    HIST = 'hist'
    "Histogram."
    STEPS = 'steps'
    "Histogram without fill"

def datahist_plot(
    ncols:int = 1,
    nrows:int = 1,
    xlim: tuple[float,float] = None,
    x_label: str = 'Bruchstückflächeninhalt $A_S$ (mm²)',
    y_format: str = "{0:.2f}",
    y_label: str = None,
    data_mode : DataHistMode = DataHistMode.PDF,
    figwidth = FigureSize.ROW1,
) -> tuple[Figure, list[Axes]]:
    """Create a figure and axes for a data histogram."""
    figsize = get_fig_width(figwidth)
    fig, axs = plt.subplots(ncols, nrows, figsize=figsize, sharex=True, sharey=True)

    if nrows == 1 and ncols == 1:
        axs = [axs]

    if xlim is not None:
        for ax in axs:
            ax.set_xlim(xlim)

    def xfmt(x):
        return f"{10**x:.0f}"
        # return f"$\\scriptsize{{log}} ({10**x:.0f})$"


    ticks = FuncFormatter(lambda x, pos: xfmt(x))
    # mticksx = FuncFormatter(lambda x, pos: f"{10**x:.0f}")
    ticksy = FuncFormatter(lambda x, pos: y_format.format(x))
    for ax in axs:
        ax.xaxis.set_major_formatter(ticks)
        # ax.xaxis.set_minor_formatter(mticksx)
        ax.yaxis.set_major_formatter(ticksy)

        ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        else:
            if data_mode == DataHistMode.PDF:
                ax.set_ylabel('Wahrscheinlichkeitsdichte $f(A_S)$')
            elif data_mode == DataHistMode.CDF:
                ax.set_ylabel('Kumulative Verteilungs Fkt. $F(A_S)$')
        ax.grid(True, which='major', axis='both')

        ax.annotate("x-Werte nach $10^x$", xy=(0.98, 0.02), color='black', xycoords="axes fraction", ha="right", va="bottom", fontsize=7)

    return fig, axs

def datahist_to_ax(
    ax: Axes,
    data: list[float],
    n_bins: int = None,
    binrange: list[float] = None,
    plot_mean: bool = True,
    label: str = None,
    color = None,
    as_log:bool = True,
    alpha: float = 0.75,
    data_mode: DataHistMode = DataHistMode.PDF,
    as_density = True,
    plot_mode: DataHistPlotMode = DataHistPlotMode.HIST,
    unit: str = "mm²",
    mean_format: str = ".1f"
) -> tuple[Any, list[float], Any]:
    """
    Plot a histogram of the data to axes ax.

    Returns:
        tuple[Any, list[float], Any]: The container, binrange and values of the histogram.
    """
    from fracsuite.core.stochastics import calculate_dmode, calculate_kde

    assert data_mode in ['pdf', 'cdf'], "data_mode must be either 'pdf' or 'cdf'."
    if n_bins is None and binrange is None:
        n_bins = general.hist_bins
    if isinstance(ax, list) and len(ax) == 1:
        ax = ax[0]

    def cvt(x):
        return np.log10(x) if as_log else x

    if not as_log:
        ticks = FuncFormatter(lambda x, pos: '{0:.00f}'.format(x))
        ax.xaxis.set_major_formatter(ticks)

    data = np.asarray(data)
    # fetch areas from splinters
    data = data[data > 0]
    # ascending sort, smallest to largest
    data.sort()

    data = cvt(data)

    if data_mode == DataHistMode.PDF:
        if binrange is None:
            binrange = np.linspace(data[0], data[-1], n_bins)
        if plot_mode == DataHistPlotMode.HIST or plot_mode == DataHistPlotMode.STEPS:
            fillcolor = norm_color(color)
            edgecolor = 'gray' if plot_mode == DataHistPlotMode.HIST else 'none'

            # density: normalize the bins data count to the total amount of data
            binned_data,edges,bin_container = ax.hist(data, bins=binrange,
                    density=as_density,
                    histtype='step' if plot_mode == DataHistPlotMode.STEPS else 'bar',
                    color=fillcolor,
                    label=label,
                    linewidth=1 if plot_mode == DataHistPlotMode.STEPS else 0.2,
                    alpha=alpha)
            color = edgecolor if plot_mode == DataHistPlotMode.STEPS else fillcolor
        elif plot_mode == DataHistPlotMode.KDE or plot_mode == DataHistPlotMode.KDE2:
            x,y = calculate_kde(data)
            gauss_kde_plot = ax.plot(x,y, label=label, color=color, alpha=1.0, zorder=0)
            color = gauss_kde_plot[0].get_color()
            if plot_mode != DataHistPlotMode.KDE2:
                ax.fill_between(x, y, 0.001, color=color, alpha=0.9, zorder=1)
            ax.set_ylim((0, ax.get_ylim()[1]))
            binned_data = x
    elif data_mode == DataHistMode.CDF:
        if binrange is None:
            binrange = np.linspace(data[0], data[-1], n_bins * 3)
        alpha = 1.0
        binned_data, edges = np.histogram(data, bins=binrange, density=as_density)
        cumsum = np.cumsum(binned_data)
        # density: normalize the bins data count to the total amount of data
        bin_container = ax.plot(binrange[:-1], cumsum / np.max(cumsum), label=label, alpha=alpha, color=norm_color(color))[0]
        color = bin_container.get_color()

    if plot_mean:
        mean_format = f'{{0:{mean_format}}}'

        most_probable_area,_ = calculate_dmode(data)
        print(most_probable_area)
        most_probable_area_value = 10**most_probable_area if as_log else most_probable_area
        most_probable_area_string = mean_format.format(most_probable_area_value)
        print(f"Most probable area: {most_probable_area_value:.2f}{unit}")
        ax.axvline(x=most_probable_area, ymin=0,ymax=100, linestyle='--', label=f"{most_probable_area_string}{unit}", color=color, alpha=alpha)

        # axd = ax.get_xlim()[1] - ax.get_xlim()[0]
        # ayd = ax.get_ylim()[1] - ax.get_ylim()[0]

        # # draw text on the plot so it doesnt overlap previous texts (from n_plots)
        # ax.text(most_probable_area + axd*0.01 , ayd * 0.03 + n_plots*0.01, f"{most_probable_area_string}{unit}", color=color, alpha=alpha, ha='left', va='center', zorder=2)


        # ax.text(most_probable_area + axd*0.01 , ayd * 0.03, f"{most_probable_area_string}{unit}", color=color, alpha=alpha, ha='left', va='center', zorder=2)
    return None, binrange, binned_data

def datahist_2d(
    data: list,
    n_bins: int = 30,
    y_stress: bool = False,
):
    """
    Plot a 2D histogram of the data.

    Args:
        data (list): The data to plot. [stress|energy, name, x_values]
        n_bins (int, optional): Number of bins. Defaults to 30.
        y_stress (bool, optional): States if the first entry of data is stress or energy. Defaults to False.
    """
    # logarthmic binning
    binrange = np.linspace(0, 2, n_bins)

    data = sorted(data, key=lambda x: x[0])

    def transform_x_value(y_value):
        return np.histogram(y_value, bins=binrange, density=True, range=(np.min(binrange), np.max(binrange)))

    y_values = [x[0] for x in data]
    names = [x[1] for x in data]
    x_values = [transform_x_value(x[2]) for x in data]


    sz = FigureSize.WIDE

    fig, axs = plt.subplots(figsize=get_fig_width(sz))
    # sort data and names for ascending stress
    axs.set_xlabel(AxLabels.SPLINTER_AREA)
    if not y_stress:
        axs.set_ylabel("$U$ (J/m²)")
    else:
        axs.set_ylabel("$\sigma_\\text{s}$ (MPa)")

    x_ticks = np.arange(0, n_bins, 10)
    siz_mod = 2 if len(x_ticks) > 10 else 1
    axs.set_xticks(x_ticks, [f'{10 ** binrange[x]:.2f}' if i % siz_mod == 0 else "" for i, x in enumerate(x_ticks)])

    str_mod = 5 if len(y_values) > 15 else 1
    axs.set_yticks(np.arange(0, len(y_values), 1),
                   [f'{np.abs(x):.2f}' if i % str_mod == 0 else "" for i, x in enumerate(y_values)])

    axy = axs.secondary_yaxis('right')
    axy.set_yticks(axs.get_yticks(), [x for i, x in enumerate(names)])

    dt = np.array(x_values)
    axim = axs.imshow(dt, cmap=modified_turbo, aspect='auto', interpolation='none')
    fig.colorbar(axim, ax=axs, orientation='vertical', label='WD $p(A_S)$', pad=0.2)
    # fig2 = plot_histograms((0,2), specimens, plot_mean=True)
    # plt.show()

    axy.set_yticks(np.linspace(axy.get_yticks()[0], axy.get_yticks()[-1], len(axs.get_yticks())))

    axs.grid(False)

    # disable minor ticks on y axis
    axs.yaxis.set_minor_locator(plt.NullLocator())
    axy.yaxis.set_minor_locator(plt.NullLocator())


    return fig,axs

def label_image(
    image,
    *labels,
    title = None,
    nums=None,
    return_fig=True,
    figwidth=FigureSize.ROW1,
):
    """
    Add labels to an image.

    Args:
        image (numpy.ndarray): The image to label.
        labels: A variable number of label-color pairs. Labels must be a multiple of 2.
        title (str, optional): The title of the plot. Defaults to None.
        nums (list, optional): A list of numbers to append to the labels. Defaults to None.
        return_fig (bool, optional): Whether to return the figure object. Defaults to True.
        fig_wfrac (float, optional): The width fraction of the figure. Defaults to 1.0.
    Returns:
        matplotlib.figure.Figure or numpy.ndarray: The labeled image as a figure object or numpy array.
    """

    assert len(labels) % 2 == 0, "Labels must be a multiple of 2."

    texts = labels[::2]
    labelcolors = labels[1::2]

    fig, ax = plt.subplots(figsize=get_fig_width(figwidth))
    ax.imshow(image)
    ax.axis('off')

    if title:
        plt.title(title)

    if nums is not None:
        ntexts = []
        for i, text in enumerate(texts):
            if len(nums) > i:
                ntexts.append(f"{text} ({nums[i]})")
            else:
                ntexts.append(text)
        texts = ntexts

    if len(labels) > 2:
        patches = [mpatches.Patch(color=norm_color(color), label=label) for label, color in zip(texts, labelcolors)]
        ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper left')

    if return_fig:
        return StateOutput(fig, figwidth)

    return StateOutput(to_img(fig), figwidth, ax = ax)


def annotate_image(
    image,
    title = None,
    cbar_title = None,
    cbar_range: tuple[float,float] = None,
    figwidth = FigureSize.ROW1,
    return_fig=True,
    clr_format: str = None,
) -> StateOutput:
    """Put a header in white text on top of the image.

    Args:
        image (Image): cv2.imread
        title (str): The title of the image.
    """
    assert cbar_range[1] >= cbar_range[0], f"Max color value must be greater than min value. Actual: {cbar_range}"

    figsize = get_fig_width(figwidth)
    fig, ax = plt.subplots(figsize=figsize)

    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False # labels along the bottom edge are off
    )

    if title is not None:
        ax.set_title(title)

    im = ax.imshow(image, cmap='turbo', aspect='equal', alpha=1, vmin=cbar_range[0], vmax=cbar_range[1])
    ax.grid(False)

    if cbar_title is not None:
        cbar = fig.colorbar(mappable=im, ax=ax, label=cbar_title)
        if cbar_range is not None:
            im.set_clim(*cbar_range)

        renew_ticks_cb(cbar)

        if clr_format is not None:
            formatter = FuncFormatter(lambda x, p: f"{{0:{clr_format}}}".format(x))
            cbar.ax.yaxis.set_major_formatter(formatter)

    if return_fig:
        return StateOutput(fig, figwidth, ax=ax)

    return StateOutput(to_img(fig), figwidth, ax=ax)

def annotate_images(
    images,
    title = None,
    cbar_title = None,
    min_value = 0,
    max_value = 1,
    figsize_cm=(12, 8),
    return_fig = False
):
    """Put a header in white text on top of the image.

    Args:
        image (Image): cv2.imread
        title (str): The title of the image.
    """
    assert max_value > min_value, "Max value must be greater than min value."
    cm = 1/2.54
    fig, axs = plt.subplots(1,len(images), figsize=(figsize_cm[0]*cm * len(images), figsize_cm[1]*cm))
    for ax in axs:
        ax.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False # labels along the bottom edge are off
        )

    if title is not None:
        fig.suptitle(title)

    for ax,image in zip(axs, images):
        im = ax.imshow(image, cmap='turbo', vmin=min_value, vmax=max_value, aspect='equal')

    if cbar_title is not None:
        fig.colorbar(mappable=im, ax=axs[:-1], label=cbar_title)

    if return_fig:
        return fig
    return to_img(fig)


def cfg_logplot(axs: Axes):
    # activate grid
    axs.grid(True, which='both', axis='both')
    # make the grid lines solid
    axs.grid(which='major', linestyle='-', linewidth='0.3', color='gray')
    axs.grid(which='minor', linestyle='-', linewidth='0.1', color='gray')

    axs.set_xscale('log')
    axs.set_yscale('log')


def transparent_line(img, pt1, pt2, color, thickness, transparency: float):
    # create a separate image for the line
    line_img = np.zeros_like(img)

    # draw the line on the separate image
    cv2.line(line_img, pt1, pt2, color, thickness)

    # blend the line image with the original image
    img2 = cv2.addWeighted(img, 1.0, line_img, transparency, 1)

    return img2

def transparent_contour(img, contour, color, thickness, transparency: float):
    # create a separate image for the line
    line_img = np.zeros_like(img)

    # draw the line on the separate image
    cv2.drawContours(line_img, [contour], -1, color, thickness)

    # blend the line image with the original image
    img2 = cv2.addWeighted(img, 1.0, line_img, transparency, 1)

    return img2



def fill_polar_cell(
    img,
    center: tuple[int,int],
    start_angle: float,
    end_angle: float,
    start_radius: float,
    end_radius: float,
    clr: tuple[int,int,int],
    alpha: float = 1,
):
    mask = np.zeros_like(img, dtype=np.uint8)
    center = (int(center[0]), int(center[1]))
    # start by drawing a circle with the end radius
    cv2.ellipse(mask, center, (int(end_radius), int(end_radius)), 0, start_angle, end_angle, clr, -1)

    # then draw a circle with the start radius
    cv2.ellipse(mask, center, (int(start_radius), int(start_radius)), 0, start_angle, end_angle, (0,0,0), -1)
    img = cv2.addWeighted(img, 1, mask, alpha, 0.5)

    return img


def shade(axs: Axes, X, y1, y2, color, alpha):
    assert len(y1) == len(y2), "y1 and y2 must have the same length."

    x = np.linspace(min(X), max(X), len(y1))
    polygon = np.column_stack([np.hstack([x, x[::-1]]), np.hstack([y1, y2[::-1]])])  # this is equivalent to fill as it makes a polygon
    col = PolyCollection([polygon], color=color, alpha=alpha)
    axs.add_collection(col, autolim=False)

def legend_without_duplicate_labels(ax, compact = False):


    handles, labels = ax.get_legend_handles_labels()
    def check_label(label, i):
        if not compact:
            return label not in labels[:i]

        else:
            start = label.split(' ')[0] if ' ' in label else label
            return label not in labels[:i] and not any([ll.startswith(start) for ll in labels[:i]])

    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if check_label(l,i)]

    if compact:
        # split all labels at either , or " "
        unique = [(h,l.split(' ')[0] if ' ' in l else l) for h,l in unique]
        unique = [(h,l.split(',')[0] if ',' in l else l) for h,l in unique]


    ax.legend(*zip(*unique))


def fit_curve(axs, x, y, func, color='k', ls='--', lw=1, pltlabel = 'Fit'):
    from scipy.optimize import curve_fit
    from fracsuite.core.stochastics import r_squared
    x = np.asarray(x)
    y = np.asarray(y)

    popt, pcov = curve_fit(func, x, y)
    y_fit = func(x, *popt)

    # use original values to calculate r²
    rsqr = r_squared(y, y_fit)

    # recalc for smooth plotting
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = func(x_fit, *popt)

    short_label = '{}: R²={:.2f}, '.format(func.__name__, rsqr)
    label = short_label
    for i, value in enumerate(popt):
        label += f'a{i+1}={value:.1f}, '
    axs.plot(x_fit, y_fit, color=color, linestyle=ls, label=pltlabel, linewidth=lw)

    axs.annotate(short_label[:-2], (x_fit[5], y_fit[5]), textcoords="offset points", xytext=(25,-5), ha='left', va='center', color=color)

    print(label[:-2])
    return y_fit, popt


def annotate_corner(
    ax,
    annotation
):
    ax.annotate(annotation, xy=(0.98, 0.02), color='black', xycoords="axes fraction", ha="right", va="bottom", fontsize=7)
