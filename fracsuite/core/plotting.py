"""
Plotting helper functions
"""

from enum import Enum
from typing import Any, Callable, TypeVar
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

import numpy as np
from fracsuite.core.coloring import get_color, norm_color, rand_col
from fracsuite.core.kernels import ImageKerneler, ObjectKerneler

from fracsuite.core.splinter import Splinter
from fracsuite.general import GeneralSettings
from fracsuite.helpers import annotate_image

general = GeneralSettings.get()

CONTOUR_ALPHA = 0.8

new_colormap  = mpl.colormaps['turbo'].resampled(7)
new_colormap.colors[0] = (1, 1, 1, 0)  # (R, G, B, Alpha)
modified_turbo = mpl.colors.LinearSegmentedColormap.from_list('modified_turbo', new_colormap.colors, 256,)
"Turbo but with starting color white."

class KernelContourMode(str, Enum):
    RECT = 'rect'
    CONTOURS = 'contours'

def plot_splinter_movavg(
    original_image: np.ndarray,
    splinters: list[Splinter],
    kernel_width: float,
    z_action: Callable[[list[Splinter]], float] = None,
    clr_label: str = None,
    no_ticks: bool = True,
    plot_vertices: bool = False,
    mode: KernelContourMode = KernelContourMode.CONTOURS,
    **kwargs
):
    """
    Create a figure that contains the kernel results as contours on
    top of the original image with a colorbar to the side.

    This plot does not contain any labels or titles except for the colorbar.
    """
    assert mode in ['contours', 'rect'], "mode must be either 'contours' or 'rect'."
    assert kernel_width > 0, "kernel_width must be greater than 0."
    assert kernel_width < np.min(original_image.shape[:2]), "kernel_width must be smaller than the image size."


    region = np.array([original_image.shape[1], original_image.shape[0]])
    # print(f'Creating intensity plot with region={region}...')

    kernel = ObjectKerneler(
        region,
        splinters,
        lambda x,r: x.in_region_px(r),
        kernel_width,
        skip_edge=False,
        skip_edge_factor=0.02
    )

    X, Y, Z = kernel.run(z_action, mode="area")

    return plot_kernel_results(original_image, clr_label, no_ticks, plot_vertices, mode, X, Y, Z)

def plot_kernel_results(original_image, clr_label, no_ticks, plot_vertices, mode, X, Y, Z):
    fig,axs = plt.subplots(figsize=general.figure_size, layout='tight')
    axs.imshow(original_image)

    if plot_vertices:
            axs.scatter(X, Y, marker='o', c='red')

    if mode == KernelContourMode.CONTOURS:
        axim = axs.contourf(X, Y, Z, cmap='turbo', alpha=CONTOUR_ALPHA)
    elif mode == KernelContourMode.RECT:
        z_im = cv2.resize(Z, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        axim = axs.imshow(z_im, cmap='turbo', alpha=CONTOUR_ALPHA)

    if clr_label is not None:

        # divider = make_axes_locatable(axs)
        # cax = divider.append_axes("right", size="15%", pad=0.1)
        cbar = fig.colorbar(axim, label=clr_label, ax=axs)
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

    fig.tight_layout()
    return fig, axs

def plot_image_kernel_contours(image: np.ndarray,
        kernel_width: float,
        z_action: Callable[[list[Splinter]], float] = None,
        clr_label="Z-Value [?]",
        plot_vertices: bool = False,
        skip_edge: bool = False,
        mode: KernelContourMode = KernelContourMode.CONTOURS,
        exclude_points: list[tuple[int,int]] = None,
        no_ticks = True,
    ):
    """Create an intensity plot of the fracture.

    Args:
        intensity_h (float): Size of the analyzed regions.
        z_action (def(list[Specimen])): The action that is called for every region.
        clr_label (str, optional): Colorbar title. Defaults to "Intensity [Splinters / Area]".

    Returns:
        Figure: A figure showing the intensity plot.
    """

    # print(f'Creating intensity plot with region={region}...')
    kernel = ImageKerneler(image, kernel_width, skip_edge=skip_edge)
    X, Y, Z = kernel.run(z_action, exclude_points=exclude_points)

    return plot_kernel_results(image, clr_label, no_ticks, plot_vertices, mode, X, Y, Z)


T2 = TypeVar('T2')
def plot_values(values: list[T2], values_func: Callable[[T2, Axes], Any]) -> tuple[Figure, Axes]:
    """Plot the values of a list of objects.

    Args:
        values (list[T2]): The values to plot.
        values_func (Callable[[T2], Any]): The function that returns the value to plot.
    """
    fig, axs = plt.subplots(1, len(values))
    for i,x in enumerate(values):
        values_func(x, axs[i])
    return fig,axs

def create_splinter_sizes_image(
    splinters: list[Splinter],
    shape: tuple[int,int, int],
    annotate: bool = True,
    annotate_title: str = "",
    with_contours: bool = False,
    crange: tuple[float,float] = None
) -> np.ndarray:
    """Create an image with the splinter sizes colored in a colormap."""
    img = np.zeros(shape, dtype=np.uint8)
    areas = [x.area for x in splinters]

    min_area = np.min(areas)
    max_area = np.max(areas)

    if crange is not None:
        min_area = crange[0]
        max_area = crange[1]

    for s in splinters:
        clr = get_color(s.area, min_value=min_area, max_value=max_area)
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

def datahist_plot(
    ncols:int = 1,
    nrows:int = 1,
    xlim: tuple[float,float] = None,
    x_format: str = "{0:.00f}",
    y_format: str = "{0:.2f}",
    data_mode = 'pdf',
    figsize=None
) -> tuple[Figure, list[Axes]]:
    """Create a figure and axes for a data histogram."""
    figsize = figsize or general.figure_size
    fig, axs = plt.subplots(ncols, nrows, figsize=figsize, sharex=True, sharey=True)

    if nrows == 1 and ncols == 1:
        axs = [axs]

    if xlim is not None:
        for ax in axs:
            ax.set_xlim(xlim)
    else:
        for ax in axs:
            ax.set_xlim((0, 2))

    ticks = FuncFormatter(lambda x, pos: x_format.format(10**x))
    ticksy = FuncFormatter(lambda x, pos: y_format.format(x))
    for ax in axs:
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticksy)

        # ax.xaxis.set_major_formatter(ScalarFormatter())
        if data_mode == 'pdf':
            ax.set_xlabel('Splinter Area $A_S$ [mm²]')
            ax.set_ylabel('PDF $P(A_S)$')
        elif data_mode == 'cdf':
            ax.set_xlabel('Splinter Area $A_S$ [mm²]')
            ax.set_ylabel('CDF $C(A_S)$')
        ax.grid(True, which='both', axis='both')

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
    data_mode = 'pdf',
    as_density = True
) -> tuple[Any, list[float], Any]:
    """
    Plot a histogram of the data to axes ax.

    Returns:
        tuple[Any, list[float], Any]: The container, binrange and values of the histogram.
    """

    assert data_mode in ['pdf', 'cdf'], "data_mode must be either 'pdf' or 'cdf'."
    if n_bins is None and binrange is None:
        n_bins = general.hist_bins

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

    if binrange is None:
        binrange = np.linspace(data[0], data[-1], n_bins)

    if data_mode == 'pdf':
        # density: normalize the bins data count to the total amount of data
        v,_,container = ax.hist(data, bins=binrange,
                density=as_density,
                color=norm_color(color),
                label=label,
                edgecolor='gray',
                linewidth=0.5,
                alpha=alpha)
    elif data_mode == 'cdf':
        alpha = 1.0
        cumsum = np.cumsum(np.histogram(data, bins=binrange, density=as_density)[0])
        # density: normalize the bins data count to the total amount of data
        container = ax.plot(binrange[:-1], cumsum / np.max(cumsum), label=label, alpha=alpha,
                            color=color)
        # _,_,container = ax.hist(data, bins=binrange,
        #         density=True,
        #         label=label,
        #         cumulative=True,
        #         alpha=alpha)

    if plot_mean:
        mean = np.mean(data)
        ax.axvline(mean, linestyle='--', label=f"Ø={mean:.2f}mm²")

    return container, binrange, v