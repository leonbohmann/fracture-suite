"""
Plotting helper functions
"""

from typing import Any, Callable, TypeVar
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

import numpy as np
from fracsuite.core.coloring import get_color, rand_col
from fracsuite.core.kernels import ImageKerneler, ObjectKerneler

from fracsuite.core.splinter import Splinter
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import annotate_image

general = GeneralSettings.get()

CONTOUR_ALPHA = 0.8

new_colormap  = mpl.colormaps['turbo'].resampled(7)
new_colormap.colors[0] = (1, 1, 1, 0)  # (R, G, B, Alpha)
modified_turbo = mpl.colors.LinearSegmentedColormap.from_list('modified_turbo', new_colormap.colors, 256,)
"Turbo but with starting color white."

def plot_splinter_kernel_contours(
    original_image: np.ndarray,
    splinters: list[Splinter],
    kernel_width: float,
    z_action: Callable[[list[Splinter]], float] = None,
    clr_label: str = None,
    no_ticks: bool = True,
    plot_vertices: bool = False,
    **kwargs
):
    """
    Create a figure that contains the kernel results as contours on
    top of the original image with a colorbar to the side.

    This plot does not contain any labels or titles except for the colorbar.
    """
    region = np.array([original_image.shape[1], original_image.shape[0]])
    # print(f'Creating intensity plot with region={region}...')

    kernel = ObjectKerneler(
        region,
        splinters,
        lambda x,r: x.in_region_px(r),
        kernel_width,
        skip_edge=True,
        skip_edge_factor=0.02
    )

    X, Y, Z = kernel.run(z_action, mode="area")

    fig,axs = plt.subplots(figsize=general.figure_size)

    axs.imshow(original_image)

    if plot_vertices:
            axs.scatter(X, Y, marker='o', c='red')

    axim = axs.contourf(X, Y, Z, cmap='turbo', alpha=CONTOUR_ALPHA)
    if clr_label is not None:
        fig.colorbar(axim, label=clr_label)

    if no_ticks:
        axs.set_xticks([])
        axs.set_yticks([])

    fig.tight_layout()
    return fig, axs

def plot_image_kernel_contours(image: np.ndarray,
                   kernel_width: float,
                    z_action: Callable[[list[Splinter]], float] = None,
                    clr_label="Z-Value [?]",
                    fig_title="Title",
                    xlabel="Pixels",
                    ylabel="Pixels",
                    plot_vertices: bool = False,
                    skip_edge: bool = False,
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
    X, Y, Z = kernel.run(z_action)

    fig,axs = plt.subplots()
    axs.imshow(image)

    if plot_vertices:
        axs.scatter(X, Y, marker='o', c='red')

    axim = axs.contourf(X, Y, Z, cmap='turbo', alpha=CONTOUR_ALPHA)
    fig.colorbar(axim, label=clr_label)
    axs.xaxis.tick_top()
    axs.xaxis.set_label_position('top')
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(f'{fig_title} (h={kernel_width:.2f})')

    fig.tight_layout()
    return fig


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


def datahist_plot(
    ncols:int = 1,
    nrows:int = 1,
    xlim: tuple[float,float] = None,
) -> tuple[Figure, list[Axes]]:
    """Create a figure and axes for a data histogram."""

    fig, axs = plt.subplots(ncols, nrows, figsize=general.figure_size, sharex=True, sharey=True)

    if nrows == 1 and ncols == 1:
        axs = [axs]

    if xlim is not None:
        for ax in axs:
            ax.set_xlim(xlim)
    else:
        for ax in axs:
            ax.set_xlim((0, 2))

    ticks = FuncFormatter(lambda x, pos: '{0:.00f}'.format(10**x))
    ticksy = FuncFormatter(lambda x, pos: '{0:.2f}'.format(x))
    for ax in axs:
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticksy)

        # ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Splinter Area [mm²]')
        ax.set_ylabel('Probability Density (Area) [-]')
        ax.grid(True, which='both', axis='both')

    return fig, axs

def datahist_to_ax(
    ax: Axes,
    data: list[float],
    n_bins: int = 20,
    plot_mean: bool = True,
    label: str = None,
    as_log:bool = True,
    alpha: float = 0.5,
    data_mode = 'pdf'
):
    """Plot a histogram of the data to axes ax."""

    assert data_mode in ['pdf', 'cdf'], "data_mode must be either 'pdf' or 'cdf'."


    def cvt(x):
        return np.log10(x) if as_log else x

    if not as_log:
        ticks = FuncFormatter(lambda x, pos: '{0:.00f}'.format(x))
        ax.xaxis.set_major_formatter(ticks)

    # fetch areas from splinters
    data = [cvt(x) for x in data if x > 0]
    # ascending sort, smallest to largest
    data.sort()

    max_data = cvt(100)
    binrange = np.linspace(0, max_data, n_bins)

    if data_mode == 'pdf':
        # density: normalize the bins data count to the total amount of data
        _,_,container = ax.hist(data, bins=binrange,
                density=True,
                label=label,
                alpha=alpha)
    elif data_mode == 'cdf':
        alpha = 1.0
        cumsum = np.cumsum(np.histogram(data, bins=binrange, density=True)[0])
        # density: normalize the bins data count to the total amount of data
        container = ax.plot(binrange[:-1], cumsum / np.max(cumsum), label=label, alpha=alpha)
        # _,_,container = ax.hist(data, bins=binrange,
        #         density=True,
        #         label=label,
        #         cumulative=True,
        #         alpha=alpha)

    if plot_mean:
        mean = np.mean(data)
        ax.axvline(mean, linestyle='--', label=f"Ø={mean:.2f}mm²")

    return container