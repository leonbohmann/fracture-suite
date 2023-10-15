"""
Plotting helper functions
"""
from __future__ import annotations

import tempfile
from enum import Enum
from typing import Any, Callable

from rich import print

import cv2
from deprecated import deprecated
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from fracsuite.core.coloring import get_color, norm_color, rand_col
from fracsuite.core.image import to_rgb
from fracsuite.core.kernels import ImageKerneler, ObjectKerneler
from fracsuite.core.splinter import Splinter
from fracsuite.general import GeneralSettings
from fracsuite.state import StateOutput

general = GeneralSettings.get()

CONTOUR_ALPHA = 0.8

new_colormap  = mpl.colormaps['turbo'].resampled(7)
new_colormap.colors[0] = (1, 1, 1, 0)  # (R, G, B, Alpha)
modified_turbo = mpl.colors.LinearSegmentedColormap.from_list('modified_turbo', new_colormap.colors, 256,)
"Turbo but with starting color white."


class FigWidth(str, Enum):
    """ Figwidth factor for different figure configurations! """
    ROW1 = 'row1'
    "The width of a figure in a row with one figure."
    ROW2 = 'row2'
    "The width of a figure in a row with two figures."
    ROW3 = 'row3'
    "The width of a figure in a row with three figures."

    @staticmethod
    def has_value(value):
        return value in FigWidth.values()

    @classmethod
    def values(cls):
        return set(item.value for item in cls)


class KernelContourMode(str, Enum):
    FILLED = 'filled'
    CONTOURS = 'contours'

    @staticmethod
    def has_value(value):
        return value in KernelContourMode._value2member_map_

    @staticmethod
    def values():
        return list(map(lambda c: c.value, KernelContourMode))


def get_fig_width(w: FigWidth, hf=None, dimf=1.1) -> float:
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
    assert FigWidth.has_value(w), f"FigWidth must be one of {FigWidth.values()}."

    factor = general.width_factors[w]
    w_pt = general.document_width_pt * factor

    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    fig_width_in = w_pt * inches_per_pt * dimf
    fig_height_in = fig_width_in * (hf if hf is not None else golden_ratio)

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
    fig.tight_layout()
    temp_file = tempfile.mkstemp("TEMP_FIG_TO_IMG.png")[1]
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return to_rgb(cv2.imread(temp_file))



def plot_image_movavg(
    image: np.ndarray,
    kw_px: float,
    z_action: Callable[[list[Splinter]], float] = None,
    clr_label=None,
    plot_vertices: bool = False,
    skip_edge: bool = False,
    mode: KernelContourMode = KernelContourMode.CONTOURS,
    exclude_points: list[tuple[int,int]] = None,
    no_ticks = True,
    figwidth = FigWidth.ROW2,
    clr_format: str = None,
    normalize: bool = True,
    crange: tuple[float,float] = None,
) -> tuple[StateOutput, Axes]:
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
    kernel = ImageKerneler(image, kw_px, skip_edge=skip_edge)
    X, Y, Z = kernel.run(z_action, exclude_points=exclude_points)

    if normalize:
        Z = Z / np.max(Z)


    return plot_kernel_results(image, clr_label, no_ticks, plot_vertices, mode, X, Y, Z, kw_px, figwidth=figwidth, clr_format=clr_format, crange=crange)

def plot_splinter_movavg(
    original_image: np.ndarray,
    splinters: list[Splinter],
    kw_px: int,
    z_action: Callable[[list[Splinter]], float] = None,
    clr_label: str = None,
    no_ticks: bool = True,
    plot_vertices: bool = False,
    skip_edge: bool = False,
    mode: KernelContourMode = KernelContourMode.CONTOURS,
    exclude_points: list[tuple[int,int]] = None,
    figwidth: FigWidth = FigWidth.ROW1,
    clr_format=None,
    normalize: bool = False,
    crange: tuple[float,float] = None,
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
    assert FigWidth.has_value(figwidth), f"FigWidth must be one of {FigWidth.values()}. plot_splinter_movavg."
    assert kw_px > 0, "kernel_width must be greater than 0."
    assert kw_px < np.min(original_image.shape[:2]), "kernel_width must be smaller than the image size."
    assert figwidth in general.width_factors, f"figwidth {figwidth} not found."

    region = (original_image.shape[1], original_image.shape[0])
    # print(f'Creating intensity plot with region={region}...')

    kernel = ObjectKerneler(
        region,
        splinters,
        collector=lambda x,r: x.in_region_px(r),
        kw_px=kw_px,
        skip_edge=skip_edge,
        skip_edge_factor=0.02,
    )

    X, Y, Z = kernel.run(z_action, mode="area", exclude_points=exclude_points)

    if normalize:
        Z = Z / np.max(Z)

    return plot_kernel_results(original_image, clr_label, no_ticks, plot_vertices, mode, X, Y, Z, kw_px, figwidth=figwidth, clr_format=clr_format, crange=crange)

def plot_kernel_results(
    original_image,
    clr_label,
    no_ticks,
    plot_vertices,
    mode: str,
    X,
    Y,
    results,
    kw_px,
    figwidth: FigWidth,
    clr_format: str = None,
    crange: tuple[float,float] = None
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

    def show_img():
        axs.imshow(original_image, interpolation='bilinear')

    def show_vertices():
        if plot_vertices:
            axs.scatter(X, Y, marker='x', c='white', s=3, linewidth=0.5)

    if mode == KernelContourMode.CONTOURS:
        axim = axs.contourf(X, Y, results, cmap='turbo', alpha=CONTOUR_ALPHA)
        show_vertices()
        show_img()
    elif mode == KernelContourMode.FILLED:
        show_vertices()
        show_img()
        results = cv2.resize(results, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        if crange is None:
            axim = axs.imshow(results, cmap='turbo', alpha=CONTOUR_ALPHA)
        else:
            axim = axs.imshow(results, cmap='turbo', alpha=CONTOUR_ALPHA, vmin=crange[0], vmax=crange[1])




    if clr_label is not None:
        # box = axs.get_position()
        # axs.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # divider = make_axes_locatable(axs)
        # cax = divider.append_axes("right", size="8%", pad=0.1)
        # cax.grid(False)
        # cbar_ax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.03, axs.get_position().height])
        cbar = fig.colorbar(axim, label=clr_label)

        renew_ticks_cb(cbar)


        if clr_format is not None:
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

    fig.tight_layout()
    # height_desired = figsize[1]  # For example, 6 inches. Adjust as needed.
    # current_size = fig.get_size_inches()
    # new_width = current_size[0] * (height_desired / current_size[1])
    # fig.set_size_inches(new_width, height_desired)
    return StateOutput(fig, figwidth, axs=axs)

def renew_ticks_cb(cbar):
    manual_ticks = [cbar.vmin, cbar.vmin + (cbar.vmax - cbar.vmin) / 2, cbar.vmax]
    # print(manual_ticks)
    cbar.ax.set_yticks(manual_ticks)
    labels = cbar.ax.get_yticklabels()

    labels[0].set_verticalalignment('bottom')
    labels[1].set_verticalalignment('center')
    labels[-1].set_verticalalignment('top')


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
    figwidth = FigWidth.ROW1,
) -> tuple[Figure, list[Axes]]:
    """Create a figure and axes for a data histogram."""
    figsize = get_fig_width(figwidth)
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


def label_image(
    image,
    *labels,
    title = None,
    nums=None,
    return_fig=True,
    figwidth=FigWidth.ROW1,
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
        fig.tight_layout()
        return StateOutput(fig, figwidth)

    return StateOutput(to_img(fig), figwidth)


def annotate_image(
    image,
    title = None,
    cbar_title = None,
    min_value = 0,
    max_value = 1,
    figwidth = FigWidth.ROW1,
    return_fig=True,
    clr_format: str = None,
) -> StateOutput:
    """Put a header in white text on top of the image.

    Args:
        image (Image): cv2.imread
        title (str): The title of the image.
    """
    assert max_value > min_value, "Max value must be greater than min value."

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

    im = ax.imshow(image, cmap='turbo', vmin=min_value, vmax=max_value, aspect='equal')
    ax.grid(False)

    if cbar_title is not None:
        cbar = fig.colorbar(mappable=im, ax=ax, label=cbar_title)
        if clr_format is not None:
            renew_ticks_cb(cbar)

            formatter = FuncFormatter(lambda x, p: f"{{0:{clr_format}}}".format(x))
            cbar.ax.yaxis.set_major_formatter(formatter)

    if return_fig:
        fig.tight_layout()
        return StateOutput(fig, figwidth)

    return StateOutput(to_img(fig), figwidth)

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
        fig.tight_layout()
        return fig
    return to_img(fig)