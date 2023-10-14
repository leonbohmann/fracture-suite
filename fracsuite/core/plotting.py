"""
Plotting helper functions
"""
from __future__ import annotations

import tempfile
from enum import Enum
from typing import Any, Callable

import cv2
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

general = GeneralSettings.get()

CONTOUR_ALPHA = 0.8

new_colormap  = mpl.colormaps['turbo'].resampled(7)
new_colormap.colors[0] = (1, 1, 1, 0)  # (R, G, B, Alpha)
modified_turbo = mpl.colors.LinearSegmentedColormap.from_list('modified_turbo', new_colormap.colors, 256,)
"Turbo but with starting color white."

def get_figure_size_fraction(wf, hf=None, dimf=1.2) -> tuple[tuple[float,float],float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
        width: float
            Document textwidth or columnwidth in pts
        fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
        fig_dim: tuple
            Dimensions of figure in inches
        fraction: float
            The fraction, so it can be appended to the filename.
    """
    # Width of figure (in pts)
    fig_width_pt = general.document_width_pt * wf

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt * dimf
    # Figure height in inches
    fig_height_in = fig_width_in * (hf if hf is not None else golden_ratio)

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def to_img(fig):
    fig.tight_layout()
    temp_file = tempfile.mkstemp("TEMP_FIG_TO_IMG.png")[1]
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return to_rgb(cv2.imread(temp_file))



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
    w_fraction = 1.0,
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
        skip_edge_factor=0.02,
    )

    X, Y, Z = kernel.run(z_action, mode="area")

    figsize, fraction = get_figure_size_fraction(w_fraction)

    return plot_kernel_results(original_image, clr_label, no_ticks, plot_vertices, mode, X, Y, Z, figsize=figsize)

def plot_kernel_results(original_image, clr_label, no_ticks, plot_vertices, mode, X, Y, Z, figsize):
    fig,axs = plt.subplots(figsize=figsize)
    axs.imshow(original_image)

    if plot_vertices:
            axs.scatter(X, Y, marker='o', c='red')

    if mode == KernelContourMode.CONTOURS:
        axim = axs.contourf(X, Y, Z, cmap='turbo', alpha=CONTOUR_ALPHA)
    elif mode == KernelContourMode.RECT:
        z_im = cv2.resize(Z, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        axim = axs.imshow(z_im, cmap='turbo', alpha=CONTOUR_ALPHA)

    if clr_label is not None:
        # box = axs.get_position()
        # axs.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # divider = make_axes_locatable(axs)
        # cax = divider.append_axes("right", size="8%", pad=0.1)
        # cax.grid(False)
        # cbar_ax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.03, axs.get_position().height])
        cbar = fig.colorbar(axim, label=clr_label)

        #TODO: CREATE manual ticks for cbar and align first bottom and last top
        for t in cbar.ax.get_yticklabels():
            t.set_horizontalalignment('right')
            t.set_x(3.5)





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
        w_fraction = 1.0,
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
    Z = Z / np.max(Z)

    figsize, _ = get_figure_size_fraction(w_fraction)
    return plot_kernel_results(image, clr_label, no_ticks, plot_vertices, mode, X, Y, Z, figsize=figsize)


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
    fig_fracw = 1.0,
) -> tuple[Figure, list[Axes]]:
    """Create a figure and axes for a data histogram."""
    figsize = get_figure_size_fraction(fig_fracw)
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
    fig_wfrac=1.0,
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

    fig, ax = plt.subplots(figsize=get_figure_size_fraction(fig_wfrac))
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
        return fig
    return to_img(fig)


def annotate_image(
    image,
    title = None,
    cbar_title = None,
    min_value = 0,
    max_value = 1,
    fig_wfrac = 1.0,
    return_fig=True,
):
    """Put a header in white text on top of the image.

    Args:
        image (Image): cv2.imread
        title (str): The title of the image.
    """
    assert max_value > min_value, "Max value must be greater than min value."

    figsize = get_figure_size_fraction(fig_wfrac)
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
    if cbar_title is not None:
        fig.colorbar(mappable=im, ax=ax, label=cbar_title)

    if return_fig:
        fig.tight_layout()
        return (fig, fig_wfrac)

    return (to_img(fig), fig_wfrac)

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