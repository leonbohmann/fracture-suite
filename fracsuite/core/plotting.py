"""
Plotting helper functions
"""

from typing import Any, Callable, TypeVar
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from fracsuite.core.coloring import get_color
from fracsuite.core.image import to_rgb

from fracsuite.core.stochastics import csintkern_image, csintkern_objects
from fracsuite.splinters.splinter import Splinter



new_colormap  = mpl.colormaps['turbo'].resampled(7)
new_colormap.colors[0] = (1, 1, 1, 0)  # (R, G, B, Alpha)
modified_turbo = mpl.colors.LinearSegmentedColormap.from_list('modified_turbo', new_colormap.colors, 256,)
"Turbo but with starting color white."

def plot_splinter_kernel_contours(original_image: np.ndarray,
                   splinters: list[Splinter],
                   kernel_width: float,
                    z_action: Callable[[list[Splinter]], float] = None,
                    clr_label="Intensity [Splinters / Area]",
                    fig_title="Fracture Intensity",
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
    region = np.array([original_image.shape[1], original_image.shape[0]])
    # print(f'Creating intensity plot with region={region}...')

    X, Y, Z = csintkern_objects(region,
                                splinters,
                                lambda x,r: x.in_region_px(r),
                                kernel_width,
                                z_action,
                                skip_edge=skip_edge)
    fig,axs = plt.subplots()
    axs.imshow(original_image)

    if plot_vertices:
        axs.scatter(X, Y, marker='o', c='red')

    axim = axs.contourf(X, Y, Z, cmap='turbo', alpha=0.5)
    fig.colorbar(axim, label=clr_label)
    axs.xaxis.tick_top()
    axs.xaxis.set_label_position('top')
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(f'{fig_title} (h={kernel_width:.2f})')

    fig.tight_layout()
    return fig

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

    X, Y, Z = csintkern_image(image,
                                kernel_width,
                                z_action,
                                skip_edge=skip_edge)

    fig,axs = plt.subplots()
    axs.imshow(image)

    if plot_vertices:
        axs.scatter(X, Y, marker='o', c='red')

    axim = axs.contourf(X, Y, Z, cmap='turbo', alpha=0.5)
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

def plotImage(img,title:str, cvt_to_rgb: bool = True, region: tuple[int,int,int,int] = None):
    if cvt_to_rgb:
        img = to_rgb(img)

    fig, axs = plt.subplots()
    axs.imshow(img)
    axs.set_title(title)

    if region is not None:
        (x1, y1, x2, y2) = region
        axs.set_xlim((x1,x2))
        axs.set_ylim((y1,y2))

    plt.show()
    plt.close(fig)


def plotImages(imgs: list[(str, Any)], region = None ):
    """Plots several images side-by-side in a subplot.

    Args:
        imgs (list[tuple[str,Any]]): List of tuples containing the title and the image to plot.
        region (x,y,w,h, optional): A specific region to draw. Defaults to None.
    """
    fig,axs  = plt.subplots(1,len(imgs), sharex='all', sharey='all')
    for i, (title, img) in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].set_title(title)
        if region is not None:
            (x1, y1, w, h) = region
            axs[i].set_xlim((x1-w//2,x1+w//2))
            axs[i].set_ylim((y1-h//2,y1+h//2))
    plt.show()


def create_splinter_sizes_image(splinters: list[Splinter], shape: tuple[int,int, int], out_file: str = None):
        img = np.zeros(shape, dtype=np.uint8)
        areas = [x.area for x in splinters]

        min_area = np.min(areas)
        max_area = np.max(areas)

        for s in splinters:
            clr = get_color(s.area, min_value=min_area, max_value=max_area, colormap_name='turbo')
            cv2.drawContours(img, [s.contour], -1, clr, -1)

        cv2.imwrite(out_file, img)