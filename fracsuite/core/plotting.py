"""
Plotting helper functions
"""

from matplotlib import pyplot as plt
import numpy as np
from fracsuite.core.image import to_rgb
from fracsuite.core.stochastics import csintkern_splinters
from fracsuite.splinters.splinter import Splinter


def plotImage(img,title:str, color: bool = True, region: tuple[int,int,int,int] = None):
    if color:
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

def plot_intensity(original_image: np.ndarray,
                   splinters: list[Splinter],
                   region_size: float,
                    z_action,
                    clr_label="Intensity [Splinters / Area]",
                    fig_title="Fracture Intensity",
                    xlabel="Pixels",
                    ylabel="Pixels",):
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

    X, Y, Z = csintkern_splinters(region, splinters, region_size, z_action)
    fig,axs = plt.subplots()
    axs.imshow(original_image)
    axim = axs.contourf(X, Y, Z, cmap='turbo', alpha=0.5)
    fig.colorbar(axim, label=clr_label)
    axs.xaxis.tick_top()
    axs.xaxis.set_label_position('top')
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(f'{fig_title} (h={region_size:.2f})')

    fig.tight_layout()
    return fig