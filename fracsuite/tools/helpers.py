import os
import traceback
import cv2
import re
import tempfile

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import numpy as np
from fracsuite.core.coloring import norm_color
from fracsuite.core.image import to_rgb

from fracsuite.tools.general import GeneralSettings

general = GeneralSettings.get()
def print_to_log(message: str):
    with open("log.txt", "a") as file:
        file.write(f"{message}\n")

def print_exc_to_log():
    print_to_log(traceback.format_exc())

def get_specimenname_from_path(path: os.PathLike) -> str | None:
    # find specimen pattern
    pattern = r'(\d+\.\d+\.[A-Za-z]\.\d+(-[^\s]+)?)'
    match = re.search(pattern, path)

    # Check if a match was found
    if match:
        return match.group(0)
    else:
        return os.path.basename(path)

def get_specimen_path(specimen_name: str) -> str:
    return os.path.join(general.base_path, specimen_name)

def find_file(path: os.PathLike, filter: str) -> str | None:
    """Searches a path for a file that matches with the filter.

    Args:
        path (os.PathLike): The base path to search in.
        filter (str): Filter.

    Returns:
        str | None: The full path to the found file or None, if not found.
    """
    if not os.path.exists(path):
        return None

    assert filter != "", "Filter must not be empty."

    filter = filter.lower().replace(".", "\.").replace("*", ".*")

    for file in os.listdir(path):
        if re.match(filter, file.lower()) is not None:
            return os.path.join(path, file)

    return None

def find_files(path: os.PathLike, filter: str) -> list[str]:
    """Searches a path for files that match with the filter.

    Args:
        path (os.PathLike): The path to search in.
        filter (str): Filter.

    Returns:
        list[str]: The full paths to the found files. Empty, if none found.
    """
    if not os.path.exists(path):
        return []
    if "*" in filter:
        filter = filter.replace(".", "\.").replace("*", ".*")
    files = []
    for file in os.listdir(path):
        if re.match(filter, file) is not None:
            files.append(os.path.join(path, file))

    return files

def checkmark(value: bool) -> str:
        return "[green]✔[/green]" if value else "[red]✗[/red]"

def to_img(fig):
    fig.tight_layout()
    temp_file = tempfile.mkstemp("TEMP_FIG_TO_IMG.png")[1]
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return to_rgb(cv2.imread(temp_file))

def label_image(image, *labels, title = None, nums=None, return_fig=True):
    assert len(labels) % 2 == 0, "Labels must be a multiple of 2."

    texts = labels[::2]
    labelcolors = labels[1::2]

    fig, ax = plt.subplots(figsize=(6, 5))
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


__backgrounds = ['black', 'white']
def annotate_image(
    image,
    title = None,
    cbar_title = None,
    min_value = 0,
    max_value = 1,
    figsize=general.figure_size,
    return_fig=True,
):
    """Put a header in white text on top of the image.

    Args:
        image (Image): cv2.imread
        title (str): The title of the image.
    """
    assert max_value > min_value, "Max value must be greater than min value."
    cm = 1
    fig, ax = plt.subplots(figsize=(figsize[0]*cm, figsize[1]*cm))

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
        return fig
    return to_img(fig)

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


def img_part(im, x, y, w, h):
    return im[y:y+h, x:x+w]

def bin_data(data, binrange) -> tuple[list[float], list[float]]:
    return np.histogram(data, binrange, density=True, range=(np.min(binrange), np.max(binrange)))


def dispImage(roi, title = ""):
    plt.imshow(roi)
    plt.title(title)
    plt.show()



def align_axis(ax0: Axes, ax: Axes):
    # Get the y-limits of the first axis
    ylim0 = ax0.get_ylim()
    # Calculate the scaling factor for the first axis
    fy0 = (ylim0[1] - ylim0[0]) / (ax0.get_ylim()[1] - ax0.get_ylim()[0])

    # Get the y-limits of the second axis
    ylim = ax.get_ylim()
    # Calculate the scaling factor for the second axis
    fy = (ylim[1] - ylim[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])

    # Calculate the new y-limits for the second axis based on the scaling factor
    ny0 = ylim[0] + (ylim0[0] - ax0.get_ylim()[0])/fy0 * fy
    ny1 = ny0 + (ylim0[1] - ylim0[0]) / fy0 * fy

    # Set the new y-limits for the second axis
    ax.set_ylim(ny0, ny1)