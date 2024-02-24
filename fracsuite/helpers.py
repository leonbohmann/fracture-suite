import os
import traceback
import re

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from fracsuite.general import GeneralSettings

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




def img_part(im, x, y, w, h):
    return im[y:y+h, x:x+w]

def bin_data(data, binrange,density=True) -> tuple[list[float], list[float]]:
    return np.histogram(data, binrange, density=density, range=(np.min(binrange), np.max(binrange)))


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