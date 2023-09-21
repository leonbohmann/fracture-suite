import os
import cv2
import re

from matplotlib import colors, pyplot as plt
import numpy as np

from fracsuite.tools.general import GeneralSettings

general = GeneralSettings.get()

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

def write_image(out_img, out_path):
    cv2.imwrite(out_path, out_img)

def get_color(value, min_value = 0, max_value = 1, colormap_name='turbo_r'):
    # Normalize the value to be in the range [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)

    # Choose the colormap
    colormap = plt.get_cmap(colormap_name, )

    # Map the normalized value to a color
    color = colormap(normalized_value)

    # Convert the RGBA color to RGB
    rgb_color = colors.to_rgba(color)[:3]

    return tuple(int(255 * channel) for channel in rgb_color)

def checkmark(value: bool) -> str:
        return "[green]✔[/green]" if value else "[red]✗[/red]"

__backgrounds = ['black', 'white']
def annotate_image(image,
                        title = None,
                        cbar = None,
                        min_value = 0,
                        max_value = 1,
                        unit = None,
                        background = 'white'):
    """Put a header in white text on top of the image.

    Args:
        image (Image): cv2.imread
        title (str): The title of the image.
    """
    assert background in __backgrounds, f"Background must be one of {__backgrounds}."
    assert max_value > min_value, "Max value must be greater than min value."


    stack = []
    # Get the dimensions of the input image
    height, width, _ = image.shape
    font_scale = min(width, height) // 500
    value_font_scale = font_scale * 0.4
    title_thickness = int(max(5, value_font_scale // 2))
    value_thickness = title_thickness // 2

    title_height = int(0.05 * height)
    colorbar_height = int(0.05 * height)

    if background == 'white':
        title_background = np.ones((title_height, width, 3), dtype=np.uint8) * 255
        colorbar_background = np.ones((colorbar_height, width, 3), dtype=np.uint8) * 255
    elif background == 'black':
        title_background = np.zeros((title_height, width, 3), dtype=np.uint8)
        colorbar_background = np.zeros((colorbar_height, width, 3), dtype=np.uint8)



    foreground_color = (255,255,255) if background == 'black' else (0,0,0)
    # Add title text above the original image
    if title is not None:
        title_font = cv2.FONT_HERSHEY_SIMPLEX
        title_size = cv2.getTextSize(title, title_font, font_scale, title_thickness)[0]
        title_x = (width - title_size[0]) // 2
        title_y = int(title_height // 2 + title_size[1] * 0.5)
        cv2.putText(title_background, title, (title_x, title_y), title_font, font_scale, foreground_color, title_thickness)
        stack.append(title_background)

    stack.append(image)

    if cbar is not None:
        # add colorbar to the background
        scale_x0 = int(width * 0.2)
        scale_width = int(width - 2 * scale_x0)  # Adjust the width as needed
        cbar_height = int(colorbar_height * 0.6)
        cbar_y0 = int(np.ceil(colorbar_height * 0.2))

        colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, 256), cbar)
        scaled_colormap = cv2.resize(colormap, (scale_width, cbar_height))
        colorbar_background[cbar_y0:-cbar_y0, scale_x0:scale_x0 + scale_width] = scaled_colormap

        min_text= f"{min_value:.2f} {unit if unit is not None else ''}"
        max_text= f"{max_value:.2f} {unit if unit is not None else ''}"
        # Add min and max value labels with adjusted font size and thickness
        value_font = cv2.FONT_HERSHEY_SIMPLEX
        value_size = cv2.getTextSize(min_text, value_font, value_font_scale, value_thickness)[0]
        value_x = int(0.05 * width)
        value_y = int(colorbar_height // 2 + value_size[1] * 0.5)
        cv2.putText(colorbar_background, min_text, (value_x, value_y), value_font, value_font_scale, foreground_color, value_thickness)

        value_size = cv2.getTextSize(max_text, value_font, value_font_scale, value_thickness)[0]
        value_x = int(width - value_size[0] - 0.05 * width)
        cv2.putText(colorbar_background, max_text, (value_x, value_y), value_font, value_font_scale, foreground_color, value_thickness)
        stack.append(colorbar_background)

    # Combine the original image and the colorbar with title
    final_image = np.vstack(stack)

    return final_image


def img_part(im, x, y, w, h):
    return im[y:y+h, x:x+w]

def bin_data(data, binrange) -> tuple[list[float], list[float]]:
    return np.histogram(data, binrange, density=True, range=(np.min(binrange), np.max(binrange)))


def dispImage(roi, title = ""):
    plt.imshow(roi)
    plt.title(title)
    plt.show()