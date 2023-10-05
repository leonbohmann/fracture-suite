"""
Coloring functions.
"""

import random

from matplotlib import colors
import matplotlib as mpl


def rand_col():
    """Generate a random color.

    Returns:
        (r,g,b): A random color.
    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def get_color(value, min_value = 0, max_value = 1, colormap_name='turbo'):
    # Normalize the value to be in the range [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)

    # Choose the colormap
    colormap = mpl.colormaps[colormap_name]

    # Map the normalized value to a color
    color = colormap(normalized_value)

    # Convert the RGBA color to RGB
    rgb_color = colors.to_rgba(color)[:3]

    return tuple(int(255 * channel) for channel in rgb_color)
