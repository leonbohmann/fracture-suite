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
    normalized_value = abs((value - min_value) / (max_value - min_value))

    # Choose the colormap
    colormap = mpl.colormaps[colormap_name]

    # Map the normalized value to a color
    color = colormap(normalized_value)

    # Convert the RGBA color to RGB
    rgb_color = colors.to_rgba(color)[:3]

    return tuple(int(255 * channel) for channel in rgb_color)

def norm_color(color: str | tuple[float,float,float] | tuple[int,int,int], base: float = 1.0):
    if color is None:
        return color

    if isinstance(color, str) and color.startswith('#'):
        return tuple(int(int(color.lstrip('#')[i:i+2], 16) / 255.0 * base) for i in (0, 2, 4))

    if isinstance(color, str) and ',' in color:
        return tuple(int(float(x) * base) for x in color.split(','))

    return color if isinstance(color, str) else tuple(x/255 for x in color)