import cv2
from matplotlib import colors
import numpy as np
import os
import matplotlib.pyplot as plt

from fracsuite.splinters.splinter import Splinter

def get_color(value, min_value = 0, max_value = 1, colormap_name='viridis'):
    # Normalize the value to be in the range [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)

    # Choose the colormap
    colormap = plt.get_cmap(colormap_name)

    # Map the normalized value to a color
    color = colormap(normalized_value)

    # Convert the RGBA color to RGB
    rgb_color = colors.to_rgba(color)[:3]

    return tuple(int(255 * channel) for channel in rgb_color)

def plot_impact_influence(size, splinters: list[Splinter], out_file, config,  updater = None):
    """Creates a 2D Image of the splinter orientation towards an impact point.

    Args:
        img0 (Image): Source image to plot the orientations on.
        splinters (list[Splinter]): List with splinters.
        out_file (str): Output figure file.
        config (AnalyzerConfig): Configuration
        size_f (float): _description_
        updater (_type_, optional): _description_. Defaults to None.
    """
    # analyze splinter orientations
    orientation_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    orients = []
    for s in splinters:
        if updater is not None:
            updater(0, 'Analyzing splinter orientation', len(splinters))
        orientation = s.measure_orientation(config)
        orients.append(orientation)
        color = get_color(orientation, colormap_name='turbo')
        cv2.drawContours(orientation_image, [s.contour], -1, color, -1)
        # p2 = (s.centroid_px + s.angle_vector * 15).astype(np.int32)
        # cv2.line(orientation_image, s.centroid_px, p2, (255,255,255), 3)
    cv2.circle(orientation_image, (np.array(config.impact_position) / config.size_factor).astype(np.uint32),
                np.min(orientation_image.shape[:2]) // 50, (255,0,0), -1)

    # save plot
    fig, axs = plt.subplots()
    axim = axs.imshow(orientation_image, cmap='turbo', vmin=0, vmax=1)
    fig.colorbar(axim, label='Strength  [-]')
    axs.xaxis.tick_top()
    axs.xaxis.set_label_position('top')
    axs.set_xlabel('Pixels')
    axs.set_ylabel('Pixels')
    axs.set_title('Splinter orientation towards impact point')
    # create a contour plot of the orientations, that is overlayed onto the original image
    if config.debug:
        fig.show()
        fig.waitforbuttonpress()
    
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)