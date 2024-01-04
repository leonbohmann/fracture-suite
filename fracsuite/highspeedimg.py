from cProfile import label
from matplotlib import pyplot as plt
import typer
import os
import cv2
import numpy as np


from fracsuite.callbacks import main_callback
from fracsuite.core.plotting import FigureSize, get_fig_width

MARKER_MAIN = (0,0,255) # red
MARKER_BRANCH = (255,0,0) # blue


app = typer.Typer(help=__doc__, callback=main_callback)

@app.command()
def analyze(
    folder_path: str,
    fps_mio: int = 2,
    frame_size_mm: tuple[int,int] = (40, 25), #40mm x 25mm
    plot_branchings: bool = False,
    extension: str = "tiff"
):
    images = []
    frame_size_px = (400,250)
    # load all image files from folder_path
    for f in os.listdir(folder_path):
        if f.endswith(f'.{extension}'):
            image = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_COLOR)
            images.append(image)

            assert all([image.shape == images[0].shape for image in images]), "All images must have the same shape."
            assert all([image.shape[0] == frame_size_px[1] and image.shape[1] == frame_size_px[0] for image in images]), f"All images must have the shape {frame_size_px}."

    print(f"Loaded {len(images)} images.")

    # calculate the time between two images
    dt = 1 / (fps_mio) # ns

    px_per_mm = frame_size_px[1] / frame_size_mm[1] # should be 10
    print(f"px_per_mm: {px_per_mm}")

    time = [] # ns
    distance = [] # mm
    branchings_time = []
    branchings = []
    current_distance = 0
    current_time = 0
    last_tip = None
    last_branchings = 0
    # iterate images and find the crack tip
    for img in images:
        crack_tips = get_pixel(img, MARKER_MAIN, px_per_mm)
        if len(crack_tips) == 0:
            continue
        crack_tip = crack_tips[0]

        branchs = get_pixel(img, MARKER_BRANCH, px_per_mm)


        if last_tip is not None:
            # calculate distance between last tip and current tip
            current_distance += np.linalg.norm(crack_tip - last_tip)
            current_time += dt

            time.append(current_time)
            distance.append(current_distance)

            if len(branchs) > last_branchings:
                branchings_time.append(current_time)
                branchings.append(len(branchs))


        last_branchings = len(branchs)
        last_tip = crack_tip

    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    axs.plot(time,distance, label="Distance")

    if plot_branchings:
        axs2 = axs.twinx()
        for bt in branchings_time:
            bline = axs.axvline(bt, color='red', linestyle='--')
        axs2.set_ylabel("Branchings")
        axs2.tick_params(axis='y', colors=bline.get_color())

    axs.set_xlabel("Time [ns]")
    axs.set_ylabel("Distance [mm]")
    plt.show()

    # calculate velocity
    distance_m = np.asarray(distance) * 1e-3
    time = np.asarray(time) * 1e-6
    velocity = np.diff(distance_m) / np.diff(time)
    velocity_time = time[:-1] + np.diff(time) / 2

    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    axs.plot(velocity_time, velocity)
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Velocity [m/s]")
    plt.show()


def get_pixel(img, marker, px_per_mm):
    mask = cv2.inRange(img, marker, marker)

    # find the crack tip
    pixels = np.argwhere(mask)

    # swap x,y
    pixels = pixels[:,::-1]

    # convert to mm
    pixels = pixels / px_per_mm
    return pixels
