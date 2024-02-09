from typing import Any
from matplotlib import pyplot as plt
import typer
import os
import cv2
import numpy as np


from fracsuite.callbacks import main_callback
from fracsuite.core.plotting import FigureSize, get_fig_width, transparent_line
from fracsuite.core.stochastics import moving_average
from fracsuite.state import State

MARKER_MAIN = (0,0,255) # red
MARKER_BRANCH = (255,0,0) # blue


app = typer.Typer(help=__doc__, callback=main_callback)

@app.command()
def vid2img(
    vid_path: str,
    every:int = 1
):
    """
    Converts a video file to a series of images.
    """
    vid_name = os.path.basename(vid_path).split('.')[0]
    vidcap = cv2.VideoCapture(vid_path)
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()
    count = 0
    while success:
        if count % every == 0:
            print(f"Frame {count}/{n_frames}")
            cv2.imwrite(os.path.join(os.path.dirname(vid_path), f"{vid_name}_{count:02d}.tiff"), image)     # save frame as TIFF file
        success, image = vidcap.read()
        count += 1

@app.command()
def extract_details(
    folder_path: str,
    frame_start: int,
    frame_end: int,
    name: str,
    region: tuple[int,int,int,int],
    extension: str = "tiff",
    region_points: bool = False
):
    if folder_path.endswith("/"):
        folder_path = folder_path[:-1]

    if region_points:
        w = region[2] - region[0]
        h = region[3] - region[1]
        # x and y are the center of the region
        x = region[0] + w/2
        y = region[1] + h/2
        region = (x,y,w,h)

    # load images from folder
    images: list[tuple[str,Any]] = []
    for f in os.listdir(folder_path):
        if f.endswith(extension):
            image = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_COLOR)

            # extract last digit from filename
            index = int(f.split('.')[0].split("_")[-1])
            images.append((f, index, image))

    # sort images by index
    images.sort(key=lambda x: x[1])

    # get index in images where index is frame_start
    start_index = next(i for i,(_,index,_) in enumerate(images) if index == frame_start)
    end_index = next(i for i,(_,index,_) in enumerate(images) if index == frame_end)

    images = images[start_index:end_index+1]

    # crop images
    x,y,w,h = region
    images = [(f, index, img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]) for f,index,img in images]

    # save images to output folder
    output_path = os.path.join(State.get_output_dir(), name)
    os.makedirs(output_path, exist_ok=True)

    # add a scale to the image
    scale = 10 # px/mm
    img_scale_f = 2
    scale_length = 2 # mm
    scale_px = scale * scale_length *img_scale_f

    for f,_,img in images:
        img = cv2.resize(img, (img.shape[1]*img_scale_f, img.shape[0]*img_scale_f))

        scale_pos = (5*img_scale_f,img.shape[0]-5*img_scale_f)

        cv2.line(img, scale_pos, (int(scale_pos[0]+scale_px), int(scale_pos[1])), (0,0,0), 2)
        _, sz = cv2.getTextSize(f"{scale_length}mm", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, f"{scale_length}mm", (int(scale_pos[0]), int(scale_pos[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imwrite(os.path.join(output_path, f) + ".png", img)




    #

@app.command()
def prepare(
    folder_path: str,
    output_name: str,
    extension: str = "tiff"
):
    """
    Prepares the images in the given folder by calculating the absolute difference
    between two images and adding a weighted version of the difference to the image.

    This enhances the crack tip surrounding and makes it easier to detect.
    """
    if folder_path.endswith("/"):
        folder_path = folder_path[:-1]

    output_path = os.path.join(folder_path, output_name)
    os.makedirs(output_path, exist_ok=True)

    # load all image files from folder_path
    images: list[tuple[str,Any]] = []
    for f in os.listdir(folder_path):
        if f.endswith(extension):
            image = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_COLOR)
            images.append((f,image))

    # now, create the absolute difference between every image
    diffs = []
    for i in range(len(images)-1):
        img1 = images[i][1]
        img2 = images[i+1][1]
        diff = cv2.absdiff(img2, img1)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        diffs.append(diff)

    # save the differences by adding a weighted version of diff to the frame
    for i in range(len(diffs)):
        print(f"Saving {i+1}/{len(diffs)}")
        img = images[i][1]
        diff = diffs[i]
        img = cv2.addWeighted(img, 1, diff, 0.1, 1)
        cv2.imwrite(os.path.join(output_path, images[i][0]), img)

@app.command()
def analyze(
    folder_path: str,
    fps_mio: int = 2,
    frame_size_mm: tuple[int,int] = (40, 25), #40mm x 25mm
    plot_branchings: bool = False,
    extension: str = "tiff"
):
    """
    Analyzes the images in the given folder and plots the results.

    Args:
        folder_path (str): The path to the folder containing the images.
        fps_mio (int, optional): Million fps. Defaults to 2.
        frame_size_mm (tuple[int,int], optional): The size of the image in mm. Defaults to (40, 25).
        extension (str, optional): Image file extension. Defaults to "tiff".
    """
    if folder_path.endswith("/"):
        folder_path = folder_path[:-1]

    folder_name = os.path.basename(folder_path)
    images: list[tuple[str,Any]] = []
    frame_size_px = (400,250)

    # calculate the time between two images
    dt = 1 / (fps_mio) # us

    px_per_mm = frame_size_px[1] / frame_size_mm[1] # should be 10
    print(f"px_per_mm: {px_per_mm}")

    ic = 0
    # load all image files from folder_path
    for f in os.listdir(folder_path):
        if f.endswith(f'.{extension}'):
            image = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_COLOR)
            images.append((f,image, ic))
            ic += 1

            # folder_name = os.path.basename(f)

            assert all([img.shape == images[0][1].shape for _,img,_ in images]), "All images must have the same shape."
            assert all([img.shape[0] == frame_size_px[1] and img.shape[1] == frame_size_px[0] for _,img,_ in images]), f"All images must have the shape {frame_size_px}."

    print(f"Loaded {len(images)} images.")



    CRACK_STARTING = 0          # crack is starting
    CRACK_RUNNING = 1           # crack is running
    CRACK_BRANCHING = 2         # crack is branching into other cracks
    CRACK_BRANCHING_ENDING = 3  # crack is branching and ending
    CRACK_ENDING = 4            # crack is just ending

    # holds the points of the individual crack systems
    crack_tip_systems: dict[tuple,Any] = {}


    time = [0] # ns

    # iterate images and find the crack tip
    for img_name, img, index in images:
        # time increment after each image
        current_time = time[-1] + dt
        time.append(current_time)

        # reshape the image to be a 2D array where each row is a pixel and each column is a color channel
        reshaped_img = img.reshape(-1, img.shape[-1])
        # find the unique colors
        unique_colors = np.unique(reshaped_img, axis=0)

        # remove all colors from unique_colors that are grayscale
        unique_colors = [tuple(clr) for clr in unique_colors if not (clr[0] == clr[1] == clr[2])]
        print(unique_colors)
        for clr in unique_colors:
            if clr not in crack_tip_systems:
                # [time, distance, status, positions, angle]
                crack_tip_systems[clr] = [[],[],[],[],[],[]]




        for crack_clr in crack_tip_systems:
            stime = crack_tip_systems[crack_clr][0]
            sdistance = crack_tip_systems[crack_clr][1]
            status = crack_tip_systems[crack_clr][2]
            positions = crack_tip_systems[crack_clr][3]
            sangle = crack_tip_systems[crack_clr][4]
            sindices = crack_tip_systems[crack_clr][5]

            # last status of the crack
            last_status = status[-1] if len(status) > 0 else CRACK_STARTING
            if last_status == CRACK_ENDING or last_status == CRACK_BRANCHING_ENDING:
                continue

            # get all crack tips of the current system
            crack_tips = get_pixel(img, crack_clr, px_per_mm)
            current_status = len(crack_tips)
            if current_status > 0:
                crack_tip = crack_tips[0]
            else:
                continue
                # raise Exception(f"Crack tip with color {crack_clr} missing in {img_name}! Maybe you forgot to indicate an ending with 4 pixels?")

            # time increment
            stime.append(current_time)
            # distance increment
            d_distance = np.linalg.norm(crack_tip - positions[-1]) if len(positions) > 0 else 0
            current_distance = sdistance[-1] + d_distance if len(sdistance) > 0 else d_distance
            sdistance.append(current_distance)

            if len(positions) > 0:
                dp = crack_tip - positions[-1]
                angle = np.arctan2(dp[1], dp[0]) * 180 / np.pi
                if len(sangle) > 0:
                    angle = np.abs(sangle[-1] - angle)
            else:
                angle = 0

            sangle.append(angle)
            status.append(current_status)
            positions.append(crack_tip)
            sindices.append(index)


    fig_width = FigureSize.ROW1HL
    # plot results
    for c,crack_sys in enumerate(crack_tip_systems):
        print(f"Crack system {crack_sys}:")
        print(f'> {len(crack_tip_systems[crack_sys][0])} timesteps')
        print(f'> {len(crack_tip_systems[crack_sys][1])} distances')
        print(f'> {len(crack_tip_systems[crack_sys][2])} status')
        print(f'> {len(crack_tip_systems[crack_sys][3])} positions')
        print(f'> {len(crack_tip_systems[crack_sys][4])} angles')

        stime = crack_tip_systems[crack_sys][0]
        sdistance = crack_tip_systems[crack_sys][1]
        status = crack_tip_systems[crack_sys][2]
        positions = crack_tip_systems[crack_sys][3]
        sangle = crack_tip_systems[crack_sys][4]

        # trace the crack tip on the full image (last)
        img = images[-1][1].copy()
        for ip in range(len(positions)-1):
            p0 = positions[ip] * px_per_mm
            p1 = positions[ip+1] * px_per_mm
            if status[ip] == CRACK_BRANCHING:
                cv2.ellipse(img, tuple(p0.astype(int)), (5,5), 0, 0, 360, (0,0,255), -1)
            cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (255,0,0), 3)

        # calculate velocity
        distance_m = np.asarray(sdistance) * 1e-3
        time_s = np.asarray(stime)
        velocity = np.diff(distance_m) / np.diff(time_s) / 1e-6
        velocity_time = time_s[:-1] + np.diff(time_s) / 2

        fig,axs = plt.subplots(1,1,figsize=get_fig_width(fig_width))

        dst_ax = axs   # axes for distance

        # distance plot
        dst_line = dst_ax.plot(stime,sdistance, label="Weg", color='orange', zorder=10)
        dst_ax.set_xlabel("Zeit [$\mu$s]")
        dst_ax.set_ylabel("Weg [mm]")

        # angle plot
        ang_ax = dst_ax.twinx()
        ang_line = ang_ax.plot(stime, sangle, label="Rel. Winkel", color='gray', linestyle='-.', linewidth=0.75)
        ang_ax.set_ylabel("Rel. Winkel [°]")
        ang_ax.tick_params(axis='y', colors='gray')
        ang_ax.grid(False)

        # velocity plot
        vel_ax = dst_ax.twinx()   # axes for velocity
        vel_ax.spines['right'].set_position(('outward', 40))  # move the right spine outward
        vel_ax.tick_params(axis='y', colors='blue')
        vline = vel_ax.plot(velocity_time, velocity, color='blue', linestyle='--', label="Geschwindigkeit", linewidth=0.75)
        vel_ax.set_ylabel("Geschwindigkeit [m/s]")
        vel_ax.grid(False)

        # create a linear fit for the distance
        fit = np.polyfit(stime, sdistance, 1)
        # get the steigung and y-axis intercept of the fit
        mean_velocity = fit[0]
        intercept = fit[1]
        # plot the fitting function
        dst_ax.plot(stime, mean_velocity*np.asarray(stime)+intercept, color='orange', linestyle='--', linewidth=0.75)
        # plot the mean_velocity as hline
        vel_ax.axhline(mean_velocity * 1e3, color='blue', linestyle='--', linewidth=0.5)
        # annotate the axhline with the mean_velocity
        vel_ax.annotate(f"{mean_velocity*1e3:.2f}m/s", (stime[-3], mean_velocity * 1e3), textcoords="offset points", xytext=(0.5,0), ha='center', color='blue')


        # mark branchings in plot as points
        for i in range(len(status)):
            if status[i] == CRACK_BRANCHING:
                dst_ax.plot(stime[i], sdistance[i], 'o', color='red', zorder=11)

        # insert tick into existing ticks for velocity
        vel_ticks = vel_ax.get_yticks()
        vel_ticks_labels = vel_ax.get_yticklabels()
        vel_ticks = np.insert(vel_ticks, 0, mean_velocity * 1e3)
        vel_ticks_labels = [f"{mean_velocity*1e3:.0f}"] + vel_ticks_labels
        vel_ax.set_yticks(vel_ticks)
        vel_ax.set_yticklabels(vel_ticks_labels)

        dst_ax.set_zorder(3)
        dst_ax.set_frame_on(False)
        dst_ax.legend([dst_line[0], ang_line[0], vline[0]], ["Weg", "Rel. Winkel", "Geschwindigkeit"], loc='best')

        State.output(fig, f"{folder_name}/sys{c}_plot", figwidth=fig_width)
        print('Mean velocity: ', mean_velocity)


    for c,crack_sys in enumerate(crack_tip_systems):
        stime = crack_tip_systems[crack_sys][0]
        sdistance = crack_tip_systems[crack_sys][1]
        status = crack_tip_systems[crack_sys][2]
        positions = np.asarray(crack_tip_systems[crack_sys][3])
        sangle = crack_tip_systems[crack_sys][4]
        sindices = crack_tip_systems[crack_sys][5]

        # find maximum region
        min_x = np.min(positions[:,0])
        max_x = np.max(positions[:,0])
        min_y = np.min(positions[:,1])
        max_y = np.max(positions[:,1])

        # convert to pixels
        min_x = int(min_x * px_per_mm)
        max_x = int(max_x * px_per_mm)
        min_y = int(min_y * px_per_mm)
        max_y = int(max_y * px_per_mm)

        for i, img_index in enumerate(sindices):
            img = images[img_index][1].copy()

            # draw the previous crack line
            for ip in range(i):
                p0 = positions[ip] * px_per_mm
                p1 = positions[ip+1] * px_per_mm
                if status[ip] == CRACK_BRANCHING:
                    cv2.ellipse(img, tuple(p0.astype(int)), (5,5), 0, 0, 360, (0,0,255), -1)
                    cv2.ellipse(img, tuple(p0.astype(int)), (5,5), 0, 0, 360, (0,0,255), 1)

                img = transparent_line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 127, 255), 2, 0.4)

            # crop the image to the region
            img = img[min_y:max_y, min_x:max_x]

            # add a scale to the image
            scale = 10 # px/mm
            img_scale_f = 2
            scale_length = 2 # mm
            scale_px = scale * scale_length *img_scale_f

            img = cv2.resize(img, (img.shape[1]*img_scale_f, img.shape[0]*img_scale_f))

            scale_pos = (5*img_scale_f,img.shape[0]-5*img_scale_f)

            cv2.line(img, scale_pos, (int(scale_pos[0]+scale_px), int(scale_pos[1])), (0,0,0), 2)
            _, sz = cv2.getTextSize(f"{scale_length}mm", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(img, f"{scale_length}mm", (int(scale_pos[0]), int(scale_pos[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            State.output_nopen(img, f"{folder_name}/sys{c}/running_{stime[i]:.2f}_{img_index:02d}", figwidth=FigureSize.ROW2)

    print(f'dt={dt}ns')
    print(f'px_per_mm={px_per_mm}')


def get_pixel(img, marker, px_per_mm):
    marker = np.asarray(marker)
    mask = cv2.inRange(img, marker, marker)

    # find the crack tip
    pixels = np.argwhere(mask)

    # swap x,y
    pixels = pixels[:,::-1]

    # convert to mm
    pixels = pixels / px_per_mm
    return pixels
