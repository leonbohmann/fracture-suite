from typing import Any
from matplotlib import pyplot as plt
import typer
import os
import cv2
import numpy as np


from fracsuite.callbacks import main_callback
from fracsuite.core.plotting import FigureSize, get_fig_width, transparent_line
from fracsuite.state import State

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
    images: list[tuple[str,Any]] = []
    frame_size_px = (400,250)

    # calculate the time between two images
    dt = 1 / (fps_mio) # ns

    px_per_mm = frame_size_px[1] / frame_size_mm[1] # should be 10
    print(f"px_per_mm: {px_per_mm}")

    ic = 0
    # load all image files from folder_path
    for f in os.listdir(folder_path):
        if f.endswith(f'.{extension}'):
            image = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_COLOR)
            images.append((f,image, ic))
            ic += 1



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
                print('Crack tips:', crack_tips)
                raise Exception(f"Crack tip with color {crack_clr} missing in {img_name}! Maybe you forgot to indicate an ending with 4 pixels?")

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
        time_s = np.asarray(stime) * 1e-6
        velocity = np.diff(distance_m) / np.diff(time_s)
        velocity_time = time_s[:-1] + np.diff(time_s) / 2

        fig,axs = plt.subplots(2,1,figsize=get_fig_width(FigureSize.ROW1))

        dst_ax = axs[0]   # axes for distance

        # image plot
        ax_big = axs[1]
        # add a scale to the image
        scale = px_per_mm # px/mm
        scale_length = 10 # mm
        scale_px = scale * scale_length
        scale_pos = (10,img.shape[0]-10)
        cv2.line(img, scale_pos, (int(scale_pos[0]+scale_px), int(scale_pos[1])), (255,255,255), 2)
        cv2.putText(img, f"{scale_length}mm", (int(scale_pos[0]+scale_px//2), int(scale_pos[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        ax_big.imshow(img)
        ax_big.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False,
            right=False,
            left=False,
            labelleft=False
        )


        # distance plot
        dst_line = dst_ax.plot(stime,sdistance, label="Distance", color='orange', zorder=10)
        dst_ax.set_xlabel("Time [ns]")
        dst_ax.set_ylabel("Distance [mm]")

        # angle plot
        ang_ax = dst_ax.twinx()
        ang_line = ang_ax.plot(stime, sangle, label="Angle", color='gray', linestyle='-.', linewidth=0.5)
        ang_ax.set_ylabel("Angle [°]")
        ang_ax.tick_params(axis='y', colors='gray')

        # velocity plot
        vel_ax = dst_ax.twinx()   # axes for velocity
        vel_ax.spines['right'].set_position(('outward', 40))  # move the right spine outward
        vel_ax.tick_params(axis='y', colors='blue')
        vline = vel_ax.plot(velocity_time, velocity, color='blue', linestyle='--', label="Velocity", linewidth=0.5)
        vel_ax.set_ylabel("Velocity [m/s]")



        dst_ax.set_zorder(3)
        dst_ax.set_frame_on(False)
        dst_ax.legend([dst_line[0], ang_line[0], vline[0]], ["Distance", "Angle", "Velocity"], loc='upper left')
        fig.tight_layout()

        State.output(fig, f"sys{c}_plot", figwidth=FigureSize.ROW1)


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
                    cv2.ellipse(img, tuple(p0.astype(int)), (2,2), 0, 0, 360, (0,0,255), -1)
                img = transparent_line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 127, 255), 2, 0.4)

            # crop the image to the region
            img = img[min_y:max_y, min_x:max_x]

            State.output_nopen(img, f"sys{c}/running_{img_index:02d}", figwidth=FigureSize.ROW2)

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
