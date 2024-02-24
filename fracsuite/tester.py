import json
import os
import re
import tkinter as tk
from functools import partial
from itertools import product
from multiprocessing import Pool
from tkinter import Checkbutton, Frame, IntVar, Scale
from typing import Annotated

import cv2
from matplotlib import pyplot as plt
import numpy as np
import typer
from PIL import Image, ImageTk
from rich import inspect, print
from rich.progress import Progress, track

from fracsuite.callbacks import main_callback
from fracsuite.core.geometry import ellipse_radius
from fracsuite.core.image import to_gray, to_rgb
from fracsuite.core.imageprocessing import preprocess_image
from fracsuite.core.preps import PrepMode, PreprocessorConfig
from fracsuite.core.specimen import Specimen
from fracsuite.core.splinter import Splinter
from fracsuite.core.stochastics import similarity, similarity_count
from fracsuite.core.vectors import angle_between
from fracsuite.helpers import find_file
from fracsuite.state import State

tester_app = typer.Typer(callback=main_callback)

def plot_counts(image):
    img = cv2.imread(image)
    threshs = []
    counts = []
    mean_areas = []

    nthreshs = []
    ncounts = []
    nmean_areas = []

    for i in track(np.linspace(3,700,10), transient=False):
        ii = int(i)
        thresh = ii + (ii+1)%2
        prep = PreprocessorConfig("test", block=thresh, c=0)
        red_overlay = preprocess_image(img, prep)
        splinters = Splinter.analyze_image(to_rgb(red_overlay), skip_preprocessing=True)
        ctrs = [x.contour for x in splinters]

        threshs.append(thresh)
        counts.append(len(ctrs))
        mean_areas.append(np.mean([x.area for x in splinters]))

    for i in track(np.linspace(1,255,10), transient=False):
        nimg_thresh = cv2.threshold(to_gray(img), int(i), 255, cv2.THRESH_BINARY)[1]
        nred_overlay = cv2.merge([nimg_thresh * 0, nimg_thresh * 0, nimg_thresh])
        nsplinters = Splinter.analyze_image(nred_overlay, skip_preprocessing=True)
        nctrs = [x.contour for x in nsplinters]

        nthreshs.append(int(i))
        ncounts.append(len(nctrs))
        nmean_areas.append(np.mean([x.area for x in nsplinters]))


    import matplotlib.pyplot as plt
    plt.plot(threshs, counts)
    plt.plot(threshs, mean_areas)
    plt.show()

    plt.plot(nthreshs, ncounts, label='Counts')
    plt.plot(nthreshs, nmean_areas, label='Areas')
    plt.legend()
    plt.show()

def resize_images(img , *others):
    # calculate scale factor so that larger side is 500px
    scale_factor = 500 / max(img.shape[0], img.shape[1])
    img = cv2.resize(img, (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor)))
    for label_img in others:
        label_img = cv2.resize(label_img, (int(label_img.shape[1]*scale_factor), int(label_img.shape[0]*scale_factor)))

    return img, *others

def evaluate_params(params, img, label_areas):
    block, c, gauss_size, gauss_sigma, lum, correct_light, clahe_strength, clahe_size = params
    prep = PreprocessorConfig(
        "test",
        block=block,
        c=c,
        gauss_size=gauss_size,
        gauss_sigma=gauss_sigma,
        lum=lum,
        correct_light=correct_light,
        clahe_strength=clahe_strength,
        clahe_size=clahe_size
    )
    splinters = Splinter.analyze_image(img, prep=prep)
    areas = [x.area for x in splinters]

    binrange = np.linspace(1, 9000, 20)
    sim = similarity(areas, label_areas, binrange=binrange, no_print=True)[-1] # absolute error in splinters
    similarity_value = sim

    return (params, similarity_value,prep)
@tester_app.command()
def best_params_mp(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    dir = os.path.dirname(image)
    label_path = find_file(dir, 'label.png')
    label_img = cv2.imread(label_path, cv2.IMREAD_COLOR)
    label_splinters = Splinter.analyze_label_image(label_img)
    label_areas = [x.area for x in label_splinters]


    # Define parameter ranges
    block_range = np.linspace(3, 101, 50)


    c_range = np.linspace(0, 5, 5)
    gauss_size_range = [(3,3), (5,5)]
    gauss_sigma_range = np.linspace(0, 10, 10)
    lum_range = [0]
    correct_light_range = [False]
    clahe_strength_range = [0] #[3, 5, 7]
    clahe_size_range = [0] # [8]

    print("Generating combinations...")


    # Generate combinations
    param_combinations = list(product(
        block_range, c_range, gauss_size_range, gauss_sigma_range,
        lum_range, correct_light_range, clahe_strength_range, clahe_size_range
    ))

    results = {}
    best_similarity = float('inf')
    best_params = None
    best_prep = None

    with Progress() as progress:
        task = progress.add_task("[cyan]Optimizing Parameters...", total=len(param_combinations))

        with Pool() as pool:
            partial_evaluate_params = partial(evaluate_params, img=img, label_areas=label_areas)
            for params, sim, prep in pool.imap(partial_evaluate_params, param_combinations):
                progress.update(task, advance=1)

                results[params] = sim

                if sim < best_similarity:
                    best_similarity = sim
                    best_params = params
                    best_prep = prep
                    progress.print("[bold green]<<< NEW BEST PARAM FOUND >>")
                    progress.print(f"{best_params}: {best_similarity:.2f}")
                    progress.update(task, description=f"Current best Similarity: {best_similarity}")


    print("Best params: ", best_params)
    print("Best similarity: ", best_similarity)

    inspect(best_prep)
    # save prep to inout image directory
    output_file = os.path.join(dir, "bestprep.json")
    with open(output_file, 'w') as f:
        d = best_prep.__json__()
        json.dump(d, f, indent=4)
    return best_params, results



@tester_app.command()
def best_params(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    dir = os.path.dirname(image)
    label_path = find_file(dir, 'label.png')
    label_img = cv2.imread(label_path, cv2.IMREAD_COLOR)
    label_splinters = Splinter.analyze_label_image(label_img)
    label_areas = [x.area for x in label_splinters]

    binrange = np.linspace(np.min(label_areas), np.max(label_areas), 20)

    # Define parameter ranges
    block_range = np.linspace(3, 500, 20)
    # make sure blockrange is odd
    block_range = block_range + (block_range+1)%2
    # remove duplicates from blockrange
    block_range = np.unique(block_range)


    c_range = np.linspace(0, 5, 5)
    gauss_size_range = [(3,3), (5,5)]
    gauss_sigma_range = np.linspace(0, 10, 10)
    lum_range = [0]
    correct_light_range = [True, False]
    clahe_strength_range = [3, 5, 7]
    clahe_size_range = [8]

    print("Generating combinations...")


    # Generate combinations
    param_combinations = list(product(
        block_range, c_range, gauss_size_range, gauss_sigma_range,
        lum_range, correct_light_range, clahe_strength_range, clahe_size_range
    ))


    best_similarity = float('-inf')  # Initialize to a high value
    best_params = None
    results = {}

    for block, c, gauss_size, gauss_sigma, lum, correct_light, clahe_strength, clahe_size in track(param_combinations):
        prep = PreprocessorConfig(
            "test",
            block=block,
            c=c,
            gauss_size=gauss_size,
            gauss_sigma=gauss_sigma,
            lum=lum,
            correct_light=correct_light,
            clahe_strength=clahe_strength,
            clahe_size=clahe_size
        )
        prep_image = preprocess_image(img, prep)
        splinters = Splinter.analyze_image(prep_image, skip_preprocessing=True)
        areas = [x.area for x in splinters]


        sim = similarity(areas, label_areas, binrange=binrange, no_print=True)
        similarity_value = sim[2]  # MSE

        param_tuple = (block, c, gauss_size, gauss_sigma, lum, correct_light, clahe_strength, clahe_size)
        results[param_tuple] = similarity_value

        if similarity_value > best_similarity:
            best_similarity = similarity_value
            best_params = param_tuple
            print(similarity_value)

    return best_params, results

@tester_app.command()
def threshold(
    source: Annotated[str, typer.Argument(help='Path to image file.')],
    region: Annotated[tuple[int,int,int,int], typer.Option(help='')] = (250,250,100,100),
    region_f: Annotated[tuple[float,float], typer.Option(help='Region center in percent.')] = None,
):
    # Initialize GUI
    root = tk.Tk()
    root.title("Adaptive Threshold GUI")

    # Initialize variables
    bilateral_filter_var = IntVar()
    normal_thresh_filter_var = IntVar()
    correct_light_var = IntVar()

    from fracsuite.core.preps import defaultPrepConfig
    prep0 = defaultPrepConfig

    dir = os.path.dirname(source)

    if (specimen := Specimen.get(source, panic=False)) is not None:
        print("[cyan]Specimen detected")
        source = specimen.get_splinter_outfile("dummy")
        # switch directory to specimen directory
        dir = os.path.dirname(source)

        img = specimen.get_fracture_image()
        px_per_mm = specimen.calculate_px_per_mm()
        print("px/mm: ", px_per_mm)
        
        # take a small portion of the image
        if region is None:
            region = np.array((250,250,500,500)) * px_per_mm
        else:
            region = np.array(region) * px_per_mm

        if region_f is not None:
            region = img.shape[1] * region_f[0], img.shape[0] * region_f[1], region[2], region[3]
            region = np.asarray(region)

        region = region.astype(np.uint32)
        print(region)
        print(img.shape)
        # get region cx cy w h from image
        img = img[region[1]-region[3]//2:region[1]+region[3]//2, region[0]-region[2]//2:region[0]+region[2]//2]

        # img = img[region[0]-region[2]//2:region[0]+region[2]//2, region[1]-region[3]//2:region[1]-region[3]//2]
        is_specimen = True

        if (pconf := specimen.get_prepconf()) is not None:
            print('Loading preprocessor config...')
            prep0 = pconf
            inspect(prep0)
    else:
        # Load image and convert to grayscale
        img = cv2.imread(source, cv2.IMREAD_COLOR)
        is_specimen = False

        if (prepfile := find_file(dir, 'prep.json')) is not None:
            print('Loading preprocessor config...')
            prep0 = PreprocessorConfig.load(prepfile)
            inspect(prep0)

    label_path = find_file(dir, 'label.png')
    label_img = None
    if label_path is not None:
        label_img = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label_splinters = Splinter.analyze_label_image(label_img)
        label_areas = [x.area for x in label_splinters]
    else:
        label_img = None

    # # resize image if it is too large
    # if img.shape[0] > 1000 and img.shape[1] > 1000:
    #     # take region of image
    #     img = img[250:750, 250:750]



    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create Frames
    threshold_frame = Frame(root)
    threshold_frame.grid(row=0, column=0)
    second_frame = Frame(root)
    second_frame.grid(row=0, column=1)
    normthresh_frame = Frame(root)
    normthresh_frame.grid(row=0, column=2)

    # Create Sliders in Threshold Frame
    block_size_slider = Scale(threshold_frame, from_=3, to_=200, orient="horizontal", label="Block Size", command=lambda x: update_image())
    block_size_slider.set(prep0.athresh_block_size)
    block_size_slider.pack()

    c_slider = Scale(threshold_frame, from_=-10, to_=10, orient="horizontal", label="C", command=lambda x: update_image())
    c_slider.pack()
    c_slider.set(prep0.athresh_c)

    sz_slider = Scale(threshold_frame, from_=1, to_=31, orient="horizontal", label="Gauss Size", command=lambda x: update_image())
    sz_slider.pack()
    sz_slider.set(prep0.gauss_size[0])
    sig_slider = Scale(threshold_frame, from_=-10, to_=28, orient="horizontal", label="Gauss Sigma", command=lambda x: update_image())
    sig_slider.pack()
    sig_slider.set(prep0.gauss_sigma)
    lum_slider = Scale(threshold_frame, from_=-255, to_=255, orient="horizontal", label="Luminance Delta", command=lambda x: update_image())
    lum_slider.pack()
    lum_slider.set(prep0.lum if prep0.lum is not None else 0)

    clahe_strength = Scale(second_frame, from_=0, to_=255, orient="horizontal", label="Clahe Strength", command=lambda x: update_image())
    clahe_strength.pack()
    clahe_strength.set(prep0.clahe_strength)
    clahe_size = Scale(second_frame, from_=3, to_=15, orient="horizontal", label="Clahe Size", command=lambda x: update_image())
    clahe_size.pack()
    clahe_size.set(prep0.clahe_size)

    similarity_label = tk.Label(second_frame, text="Similarity: ")
    similarity_label.pack()

    # Create Sliders in Bilateral Frame (Initially Hidden)
    lower_slider = Scale(normthresh_frame, from_=-1, to_=255, orient="horizontal", label="Lower Bound", command=lambda x: update_image())
    upper_slider = Scale(normthresh_frame, from_=1, to_=255, orient="horizontal", label="Max Value", command=lambda x: update_image())
    lower_slider.set(prep0.nthresh_lower)
    upper_slider.set(prep0.nthresh_upper)
    lower_slider.pack()
    upper_slider.pack()

    # Create Image Display
    label_field = tk.Label(root)
    label_field.grid(row=1, columnspan=3)


    # Create Checkbox for bilateral filter
    norm_thresh_check = tk.Checkbutton(normthresh_frame, text="Use normal Thresh", variable=normal_thresh_filter_var, command=lambda: update_image())
    norm_thresh_check.pack()
    normal_thresh_filter_var.set(True if prep0.mode == PrepMode.NORMAL else False)

    # Create Checkbox for bilateral filter
    correct_light_check = tk.Checkbutton(threshold_frame, text="Correct light", variable=correct_light_var, command=lambda: update_image())
    correct_light_check.pack()
    correct_light_var.set(prep0.correct_light)

    def load_state(prep: PreprocessorConfig):
        block_size_slider.set(prep.athresh_block_size)
        c_slider.set(prep.athresh_c)
        sz_slider.set(prep.gauss_size[0])
        sig_slider.set(prep.gauss_sigma)

        lum_slider.set(prep.lum if prep.lum is not None else 0)
        clahe_strength.set(prep.clahe_strength)
        clahe_size.set(prep.clahe_size)

        if prep.mode == PrepMode.NORMAL:
            normal_thresh_filter_var.set(True)
        else:
            normal_thresh_filter_var.set(False)

        if prep.correct_light:
            correct_light_var.set(True)
        else:
            correct_light_var.set(False)

        lower_slider.set(prep.nthresh_lower)
        upper_slider.set(prep.nthresh_upper)

        root.update()


    def save_state():
        blockSize = block_size_slider.get()
        C = c_slider.get()
        sz = sz_slider.get()
        sig = sig_slider.get()
        if blockSize % 2 == 0:  # Ensure it's odd
            blockSize += 1
        if sz % 2 == 0:  # Ensure it's odd
            sz += 1
        lum = lum_slider.get()

        cl_strength = clahe_strength.get()
        cl_size = clahe_size.get()

        if not is_specimen:
            name = os.path.basename(source)
            if "." in name:
                name = name[:name.rindex(".")]
            name = name + "_prep"
        else:
            name = specimen.name + "_prep"

        prep = PreprocessorConfig(
            name,
            mode=PrepMode.ADAPTIVE if not normal_thresh_filter_var.get() else PrepMode.NORMAL,
            block=blockSize,
            c=C,
            gauss_size=(sz,sz),
            gauss_sigma=sig,
            lum = lum,
            clahe_size=cl_size,
            clahe_strength=cl_strength,
            correct_light=correct_light_var.get(),
        )

        with open(State.get_output_file('prep_config.json'), 'w') as f:
            d = prep.__json__()
            d['meant_for'] = os.path.basename(source)
            json.dump(d, f, indent=4)

        if is_specimen:
            output_file = specimen.get_splinter_outfile("prep.json")
            with open(output_file, 'w') as f:
                d = prep.__json__()
                json.dump(d, f, indent=4)
        else:
            output_file = os.path.join(dir, name) + ".json"
            with open(output_file, 'w') as f:
                d = prep.__json__()
                json.dump(d, f, indent=4)


    save_state_button = tk.Button(threshold_frame, text="Save State", command=save_state)
    save_state_button.pack()

    def update_image():
        blockSize = block_size_slider.get()
        thresh_C = c_slider.get()

        sz = sz_slider.get()
        sig = sig_slider.get()
        if blockSize % 2 == 0:  # Ensure it's odd
            blockSize += 1
        if sz % 2 == 0:  # Ensure it's odd
            sz += 1
        lum = lum_slider.get()

        sz_slider.set(sz)
        block_size_slider.set(blockSize)

        cl_strength = clahe_strength.get()
        cl_size = clahe_size.get()

        prep = PreprocessorConfig(
            "test",
            block=blockSize,
            c=thresh_C,
            gauss_size=(sz,sz),
            gauss_sigma=sig,
            lum=lum,
            correct_light=correct_light_var.get(),
            clahe_strength=cl_strength,
            clahe_size=cl_size,
            mode=PrepMode.ADAPTIVE if not normal_thresh_filter_var.get() else PrepMode.NORMAL,
            nthresh_lower=lower_slider.get(),
            nthresh_upper=upper_slider.get()
        )
        img_processed = preprocess_image(img, prep)


        img_thresh = img_processed
        red_overlay = cv2.merge([img_thresh * 1, img_thresh* 0, img_thresh* 0])

        label = None
        if label_img is not None:
            label = cv2.threshold(label_img, 0, 255, cv2.THRESH_BINARY)[1]
            label = 255-to_gray(label)
            green_overlay = cv2.merge([label * 0, label* 1, label* 0])


        # Blend the original image with the red overlay
        blended_img = cv2.addWeighted(to_rgb(img_gray), 1, red_overlay, 0.3, 0)
        splinters = Splinter.analyze_image(red_overlay, skip_preprocessing=True)
        ctrs = [x.contour for x in splinters]
        if len(ctrs) > 0:
            ctrs_img = np.zeros(blended_img.shape)
            cv2.drawContours(ctrs_img, ctrs, -1, (0,0,255), 2)

        State.output_nopen(ctrs_img, 'contours', force_delete_old=True, no_print=True)

        if label is not None:
            area0 = np.array([x.area for x in splinters])
            binrange = np.linspace(np.min(label_areas), np.max(label_areas), 20)


            sim = similarity(area0, label_areas, binrange=binrange, no_print=True)
            sim = sim[-1] # absolute error
            sim = sim
            # print text to similarrity label
            similarity_label.config(text=f"Difference: {sim:.0f} Spl.")



        # Blend the original image with the red overlay
        if label is not None:
            blended_img[np.all(green_overlay == (0,255,0), axis=-1)] = green_overlay[np.all(green_overlay == (0,255,0), axis=-1)]
        blended_img[np.all(ctrs_img == (0,0,255), axis=-1)] = ctrs_img[np.all(ctrs_img == (0,0,255), axis=-1)]


        blended_img = resize_images(blended_img)[0]
        img_pil = Image.fromarray(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label_field.config(image=img_tk)
        label_field.image = img_tk

    # First display
    # update_image()
    load_state(prep0)

    root.mainloop()


@tester_app.command()
def preprocess(name):
    specimen = Specimen.get(name)
    fracture_image = specimen.get_fracture_image()
    px_per_mm = specimen.calculate_px_per_mm()
    prep = specimen.get_prepconf()
    splinters = Splinter.analyze_image(fracture_image, prep=prep, px_per_mm=px_per_mm)

@tester_app.command()
def roundrect(angle: int = 100):
    # create an image and draw a couple of rotated rectangles with different angles on it
    img = np.ones((200,200,3), np.uint8) * 255

    # draw a rectangle with a 40 degree angle
    rotrect = ((100,100),(20,50),25)
    box = np.int0(cv2.boxPoints(rotrect))
    cv2.drawContours(img,[box],0,(0,0,255),2)
    print('Width', 50)

    # draw a rectangle with a 60 degree angle
    rotrect = ((100,100),(50,20),60)
    box = np.int0(cv2.boxPoints(rotrect))
    cv2.drawContours(img,[box],0,(0,255,0),2)
    print('Width', 50)


    # plt.imshow(img)
    # plt.show()

    img = np.zeros((200,200,3), np.uint8)

    # draw an ellipse into the image
    ellipse = (100,100), (100,50), angle
    cv2.ellipse(img, ellipse, (255,255,255), -1)

    # perfrom contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw the contours into the image
    cv2.drawContours(img, contours, -1, (0,255,0), 1)
    # plt.imshow(img)
    # plt.show()

    img = np.zeros((200,200,3), np.uint8)
    cv2.ellipse(img, ellipse, (255,255,255), -1)

    # fit an ellipse to the contours
    ellipse_fit = cv2.fitEllipse(contours[0])
    # draw the fitted ellipse into the image
    cv2.ellipse(img, ellipse_fit, (0,0,255), 1)

    # find major axis and minor axis vectors
    a = ellipse_fit[1][0] / 2
    b = ellipse_fit[1][1] / 2
    major_axis = (a * np.cos(np.deg2rad(ellipse_fit[2])), a * np.sin(np.deg2rad(ellipse_fit[2])))
    minor_axis = (b * np.cos(np.deg2rad(ellipse_fit[2] + 90)), b * np.sin(np.deg2rad(ellipse_fit[2] + 90)))
    # plot them to the image
    center = (int(ellipse_fit[0][0]), int(ellipse_fit[0][1]))
    cv2.arrowedLine(img, center, (int(center[0] + major_axis[0]), int(center[1] + major_axis[1])), (125,0,255), 2)
    cv2.arrowedLine(img, center, (int(center[0] + minor_axis[0]), int(center[1] + minor_axis[1])), (0,125,255), 2)

    print('Fitted', ellipse_fit)
    print('angle', np.degrees(angle_between(major_axis, [1,0])))


    plt.imshow(img)
    plt.show()



@tester_app.command()
def alignment(
    A : Annotated[tuple[float,float], typer.Option(help="The first vector")] = (0,0),
    B : Annotated[tuple[float,float], typer.Option(help="The first vector")] = (0,0),
):
    A = np.array(A)
    B = np.array(B)

    dot = np.dot(A, B)
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    a = np.abs(dot / (magA * magB))
    print('1-a=', 1 - a)
    print('a=',a)

@tester_app.command()
def ellipse_r(
    a:float,
    b:float,
    theta:float,
):
    """Returns the radius of an ellipse at a given angle."""
    theta = np.deg2rad(theta)
    r = ellipse_radius(a,b,theta)
    print(r)

@tester_app.command()
def angle(
    x1:float,
    y1:float,
    x2:float,
    y2:float,
):
    A = np.array((x1,y1))
    B = np.array((x2,y2))
    theta = angle_between(A,B)
    print(np.rad2deg(theta))

if __name__ == "__main__":
    typer.run(threshold)
