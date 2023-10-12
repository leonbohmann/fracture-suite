import json
import os
import re
import tkinter as tk
from functools import partial
from itertools import product
from multiprocessing import Pool
from tkinter import Checkbutton, Frame, IntVar, Scale

import cv2
import numpy as np
import typer
from PIL import Image, ImageTk
from rich import print
from rich.progress import Progress, track

from fracsuite.callbacks import main_callback
from fracsuite.core.image import to_gray, to_rgb
from fracsuite.core.imageprocessing import preprocess_image
from fracsuite.core.preps import PrepMode, PreprocessorConfig
from fracsuite.core.specimen import Specimen
from fracsuite.core.splinter import Splinter
from fracsuite.core.stochastics import similarity, similarity_count
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

def resize_images(img , label_img):
    # calculate scale factor so that larger side is 500px
    scale_factor = 500 / max(img.shape[0], img.shape[1])
    img = cv2.resize(img, (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor)))
    if label_img is not None:
        label_img = cv2.resize(label_img, (int(label_img.shape[1]*scale_factor), int(label_img.shape[0]*scale_factor)))

    return img, label_img

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
    sim = similarity_count(areas, label_areas, binrange=binrange)
    similarity_value = sim

    return (params, similarity_value)
@tester_app.command()
def best_params_mp(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    dir = os.path.dirname(image)
    label_path = find_file(dir, 'label.png')
    label_img = cv2.imread(label_path, cv2.IMREAD_COLOR)
    label_splinters = Splinter.analyze_label_image(label_img)
    label_areas = [x.area for x in label_splinters]


    # Define parameter ranges
    block_range = np.linspace(3, 250, 20)


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
    best_similarity = float('-inf')
    best_params = None

    with Progress() as progress:
        task = progress.add_task("[cyan]Optimizing Parameters...", total=len(param_combinations))

        with Pool() as pool:
            partial_evaluate_params = partial(evaluate_params, img=img, label_areas=label_areas)
            for params, sim in pool.imap(partial_evaluate_params, param_combinations):
                progress.update(task, advance=1)

                results[params] = sim

                if sim > best_similarity:
                    best_similarity = sim
                    best_params = params
                    progress.print("[bold green]<<< NEW BEST PARAM FOUND >>")
                    progress.print(f"{best_params}")
                    progress.update(task, description=f"Current best Similarity: {best_similarity}")


    print("Best params: ", best_params)
    print("Best similarity: ", best_similarity)
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
def threshold(image):
    # Initialize GUI
    root = tk.Tk()
    root.title("Adaptive Threshold GUI")

    # Initialize variables
    bilateral_filter_var = IntVar()
    normal_thresh_filter_var = IntVar()
    correct_light_var = IntVar()

    if re.match(r'.*\..*\..*\..*', image):
        print("[cyan]Specimen detected")
        specimen = Specimen.get(image)
        image = specimen.get_splinter_outfile("dummy")
        img = specimen.get_fracture_image()
        # take a small portion of the image
        img = img[500:1000, 500:1000]
        is_specimen = True
    else:
        # Load image and convert to grayscale
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        is_specimen = False

    # plot_counts(img)

    dir = os.path.dirname(image)
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
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    blockSize = int(40*edge_density)
    print('Edge density blocksize: ', blockSize)
    # Create Frames
    threshold_frame = Frame(root)
    threshold_frame.grid(row=0, column=0)
    second_frame = Frame(root)
    second_frame.grid(row=0, column=1)
    normthresh_frame = Frame(root)
    normthresh_frame.grid(row=0, column=2)

    # Create Sliders in Threshold Frame
    block_size_slider = Scale(threshold_frame, from_=3, to_=500, orient="horizontal", label="Block Size", command=lambda x: update_image())
    block_size_slider.pack()
    block_size_slider.set(213)

    c_slider = Scale(threshold_frame, from_=-10, to_=10, orient="horizontal", label="C", command=lambda x: update_image())
    c_slider.pack()

    sz_slider = Scale(threshold_frame, from_=1, to_=31, orient="horizontal", label="Gauss Size", command=lambda x: update_image())
    sz_slider.pack()
    sz_slider.set(5)
    sig_slider = Scale(threshold_frame, from_=-10, to_=28, orient="horizontal", label="Gauss Sigma", command=lambda x: update_image())
    sig_slider.pack()
    sig_slider.set(1)
    lum_slider = Scale(threshold_frame, from_=-255, to_=255, orient="horizontal", label="Luminance Delta", command=lambda x: update_image())
    lum_slider.pack()
    lum_slider.set(0)

    clahe_strength = Scale(second_frame, from_=0, to_=255, orient="horizontal", label="Clahe Strength", command=lambda x: update_image())
    clahe_strength.pack()
    clahe_strength.set(5)
    clahe_size = Scale(second_frame, from_=3, to_=15, orient="horizontal", label="Clahe Strength", command=lambda x: update_image())
    clahe_size.pack()
    clahe_size.set(8)

    similarity_label = tk.Label(second_frame, text="Similarity: ")
    similarity_label.pack()

    # Create Sliders in Bilateral Frame (Initially Hidden)
    lower_slider = Scale(normthresh_frame, from_=-1, to_=255, orient="horizontal", label="Lower Bound", command=lambda x: update_image())
    upper_slider = Scale(normthresh_frame, from_=1, to_=255, orient="horizontal", label="Max Value", command=lambda x: update_image())
    upper_slider.set(255)
    lower_slider.pack()
    upper_slider.pack()

    # Create Image Display
    label_field = tk.Label(root)
    label_field.grid(row=1, columnspan=3)


    # Create Checkbox for bilateral filter
    norm_thresh_check = tk.Checkbutton(normthresh_frame, text="Use normal Thresh", variable=normal_thresh_filter_var, command=lambda: update_image())
    norm_thresh_check.pack()

    # Create Checkbox for bilateral filter
    correct_light_check = tk.Checkbutton(threshold_frame, text="Correct light", variable=correct_light_var, command=lambda: update_image())
    correct_light_check.pack()

    def save_state():
        blockSize = block_size_slider.get()
        C = (float(c_slider.get()) / 50.0 )* 2.0
        sz = sz_slider.get()
        sig = sig_slider.get()
        if blockSize % 2 == 0:  # Ensure it's odd
            blockSize += 1
        if sz % 2 == 0:  # Ensure it's odd
            sz += 1
        lum = lum_slider.get()

        cl_strength = clahe_strength.get()
        cl_size = clahe_size.get()

        prep = PreprocessorConfig(
            specimen.name + "_prep",
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
            d['meant_for'] = os.path.basename(image)
            json.dump(d, f, indent=4)

        if is_specimen:
            output_file = specimen.get_splinter_outfile("prep.json")
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
            label = to_gray(label)
            green_overlay = cv2.merge([label * 0, label* 1, label* 0])


        # Blend the original image with the red overlay
        blended_img = cv2.addWeighted(to_rgb(img_processed), 1, red_overlay, 0.3, 0)
        splinters = Splinter.analyze_image(red_overlay, skip_preprocessing=True)
        ctrs = [x.contour for x in splinters]
        if len(ctrs) > 0:
            ctrs_img = np.zeros(blended_img.shape)
            cv2.drawContours(ctrs_img, ctrs, -1, (0,0,255), 1)

        State.output_nopen(ctrs_img, 'contours', force_delete_old=True, no_print=True)

        if label is not None:
            area0 = np.array([x.area for x in splinters])
            binrange = np.linspace(np.min(label_areas), np.max(label_areas), 20)


            sim = similarity(area0, label_areas, binrange=binrange, no_print=True)
            sim = sim[2]
            # print text to similarrity label
            similarity_label.config(text=f"Similarity: {sim:.2f}%")



        # Blend the original image with the red overlay
        if label is not None:
            blended_img[np.all(green_overlay == (0,255,0), axis=-1)] = green_overlay[np.all(green_overlay == (0,255,0), axis=-1)]
        blended_img[np.all(ctrs_img == (0,0,255), axis=-1)] = ctrs_img[np.all(ctrs_img == (0,0,255), axis=-1)]


        blended_img = resize_images(blended_img, None)[0]
        img_pil = Image.fromarray(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label_field.config(image=img_tk)
        label_field.image = img_tk


    # First display
    update_image()

    root.mainloop()

if __name__ == "__main__":
    typer.run(threshold)