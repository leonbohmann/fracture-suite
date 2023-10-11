import json
import os
import numpy as np
import typer
import cv2
import tkinter as tk
from tkinter import Scale, Checkbutton, IntVar, Frame
from PIL import Image, ImageTk
from rich.progress import track
from fracsuite.core.image import to_gray, to_rgb
from fracsuite.core.imageprocessing import preprocess_image
from fracsuite.core.preps import PreprocessorConfig
from fracsuite.core.splinter import Splinter
from fracsuite.tools.callbacks import main_callback
from fracsuite.tools.state import State

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
        splinters = Splinter.analyze_image(to_rgb(red_overlay))
        ctrs = [x.contour for x in splinters]

        threshs.append(thresh)
        counts.append(len(ctrs))
        mean_areas.append(np.mean([x.area for x in splinters]))

    for i in track(np.linspace(1,255,10), transient=False):
        nimg_thresh = cv2.threshold(to_gray(img), int(i), 255, cv2.THRESH_BINARY)[1]
        nred_overlay = cv2.merge([nimg_thresh * 0, nimg_thresh * 0, nimg_thresh])
        nsplinters = Splinter.analyze_image(nred_overlay)
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



@tester_app.command()
def threshold(image):
    # Initialize GUI
    root = tk.Tk()
    root.title("Adaptive Threshold GUI")

    # Initialize variables
    bilateral_filter_var = IntVar()
    normal_thresh_filter_var = IntVar()

    # Load image and convert to grayscale
    img = cv2.imread(image)
    # plot_counts(image)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    blockSize = int(40*edge_density)
    print(blockSize)
    # Create Frames
    threshold_frame = Frame(root)
    threshold_frame.grid(row=0, column=0)
    bilateral_frame = Frame(root)
    bilateral_frame.grid(row=0, column=1)
    normthresh_frame = Frame(root)
    normthresh_frame.grid(row=0, column=2)

    # Create Sliders in Threshold Frame
    block_size_slider = Scale(threshold_frame, from_=3, to_=500, orient="horizontal", label="Block Size", command=lambda x: update_image())
    block_size_slider.pack()
    block_size_slider.set(213)

    c_slider = Scale(threshold_frame, from_=-50, to_=50, orient="horizontal", label="C", command=lambda x: update_image())
    c_slider.pack()

    sz_slider = Scale(threshold_frame, from_=1, to_=31, orient="horizontal", label="Gauss Size", command=lambda x: update_image())
    sz_slider.pack()
    sz_slider.set(5)
    sig_slider = Scale(threshold_frame, from_=-10, to_=28, orient="horizontal", label="Gauss Sigma", command=lambda x: update_image())
    sig_slider.pack()
    sig_slider.set(1)

    # Create Sliders in Bilateral Frame (Initially Hidden)
    d_slider = Scale(bilateral_frame, from_=1, to_=200, orient="horizontal", label="Diameter", command=lambda x: update_image())
    sigma_color_slider = Scale(bilateral_frame, from_=10, to_=200, orient="horizontal", label="SigmaColor", command=lambda x: update_image())
    sigma_space_slider = Scale(bilateral_frame, from_=10, to_=200, orient="horizontal", label="SigmaSpace", command=lambda x: update_image())

    # Create Sliders in Bilateral Frame (Initially Hidden)
    lower_slider = Scale(normthresh_frame, from_=1, to_=255, orient="horizontal", label="Lower Bound", command=lambda x: update_image())
    upper_slider = Scale(normthresh_frame, from_=1, to_=255, orient="horizontal", label="Max Value", command=lambda x: update_image())
    upper_slider.set(255)

    # Create Image Display
    label_img = tk.Label(root)
    label_img.grid(row=1, columnspan=3)

    def toggle_bilateral():
        if bilateral_filter_var.get():
            d_slider.pack()
            sigma_color_slider.pack()
            sigma_space_slider.pack()
        else:
            d_slider.pack_forget()
            sigma_color_slider.pack_forget()
            sigma_space_slider.pack_forget()

        if normal_thresh_filter_var.get():
            lower_slider.pack()
            upper_slider.pack()
        else:
            lower_slider.pack_forget()
            upper_slider.pack_forget()

        update_image()

    # Create Checkbox for bilateral filter
    bilateral_filter_check = tk.Checkbutton(bilateral_frame, text="Apply Bilateral Filter", variable=bilateral_filter_var, command=toggle_bilateral)
    bilateral_filter_check.pack()

    # Create Checkbox for bilateral filter
    norm_thresh_check = tk.Checkbutton(normthresh_frame, text="Use normal Thresh", variable=normal_thresh_filter_var, command=toggle_bilateral)
    norm_thresh_check.pack()



    def save_state():
        blockSize = block_size_slider.get()
        C = (float(c_slider.get()) / 50.0 )* 2.0
        sz = sz_slider.get()
        sig = sig_slider.get()
        if blockSize % 2 == 0:  # Ensure it's odd
            blockSize += 1
        if sz % 2 == 0:  # Ensure it's odd
            sz += 1

        prep = PreprocessorConfig(
            "test",
            block=blockSize,
            c=C,
            gauss_size=(sz,sz),
            gauss_sigma=sig
        )

        with open(State.get_output_file('prep_config.json'), 'w') as f:
            d = prep.__json__()
            d['meant_for'] = os.path.basename(image)
            json.dump(d, f, indent=4)

    save_state_button = tk.Button(threshold_frame, text="Save State", command=save_state)
    save_state_button.pack()

    def update_image():
        blockSize = block_size_slider.get()
        thresh_C = (float(c_slider.get()) / 50.0) * 2.0

        sz = sz_slider.get()
        sig = sig_slider.get()
        if blockSize % 2 == 0:  # Ensure it's odd
            blockSize += 1
        if sz % 2 == 0:  # Ensure it's odd
            sz += 1

        sz_slider.set(sz)
        block_size_slider.set(blockSize)

        if bilateral_filter_var.get():
            d = d_slider.get()
            sigmaColor = sigma_color_slider.get()
            sigmaSpace = sigma_space_slider.get()
            img_processed = cv2.bilateralFilter(img_gray, d, sigmaColor, sigmaSpace)
        elif normal_thresh_filter_var.get():
            lower = lower_slider.get()
            upper = upper_slider.get()
            img_processed = cv2.threshold(img_gray, lower, upper, cv2.THRESH_BINARY)[1]
        else:
            prep = PreprocessorConfig("test", block=blockSize, c=thresh_C, gauss_size=(sz,sz), gauss_sigma=sig)
            img_processed = preprocess_image(img, prep)


        img_thresh = img_processed
        red_overlay = cv2.merge([img_thresh * 1, img_thresh* 0, img_thresh* 0])

        # Blend the original image with the red overlay
        blended_img = cv2.addWeighted(img, 1, red_overlay, 0.3, 0)
        ctrs = [x.contour for x in Splinter.analyze_image(red_overlay)]
        if len(ctrs) > 0:
            ctrs_img = np.zeros(blended_img.shape)
            cv2.drawContours(ctrs_img, ctrs, -1, (0,0,255), 1)

        State.output_nopen(ctrs_img, 'contours', force_delete_old=True, no_print=True)

        # Blend the original image with the red overlay
        blended_img[np.all(ctrs_img == (0,0,255), axis=-1)] = ctrs_img[np.all(ctrs_img == (0,0,255), axis=-1)]
        img_pil = Image.fromarray(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label_img.config(image=img_tk)
        label_img.image = img_tk



    # First display
    update_image()

    root.mainloop()

if __name__ == "__main__":
    typer.run(threshold)
