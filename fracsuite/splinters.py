"""
Splinter analyzation tools.
"""

import multiprocessing.shared_memory as sm
import os
import pickle
import re
import shutil
import sys
from itertools import groupby
from typing import Annotated, Any, Callable

import cv2
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path
import numpy as np
import numpy.typing as npt
import typer
from matplotlib import pyplot as plt
from rich import inspect, print
from rich.progress import track

from fracsuite.callbacks import main_callback
from fracsuite.core.calculate import pooled
from fracsuite.core.coloring import get_color, rand_col
from fracsuite.core.detection import attach_connections, get_adjacent_splinters_parallel
from fracsuite.core.image import put_text, to_gray, to_rgb
from fracsuite.core.imageplotting import plotImage, plotImages
from fracsuite.core.imageprocessing import crop_matrix, crop_perspective
from fracsuite.core.kernels import ObjectKerneler
from fracsuite.core.plotting import (
    DataHistMode,
    DataHistPlotMode,
    FigureSize,
    KernelContourMode,
    annotate_image,
    create_splinter_colored_image,
    create_splinter_sizes_image,
    datahist_plot,
    datahist_to_ax,
    get_fig_width,
    label_image,
    modified_turbo,
    plot_image_movavg,
    plot_splinter_movavg,
)
from fracsuite.core.preps import defaultPrepConfig
from fracsuite.core.progress import get_progress
from fracsuite.core.specimen import Specimen
from fracsuite.core.splinter import Splinter
from fracsuite.core.stochastics import similarity
from fracsuite.general import GeneralSettings
from fracsuite.helpers import bin_data, find_file, find_files
from fracsuite.state import State, StateOutput

app = typer.Typer(help=__doc__, callback=main_callback)

general = GeneralSettings.get()
IMNAME = "fracture_image"

@app.command()
def gen(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
    realsize: Annotated[tuple[float, float], typer.Option(help='Real size of specimen in mm.')] = (-1, -1),
    quiet: Annotated[bool, typer.Option(help='Do not ask for confirmation.')] = False,
    all: Annotated[bool, typer.Option(help='Generate splinters for all specimens.')] = False,
    all_exclude: Annotated[str, typer.Option(help='Exclude specimens from all.')] = None,
    all_skip_existing: Annotated[bool, typer.Option(help='Skip specimens that already have splinters.')] = False,
    from_label: Annotated[bool, typer.Option(help='Generate splinters from labeled image.')] = False,
):
    """Generate the splinter data for a specific specimen."""
    if realsize[0] == -1 or realsize[1] == -1:
        realsize = None

    if not all:
        specimen = Specimen.get(specimen_name, load=False)
        specimens = [specimen]
    else:
        def exclude(specimen: Specimen):
            if not specimen.has_fracture_scans:
                return False
            if specimen.has_splinters and all_skip_existing:
                return False
            return re.search(all_exclude, specimen.name) is None
        specimens = Specimen.get_all_by(decider=exclude, lazyload=True)
    with get_progress(total=len(specimens)) as progress:
        progress.set_total(len(specimens))
        for specimen in specimens:
            if specimen.has_splinters:
                progress.pause()
                if not quiet and not typer.confirm(f"> Specimen '{specimen.name}' already has splinters. Overwrite?"):
                    progress.resume()
                    return
                progress.resume()

            fracture_image = specimen.get_fracture_image()
            px_per_mm = specimen.calculate_px_per_mm(realsize)
            prep = specimen.get_prepconf()
            # generate splinters for the specimen
            print(f'Using  px_per_mm = {px_per_mm:.2f}')
            print(f'            prep = "{prep.name}"')

            print('Running analysis...')
            if not from_label:
                splinters = Splinter.analyze_image(fracture_image, px_per_mm=px_per_mm, prep=prep)
            elif from_label:
                label_img = specimen.get_label_image()
                p = defaultPrepConfig
                p.max_area = 1e15
                splinters = Splinter.analyze_label_image(label_img, px_per_mm=px_per_mm, prep=p)

            # save splinters to specimen
            output_file = specimen.get_splinter_outfile("splinters_v2.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(splinters, f)
            print(f'Saved splinters to "{output_file}"')
            progress.advance()


def plot_touching_len(specimen, splinters):
    out_img = specimen.get_fracture_image()
    touches = [len(x.adjacent_splinter_ids) for x in splinters]
    max_adj = np.max(touches)
    min_adj = np.min(touches)

    print(f"Max touching splinters: {max_adj}")
    print(f"Min touching splinters: {min_adj}")
    print(f"Mean touching splinters: {np.mean(touches)}")

    for splinter in track(splinters, description="Drawing touching splinters...", transient=True):
        clr = get_color(len(splinter.adjacent_splinter_ids), min_adj, max_adj)
        cv2.drawContours(out_img, [splinter.contour], 0, clr, -1)


    out_img = annotate_image(
        out_img,
        cbar_title="Amount of touching splinters",
        clr_format=".2f",
        min_value=min_adj,
        max_value=max_adj,
        figwidth=FigureSize.ROW2,
    )

    return out_img

@app.command()
def gen_adjacent_all(
    input_folder: str
):
    """
    Loads all splinters from the input folder, generates adjacent ids
    and overwrites the original file with the new data.
    """
    # get all pkl files in input_folder
    files = find_files(input_folder, "*.pkl")
    for file in files:
        with open(file, 'rb') as f:
            splinters = pickle.load(f)

        adjacent_ids = get_adjacent_splinters_parallel(splinters, (4000,4000))

        # output file is the same as input file but with _adjacent appended
        attach_connections(splinters, adjacent_ids)

        with open(file, 'wb') as f:
            pickle.dump(splinters, f)

@app.command()
def gen_adjacent(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load or a pkl file.')],
):
    """Modify the existing splinters and attach adjacent splinters into them."""

    if re.match(r'.*\..*\..*\..*', specimen_name):
        specimen = Specimen.get(specimen_name, load=False)
        assert specimen.has_splinters, "Specimen has no splinters."
        specimen.load_splinters()
    elif specimen_name.endswith(".pkl"):
        raise Exception("Generating adjacency from .pkl file is not supported yet.")
        # with open(specimen_name, 'rb') as f:
        #     splinters = pickle.load(f)
        # output_splinter_file = os.path.join(os.path.dirname(specimen_name), Specimen.adjacency_file)
    else:
        raise Exception("Invalid input. Neither .pkl file nor specimen name.")

    adjacent_ids = get_adjacent_splinters_parallel(
        specimen.splinters,
        specimen.get_fracture_image().shape[:2]
    )
    attach_connections(specimen.splinters, adjacent_ids)
    with open(specimen.splinters_file, 'wb') as f:
        pickle.dump(specimen.splinters, f)


@app.command()
def plot_adjacent(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
    n_range: Annotated[tuple[int, int], typer.Option(help='Range of splinters to analyze.')] = (0, 20),
):
    specimen = Specimen.get(specimen_name)
    assert specimen.has_splinters, "Specimen has no splinters."
    splinters = specimen.splinters

    outp = plot_touching_len(specimen, splinters)
    State.output(outp, spec=specimen, to_additional=True)


    lens = [len(x.adjacent_splinter_ids) for x in splinters]
    fig, axs = datahist_plot(
        x_label='Amount of edges $N_e$',
        y_label='Probability Density $p(N_e)$',
        figwidth=FigureSize.ROW1,
    )

    br = np.linspace(np.min(lens)-0.5, np.max(lens)+0.5, np.max(lens)-np.min(lens)+2)
    datahist_to_ax(
        axs[0],
        lens,
        binrange=br,
        data_mode=DataHistMode.PDF,
        plot_mode=DataHistPlotMode.HIST,
        alpha = 1.0,
        unit = "",
        as_log=False,
        mean_format=".0f",
    )
    axs[0].autoscale()
    axs[0].set_xlim(n_range)

    State.output(fig, spec=specimen, to_additional=True)

@app.command()
def plot_adjacent_detail(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
    n: Annotated[int, typer.Option(help='Splinter to analyze.')] = 5,
):
    """
    Plots the adjacent detail of a splinter in a specimen.

    Args:
        specimen_name (str): Name of the specimen to load.
        n (int, optional): Amount of splinters to analyze. Defaults to 5.
    """
    specimen = Specimen.get(specimen_name)
    assert specimen.has_splinters, "Specimen has no splinters."
    assert specimen.has_adjacency, "Specimen has no adjacency data."
    splinters = specimen.splinters

    # create example images
    for i in range(n):
        im0 = specimen.get_fracture_image()
        rnd_i = np.random.randint(0, len(splinters))
        splinter = splinters[rnd_i]
        print(f'Random splinter: {rnd_i}, Touching Splinters: {len(splinter.adjacent_splinter_ids)}')

        for ij, j in enumerate(splinter.adjacent_splinter_ids):
            cv2.drawContours(im0, [splinters[j].contour], 0, (0, 125, 255), 1)

        # draw splinter in red
        cv2.drawContours(im0, [splinter.contour], 0, (0, 0, 255), 2)


        # retrieve region around splinter
        x0, y0, w, h = cv2.boundingRect(splinter.contour)
        # enlarge bounding rect and make sure that it is inside the image
        x0 -= 50
        y0 -= 50
        w += 100
        h += 100
        w = min(im0.shape[1], w)
        h = min(im0.shape[0], h)
        x = max(0, x0)
        x = min(im0.shape[1]-w, x)
        y = max(0, y0)
        y = min(im0.shape[0]-h, y)
        im0 = im0[y:y+h, x:x+w]
        im0 = cv2.resize(im0, (0, 0), fx=1.5, fy=1.5)

        for ij, j in enumerate(splinter.adjacent_splinter_ids):
            t_point = ((splinters[j].centroid_px[0]-x)*1.5, (splinters[j].centroid_px[1]-y)*1.5)
            text = f'{ij+1}'
            # calculate text size so its always the same size
            put_text(text, im0, t_point, clr = (255,255,255))

        State.output(im0, 'splinter_detail',spec=specimen, to_additional=True, mods=[i])

@app.command()
def draw_contours(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
):
    specimen = Specimen.get(specimen_name)
    assert specimen.has_splinters, "Specimen has no splinters."
    splinters = specimen.splinters

    out_img = specimen.get_fracture_image()
    for splinter in track(splinters, description="Drawing contours...", transient=True):
        clr = rand_col()
        cv2.drawContours(out_img, [splinter.contour], 0, clr, 1)

    State.output(out_img, 'contours',spec=specimen, to_additional=True, figwidth=FigureSize.ROW1)

def check_chunk(i, chunksize, p_len):
    points_sm = sm.SharedMemory(name='points')
    points = np.ndarray(p_len, dtype=np.uint8, buffer=points_sm.buf)

    i0 = i * chunksize
    matches = []
    for j in range(i0, i0+chunksize):
        if j >= len(points):
            break

        p1 = points[j]
        for k in range(j+1, len(points)):
            p2 = points[k]
            if p1[0] == p2[0] and p1[1] == p2[1]:
                matches.append((j,k))

    return matches

@app.command()
def export_files(
    output_path: Annotated[str, typer.Argument(help='Path to output file.')],
):
    specimens: list[Specimen] = Specimen.get_all_by(lambda x: x.has_splinters)

    for specimen in specimens:
        print(f"Loaded splinters for {specimen.name}")

        out_file = specimen.get_splinter_outfile("splinters_v2.pkl")
        # copy the file to the output_path
        shutil.copy(out_file, output_path)
        os.rename(f"{output_path}/splinters_v2.pkl", f"{output_path}/{specimen.name}.pkl")

@app.command()
def import_files(
    input_folder: Annotated[str, typer.Argument(help='Path to input folder.')],
):
    # get all pkl files in input_folder
    files = find_files(input_folder, "*.pkl")
    for file in files:
        specimen_name = os.path.basename(file).replace(".pkl", "")
        specimen = Specimen.get(specimen_name, False)
        assert specimen.has_splinters, "Specimen has no splinters."

        # copy the file into specimen splinter folder
        shutil.copy(file, specimen.get_splinter_outfile("splinters_v2.pkl"))
        print(f"Imported adjacent file for {specimen_name}.")

@app.command()
def show_prep():
    """Show the default preprocessing configuration."""
    inspect(defaultPrepConfig)


@app.command(name='norm')
def count_splinters_in_norm_region(
        specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
        norm_region_center: Annotated[
            tuple[int, int], typer.Option(help='Center of the norm region in mm.', metavar='X Y')] = (400, 400),
        norm_region_size: Annotated[
            tuple[int, int], typer.Option(help='Size of the norm region in mm.', metavar='W H')] = (50, 50),
) -> float:
    specimen = Specimen.get(specimen_name)
    assert specimen is not None, "Specimen not found."

    s_count, splinters_in_region = specimen.calculate_esg_norm(norm_region_center, norm_region_size)

    print(f'Splinters in norm region: {s_count}')

    detail, overview = specimen.plot_region_count(norm_region_center, norm_region_size, splinters_in_region, s_count)

    State.output(detail, figwidth=FigureSize.ROW2, spec=specimen, to_additional=True, mods=['detail'])
    State.output(overview, figwidth=FigureSize.ROW2, spec=specimen, to_additional=True, mods=['overview'])
    # w0, h0 = int(25*f), int(25*f)
    # w, h = int(normed_image.shape[1]), int(normed_image.shape[0])

    # circ = Circle(impact_pos, 100*f, fill=False)
    # rect0 = Rectangle((0,0), w0, h, fill=False)
    # rect1 = Rectangle((0,0), w, h0, fill=False)
    # rect2 = Rectangle((w-w0,0), w0, h, fill=False)
    # rect3 = Rectangle((0,h-h0), w, h0, fill=False)

    # hatches = [circ, rect0, rect1, rect2, rect3]

    # for h in hatches:
    #     h.set_hatch('x')
    #     h.set_alpha(0.5)
    #     h.set_edgecolor('red')
    #     h.set_linewidth(20)


    # fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW2))
    # axs.axis('off')
    # axs.imshow(normed_image)
    # for h in hatches:
    #     axs.add_patch(h)
    # State.output(StateOutput(fig, figwidth=FigureSize.ROW2), spec=specimen, to_additional=True, mods=['overview'])




    return s_count


@app.command()
def roughness_f(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
    w_mm: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 50,
    n_points: Annotated[int, typer.Option(help='Amount of points in kerneler.')] = 50,
):
    """Create a contour plot of the roughness on the specimen.

    Args:
        specimen_name (str, optional): Name of the specimens to load.
    """

    def roughness_function(splinters: list[Splinter]):
        return np.mean([splinter.calculate_roughness() for splinter in splinters])

    # create contour plot of roughness
    specimen = Specimen.get(specimen_name)
    assert specimen is not None, "Specimen not found."

    fig = plot_splinter_movavg(
        specimen.get_fracture_image(),
        splinters=specimen.splinters,
        kw_px=w_mm * specimen.calculate_px_per_mm(),
        n_points=n_points,
        z_action=roughness_function,
        clr_label='Mean roughness $\\bar{\lambda}_r$',
        figwidth=FigureSize.ROW2,
        clr_format=".2f",
        mode=KernelContourMode.FILLED,
    )

    State.output(fig, spec=specimen, to_additional=True)


@app.command()
def roundness_f(
        specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
        w_mm: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 50,
        as_contours: Annotated[bool, typer.Option(help='Plot the kernel as contours.')] = False,
        n_points: Annotated[int, typer.Option(help='Amount of points to evaluate.')] = general.n_points_kernel,
):
    """Create a contour plot of the roundness on the specimen.

    Args:
        specimen_name (str, optional): Name of the specimens to load.
    """

    def roundness_function(splinters: list[Splinter]):
        return np.mean([splinter.calculate_roundness() for splinter in splinters])

    # create contour plot of roughness
    specimen = Specimen.get(specimen_name)
    assert specimen is not None, "Specimen not found."

    fig = plot_splinter_movavg(
        specimen.get_fracture_image(),
        splinters=specimen.splinters,
        kw_px=w_mm * specimen.calculate_px_per_mm(),
        n_points=n_points,
        z_action=roundness_function,
        clr_label='Mean Roundness $\\bar{\lambda}_c$',
        mode=KernelContourMode.FILLED if not as_contours else KernelContourMode.CONTOURS,
        figwidth=FigureSize.ROW2,
        clr_format=".2f",
    )

    fig.overlayImpact(specimen)
    State.output(fig, spec=specimen, to_additional=True)


@app.command()
def roughness(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')]):
    """Plot the roughness of a specimen."""

    specimen = Specimen.get(specimen_name)

    out_img = specimen.get_fracture_image()
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

    rough = [splinter.calculate_roughness() for splinter in specimen.splinters]
    max_r = np.max(rough)
    min_r = np.min(rough)

    print(f"Max roughness: {max_r}")
    print(f"Min roughness: {min_r}")
    print(f"Mean roughness: {np.mean(rough)}")
    print(f"Median roughness: {np.median(rough)}")

    # d = np.median(rough) - min_r
    # max_r = np.median(rough) + d

    for splinter in track(specimen.splinters, description="Calculating roughness", transient=True):
        roughness = splinter.calculate_roughness()
        clr = get_color(roughness, min_r, max_r)

        cv2.drawContours(out_img, [splinter.contour], 0, clr, -1)

    out_img = annotate_image(
        out_img,
        cbar_title="Roughness $\lambda_r$",
        clr_format=".2f",
        min_value=min_r,
        max_value=max_r,
        figwidth=FigureSize.ROW2,
    )

    out_img.overlayImpact(specimen)
    State.output(out_img, spec=specimen, to_additional=True)


@app.command()
def roundness(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')]):
    """Plot the roundness of a specimen."""

    specimen = Specimen.get(specimen_name)

    out_img = specimen.get_fracture_image()

    rounds = [splinter.calculate_roundness() for splinter in specimen.splinters]
    max_r = np.max(rounds)
    min_r = np.min(rounds)

    print(f"Max roundness: {max_r}")
    print(f"Min roundness: {min_r}")
    print(f"Mean roundness: {np.mean(rounds)}")

    # scale max and min roundness to +- 60% around mean
    # max_r = np.mean(rounds) + np.mean(rounds) * 0.6
    # min_r = np.mean(rounds) - np.mean(rounds) * 0.6

    for splinter in track(specimen.splinters):
        r = splinter.calculate_roundness()
        clr = get_color(r, min_r, max_r)

        cv2.drawContours(out_img, [splinter.contour], 0, clr, -1)

    out_img = annotate_image(
        out_img,
        cbar_title="Roundness $\lambda_c$",
        min_value=min_r,
        max_value=max_r,
        figwidth=FigureSize.ROW2,
        clr_format=".2f"
    )


    out_img.overlayImpact(specimen)
    State.output(out_img, spec=specimen, to_additional=True)


def str_to_intlist(input: str) -> list[int]:
    if isinstance(input, int):
        return [input]

    return [int(x) for x in input.split(",")]


def specimen_parser(input: str):
    return input


def sort_two_arrays(array1, array2, reversed=False, keyoverride=None) -> tuple[list, list]:
    # Combine x and y into pairs
    pairs = list(zip(array1, array2))
    # Sort the pairs based on the values in x
    sorted_pairs = sorted(pairs, key=keyoverride or (lambda pair: pair[0]), reverse=reversed)
    # Separate the sorted pairs back into separate arrays
    return zip(*sorted_pairs)


@app.command(name='sigmasize')
def size_vs_sigma(xlim: Annotated[tuple[float, float], typer.Option(help='X-Limits for plot')] = (0, 2),
                  thickness: Annotated[list[int], typer.Option(help='Thickness of specimens.', parser=str_to_intlist,
                                                               metavar='4,8,12')] = [8],
                  break_pos: Annotated[
                      str, typer.Option(help='Break position.', metavar='[corner, center]')] = "corner",
                  more_data: Annotated[bool, typer.Option('--moredata',
                                                          help='Write specimens sig_h and thickness into legend.')] = False,
                  nolegend: Annotated[
                      bool, typer.Option('--nolegend', help='Dont display the legend on the plot.')] = False, ):
    """Plot the mean splinter size against the stress."""
    thickness = thickness[0]
    from scipy.optimize import curve_fit

    def decider(spec: Specimen):
        if not spec.has_splinters:
            return False
        if not spec.has_scalp:
            return False
        if spec.boundary == "":
            return False
        if spec.thickness not in thickness:
            return False
        if spec.settings["break_pos"] != break_pos:
            return False

        return True

    def specimen_value(spec: Specimen):
        return (spec.boundary, np.mean([x.area for x in spec.splinters]), np.abs(spec.U))

    specimens = Specimen.get_all_by(decider, lazyload=False)

    specimens = [specimen_value(x) for x in specimens]
    specimens = sorted(specimens, key=lambda x: x[0])
    # group specimens by boundary conditions
    t = groupby(specimens, lambda x: x[0])

    markers = {
        'A': 'o',
        'B': 's',
        'Z': 'v',
    }

    labels = {
        'A': 'All-side',
        'B': 'Bedded',
        'Z': 'Two-side',
    }

    min_sig = np.min([x[2] for x in specimens])
    max_sig = np.max([x[2] for x in specimens])

    max_size = np.max([x[1] for x in specimens])
    min_size = np.min([x[1] for x in specimens])

    fig, ax = plt.subplots()
    # create a plot for each group
    for boundary, group in t:

        print(f"Processing boundary '{boundary}'...")
        sizes = []
        stresses = []
        for s in group:
            sizes.append(s[1])
            stresses.append(s[2])

        stresses = np.array(stresses, dtype=np.float64)
        sizes = np.array(sizes, dtype=np.float64)

        ps = ax.scatter(stresses, sizes, marker=markers[boundary], label=labels[boundary])

        def func(x, a, b, c):
            return a + (b / (c + x))
            # return a*np.exp(-b*x) + c
            # return a * x ** 2 + b*x + c

        params = curve_fit(func, stresses, sizes, bounds=([0, 0, 0], [np.min(sizes), np.inf, np.inf]))

        s_inter = np.linspace(min_sig, max_sig, 50)
        ax.plot(s_inter, func(s_inter, *params[0]), '--', color=ps.get_facecolor()[0], linewidth=0.5)

        # p = np.polyfit(stresses, sizes, 4)
        # s_inter = np.linspace(stresses[0], stresses[-1], 100)
        # plt.plot(s_inter, np.polyval(p, s_inter), '--', color=ps.get_facecolor()[0])
    ax.set_xlabel("Strain Energy  [J/m²]")
    ax.set_ylabel("Mean splinter size [mm²]")
    ax.set_ylim((min_size - 2, max_size + 2))
    if not nolegend:
        ax.legend(loc='best')
    ax.grid(True, which='both', axis='both')
    fig.tight_layout()

    State.output(fig, 'stress_vs_size')


def diag_dist_specimen_intensity_func(
        specimen: Specimen,
        kernel_width=100,
        n_points=100
) -> tuple[npt.ArrayLike, Specimen]:
    """used in diag_dist to calculate the intensity of a specimen"""
    # calculate intensities
    img = specimen.get_fracture_image()
    kernel = ObjectKerneler(
        img.shape[:2],
        specimen.splinters,
        lambda x, r: x.in_region_px(r),
        kw_px=kernel_width,
        skip_edge=True,
    )

    intensity = kernel.run(lambda x: len(x), n_points=n_points, mode='diag')

    intensity = np.array(intensity)
    # intensity = intensity / np.max(intensity)

    return intensity, specimen


@app.command()
def log2dhist_diag(
        names: Annotated[str, typer.Option(help='Name filter. Can use wildcards.', metavar='*')] = "*",
        sigmas: Annotated[str, typer.Option(
            help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120" or "all").',
            metavar='s, s1-s2, all')] = None,
        delta: Annotated[float, typer.Option(help='Additional range for sigmas.')] = 10,
        out: Annotated[str, typer.Option(help='Output file.')] = None,
        w_mm: Annotated[int, typer.Option(help='Intensity kernel width.')] = 50,
        n_points: Annotated[int, typer.Option(help='Amount of points on the diagonal to evaluate.')] = 100,
        y_stress: Annotated[bool, typer.Option(help='Plot sigma instead of energy on y-axis.')] = False,
):
    """
    Same as log2dhist but with a kernel running on the diagonal.
    """

    filter = create_filter_function(names, sigmas, sigma_delta=delta)

    specimens: list[Specimen] = Specimen.get_all_by(filter, lazyload=False)

    fig, axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW2))

    # check that all specimens have the same size
    size0 = specimens[0].splinters_data['cropsize']
    specs0 = []
    for s in specimens:
        if s.splinters_data['cropsize'] != size0:
            print(f"Specimen '{s.name}' has different size than others. Skipping.")
        else:
            specs0.append(s)
    specimens = specs0

    data = []
    stress = []
    with get_progress() as progress:
        an_task = progress.add_task("Loading splinters...", total=len(specimens))

        sf = specimens[0].calculate_px_per_mm()
        assert np.all(
            [x.calculate_px_per_mm() == sf for x in specimens]), "Not all specimens have the same size factor."

        w_px = w_mm / sf

        for intensity, specimen in pooled(specimens, diag_dist_specimen_intensity_func,
                                          advance=lambda: progress.advance(an_task),
                                          kernel_width=w_px,
                                          n_points=n_points):
            intensity = intensity / np.max(intensity)

            data.append((intensity, specimen.name))
            if not y_stress:
                stress.append(specimen.U_d)
            else:
                stress.append(specimen.sig_h)

    # sort data and names for ascending stress
    stress, data = sort_two_arrays(stress, data, True)
    names = [x[1] for x in data]
    data = [x[0] for x in data]
    axs.set_xlabel(r"Diagonal Distance $\xi$ [-]")

    if not y_stress:
        axs.set_ylabel("Strain Energy Density [J/m³]")
    else:
        axs.set_ylabel("Surface Stress [MPa]")

    str_mod = 5 if len(stress) > 15 else 1
    x_ticks = np.linspace(0, n_points, 11)
    print(x_ticks)
    axs.set_xticks(x_ticks, [f'{x / n_points:.2f}' for i, x in enumerate(x_ticks)])
    axs.set_yticks(np.arange(0, len(stress), 1),
                   [f'{np.abs(x):.2f}' if i % str_mod == 0 else "" for i, x in enumerate(stress)])

    axy = axs.secondary_yaxis('right')
    axy.set_yticks(axs.get_yticks(), [x for i, x in enumerate(names)])

    dt = np.array(data)
    axim = axs.imshow(dt, cmap=modified_turbo, aspect='auto', interpolation='none')
    fig.colorbar(axim, ax=axs, orientation='vertical', label='Relative Intensity', pad=0.2)

    axs.set_xlim((-0.01, n_points + 0.01))
    # fig2 = plot_histograms((0,2), specimens, plot_mean=True)
    # plt.show()

    # axy.set_yticks(np.linspace(axy.get_yticks()[0], axy.get_yticks()[-1], len(axs.get_yticks())))
    fig.tight_layout()

    State.output(fig)


@app.command(name="log2dhist")
def log_2d_histograms(
        names: Annotated[str, typer.Option(help='Name filter. Can use wildcards.', metavar='*')] = "*",
        sigmas: Annotated[str, typer.Option(
            help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120" or "all").',
            metavar='s, s1-s2, all')] = None,
        exclude: Annotated[str, typer.Option(help='Exclude specimen names matching this.')] = None,
        delta: Annotated[float, typer.Option(help='Additional range for sigmas.')] = 10,
        y_stress: Annotated[bool, typer.Option(help='Show stress on y axis instead of energy.')] = False,
        maxspecimen: Annotated[int, typer.Option(help='Maximum amount of specimens.')] = 50,
        normalize: Annotated[bool, typer.Option(help='Normalize histograms.')] = False,
        n_bins: Annotated[int, typer.Option(help='Number of bins for histogram.')] = 60):
    """Plot a 2D histogram of splinter sizes and stress."""

    filter = create_filter_function(names, sigmas, delta,
                                    exclude=exclude,
                                    needs_scalp=False,
                                    needs_splinters=True)

    specimens: list[Specimen] = Specimen.get_all_by(filter, max_n=maxspecimen, lazyload=False)

    assert len(specimens) > 0, "[red]No specimens loaded.[/red]"

    binrange = np.linspace(0, 2, n_bins)
    fig, axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1H))

    data = []
    stress = []

    with get_progress() as progress:
        an_task = progress.add_task("Loading splinters...", total=len(specimens))
        for specimen in specimens:

            areas = [np.log10(x.area) for x in specimen.splinters if x.area > 0]
            # ascending sort, smallest to largest
            areas.sort()

            hist, edges = bin_data(areas, binrange)

            hist = np.array(hist)
            if normalize:
                hist = hist / np.max(hist)

            data.append((hist, specimen.name))
            if not y_stress:
                stress.append(specimen.U_d)
            else:
                stress.append(specimen.sig_h)

            progress.update(an_task, advance=1)

    # sort data and names for ascending stress
    stress, data = sort_two_arrays(stress, data, True)
    names = [x[1] for x in data]
    data = [x[0] for x in data]
    axs.set_xlabel("Splinter Area $A_S$ [mm²]")
    if not y_stress:
        axs.set_ylabel("Strain Energy [J/m²]")
    else:
        axs.set_ylabel("Surface Stress [MPa]")

    str_mod = 5 if len(stress) > 15 else 1
    x_ticks = np.arange(0, n_bins, 5)
    siz_mod = 2 if len(x_ticks) > 10 else 1
    axs.set_xticks(x_ticks, [f'{10 ** edges[x]:.2f}' if i % siz_mod == 0 else "" for i, x in enumerate(x_ticks)])
    axs.set_yticks(np.arange(0, len(stress), 1),
                   [f'{np.abs(x):.2f}' if i % str_mod == 0 else "" for i, x in enumerate(stress)])

    axy = axs.secondary_yaxis('right')
    axy.set_yticks(axs.get_yticks(), [x for i, x in enumerate(names)])

    dt = np.array(data)
    axim = axs.imshow(dt, cmap=modified_turbo, aspect='auto', interpolation='none')
    fig.colorbar(axim, ax=axs, orientation='vertical', label='PD $p(A_S)$', pad=0.2)
    # fig2 = plot_histograms((0,2), specimens, plot_mean=True)
    # plt.show()

    axy.set_yticks(np.linspace(axy.get_yticks()[0], axy.get_yticks()[-1], len(axs.get_yticks())))
    fig.tight_layout()

    if sigmas is not None:
        out_name = f"{sigmas[0]}_{sigmas[1]}"
    elif names is not None:
        out_name = f"{names[0]}"

    axs.grid(False)

    State.output(fig, f'loghist2d_{out_name}', to_additional=True)


def create_filter_function(name_filter,
                           sigmas=None,
                           sigma_delta=10,
                           exclude: str = None,
                           needs_scalp=True,
                           needs_splinters=True
                           ) -> Callable[[Specimen], bool]:
    """Creates a filter function for specimens.

    Args:
        names (str): String wildcard to match specimen names.
        sigmas (str): String with sigma range.
        sigma_delta (int, optional): If a single sigma value is passed, this range is added around the value. Defaults to 10.
        exclude (str, optional): Name filter to exclude. Defaults to None.
        needs_scalp (bool, optional): The specimen needs valid scalp data. Defaults to True.
        needs_splinters (bool, optional): The specimen needs valid splinter data. Defaults to True.

    Returns:
        Callable[[Specimen], bool]: Modified names, sigmas and filter function.
    """

    def in_names_wildcard(s: Specimen, filter: str) -> bool:
        return re.match(filter, s.name) is not None

    def in_names_list(s: Specimen, filter: list[str]) -> bool:
        return s.name in filter

    def all_names(s, filter) -> bool:
        return True

    name_filter_function: Callable[[Specimen, Any], bool] = None

    # create name_filter_function based on name_filter
    if name_filter is not None and "," in name_filter:
        name_filter = name_filter.split(",")
        print(f"Searching for specimen whose name is in: '{name_filter}'")
        name_filter_function = in_names_list
    elif name_filter is not None and " " in name_filter:
        name_filter = name_filter.split(" ")
        print(f"Searching for specimen whose name is in: '{name_filter}'")
        name_filter_function = in_names_list
    elif name_filter is not None and "*" not in name_filter:
        name_filter = [name_filter]
        print(f"Searching for specimen whose name is in: '{name_filter}'")
        name_filter_function = in_names_list
    elif name_filter is not None and "*" in name_filter:
        print(f"Searching for specimen whose name matches: '{name_filter}'")
        name_filter = name_filter.replace(".", "\.").replace("*", ".*").replace('!', '|')
        name_filter_function = in_names_wildcard
    elif name_filter is None:
        name_filter = ".*"
        print("[green]All[/green] specimen names included!")
        name_filter_function = all_names

    if sigmas is not None:
        if "-" in sigmas:
            sigmas = [float(s) for s in sigmas.split("-")]
        elif sigmas == "all":
            sigmas = [0, 1000]
        else:
            sigmas = [float(sigmas), float(sigmas)]
            sigmas[0] = max(0, sigmas[0] - sigma_delta)
            sigmas[1] += sigma_delta

        print(f"Searching for splinters with stress in range {sigmas[0]} - {sigmas[1]}")

    def filter_specimens(specimen: Specimen):
        if exclude is not None and re.match(exclude, specimen.name):
            return False
        elif not name_filter_function(specimen, name_filter):
            return False
        elif needs_scalp and not specimen.has_scalp:
            print(f"Specimen '{specimen.name}' has no scalp data. Skipping.")
            return False
        elif sigmas is not None:
            return sigmas[0] <= abs(specimen.scalp.sig_h) <= sigmas[1]
        elif needs_splinters and not specimen.has_splinters:
            print(f"Specimen '{specimen.name}' has no splinter data. Skipping.")
            return False

        return True

    return filter_specimens


@app.command()
def log_histograms(
    names: Annotated[str, typer.Argument(help='Names of specimens to load')],
    sigmas: Annotated[str, typer.Argument(help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120").')] = None,
    n_bins: Annotated[int, typer.Option(help='Number of bins for histogram.')] = general.hist_bins,
    plot_mean: Annotated[bool, typer.Option('--plot-mean', help='Plot mean splinter size.')] = False,
    data_mode: Annotated[DataHistMode, typer.Option(help='Mode for histogram. Either pdf or cdf.')] = 'pdf',
    plot_mode: Annotated[DataHistPlotMode, typer.Option(help='Data mode.')] = DataHistPlotMode.HIST,
    legend: Annotated[str, typer.Option(help='Legend style (0: Name, 1: Sigma, 2: Dicke, 3: Mean-Size).')] = None,
    xlim: Annotated[tuple[float, float], typer.Option(help='X-Limits for plot')] = (0, 2),
    figwidth: Annotated[FigureSize, typer.Option(help='Width of the figure.')] = FigureSize.ROW2,
):
    """Plot logaritmic histograms of splinter sizes for specimens."""
    filter = create_filter_function(names, sigmas, needs_scalp=False, needs_splinters=True)
    specimens = Specimen.get_all_by(filter, lazyload=False)

    if len(specimens) == 0:
        print("[red]No specimens loaded.[/red]")
        return

    if legend is not None:
        def legend_by_string(x: Specimen):
            return legend.format(x.name, x.sig_h, x.thickness, np.mean([x.area for x in x.splinters]))

        legend_f = legend_by_string
    else:
        def legend_f(x):
            return ''

    fig, axs = datahist_plot(xlim=xlim, figwidth=figwidth, data_mode=data_mode)

    a = 0.7 if len(specimens) > 1 else 1

    # br is created once, so that the bin-ranges match for all specimens
    br = None
    for specimen in specimens:
        areas = [x.area for x in specimen.splinters]
        _, br0, _ = datahist_to_ax(
            axs[0],
            areas,
            alpha=a,
            n_bins=n_bins,
            binrange=br,
            plot_mean=plot_mean,
            label=legend_f(specimen),
            data_mode=data_mode,
            plot_mode=plot_mode
        )
        if br is None:
            br = br0


    if legend is not None and len(specimens) > 1:
        fig.legend(loc='upper left', bbox_to_anchor=(1.05, 1), bbox_transform=axs[0].transAxes)

    output = StateOutput(fig, figwidth)
    if len(specimens) == 1:
        State.output(output, 'loghist', spec=specimens[0], to_additional=True, mods=[data_mode])
    else:
        State.output(output, to_additional=True, mods=[data_mode])

    disp_mean_sizes(specimens)


def disp_mean_sizes(specimens: list[Specimen]):
    """Displays mean splinter sizes.

    Args:
        specimens (list[Specimen]): Specimens to display.
    """
    print("* Mean splinter sizes:")
    for specimen in specimens:
        print(f"\t '{specimen.name}' ({specimen.scalp.sig_h:.2f}): {np.mean([x.area for x in specimen.splinters]):.2f}")


@app.command()
def splinter_orientation_f(
        specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
        w_mm: Annotated[float, typer.Option(help='Width of kernel in mm.')] = 50,
        n_points: Annotated[int, typer.Option(help='Amount of points to use.')] = general.n_points_kernel,
        as_contours: Annotated[bool, typer.Option(help='Plot the kernel as contours.')] = False,
        plot_vertices: Annotated[bool, typer.Option(help='Plot the kernel points.')] = False,
        exclude_points: Annotated[bool, typer.Option(help='Exclude points from the kernel.')] = False,
        plot_kernel: Annotated[bool, typer.Option(help='Plot the kernel rectangle.')] = False,
        skip_edge: Annotated[
            bool, typer.Option(help='Skip one row of the edges when calculating intensities.')] = False,
        figwidth: Annotated[FigureSize, typer.Option(help='Fraction of kernel width to use.')] = FigureSize.ROW2,
):
    specimen = Specimen.get(specimen_name)
    impact_pos = specimen.get_impact_position()
    def mean_orientations(splinters: list[Splinter]):
        orientations = np.array([x.measure_orientation(impact_pos) for x in splinters ])
        return np.mean(orientations[~np.isnan(orientations)])

    w_px = int(w_mm * specimen.calculate_px_per_mm())

    fig_output = plot_splinter_movavg(
        specimen.get_fracture_image(),
        specimen.splinters,
        plot_kernel=plot_kernel,
        plot_vertices=plot_vertices,
        exclude_points=[specimen.get_impact_position(True)] if exclude_points else None,
        skip_edge=skip_edge,
        fill_skipped_with_mean=True,
        n_points=n_points,
        kw_px=w_px,
        z_action=mean_orientations,
        clr_label="Mean Orientation Strength $\\bar{\Delta}$",
        mode=KernelContourMode.FILLED if not as_contours else KernelContourMode.CONTOURS,
        figwidth=figwidth,
        clr_format='.1f',
        normalize=True,
    )

    fig_output.overlayImpact(specimen)
    State.output(fig_output, spec=specimen, to_additional=True)


@app.command()
def splinter_orientation(specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')]):
    """Plot the orientation of splinters."""
    specimen = Specimen.get(specimen_name)

    impact_pos = specimen.get_impact_position()
    splinters = specimen.splinters

    # analyze splinter orientations
    orientation_image = np.zeros_like(specimen.get_fracture_image(), dtype=np.uint8)
    for s in track(splinters):
        orientation = s.measure_orientation(impact_pos)
        color = get_color(orientation)
        cv2.drawContours(orientation_image, [s.contour], -1, color, -1)

    # draw splinter contour lines
    cv2.drawContours(orientation_image, [x.contour for x in splinters], -1, (0, 0, 0), 1)

    orientation_fig = annotate_image(
        orientation_image,
        cbar_title='Orientation Strength $\Delta$',
        min_value=0,
        max_value=1,
        figwidth=FigureSize.ROW2,
        clr_format='.1f'
    )

    orientation_fig.overlayImpact(specimen)

    State.output(orientation_fig, spec=specimen, to_additional=True)


@app.command()
def fracture_intensity_img(
        specimen_name: str,
        w_mm: Annotated[int, typer.Option(help='Kernel width.')] = 50,
        n_points: Annotated[int, typer.Option(help='Amount of points to use.')] = general.n_points_kernel,
        as_contours: Annotated[bool, typer.Option(help='Plot the kernel as contours.')] = False,
        plot_vertices: Annotated[bool, typer.Option(help='Plot the kernel points.')] = False,
        exclude_points: Annotated[bool, typer.Option(help='Exclude points from the kernel.')] = False,
        plot_kernel: Annotated[bool, typer.Option(help='Plot the kernel rectangle.')] = False,
        skip_edge: Annotated[bool, typer.Option(help='Skip one row of the edges when calculating intensities.')] = True,
        figwidth: Annotated[FigureSize, typer.Option(help='Fraction of kernel width to use.')] = FigureSize.ROW2,
):
    """
    Plot the intensity of the fracture image.

    Basically the same as fracture-intensity, but performs operations on the image
    instead of the splinters.

    The edges are skipped by default, because most often the edges are too black because of the
    glass ply itself. Therefor the edges are not representative of the fracture morphology.

    Intensity here is the mean value of the image part (defined by kernel_width).
    Higher Intensity => Darker image part (more cracks)
    Lower Intensity => Brighter image part (less crack)

    Args:
        specimen_name (str): Name of specimen to load.
        kernel_width (int, optional): Grid size. Defaults to 100.
    """
    specimen = Specimen.get(specimen_name)

    def mean_img_value(img_part):
        # return np.mean(img_part)
        # count black pixels
        binary_image = (img_part < 127).astype(np.uint8)
        return np.sum(binary_image == 1)

    img = specimen.get_fracture_image()
    img = to_gray(img)
    # img = preprocess_image(img, specimen.splinter_config)
    w_px = int(w_mm * specimen.calculate_px_per_mm())
    output = plot_image_movavg(
        img,
        w_px,
        n_points,
        z_action=mean_img_value,
        clr_label="Black Pixels Value [$N_{BP}/A/N_t$]",
        mode=KernelContourMode.FILLED if not as_contours else KernelContourMode.CONTOURS,
        skip_edge=skip_edge,
        exclude_points=[specimen.get_impact_position(True)] if exclude_points else None,
        figwidth=figwidth,
        plot_vertices=plot_vertices,
        plot_kernel=plot_kernel,
        clr_format='.1f',
        transparent_border=True,
        fill_skipped_with_mean=True,
        normalize=True
    )
    output.overlayImpact(specimen)
    State.output(output, spec=specimen, to_additional=True)



@app.command()
def fracture_intensity_f(
        specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
        w_mm: Annotated[int, typer.Option(help='Kernel width.')] = 50,
        n_points: Annotated[int, typer.Option(help='Amount of points to use.')] = general.n_points_kernel,
        plot_vertices: Annotated[bool, typer.Option(help='Plot the kernel points.')] = False,
        plot_kernel: Annotated[bool, typer.Option(help='Plot the kernel rectangle.')] = False,
        as_contours: Annotated[bool, typer.Option(help='Plot the kernel as contours.')] = False,
        skip_edge: Annotated[
            bool, typer.Option(help='Skip one row of the edges when calculating intensities.')] = False,
        exclude_points: Annotated[bool, typer.Option(help='Exclude points from the kernel.')] = False,
        figwidth: Annotated[FigureSize, typer.Option(help='Fraction of kernel width to use.')] = FigureSize.ROW2,
):
    """Plot the intensity of the fracture morphology."""

    specimen = Specimen.get(specimen_name)
    w_px = int(w_mm * specimen.calculate_px_per_mm())
    print(f"Kernel width: {w_mm} mm")

    original_image = specimen.get_fracture_image()

    # this counts the splinters in the kernel by default
    fig = plot_splinter_movavg(
        original_image,
        specimen.splinters,
        w_px,
        n_points,
        z_action=lambda x: len(x),
        plot_vertices=plot_vertices,
        plot_kernel=plot_kernel,
        clr_label="Fracture Intensity $\Gamma$ [$N_S/A$]",  # , $w_A,h_A$={w_mm}mm
        mode=KernelContourMode.FILLED if not as_contours else KernelContourMode.CONTOURS,
        exclude_points=[specimen.get_impact_position(True)] if exclude_points else None,
        skip_edge=skip_edge,
        figwidth=figwidth,
        clr_format='.0f',
        fill_skipped_with_mean=False,
        transparent_border=False,
    )

    mods = []
    if as_contours:
        mods.append("contours")
    if plot_vertices:
        mods.append("vertices")

    fig.overlayImpact(specimen)
    State.output(fig, spec=specimen, to_additional=True, mods=mods)


@app.command()
def create_voronoi(specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')], ):
    # specimen = fetch_specimens(specimen_name, general.base_path)
    # assert specimen is not None, "Specimen not found."

    # def __create_voronoi(self, config: AnalyzerConfig):
    #     centroids = np.array([x.centroid_px for x in self.splinters if x.has_centroid])
    #     voronoi = Voronoi(centroids)

    #     voronoi_img = np.zeros_like(self.original_image, dtype=np.uint8)
    #     if not is_gray(voronoi_img):
    #         voronoi_img = cv2.cvtColor(voronoi_img, cv2.COLOR_BGR2GRAY)
    #     for i, r in enumerate(voronoi.regions):
    #         if -1 not in r and len(r) > 0:
    #             polygon = [voronoi.vertices[i] for i in r]
    #             polygon = np.array(polygon, dtype=int)
    #             cv2.polylines(voronoi_img, [polygon], isClosed=True, color=255, thickness=2)

    #     cv2.imwrite(self.__get_out_file(f"voronoi_img.{config.ext_imgs}"), voronoi_img)
    #     fig = voronoi_plot_2d(voronoi, show_points=True, point_size=5, show_vertices=False, line_colors='red')
    #     plt.imshow(self.original_image)
    #     plt.axis('off')  # Turn off axis labels and ticks
    #     plt.title('Voronoi Plot Overlay on Image')
    #     if config.debug:
    #         plt.show()
    #     fig.savefig(self.__get_out_file(f"voronoi.{config.ext_plots}"))

    #     # optimal_h = estimate_optimal_h(events, region)
    #     # print(f'Optimal h: {optimal_h}')

    #     #TODO: compare voronoi splinter size distribution to actual distribution
    #     # X,Y,Z = csintkern(events, region, 500)
    #     self.create_intensity_plot(config.intensity_h, config)
    pass


def get_detection_rate(splinters: list[Splinter], real_size: tuple[float, float]) -> float:
    #############
    # check percentage of detected splinters
    total_area = np.sum([x.area for x in splinters])

    total_img_size = real_size[0] * real_size[1]

    p = total_area / total_img_size * 100

    return p


@app.command(name='crop-frac')
def crop_fracture_morph(
        specimen_name: Annotated[str, typer.Option(help='Name of specimen to load')] = "",
        all: Annotated[bool, typer.Option('--all', help='Perform this action on all specimen.')] = False,
        rotate: Annotated[bool, typer.Option('--rotate', help='Rotate image by 90°.')] = False,
        crop: Annotated[bool, typer.Option('--crop', help='Crop the image.')] = True,
        size: Annotated[
            tuple[int, int], typer.Option(help='Image size.', metavar='Y X')] = general.default_image_size_px,
        rotate_only: Annotated[
            bool, typer.Option('--rotate-only', help='Only rotate image by 90°, skip cropping.')] = False,
        resize_only: Annotated[bool, typer.Option('--resize_only', help='Only resize the image to 4000px².')] = False,
):
    f"""Crop and resize fracture morphology images. Can run on all specimens, several or just one single one.

    Args:
        specimen_name (Annotated[str, typer.Option, optional): The specimen names. Defaults to 'Name of specimen to load')]
        all (bool, optional): Run the method on all specimens. Defaults to False.
        rotate (bool, optional): Rotate the input image 90° CCW. Defaults to False.
        crop (bool, optional): Crop the input image to ply bounds. Defaults to True.
        size (tuple[int,int], optional): Size of the image. Defaults to {general.default_image_size_px}.
        rotate_only (bool, optional): Only rotate the images. Defaults to False.
        resize_only (bool, optional): Only resizes the images. Defaults to False.
    """
    from stat import S_IREAD, S_IRGRP, S_IROTH
    if all:
        specimens = Specimen.get_all()
    elif isinstance(specimen_name, Specimen):
        specimens = [specimen_name]
    else:
        specimens = Specimen.get_all(specimen_name)

    for specimen in track(specimens):
        path = specimen.fracture_morph_dir
        if not os.path.exists(path):
            continue

        imgs = [(x, cv2.imread(x, cv2.IMREAD_GRAYSCALE)) for x in find_files(path, '*.bmp')]

        if len(imgs) == 0:
            continue

        img0 = [y for x, y in imgs if "Transmission" in x][0]
        _, M0 = crop_perspective(img0, size, False, True)

        for file, img in imgs:
            if not os.access(file, os.W_OK):
                print(f"Skipping '{os.path.basename(file)}', no write access.")
                continue

            if resize_only:
                img = cv2.resize(img, size)

            if not rotate_only and crop and not resize_only:
                img = crop_matrix(img, M0, size)

            if (rotate or rotate_only) and not resize_only:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(file, img)
            os.chmod(os.path.join(path, file), S_IREAD | S_IRGRP | S_IROTH)

        # elif file.endswith(".bmp") and not os.access(os.path.join(path, file), os.W_OK):
        #     os.chmod(os.path.join(path, file), S_IWRITE)


@app.command()
def watershed(
        name: Annotated[str, typer.Argument(help='Name of the specimen.', metavar='*.*.*.*')],
        debug: Annotated[bool, typer.Option(help='Show debug plots.')] = False,
):
    """Raw comparison of existing splinter data with watershed algorithm."""
    # TODO: Check the individual steps of the watershed algorithm
    #   1. Check Background identification, this should be 0 so that all available space
    #       is used for the watershed algorithm and the markers
    #   2. Check the distance transform, this should identify all splinters and not
    #       create any false positives or miss small splinters
    #
    #   Once this works, validate the detection of dark spots and compare them
    #   to the original algorithm!

    specimen = Specimen.get(name)

    image = specimen.get_fracture_image()

    assert image is not None, "No fracture image found."

    size_factor = specimen.calculate_px_per_mm()
    print(size_factor)
    splinters = Splinter.analyze_image(image, px_per_mm=size_factor)

    # ORIGINAL output
    sp_img = cv2.imread(specimen.get_splinter_outfile("img_filled.png"))
    size_img_file = find_file(specimen.get_splinter_outfile(""), "img_splintersizes")
    if not size_img_file:
        size_img_file = specimen.get_splinter_outfile("img_splintersizes.png")
        create_splinter_sizes_image(specimen.splinters, image.shape, size_img_file)
    sz_img = cv2.imread(size_img_file, cv2.IMREAD_COLOR)

    m_img = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(m_img, [x.contour for x in splinters], -1, (255, 255, 255), 1)
    cmp_image = cv2.addWeighted(image, 1.0, sp_img, 0.2, 0)
    cmp_image = cv2.addWeighted(cmp_image, 1, m_img, 1, 0)
    if debug:
        plotImages((("Original", image), ("Comparison", cmp_image), ("Splinter Sizes", sz_img)))
        State.output(cmp_image)

    ## create splinter size image
    sz_image2 = create_splinter_sizes_image(
        splinters,
        image.shape,
        annotate=True,
        annotate_title="Watershed",
        with_contours=True)
    if debug:
        State.output(sz_image2)

    rnd_splinters = create_splinter_colored_image(
        splinters,
        image.shape,
        specimen.get_splinter_outfile("img_filled_watershed.png"),
    )

    if debug:
        plotImages([("Splinter Sizes", sz_img), ("Splinter Sizes Watershed", sz_image2)])

    # overlay contours over rnd splinters
    rnd_splinters = cv2.addWeighted(rnd_splinters, 1.0, to_rgb(m_img), 1.0, 0)
    if debug:
        plotImage(rnd_splinters, "Splinters Watershed")

    # plot splinter histograms
    fig, axs = datahist_plot()
    ax = axs[0]
    ax.set_xlabel("Splinter Size $A_S$ [px²]")
    ax.set_ylabel("PDF $P(A_S)$")
    _, br = datahist_to_ax(ax, [x.area for x in splinters], general.hist_bins, as_log=True, alpha=0.9,
                           label='Watershed', plot_mean=False)
    datahist_to_ax(ax, [x.area for x in specimen.splinters], binrange=br, as_log=True, label='Original',
                   plot_mean=False)
    ax.legend()
    ax.autoscale()
    fig.tight_layout()

    State.output(fig, specimen, override_name="splinter_sizes_watershed")
    plt.close(fig)

    fig, axs = datahist_plot(xlim=(0, 2))
    ax = axs[0]
    ax.set_xlabel("Splinter Size [px²]")
    ax.set_ylabel("CDF")
    datahist_to_ax(ax, [x.area for x in splinters], 50, as_log=True, alpha=0.9, label='Watershed', plot_mean=False,
                   data_mode='cdf')
    datahist_to_ax(ax, [x.area for x in specimen.splinters], 50, as_log=True, label='Original', plot_mean=False,
                   data_mode='cdf')
    ax.legend()
    ax.autoscale()
    fig.tight_layout()
    State.output(fig, specimen, override_name="splinter_sizes_watershed_cdf")


@app.command()
def compare_manual(
        folder: Annotated[str, typer.Argument(help=f'Subfolder of "{State.get_output_dir()}" that contains images.')],
        x_range: Annotated[str, typer.Option(help='Comma seperated x bounds.')] = None,
        y_range: Annotated[str, typer.Option(help='Comma seperated y bounds.')] = None,
        no_ylabs: Annotated[bool, typer.Option(help='Remove y ticks.')] = False,
):
    """
        Compare the results of different methods for detecting splinters in an image.

        Args:
            folder: The name of the subfolder inside of `State.get_output_dir()`

        Raises:
            Files "input", "marked", and "label" must be present in the folder.
        """

    test_dir = os.path.join(State.get_input_dir(), folder)

    input_img_path = find_file(test_dir, "input")
    marked_img_path = find_file(test_dir, "marked")
    label_img_path = find_file(test_dir, "label")

    assert input_img_path is not None, "No input image found."
    assert marked_img_path is not None, "No marked image found."
    assert label_img_path is not None, "No label image found."

    input_img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
    marked_img = cv2.imread(marked_img_path, cv2.IMREAD_COLOR)
    label_img = cv2.imread(label_img_path, cv2.IMREAD_COLOR)

    label_splinters = Splinter.analyze_label_image(label_img)

    # get splinters from labeled image
    manual_splinters = Splinter.analyze_marked_image(
        marked_img,
        input_img,
        px_per_mm=1,
    )

    # get splinters from watershed
    splinters = Splinter.analyze_image(input_img)

    # get splinters from legacy method
    with open(find_file(test_dir, "splinters"), 'rb') as f:
        # import to redirect pickle import
        import fracsuite.core.splinter as splt
        sys.modules['fracsuite.splinters.splinter'] = splt
        legacy_splinters: list[Splinter] = pickle.load(f, fix_imports=True)

    cont_img_alg = cv2.drawContours(np.zeros_like(input_img), [x.contour for x in splinters], -1, (255, 0, 0), 3)
    cont_img_man = cv2.drawContours(np.zeros_like(input_img), [x.contour for x in manual_splinters], -1, (0, 255, 0), 3)
    cont_img_leg = cv2.drawContours(np.zeros_like(input_img), [x.contour for x in legacy_splinters], -1, (0, 255, 0), 3)
    cont_img_lab = cv2.drawContours(np.zeros_like(input_img), [x.contour for x in label_splinters], -1, (0, 255, 0), 3)

    # test = cv2.addWeighted(input_img, 1, cont_img_lab, 1.0, 0)
    # plotImage(test, "test", force=True)

    cont_diff = cv2.addWeighted(input_img, 1, cont_img_alg, 1.0, 0)
    cont_diff = cv2.addWeighted(cont_diff, 1, cont_img_man, 1.0, 0)
    cont_diff_leg = cv2.addWeighted(input_img, 1, cont_img_alg, 1.0, 0)
    cont_diff_leg = cv2.addWeighted(cont_diff_leg, 1, cont_img_leg, 1.0, 0)
    cont_diff_lab = cv2.addWeighted(input_img, 1, cont_img_alg, 1.0, 0)
    cont_diff_lab = cv2.addWeighted(cont_diff_lab, 1, cont_img_lab, 1.0, 0)

    alg_man = cv2.absdiff(cont_img_alg, cont_img_man)
    yellow_pixels = np.all(alg_man == (255, 255, 0), axis=-1)

    alg_leg = cv2.absdiff(cont_img_alg, cont_img_leg)
    yellow_pixels_leg = np.all(alg_leg == (255, 255, 0), axis=-1)

    alg_lab = cv2.absdiff(cont_img_alg, cont_img_lab)
    yellow_pixels_lab = np.all(alg_lab == (255, 255, 0), axis=-1)

    matching_color = (0, 120, 255)
    diff_matching = np.zeros_like(input_img)
    diff_matching[yellow_pixels] = matching_color
    cont_diff[yellow_pixels] = (0, 0, 0)
    cont_diff = cv2.addWeighted(cont_diff, 1, diff_matching, 1.0, 0)

    diff_leg_matching = np.zeros_like(input_img)
    diff_leg_matching[yellow_pixels_leg] = matching_color
    cont_diff_leg[yellow_pixels_leg] = (0, 0, 0)
    cont_diff_leg = cv2.addWeighted(cont_diff_leg, 1, diff_leg_matching, 1.0, 0)

    diff_lab_matching = np.zeros_like(input_img)
    diff_lab_matching[yellow_pixels_lab] = matching_color
    cont_diff_lab[yellow_pixels_lab] = (0, 0, 0)
    cont_diff_lab = cv2.addWeighted(cont_diff_lab, 1, diff_lab_matching, 1.0, 0)

    # cont_diff = cv2.absdiff(cont_img_alg, cont_img_man)
    # cont_diff_leg = cv2.absdiff(cont_img_alg, cont_img_leg)

    # cont_diff[np.all(cont_diff == (255,255,0), axis=-1)] = (0,0,0)
    # cont_diff_leg[np.all(cont_diff_leg == (255,255,0), axis=-1)] = (0,0,0)

    # thresh_diff = thresh.copy()
    # thresh_diff[np.all(cont_diff != (0,0,0), axis=-1)] = (0,0,0)
    # cont_diff = cv2.addWeighted(thresh_diff, 1, cont_diff, 1.0, 0)
    # thresh_diff_leg = thresh.copy()
    # thresh_diff_leg[np.all(cont_diff_leg != (0,0,0), axis=-1)] = (0,0,0)
    # cont_diff_leg = cv2.addWeighted(thresh_diff_leg, 1, cont_diff_leg, 1.0, 0)

    # cont_img_alg1 = cont_img_alg.copy()
    # cont_img_alg1[np.all(cont_img_man != (0,0,0), axis=-1)] = (0,0,255)

    cmp_alg_man = label_image(
        cont_diff,
        'Algorithm', 'red',
        'Manual', 'green',
        'Identical', matching_color,
        nums=[len(splinters), len(manual_splinters)],
    )
    cmp_alg_leg = label_image(
        cont_diff_leg,
        'Algorithm', 'red',
        'Legacy', 'green',
        'Identical', matching_color,
        nums=[len(splinters), len(legacy_splinters)],
    )
    cmp_alg_lab = label_image(
        cont_diff_lab,
        'Algorithm', 'red',
        'Labeled', 'green',
        'Identical', matching_color,
        nums=[len(splinters), len(label_splinters)],
    )

    nr = folder.replace("test", "")

    State.output_nopen(cont_img_alg, folder, 'watershed_contour')

    State.output_nopen(cont_diff, folder, f'{nr}_compare_contours_watershed_manual_nolegend', cvt_rgb=True)
    State.output(cont_diff_lab, folder, f'{nr}_compare_contours_watershed_label_nolegend', to_additional=True,
                 cvt_rgb=True)
    State.output_nopen(cont_diff_leg, folder, f'{nr}_compare_contours_watershed_legacy_nolegend', cvt_rgb=True)

    State.output_nopen(cmp_alg_man, folder, f'{nr}_compare_contours_watershed_manual')
    State.output(cmp_alg_lab, folder, f'{nr}_compare_contours_watershed_label', to_additional=True)
    State.output_nopen(cmp_alg_leg, folder, f'{nr}_compare_contours_watershed_legacy')

    # use this for better visibility
    hist_bins = 20
    figwidth = FigureSize.ROW3

    fig, axs = datahist_plot(
        y_format='{0:.0f}',
        figwidth=figwidth,
    )
    ax = axs[0]
    ax.set_xlabel("Splinter Size $A_S$ [px²]")
    # ax.set_ylabel("PDF $P(A_S)$ [1/px²]")
    ax.set_ylabel("$N(A_S)$")
    area0 = np.array([x.area for x in splinters])
    area1 = np.array([x.area for x in manual_splinters])

    if x_range is None:
        _, br, d1 = datahist_to_ax(ax, area0, n_bins=hist_bins, label='Algorithm', color='red', plot_mean=False,
                                   as_density=False, plot_mode=DataHistPlotMode.HIST)
    else:
        low, up = [float(x) for x in x_range.split(':')]
        low = low if low > 0 else 1
        br = np.linspace(np.log10(low) if low > 0 else 1, np.log10(up), hist_bins)
        _, _, d1 = datahist_to_ax(ax, area0, binrange=br, label='Algorithm', color='red', plot_mean=False,
                                  as_density=False, plot_mode=DataHistPlotMode.HIST)

    # datahist_to_ax(ax, [x.area for x in manual_splinters], n_bins=hist_bins, label='Manual', plot_mean=False)
    # datahist_to_ax(ax, [x.area for x in legacy_splinters], n_bins=hist_bins, label='Legacy', plot_mean=False)
    _, _, d2 = datahist_to_ax(ax, area1, binrange=br, label='Manual', color='green', plot_mean=False, as_density=False, plot_mode=DataHistPlotMode.HIST)
    delta_area = np.abs(d1 - d2)
    plt.bar(br[:-1], delta_area, width=np.diff(br), align="edge", alpha=1, label="Difference", color="blue")
    ax.autoscale()
    if x_range is not None:
        ax.set_xlim((np.log10(low), np.log10(up)))

    if y_range is not None:
        low, up = [float(x) for x in y_range.split(':')]

        ax.set_ylim((low, up))

    # ax.legend(loc='upper left')
    if no_ylabs:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    State.output(
        StateOutput(fig, figwidth),
        folder,
        f"{nr}_splinter_sizes_compare",
        to_additional=True
    )

    if x_range is None:
        binrange = np.linspace(np.min(area0), np.max(area0), hist_bins)
    else:
        low, up = [float(x) for x in x_range.split(':')]
        binrange = np.linspace(low, up, hist_bins)

    sims = similarity(
        area0,
        area1,
        binrange
    )

    data_file = State.get_output_file(folder, "data.txt")
    print(f"Writing data to '{data_file}'")
    with open(data_file, 'w') as f:
        f.write('Similarities:\n')
        f.write(f'Pearson: {sims[0]}\n')
        f.write(f'MSE: {sims[1]}\n')
        f.write(f'KS: {sims[2]}\n')
        f.write(f'Count: {sims[3]}\n')


@app.command()
def extract_labels(
        specimen_name: str,
        out_dir: Annotated[str, typer.Argument(help='Output directory.')],
        n_side: Annotated[int, typer.Option(help='Amount of images per side to extract.')] = 10,
):
    """Splits the specimen fracture image into n_side x n_side images and extracts the labels."""
    specimen = Specimen.get(specimen_name)

    transmission_image = specimen.get_transmission_image()
    frac_image = specimen.get_fracture_image()
    # draw contours onto contour image
    ctr_image = np.zeros(frac_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(ctr_image, [x.contour for x in specimen.splinters], -1, 255, 1)

    # split image into n regions
    im_w, im_h = frac_image.shape[:2]
    d_w = im_w // n_side
    d_h = im_h // n_side

    # extract labels
    for i in range(n_side):
        for j in range(n_side):
            x = i * d_w
            y = j * d_h
            img = frac_image[x:x + d_w, y:y + d_h]
            ctr_img = ctr_image[x:x + d_w, y:y + d_h]
            orig_img = transmission_image[x:x + d_w, y:y + d_h]

            cv2.imwrite(os.path.join(out_dir, f"img_{i}_{j}.jpg"), img)
            cv2.imwrite(os.path.join(out_dir, f"lab_{i}_{j}.jpg"), ctr_img)
            cv2.imwrite(os.path.join(out_dir, f"orig_{i}_{j}.jpg"), orig_img)

@app.command()
def extract_all_labels(
        out_dir: Annotated[str, typer.Argument(help='Output directory.')],
        n_side: Annotated[int, typer.Option(help='Amount of images per side to extract.')] = 10,
):
    specimens = Specimen.get_all()

    for spec in specimens:
        if not spec.has_fracture_scans or not spec.has_splinters:
            continue

        out_dir2 = os.path.join(out_dir, spec.name)
        os.makedirs(out_dir2, exist_ok=True)
        extract_labels(spec.name, out_dir2, n_side)

@app.command()
def annotate_impact(
    specimen_name: str
):
    """Annotate the impact position on the fracture image."""
    specimen = Specimen.get(specimen_name)

    frac_image = specimen.get_fracture_image()
    impact_pos = specimen.get_impact_position(in_px=True, as_tuple=True)
    fall_height = specimen.get_fall_height_m()

    f = specimen.calculate_px_per_mm()

    # cv2.circle(frac_image, impact_pos, 10, (0, 0, 255), -1)
    # extract 100px around impact position
    d_mm = 150
    x, y = impact_pos
    w,h = int(d_mm * f), int(d_mm * f)
    print(w,h)

    # clip x y w and h so it does not exceed image bounds
    x = max(w//2, x)
    y = max(h//2, y)
    x = min(frac_image.shape[1], x)
    y = min(frac_image.shape[0], y)

    impact_image = frac_image[y-h//2:y+h//2, x-w//2:x+w//2]

    # draw white rectangle to put text on
    cv2.rectangle(impact_image, (0, 0), (350, 120), (255, 255, 255), -1)
    # annotate fall height
    cv2.putText(impact_image, f"Thickness: {specimen.measured_thickness:.2f}mm", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(impact_image, f"Pre-Stress: {np.abs(specimen.sig_h):.0f}MPa", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(impact_image, f"Fall Height: {fall_height:.2f}m", (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(impact_image, f"Impact: {specimen.get_impact_position_name()}", (5, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    State.output(impact_image, spec=specimen, override_name="impact_annotated", figwidth=FigureSize.ROW1)