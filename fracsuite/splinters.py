"""
Splinter analyzation tools.
"""

from enum import Enum
from fracsuite.core.logging import debug
import multiprocessing.shared_memory as sm
import os
import pickle
import re
import shutil
import sys
from itertools import groupby
from typing import Annotated, Any, Callable

import cv2
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
import typer
from matplotlib import pyplot as plt
from rich import inspect, print
from rich.progress import track

from fracsuite.callbacks import main_callback
from fracsuite.core.arrays import sort_arrays, sort_two_arrays
from fracsuite.core.calculate import pooled
from fracsuite.core.coloring import get_color, norm_color, rand_col
from fracsuite.core.detection import attach_connections, get_adjacent_splinters_parallel
from fracsuite.core.image import FontSize, put_scale, put_text, to_gray, to_rgb
from fracsuite.core.imageplotting import plotImage, plotImages
from fracsuite.core.imageprocessing import crop_matrix, crop_perspective
from fracsuite.core.kernels import ObjectKerneler
from fracsuite.core.navid_results import navid_nfifty_ud, navid_nfifty
from fracsuite.core.plotting import (
    AxLabels,
    DataHistMode,
    DataHistPlotMode,
    FigureSize,
    KernelContourMode,
    annotate_image,
    cfg_logplot,
    create_splinter_colored_image,
    create_splinter_sizes_image,
    datahist_plot,
    datahist_to_ax,
    get_fig_width,
    label_image,
    modified_turbo,
    plot_image_movavg,
    plot_kernel_results,
    plot_splinter_movavg,
)
from fracsuite.core.preps import PreprocessorConfig, defaultPrepConfig
from fracsuite.core.progress import get_progress
from fracsuite.core.specimen import Specimen, SpecimenBoundary
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stochastics import similarity
from fracsuite.core.vectors import alignment_cossim
from fracsuite.general import GeneralSettings
from fracsuite.helpers import bin_data, find_file, find_files
from fracsuite.state import State, StateOutput

from scipy.optimize import curve_fit

app = typer.Typer(help=__doc__, callback=main_callback)

general = GeneralSettings.get()
IMNAME = "fracture_image"

@app.command()
def gen(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
    realsize: Annotated[tuple[float, float], typer.Option(help='Real size of specimen in mm. -1 is used to indicate auto, then the specimen realsize is used.')] = (-1, -1),
    quiet: Annotated[bool, typer.Option(help='Do not ask for confirmation.')] = False,
    all: Annotated[bool, typer.Option(help='Generate splinters for all specimens.')] = False,
    all_exclude: Annotated[str, typer.Option(help='Exclude specimens from all.')] = None,
    all_skip_existing: Annotated[bool, typer.Option(help='Skip specimens that already have splinters.')] = False,
    from_label: Annotated[bool, typer.Option(help='Generate splinters from labeled image.')] = False,
    use_default_prep: Annotated[bool, typer.Option(help='Use default prep config.')] = False,
):
    """Generate the splinter data for a specific specimen."""
    if realsize[0] == -1 or realsize[1] == -1:
        realsize = None

    specimens: list[Specimen]

    if not all:
        filter = create_filter_function(specimen_name, needs_splinters=False, needs_scalp=False)

        specimens = Specimen.get_all_by(filter, load=True)
    else:
        def exclude(specimen: Specimen):
            if not specimen.has_fracture_scans:
                return False
            if specimen.has_splinters and all_skip_existing:
                return False
            if all_exclude is None:
                return True

            return re.search(all_exclude, specimen.name) is None
        specimens = Specimen.get_all_by(decider=exclude, load=True)


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
            if use_default_prep:
                prep = defaultPrepConfig

            # generate splinters for the specimen
            print("---        Summary         --")
            print(f'       px_per_mm = {px_per_mm:.2f}')
            print(f'            prep = "{prep.name}"')
            print(f'       real_size = {realsize}')

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
            specimen.splinter_area = np.sum([x.area for x in splinters])
            progress.advance()


def plot_touching_len(specimen, splinters) -> StateOutput:
    """Returns an image with the splinters colored after their amount of touching splinters."""
    output_image0 = np.zeros_like(specimen.get_fracture_image(), dtype=np.uint8)

    touches = [len(x.adjacent_splinter_ids) for x in splinters]
    max_adj = np.max(touches)
    min_adj = np.min(touches)

    if np.sum(touches) == 0:
        # print red error
        print(f"[red]Error: No touching splinters found for {specimen.name}.[/red")


    print(f"Max touching splinters: {max_adj}")
    print(f"Min touching splinters: {min_adj}")
    print(f"Mean touching splinters: {np.mean(touches)}")

    for splinter in track(splinters, description="Drawing touching splinters...", transient=True):
        clr = get_color(len(splinter.adjacent_splinter_ids), min_adj, max_adj)
        cv2.drawContours(output_image0, [splinter.contour], 0, clr, -1)

    # draw white contour around splinters for better visibility
    cv2.drawContours(output_image0, [x.contour for x in splinters], -1, (0, 0, 0), 1)

    output_image0 = annotate_image(
        output_image0,
        cbar_title="Amount of touching splinters",
        clr_format=".2f",
        cbar_range=(min_adj,max_adj),
        figwidth=FigureSize.ROW2,
    )

    return output_image0

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

    # create filled image with splinters colored after their amount of touching splinters
    outp = plot_touching_len(specimen, splinters)
    outp.overlayImpact(specimen)
    State.output(outp, "adjacent_plot", spec=specimen, to_additional=True)

    # create probability histogram plot of touching splinters
    lens = [len(x.adjacent_splinter_ids) for x in splinters]
    fig, axs = datahist_plot(
        x_label='Kantenanzahl $N_\\text{e}$',
        y_label='Wahrscheinlichkeitsdichte $f(N_\\text{e})$',
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

    State.output(fig, 'adjacent_pdf', spec=specimen, to_additional=True, figwidth=FigureSize.ROW1)

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
def nielsen_n50(

):
    def n50(sig_s: float):
        return (sig_s*0.5/14.96) ** 4


    specimens: list[Specimen] = Specimen.get_all_by(lambda x: x.has_splinters, load=True)
    x = np.zeros(len(specimens))
    y_real = np.zeros(len(specimens))
    y_nielsen = np.zeros(len(specimens))

    for i, specimen in enumerate(specimens):
        sig_s = np.abs(specimen.sig_h)

        real_n50 = specimen.calculate_nfifty_count()
        nielsen_spec_n50 = n50(sig_s)

        x[i] = sig_s
        y_real[i] = real_n50
        y_nielsen[i] = nielsen_spec_n50

    sz = FigureSize.ROW2
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    axs.scatter(x, y_real, label='Real')

    axs.set_xlabel('Stress $\sigma_s$ [MPa]')
    axs.set_ylabel('N50 [-]')

    # nielsen fit
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = n50(x_fit)
    axs.plot(x_fit, y_fit, label='Nielsen')

    # # real fit
    # popt,pcov = curve_fit(fitter, x, y_real)
    # y_fit = fitter(x_fit, *popt)
    # axs.plot(x_fit, y_fit, label='Real')


    axs.legend()
    State.output(fig, 'nielsen_n50', to_additional=True, figwidth=sz)


@app.command()
def draw_contours(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
    fill: Annotated[bool, typer.Option(help='Fill contours.')] = False,
    ls: Annotated[int, typer.Option(help='Line size.')] = 2,
    color: Annotated[str, typer.Option(help='Color of the contours. None means random color for each splinter.')] = None,
    label: Annotated[bool, typer.Option(help='Use a black background instead of the specimens fracture image.')] = False,
):
    specimen = Specimen.get(specimen_name)
    assert specimen.has_splinters, "Specimen has no splinters."
    splinters = specimen.splinters

    if not label:
        out_img = specimen.get_fracture_image()
    else:
        out_img = np.zeros_like(specimen.get_fracture_image(), dtype=np.uint8)

    with get_progress(total=len(splinters), title='Drawing contours') as progress:
        for splinter in splinters:
            if color is None:
                clr = rand_col()
            else:
                clr = norm_color(color, 255)

            cv2.drawContours(out_img, [splinter.contour], 0, clr, ls if not fill else -1)

            progress.advance()

    State.output(out_img, 'contours', spec=specimen, to_additional=True)

@app.command()
def draw_centroids(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
):
    specimen = Specimen.get(specimen_name)
    assert specimen.has_splinters, "Specimen has no splinters."
    splinters = specimen.splinters

    out_img = specimen.get_fracture_image()
    for splinter in track(splinters, description="Drawing contours...", transient=True):
        clr = (0,125,255)
        point = splinter.centroid_px
        cv2.circle(out_img, (int(point[0]), int(point[1])), 5, clr, -1)

    State.output(out_img, 'centroids', spec=specimen, to_additional=True)

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
def extract_details(
    names: Annotated[list[str], typer.Argument(help='Names of specimens to load')],
    region: Annotated[tuple[int, int, int, int], typer.Option(help='Region to extract. X Y W H.')] = (1250, 1250, 200, 200),
    fontscale: Annotated[str, typer.Option(help='Font scale.')] = FontSize.HUGEXL,
):
    filterf = create_filter_function(names, needs_splinters=True, needs_scalp=False)
    specimens: list[Specimen] = Specimen.get_all_by(filterf, load=True)

    assert region is not None, "Region must be specified."

    rsz_factor = 5

    for specimen in specimens:
        # get fracture image and take region
        fracture_image = specimen.get_fracture_image()

        x, y, w, h = region
        fracture_image = fracture_image[y:y+h, x:x+w]

        # calculate real size of region
        px_per_mm = specimen.calculate_px_per_mm()
        w_mm = w * rsz_factor / px_per_mm
        pxpmm = w * rsz_factor / w_mm
        scale_length_val = 20
        scale_length_px = scale_length_val * pxpmm * rsz_factor

        # upscale image using rsz_factor
        fracture_image = cv2.resize(fracture_image, (0, 0), fx=rsz_factor, fy=rsz_factor)

        # add text to center of line
        fracture_image = put_scale(scale_length_val/10,scale_length_px, fracture_image, clr=(0,0,0), sz=FontSize[fontscale.upper()])

        # save image
        State.output(fracture_image, f"extracted_{specimen.name}_{specimen.sig_h:.0f}", spec=specimen, to_additional=True, figwidth=FigureSize.ROW3)



@app.command()
def show_prep(specimen_name: str = None):
    """Show the default preprocessing configuration."""
    if specimen_name is None:
        inspect(defaultPrepConfig)

    else:
        specimen = Specimen.get(specimen_name)
        inspect(specimen.get_prepconf())


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
    plt_prop(specimen_name, prop=SplinterProp.ROUGHNESS)


@app.command()
def roundness(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')]):
    """Plot the roundness of a specimen."""
    plt_prop(specimen_name, prop=SplinterProp.ROUNDNESS)

@app.command()
def plt_prop_f2(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
    prop: Annotated[SplinterProp, typer.Argument(help='Property to plot.')],
    n_points: Annotated[int, typer.Option(help='Amount of points to evaluate.')] = 25,
    w_mm: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 50,
    smooth: Annotated[bool, typer.Option(help='Smooth the plot.')] = True,
):

    specimen = Specimen.get(specimen_name)

    X,Y,Z,Zstd = specimen.calculate_2d(prop, w_mm, n_points)

    output = plot_kernel_results(
        specimen.get_fracture_image(),
        Splinter.get_property_label(prop),
        True,
        False,
        KernelContourMode.FILLED,
        X, Y, Z,
        0,
        FigureSize.ROW2,
        crange=(np.nanmin(Z), np.nanmax(Z)),
        clr_format=".2f",
        smooth=smooth,
        fill_skipped_with_mean=False,
    )
    output.overlayImpact(specimen)
    State.output(output, f'{prop}_2d',spec=specimen, to_additional=True)

@app.command()
def plt_prop_f(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
    prop: Annotated[SplinterProp, typer.Argument(help='Property to plot.')],
    w_mm: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 50,
    as_contours: Annotated[bool, typer.Option(help='Plot the kernel as contours.')] = False,
    n_points: Annotated[int, typer.Option(help='Amount of points to evaluate.')] = general.n_points_kernel,
    plot_kernel: Annotated[bool, typer.Option(help='Plot the kernel.')] = False,
    plot_vertices: Annotated[bool, typer.Option(help='Plot the vertices.')] = False,
    exclude_points: Annotated[bool, typer.Option(help='Exclude impact point.')] = False,
    skip_edge: Annotated[bool, typer.Option(help='Skip edge.')] = False,
    figwidth: Annotated[FigureSize, typer.Option(help='Figure width.')] = FigureSize.ROW2,
):
    specimen = Specimen.get(specimen_name)
    pxpmm = specimen.calculate_px_per_mm()
    impact_pos = specimen.get_impact_position()
    def mean_value(splinters: list[Splinter]):
        values = np.array([x.get_splinter_data(prop, impact_pos, pxpmm) for x in splinters])
        return np.mean(values[~np.isnan(values)])

    w_px = int(w_mm * specimen.calculate_px_per_mm())

    clr_label = Splinter.get_property_label(prop)
    clr_label = clr_label[0].lower() + clr_label[1:]
    clr_label = "Normalized " + clr_label
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
        z_action=mean_value,
        clr_label=clr_label,
        mode=KernelContourMode.FILLED if not as_contours else KernelContourMode.CONTOURS,
        figwidth=figwidth,
        clr_format='.1f',
        normalize=True
    )

    fig_output.overlayImpact(specimen)
    State.output(fig_output, f'movavg_{prop}',spec=specimen, to_additional=True)

@app.command()
def plt_prop(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
    prop: Annotated[SplinterProp, typer.Argument(help='Property to plot.')],
):
    """
    Create a plot of a specimen where the splinters are filled with a color representing the selected property.
    """

    specimen = Specimen.get(specimen_name)

    out_img = specimen.get_fracture_image()
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

    ip = specimen.get_impact_position()
    px_p_mm = specimen.calculate_px_per_mm()
    prop_values_dict = {}
    for splinter in specimen.splinters:
        val = splinter.get_splinter_data(prop=prop, ip_mm=ip,px_p_mm=px_p_mm)
        if np.isfinite(val):
            prop_values_dict[splinter.ID] = val

    prop_values_list = list(prop_values_dict.values())

    max_prop = np.max(prop_values_list)
    min_prop = np.min(prop_values_list)
    mean_prop = np.mean(prop_values_list)

    print(f'# {prop.name}')
    print(f"Max:    {max_prop:>15.2f}")
    print(f"Min:    {min_prop:>15.2f}")
    print(f"Mean:   {mean_prop:>15.2f}")
    print(f"Median: {np.median(prop_values_list):>15.2f}")

    # d = np.median(rough) - min_r
    # max_r = np.median(rough) + d
    overlay_img = np.zeros_like(out_img, dtype=np.uint8)

    for splinter in track(specimen.splinters, description=f"Calculating {prop}", transient=True):
        if splinter.ID not in prop_values_dict:
            continue

        splinter_prop_value = prop_values_dict[splinter.ID]
        clr = get_color(splinter_prop_value, min_prop, max_prop)

        cv2.drawContours(overlay_img, [splinter.contour], 0, clr, -1)

    cv2.drawContours(overlay_img, [x.contour for x in specimen.splinters], -1, (0, 0, 0), 1)
    out_img = cv2.addWeighted(out_img, 0.3, overlay_img, 1, 1)

    State.output(out_img, f'{prop}_filled_raw', spec=specimen, to_additional=True, figwidth=FigureSize.IMG)

    clr_label = Splinter.get_property_label(prop)
    out_img = annotate_image(
        out_img,
        cbar_title=clr_label,
        clr_format=".1f",
        cbar_range=(min_prop, max_prop),
        figwidth=FigureSize.ROW2,
    )

    out_img.overlayImpact(specimen)
    State.output(out_img, f'{prop}_filled', spec=specimen, to_additional=True)

def str_to_intlist(input: str) -> list[int]:
    if isinstance(input, int):
        return [input]

    return [int(x) for x in input.split(",")]


def specimen_parser(input: str):
    return input




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
        if spec.boundary == SpecimenBoundary.Unknown:
            return False
        if spec.thickness not in thickness:
            return False
        if spec.settings["break_pos"] != break_pos:
            return False

        return True

    def specimen_value(spec: Specimen):
        return (spec.boundary, np.mean([x.area for x in spec.splinters]), np.abs(spec.U))

    specimens = Specimen.get_all_by(decider, load=False)

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

    specimens: list[Specimen] = Specimen.get_all_by(filter, load=False)

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
        n_bins: Annotated[int, typer.Option(help='Number of bins for histogram.')] = 60,
        energy: Annotated[float, typer.Option(help='Energy range.')] = None,
    ):
    """Plot a 2D histogram of splinter sizes and stress."""

    filter = create_filter_function(names, sigmas, delta, energy=energy,
                                    exclude=exclude,
                                    needs_scalp=True,
                                    needs_splinters=True)

    specimens: list[Specimen] = Specimen.get_all_by(filter, max_n=maxspecimen, load=True)

    assert len(specimens) > 0, "[red]No specimens loaded.[/red]"

    sz = FigureSize.WIDE
    binrange = np.linspace(0, 2, n_bins)
    fig, axs = plt.subplots(figsize=get_fig_width(sz))

    data = []
    stress = []

    with get_progress() as progress:
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
                stress.append(specimen.U)
            else:
                stress.append(specimen.sig_h)

            progress.advance()

    # sort data and names for ascending stress
    stress, data = sort_two_arrays(stress, data, True)
    names = [x[1] for x in data]
    data = [x[0] for x in data]
    axs.set_xlabel(AxLabels.SPLINTER_AREA)
    if not y_stress:
        axs.set_ylabel("$U$ (J/m²)")
    else:
        axs.set_ylabel("$\sigma_\\text{s}$ (MPa)")

    x_ticks = np.arange(0, n_bins, 10)
    siz_mod = 2 if len(x_ticks) > 10 else 1
    axs.set_xticks(x_ticks, [f'{10 ** edges[x]:.2f}' if i % siz_mod == 0 else "" for i, x in enumerate(x_ticks)])

    str_mod = 5 if len(stress) > 15 else 1
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

    if sigmas is not None:
        out_name = f"{sigmas[0]}_{sigmas[1]}"
    elif names is not None:
        out_name = f"{names[0]}"

    axs.grid(False)

    # disable minor ticks on y axis
    axs.yaxis.set_minor_locator(plt.NullLocator())
    axy.yaxis.set_minor_locator(plt.NullLocator())


    State.output(fig, f'loghist2d_{out_name}', to_additional=True, figwidth=sz)


def create_filter_function(name_filter,
                           sigmas=None,
                           sigma_delta=10,
                           energy=None,
                           exclude: str = None,
                           needs_scalp=True,
                           needs_splinters=True,
                           needs_fracture_scans=True
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

    def custom_regex_filter(s: Specimen, filter: str) -> bool:
        # format: t.sigma.b.nbr
        values = filter.split(".")
        # print(values)
        t, sigma, b, nbr = values


        rt = ""
        rsigma = ""
        rb = ""
        rnbr = ""

        if ":" in t:
            t = t.split(":")
            # create regex pattern which matches any possible t values
            rt = "(" + "|".join(t) + ")"
        else:
            rt = t.replace(".", "\.").replace("*", ".*").replace('!', '|')

        if ":" in sigma:
            sigma = sigma.split(":")
            # create regex pattern which matches any possible sigma values
            rsigma = "(" + "|".join(sigma) + ")"
        else:
            rsigma = sigma.replace(".", "\.").replace("*", ".*").replace('!', '|')

        if ":" in b:
            b = b.split(":")
            # create regex pattern which matches any possible b values
            rb = "(" + "|".join(b) + ")"
        else:
            rb = b.replace(".", "\.").replace("*", ".*").replace('!', '|')

        if ":" in nbr:
            nbr = nbr.split(":")
            # create regex pattern which matches any possible nbr values
            rnbr = "(" + "|".join(nbr) + ")"
        else:
            rnbr = nbr.replace(".", "\.").replace("*", ".*").replace('!', '|')


        regex = f"{rt}\.{rsigma}\.{rb}\.{rnbr}"
        # print(regex)
        return re.match(regex, s.name) is not None

    name_filter_function: Callable[[Specimen, Any], bool] = None

    # create name_filter_function based on name_filter
    if name_filter is not None and "," in name_filter:
        name_filter = name_filter.split(",")
        print(f"Searching for specimen whose name is in: {name_filter}")
        name_filter_function = in_names_list
    elif name_filter is not None and " " in name_filter:
        name_filter = name_filter.split(" ")
        print(f"Searching for specimen whose name is in: {name_filter}")
        name_filter_function = in_names_list
    elif isinstance(name_filter, list):
        print(f"Searching for specimen whose name is in: {name_filter}")
        name_filter_function = in_names_list
    elif name_filter is not None and all([c not in "*[]^\\" for c in name_filter]):
        name_filter = [name_filter]
        print(f"Searching for specimen whose name is in: {name_filter}")
        name_filter_function = in_names_list
    elif name_filter is not None and ":" in name_filter:
        name_filter_function = custom_regex_filter
    elif name_filter is not None and any([c in "*[]^\\" for c in name_filter]):
        print(f"Searching for specimen whose name matches: {name_filter}")
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
            debug(f"Specimen '{specimen.name}' has no scalp data. Skipping.")
            return False
        elif needs_splinters and not specimen.has_splinters:
            debug(f"Specimen '{specimen.name}' has no splinter data. Skipping.")
            return False
        elif sigmas is not None:
            return sigmas[0] <= abs(specimen.scalp.sig_h) <= sigmas[1]
        elif energy is not None:
            return abs(specimen.U-energy) < 0.05 * energy
        elif needs_fracture_scans and not specimen.has_fracture_scans:
            debug(f"Specimen '{specimen.name}' has no fracture scans. Skipping.")
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
    plot_mode: Annotated[DataHistPlotMode, typer.Option(help='Histograms or KDE Estimation, only applies to data_mode=pdf. If not specified and PDF-Mode, plot_mode is HIST.')] = None,
    legend: Annotated[str, typer.Option(help='Legend style (0: Name, 1: Sigma, 2: Dicke, 3: Mean-Size).')] = None,
    xlim: Annotated[tuple[float, float], typer.Option(help='X-Limits for plot')] = (0, 2),
    figwidth: Annotated[FigureSize, typer.Option(help='Width of the figure.')] = FigureSize.ROW2,
):
    """Plot logaritmic histograms of splinter sizes for specimens."""

    if data_mode == DataHistMode.CDF and plot_mode is not None:
        # print info that plot_mode is ignored
        print("[cyan]Plot mode is ignored when using CDF mode.[/cyan]")
    elif data_mode == DataHistMode.PDF and plot_mode is None:
        plot_mode = DataHistPlotMode.HIST


    filter = create_filter_function(names, sigmas, needs_scalp=False, needs_splinters=True)
    specimens = Specimen.get_all_by(filter, load=True)

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

    fig, axs = datahist_plot(figwidth=figwidth, data_mode=data_mode)

    a = 2 / len(specimens) if len(specimens) > 1 else 1

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
        clr_label="Normalisierte Ausrichtung $\\bar{\Delta}$",
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


    if not State.debug:
        plt_prop(specimen_name, prop=SplinterProp.ORIENTATION)
        return

    specimen = Specimen.get(specimen_name)

    impact_pos = specimen.get_impact_position()
    splinters = specimen.splinters

    # analyze splinter orientations
    debug_img = specimen.get_fracture_image()
    n = 0
    for s in track(splinters):
        if n % 10 == 0:
            # draw splinter contour into image
            cv2.drawContours(debug_img, [s.contour], -1, (0, 0, 255), 1)
            A = impact_pos - s.centroid_mm

            # draw splinter ellipse
            ellipse = cv2.fitEllipse(s.contour)
            cv2.ellipse(debug_img, ellipse, (0, 255, 255), 1)

            # get bounding box major axis
            major_axis_angle = np.deg2rad(ellipse[2])
            major_axis_vector = np.array([-np.sin(major_axis_angle), np.cos(major_axis_angle)])
            p0 = np.array(ellipse[0])
            p1 = p0 + major_axis_vector * ellipse[1][0] * 1.2
            cv2.line(debug_img, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 255, 255), 1)
            c0 = p0 - A * 20 / np.linalg.norm(A)
            c1 = p0
            cv2.line(debug_img, tuple(c0.astype(int)), tuple(c1.astype(int)), (255, 255, 0), 1)


            # # draw splinter bounding box
            # (x,y),(w,h),a = cv2.minAreaRect(s.contour)
            # box = cv2.boxPoints(((x,y),(w,h),a))
            # box = np.int0(box)
            # cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 1)

            # # get bounding box major axis
            # major_axis_angle = np.deg2rad(a)
            # major_axis_vector = np.array([np.cos(major_axis_angle), np.sin(major_axis_angle)])
            # p0 = np.array([x,y])
            # p1 = p0 + major_axis_vector * w * 1.2
            # cv2.line(debug_img, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 255, 0), 1)

            try:
                # extract splinter with its bounding box
                bbox = cv2.boundingRect(s.contour)
                # enlarge bbox by 20px
                bbox = (bbox[0]-20, bbox[1]-20, bbox[2]+40, bbox[3]+40)
                bbox_img = debug_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                bbox_w, bbox_h = bbox_img.shape[:2]

                bbox_img = cv2.resize(bbox_img, (int(bbox_h*4), int(bbox_w*4)), interpolation=cv2.INTER_NEAREST)
                bbox_w, bbox_h = bbox_img.shape[:2]

                h0 = 15
                cv2.rectangle(bbox_img, (0, 0), (250, 3*h0), (255, 255, 255), -1)
                # cv2.putText(bbox_img, f"{a:.0f} deg", (5, h0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # cv2.putText(bbox_img, f"{w:.0f}x{h:.0f}", (5, 2*h0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # B = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
                # cv2.putText(bbox_img, f"{alignment_sim(A,B)*100:.2f}%", (5, 3*h0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # cv2.putText(bbox_img, f"Angle: {ellipse[2]:.0f} deg", (5, h0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                w,h = ellipse[1]
                B = np.array([-np.sin(np.deg2rad(ellipse[2])), np.cos(np.deg2rad(ellipse[2]))])
                cv2.putText(bbox_img, f"Cosine-Similarity: {alignment_cossim(A,B)*100:.2f}%", (5, 1*h0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(bbox_img, f"Orientation strength: {s.measure_orientation(impact_pos):.2f}", (5, 2*h0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # save image to debug folder
                out_file = State.get_general_outputfile(f"debug/{specimen.name}_{n}.png")
                cv2.imwrite(out_file, bbox_img)
            except Exception as e:
                print(f"Error: {e}")
        n += 1

@app.command()
def get_energy(
    specimen_name: str = typer.Argument(help='Name of specimen to load'),
):
    specimen = Specimen.get(specimen_name, load=True)
    print(f"U_d={specimen.U_d:.2f} J/m²")
    print(f"U={specimen.U:.2f} J/m²")

@app.command()
def ud(sigma: float) -> float:
    nue = 0.23
    E = 70e3
    print("Strain Energy Density [J/m³]")
    print(f"U_d={1e6/5 * (1-nue)/E * (sigma ** 2):.2f} J/m²")

@app.command()
def u(sigma: float, thickness: float) -> float:
    nue = 0.23
    E = 70e3
    thickness = float(thickness) * 1e-3
    print("Strain Energy [J/m²]")
    print(f"U={thickness * 1e6/5 * (1-nue)/E * sigma ** 2:.2f} J/m²")


@app.command()
def kde_impact_layer(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
    mode: Annotated[SplinterProp, typer.Option(help='Mode for the aspect ratio.')] = 'asp',
):
    specimen = Specimen.get(specimen_name)
    px_per_mm = specimen.calculate_px_per_mm()
    ip = specimen.get_impact_position()

    # precomputed data
    data = {}
    for splinter in specimen.splinters:
        data[splinter] = splinter.get_splinter_data(prop=mode, px_p_mm=px_per_mm, ip_mm=ip)

    def splinter_data_getter(splinters: list[Splinter]):
        if len(splinters) == 0:
            return np.nan

        r = 0
        for s in splinters:
            r += data[s]

        return r / len(splinters)

    clr_label = Splinter.get_property_label(mode, row3=False)

    w_mm = 50
    n_points = 25

    soutput = plot_splinter_movavg(
        specimen.get_fracture_image(),
        specimen.splinters,
        exclude_points=None,
        skip_edge=False,
        fill_skipped_with_mean=True,
        n_points=n_points,
        kw_px=w_mm*px_per_mm,
        z_action=splinter_data_getter,
        clr_label=clr_label,
        mode=KernelContourMode.FILLED,
        figwidth=FigureSize.ROW2,
        clr_format='.1f',
        normalize=False,
    )
    soutput.overlayImpact(specimen)
    State.output(soutput,f"impact-layer_{mode}" ,spec=specimen, to_additional=True)


    img = create_splinter_sizes_image(
        specimen.splinters,
        specimen.get_fracture_image().shape,
        annotate=False,
        with_contours=True,
        annotate_title=f"Splinter {mode}",
        s_value=lambda x: x.get_splinter_data(mode=mode, px_p_mm=px_per_mm, ip=ip),
    )

    minval = np.min([x for x in data.values() if np.isfinite(x)])
    maxval = np.max([x for x in data.values() if np.isfinite(x)])
    output = annotate_image(
        img,
        cbar_title=clr_label,
        figwidth=FigureSize.ROW2,
        clr_format='.1f',
        min_value = minval,
        max_value = maxval,
    )
    output.overlayImpact(specimen)
    State.output(output, f"impact-layer_{mode}_splinters", spec=specimen, to_additional=True)


class EnergyUnit(str,Enum):
    U = "U"
    "Strain Energy [J/m²]"
    UD = "Ud"
    "Strain Energy Density [J/m³]"
    Ut = "Ut"
    "Tensile Strain Energy [J/m²]"
    UDt = "Udt"
    "Tensile Strain Energy Density [J/m³]"


@app.command()
def test_navid_nfifty():
    sz = FigureSize.ROW2
    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    # get navids ud
    from fracsuite.core.navid_results import nfifty_ud
    ud_x = nfifty_ud[:,0]
    ud_y = nfifty_ud[:,1]

    ud0 = navid_nfifty_ud()
    ud0_x = ud0[:,0]
    ud0_y = ud0[:,1]

    def ud(x):
        return 0.255*x**2+109.28*x+5603
    xs = np.linspace(1,200,100)
    yud = ud(xs)

    # plot the xs,yud to the background
    axs.plot(xs,yud, color='k', label="Fit")

    axs.scatter(ud_x, ud_y, label="Original", marker='o', facecolors='none', edgecolors='r')


    for it,t in enumerate([4,8,12]):
        mask = ud0[:,2] == t
        ud0_x = ud0[mask,0]
        ud0_y = ud0[mask,1]

        axs.scatter(ud0_x, ud0_y, label=f"{t}mm from U", marker='osv'[it], facecolors='none', edgecolors='b')

    axs.set_xlabel("Bruchstückdichte N50")
    axs.set_ylabel("Formänderungsenergiedichte $U_d$ [J/m³]")
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.legend()
    State.output(fig, "ud", to_additional=True,figwidth=sz)

    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    def u4(x):
        return 0.58*x+49.47
    def u8(x):
        return 1.14*x+49.51
    def u12(x):
        return 1.92*x+48.24

    tf = [
        u4,u8,u12
    ]

    # now plot all thicknesses from u
    for it, t in enumerate([4,8,12]):
        navid_n50 = navid_nfifty(t)
        navid_x = navid_n50[:,0]
        navid_y = navid_n50[:,1]
        # navids points
        axs.scatter(
            navid_x,
            navid_y,
            marker='osv'[it],
            facecolors='none',
            label=f'{t:.0f}mm',
            edgecolors='rgb'[it],
            linewidth=1,
        )


        ys = tf[it](xs)
        axs.plot(xs,ys, color='rgb'[it])

    axs.set_xlabel("Bruchstückdichte N50")
    axs.set_ylabel("Formänderungsenergie $U$ [J/m²]")

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.legend()
    State.output(fig, "u", to_additional=True,figwidth=sz)
@app.command()
def nfifty(
    bound: Annotated[str, typer.Option(help='Boundary of the specimen.')] = None,
    break_pos: Annotated[str, typer.Option(help='Break position.')] = 'corner',
    unit: Annotated[EnergyUnit, typer.Option(help='Energy unit.')] = EnergyUnit.U,
    recalc: Annotated[bool, typer.Option(help='Recalculate N50.')] = False,
    use_mean: Annotated[bool, typer.Option(help='Use mean splinter size.')] = False,
):
    bid = {
        'A': 1,
        'B': 2,
        'Z': 3,
    }

    tcolors = {
        1: 'green',
        2: 'red',
        3: 'blue',
    }
    bname = {
        1: 'A',
        2: 'B',
        3: 'Z',
    }
    bmarkers ={
        1: 'o',
        2: 'v',
        3: 'X',
    }
    thicknesses = [4, 8, 12]

    def add_filter(specimen: Specimen):
        if break_pos is not None and specimen.break_pos != break_pos:
            return False
        if specimen.boundary == SpecimenBoundary.Unknown:
            return False

        if not specimen.has_splinters:
            return False

        if bound is not None and specimen.boundary != bound:
            return False
        if specimen.U_d is None or not np.isfinite(specimen.U_d):
            return False

        return True

    specimens: list[Specimen] = Specimen.get_all_by(add_filter , load=True)

    centers = [
        [450,50],
        [50,450],
        [450,450],
        [200,200],
        [50,200]
    ]

    results = np.zeros((len(specimens), 7), dtype=np.float64)
    with get_progress(title='Working on specimens...') as progress:
        for i, specimen in enumerate(specimens):
            if not use_mean:
                nfifty = specimen.calculate_nfifty_count(centers, (50,50), force_recalc=recalc)
            else:
                nfifty = specimen.calculate_intensity(force_recalc=recalc)
            results[i,:] = (specimen.U, specimen.U_d, specimen.calculate_energy(), specimen.calculate_energy_density() , specimen.thickness, bid[specimen.boundary], nfifty)

            progress.advance()


    idd = {
        EnergyUnit.U: 0,
        EnergyUnit.UD: 1,
        EnergyUnit.Ut: 2,
        EnergyUnit.UDt: 3,
    }

    id = idd[unit]

    id_name = {
        0: "Formänderungsenergie $U$ [J/m²]",
        1: "Formänderungsenergiedichte $U_d$ [J/m³]",
        2: "Effektive Formänderungsenergie $U_t$ [J/m²]",
        3: "Effektive Formänderungsenergiedichte $U_{dt}$ [J/m³]",
    }

    def U4(x):
        return 0.58 *x + 49.47

    def U8(x):
        return 1.14 * x + 49.51

    def U12(x):
        return 1.92 * x + 48.24

    def UD(x):
        return 0.255 * x ** 2 + 109.28 * x + 5603.2


    sz = FigureSize.ROW3
    lw = 0.3 # line width for scatter plots

    if sz == FigureSize.ROW3:
        for idn in id_name:
            id_name[idn] = " ".join(id_name[idn].split(" ")[-2:])

    # hard coded n50 range
    min_N50 = 0
    max_N50 = 400

    axs: Axes
    fig, axs = plt.subplots(figsize=get_fig_width(sz))
    cfg_logplot(axs)

    if unit == EnergyUnit.UD or unit == EnergyUnit.UDt:
        navid_n50 = navid_nfifty_ud()

        for ith, th in enumerate([4,8,12]):
            navid_r = navid_n50[navid_n50[:,2] == th]

            navid_x = navid_r[:,0] #n50
            navid_y = navid_r[:,1] #u|ud

            # navids points
            axs.scatter(
                navid_x,
                navid_y,
                marker='osv'[ith],
                facecolors='grb'[ith],
                edgecolors='none',
                label=f'{th:.0f}mm (old)',
                linewidth=lw,
                alpha = 0.4
            )

    for it, thick in enumerate(thicknesses):
        clr = tcolors[it+1]
        mask = results[:,4] == thick
        # create a fitting curve
        x = results[mask,-1]
        y = results[mask,id]

        # get results from navid
        if unit == EnergyUnit.U or unit == EnergyUnit.Ut:
            navid_n50 = navid_nfifty(thick, as_ud=(unit == EnergyUnit.UD or unit == EnergyUnit.UDt))
            navid_x = navid_n50[:,0]
            navid_y = navid_n50[:,1]
            # navids points
            axs.scatter(
                navid_x,
                navid_y,
                marker='o',
                facecolors='none',
                label=f'{thick:.0f}mm (old)',
                edgecolors=clr,
                linewidth=lw,
                alpha=0.4
            )
        elif unit == EnergyUnit.UD or unit == EnergyUnit.UDt:
            navid_r = navid_n50[navid_n50[:,2] == thick]

            navid_x = navid_r[:,0] #n50
            navid_y = navid_r[:,1] #ud

        if len(y) > 0:
            # fit a curve
            def func(x, a, b):
                return np.float64(a * x + b)

            x = np.concatenate([x,navid_x])
            y = np.concatenate([y,navid_y])
            p = np.column_stack([x,y])

            print(p.shape)
            print(p[0])
            # sort for x
            p = p[p[:,0].argsort()]

            x = p[:,0]
            y = p[:,1]
            popt, pcov = curve_fit(func, x.astype(np.float64), y.astype(np.float64), p0=(1, 1))
            print(f'{thick}mm Fitting cov:', pcov)

            # plot the curve
            x = np.linspace(np.min(x), np.max(x), 100)
            y = func(x, *popt)
            axs.plot(x, y, linestyle=(0,(1,1)), color=clr)

            # calculate r^2
            residuals = y - func(x, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y-np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f'{thick}mm r^2:', r_squared)

        # navid_n50, navid_ud, navid_u = navid_nfifty_interpolated(thick)
        # if unit == EnergyUnit.U:
        #     # print(navid_n50, navid_ud, navid_u)
        #     plt.plot(navid_n50, navid_u, linestyle='-.', color=clr, label=f"{thick}mm (Appendix)", linewidth=0.6)
        # elif unit == EnergyUnit.UD:
        #     plt.plot(navid_n50, navid_ud, linestyle='-.', color=clr, label=f"{thick}mm (Appendix)", linewidth=0.6)



        for b in bmarkers:
            mask = (results[:,4] == thick) & (results[:,-2] == b)
            ms = '*'
            axs.scatter(results[mask,-1], results[mask,id], marker=ms, linewidth=lw, color=clr)


    ux = np.linspace(min_N50, max_N50, 100)
    u4y = U4(ux)
    u8y = U8(ux)
    u12y = U12(ux)
    udy = UD(ux)

    if id == 0:
        axs.plot(ux, u4y, linestyle='--', color=tcolors[1], alpha=0.4)
        axs.plot(ux, u8y, linestyle='--', color=tcolors[2], alpha=0.4)
        axs.plot(ux, u12y, linestyle='--', color=tcolors[3], alpha=0.4)

    elif id == 1:
        axs.plot(ux, udy, label="P-Mogh. (2020)", linestyle='--', color='k', alpha=0.4)
        # fit a curve into all scattered points
        x = np.concatenate([results[:,-1], navid_x])
        y = np.concatenate([results[:,id], navid_y])

        # sort for x
        x,y = sort_arrays(x,y)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # fit a curve to x and y
        def func(x, a, b):
            return np.float64(a * x + b)

        popt, pcov = curve_fit(func, x.astype(np.float64), y.astype(np.float64), p0=(1, 1))

        # plot the curve
        x = np.linspace(np.min(x), np.max(x), 100)
        y = func(x, *popt)
        # axs.plot(x, y, linestyle=(0,(1,1)), color='k', alpha=1, label='Bohmann (2024)')

    # labeling for leons data
    for b,t in zip(bid.values(), thicknesses):
        axs.scatter([], [], label=f"{t}mm", marker='x', color=tcolors[b])
    for b,t in zip(bid.values(), thicknesses):
        axs.plot([],[], label=f"{t}mm", color=tcolors[b])

    axs.set_ylabel(id_name[id])
    axs.set_xlabel("Bruchstückdichte $N_\\text{50}$")
    # axs.legend(loc='best')

    y_max = np.max(results[:,id])
    y_max = 10 ** np.ceil(np.log10(y_max))
    # Anpassen der Y-Achsen-Grenzen
    axs.set_ylim(bottom=axs.get_ylim()[0], top=y_max)

    name = 'nfifty' if not use_mean else 'nperwindow'
    State.output(StateOutput(fig, sz), f'{name}_{bound}_{break_pos}_{unit}', to_additional=True)

    if State.debug:
        for i in range(len(specimens)):
            res = results[i,:]
            print(specimens[i].name, "U:", res[0], "U_d:", res[1], "U_t", res[2], "Ud_t", res[3] , "Thickness:", res[4], "Boundary:", res[3], "N50:", res[4])

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
        clr_label="Schwarzwert $N_\\text{schw. Pxl}$",
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

    original_image = specimen.get_fracture_image(as_rgb=False)

    # this counts the splinters in the kernel by default
    fig = plot_splinter_movavg(
        original_image,
        specimen.splinters,
        w_px,
        n_points,
        z_action=lambda x: len(x),
        plot_vertices=plot_vertices,
        plot_kernel=plot_kernel,
        clr_label="Bruchstückdichte $N_\\text{50}=N_\\text{S}/A_\\text{50x50mm}$",  # , $w_A,h_A$={w_mm}mm
        mode=KernelContourMode.FILLED if not as_contours else KernelContourMode.CONTOURS,
        exclude_points=[specimen.get_impact_position(True)] if exclude_points else None,
        skip_edge=skip_edge,
        figwidth=figwidth,
        clr_format='.1f',
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
        rotate: Annotated[bool, typer.Option('--rotate', help='Rotate the input image 90° CCW. Defaults to False.')] = False,
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
    if all:
        specimens = Specimen.get_all()
    elif isinstance(specimen_name, Specimen):
        specimens = [specimen_name]
    else:
        specimens = Specimen.get_all(specimen_name)

    for specimen in track(specimens):
        specimen.transform_fracture_images(
            resize_only=resize_only,
            rotate_only=rotate_only,
            rotate=rotate,
            crop=crop,
            size_px=size,
        )
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

    config = None
    # try to find preprocessor config in input directory
    if (config_file := find_file(test_dir, "*_prep.json")) is not None:
        config = PreprocessorConfig.load(config_file)
        print(f"Loaded config from {config_file}")

    # get splinters from watershed
    splinters = Splinter.analyze_image(input_img, prep=config)

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
