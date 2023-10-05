"""
Splinter analyzation tools.
"""

import os
from itertools import groupby
import re
import subprocess
from typing import Annotated, Any, Callable

import cv2
import numpy as np
import numpy.typing as npt
import typer
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rich import print
from rich.progress import track
from fracsuite.core.calculate import pooled

from fracsuite.core.plotting import (
    create_colored_splinter_image,
    datahist_plot,
    plot_image_kernel_contours,
    plot_splinter_kernel_contours,
    create_splinter_sizes_image,
    datahist_to_ax
)

from fracsuite.core.image import to_rgb
from fracsuite.core.imageplotting import plotImage, plotImages
from fracsuite.core.progress import get_progress
from fracsuite.core.plotting import modified_turbo
from fracsuite.core.stochastics import csintkern_objects_diagonal
from fracsuite.core.coloring import get_color
from fracsuite.splinters.processing import crop_matrix, crop_perspective, preprocess_image
from fracsuite.splinters.splinter import Splinter
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import annotate_image, bin_data, find_file, find_files, write_image
from fracsuite.tools.maincallback import main_callback
from fracsuite.tools.specimen import Specimen



app = typer.Typer(help=__doc__, callback=main_callback)

general = GeneralSettings.get()

def finalize(out_name: str):
    print(f"Saved to '{out_name}'.")
    subprocess.Popen(['start', '', '/b', out_name], shell=True)
@app.command(name='norm')
def count_splinters_in_norm_region(
        specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
        norm_region_center: Annotated[tuple[int,int], typer.Option(help='Center of the norm region in mm.', metavar='X Y')] = (400,400),
        norm_region_size: Annotated[tuple[int,int], typer.Option(help='Size of the norm region in mm.', metavar='W H')] = (50,50),
    ) -> float:
    specimen = Specimen.get(specimen_name)
    assert specimen is not None, "Specimen not found."

    # create rectangle around args.normregioncenter with 5x5cm size
    # and count splinters in it
    x,y = norm_region_center
    w,h = norm_region_size
    x1 = x - w // 2
    x2 = x + w // 2
    y1 = y - h // 2
    y2 = y + h // 2

    s_count = 0
    # count splinters in norm region
    for s in specimen.splinters:
        if s.in_region((x1,y1,x2,y2)):
            s_count += 1

    # print(f'Splinters in norm region: {s_count}')

    # transform to real image size
    x1 = int(x1 // specimen.splinter_config.size_factor)
    x2 = int(x2 // specimen.splinter_config.size_factor)
    y1 = int(y1 // specimen.splinter_config.size_factor)
    y2 = int(y2 // specimen.splinter_config.size_factor)

    orig_image = to_rgb(specimen.get_fracture_image())
    filled_image = to_rgb(specimen.get_filled_image())

    # get norm region from original image (has to be grayscale for masking)
    norm_region_mask = np.zeros_like(cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY))
    cv2.rectangle(norm_region_mask, (x1,y1), (x2,y2), 255, -1)
    # create image parts
    normed_image = cv2.bitwise_and(filled_image, filled_image, mask=norm_region_mask)
    normed_image_surr = orig_image #cv2.bitwise_and(self.original_image, self.original_image, mask=norm_region_inv)
    # add images together
    normed_image = cv2.addWeighted(normed_image, 0.3, normed_image_surr, 1.0, 0)
    cv2.rectangle(normed_image, (x1,y1), (x2,y2), (255,0,0), 5)
    cv2.putText(normed_image, f'{s_count}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,0,255), 20)
    outname = os.path.join(specimen.splinters_path, f"norm_count.{general.image_extension}")
    cv2.imwrite(outname, cv2.resize(normed_image, (0,0), fx=0.5, fy=0.5))

    specimen.set_setting('esg_count', s_count)

    finalize(outname)

@app.command()
def roughness_f(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
                kernel_width: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 200,):
    """Create a contour plot of the roughness on the specimen.

    Args:
        specimen_name (str, optional): Name of the specimens to load.
    """
    def roughness_function(splinters: list[Splinter]):
        return np.mean([splinter.calculate_roughness() for splinter in splinters])

    # create contour plot of roughness
    specimen = Specimen.get(specimen_name)
    assert specimen is not None, "Specimen not found."

    fig = plot_splinter_kernel_contours(specimen.get_fracture_image(),
                                        splinters=specimen.splinters,
                                        kernel_width=kernel_width,
                                        z_action=roughness_function,
                                        clr_label='Mean roughness',
                                        fig_title='Splinter Roughness')

    out_path = os.path.join(specimen.splinters_path, f"fig_roughintensity.{general.image_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    finalize(out_path)

@app.command()
def roundness_f(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
                kernel_width: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 200,):
    """Create a contour plot of the roundness on the specimen.

    Args:
        specimen_name (str, optional): Name of the specimens to load.
    """
    def roundness_function(splinters: list[Splinter]):
        return np.mean([splinter.calculate_roundness() for splinter in splinters])

    # create contour plot of roughness
    specimen = Specimen.get(specimen_name)
    assert specimen is not None, "Specimen not found."

    fig = plot_splinter_kernel_contours(specimen.get_fracture_image(),
                                        splinters=specimen.splinters,
                                        kernel_width=kernel_width,
                                        z_action=roundness_function,
                                        clr_label='Mean roughness',
                                        fig_title='Splinter Roughness')

    out_path = general.get_output_file( specimen_name, "fracture", "splinter", f"fig_roundintensity.{general.plot_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    finalize(out_path)

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

    # plot an overlay of the colorbar into the image on the right side
    # colorbar = np.zeros((out_img.shape[0], 50, 3), dtype=np.uint8)
    # for i in range(out_img.shape[0]):
    #     clr = get_color(i, 0, out_img.shape[0])
    #     colorbar[i] = clr
    # out_img = np.concatenate((out_img, colorbar), axis=1)
    out_img = annotate_image(out_img, cbar_title="Roughness", min_value=min_r, max_value=max_r)
    out_path = general.get_output_file(f"{specimen.name}.{general.image_extension}")
    write_image(out_img, out_path)

    finalize(out_path)


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
    max_r = np.mean(rounds) + np.mean(rounds) * 0.6
    min_r = np.mean(rounds) - np.mean(rounds) * 0.6


    for splinter in track(specimen.splinters):
        r = splinter.calculate_roundness()
        clr = get_color(r, min_r, max_r)

        cv2.drawContours(out_img, [splinter.contour], 0, clr, -1)

    out_img = annotate_image(out_img,
                                  cbar_title="Roundness",
                                  min_value=min_r,
                                  max_value=max_r,)
    out_path = general.get_output_file(f"{specimen.name}.{general.image_extension}")
    cv2.imwrite(out_path, out_img)

    finalize(out_path)


def str_to_intlist(input: str) -> list[int]:
    if isinstance(input, int):
        return [input    ]

    return [int(x) for x in input.split(",")]

def specimen_parser(input: str):
    return input

def sort_two_arrays(array1, array2, reversed = False) -> tuple[list, list]:
    # Combine x and y into pairs
    pairs = list(zip(array1, array2))
    # Sort the pairs based on the values in x
    sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=reversed)
    # Separate the sorted pairs back into separate arrays
    return zip(*sorted_pairs)

@app.command(name='sigmasize')
def size_vs_sigma(xlim: Annotated[tuple[float,float], typer.Option(help='X-Limits for plot')] = (0, 2),
                  thickness: Annotated[list[int], typer.Option(help='Thickness of specimens.', parser=str_to_intlist, metavar='4,8,12')] = [8],
                  break_pos: Annotated[str, typer.Option(help='Break position.', metavar='[corner, center]')] = "corner",
                  more_data: Annotated[bool, typer.Option('--moredata', help='Write specimens sig_h and thickness into legend.')] = False,
                  nolegend: Annotated[bool, typer.Option('--nolegend', help='Dont display the legend on the plot.')] = False,):
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
            return a+(b/(c+x))
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
    ax.set_ylim((min_size-2, max_size+2))
    if not nolegend:
        ax.legend(loc='best')
    ax.grid(True, which='both', axis='both')
    fig.tight_layout()
    out_name = general.get_output_file( f"stress_vs_size.png")
    fig.savefig(out_name)

    finalize(out_name)

def diag_dist_specimen_intensity_func(
    specimen: Specimen,
    kernel_width=100,
    n_points=100
) -> tuple[npt.ArrayLike, Specimen]:
    """used in diag_dist to calculate the intensity of a specimen"""
    # calculate intensities
    img = specimen.get_fracture_image()
    intensity = csintkern_objects_diagonal(
        img.shape[:2],
        specimen.splinters,
        lambda x,r: x.in_region_px(r),
        kernel_width=kernel_width,
        n_points=n_points
        )

    intensity = np.array(intensity)
    # intensity = intensity / np.max(intensity)

    return intensity, specimen

@app.command()
def diag_dist(
    names: Annotated[str, typer.Option(help='Name filter. Can use wildcards.', metavar='*')] = "*",
    sigmas: Annotated[str, typer.Option(help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120" or "all").', metavar='s, s1-s2, all')] = None,
    delta: Annotated[float, typer.Option(help='Additional range for sigmas.')] = 10,
    out: Annotated[str, typer.Option(help='Output file.')] = None,
    kernel_width: Annotated[int, typer.Option(help='Intensity kernel width.')] = 200,
    n_points: Annotated[int, typer.Option(help='Amount of points on the diagonal to evaluate.')] = 100,
    y_stress: Annotated[bool, typer.Option(help='Plot sigma instead of energy on y-axis.')] = False,
):


    filter = create_filter_function(names, sigmas, sigma_delta=delta)

    specimens: list[Specimen] = Specimen.get_all_by(filter, lazyload=False)

    fig, axs = plt.subplots(figsize=(9, 3))

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



        for intensity, specimen in pooled(specimens, diag_dist_specimen_intensity_func,
                                          advance = lambda: progress.advance(an_task),
                                          kernel_width=kernel_width,
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
    axs.set_xticks(x_ticks, [f'{x/n_points:.2f}' for i,x in enumerate(x_ticks)])
    axs.set_yticks(np.arange(0, len(stress), 1), [f'{np.abs(x):.2f}' if i % str_mod == 0 else "" for i,x in enumerate(stress)])

    axy = axs.secondary_yaxis('right')
    axy.set_yticks(axs.get_yticks(), [x  for i,x in enumerate(names)])

    dt = np.array(data)
    axim = axs.imshow(dt, cmap=modified_turbo, aspect='auto', interpolation='none')
    cbar = fig.colorbar(axim, ax=axs, orientation='vertical', label='Relative Intensity', pad=0.2)

    axs.set_xlim((-0.01, n_points+0.01))
    # fig2 = plot_histograms((0,2), specimens, plot_mean=True)
    # plt.show()

    # axy.set_yticks(np.linspace(axy.get_yticks()[0], axy.get_yticks()[-1], len(axs.get_yticks())))
    fig.tight_layout()

    out_name = general.get_output_file("diag_dist.png" if out is None else out)
    fig.savefig(out_name)
    finalize(out_name)



@app.command(name="log2dhist")
def log_2d_histograms(
    names: Annotated[str, typer.Option(help='Name filter. Can use wildcards.', metavar='*')] = "*",
    sigmas: Annotated[str, typer.Option(help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120" or "all").', metavar='s, s1-s2, all')] = None,
    exclude: Annotated[str, typer.Option(help='Exclude specimen names matching this.')] = None,
    delta: Annotated[float, typer.Option(help='Additional range for sigmas.')] = 10,
    y_stress: Annotated[bool, typer.Option(help='Show stress on y axis instead of energy.')] = False,
    maxspecimen: Annotated[int, typer.Option(help='Maximum amount of specimens.')] = 50,
    n_bins: Annotated[int, typer.Option(help='Number of bins for histogram.')] = 60):
    """Plot a 2D histogram of splinter sizes and stress."""

    filter = create_filter_function(names, sigmas, delta,
                                           exclude=exclude,
                                           needs_scalp=False,
                                           needs_splinters=True)

    specimens: list[Specimen] = Specimen.get_all_by(filter, max_n=maxspecimen, lazyload=False)

    assert len(specimens) > 0, "[red]No specimens loaded.[/red]"

    binrange = np.linspace(0,2,n_bins)
    fig, axs = plt.subplots(figsize=(9, 3))

    data = []
    stress = []

    with get_progress() as progress:
        an_task = progress.add_task("Loading splinters...", total=len(specimens))
        for specimen in specimens:

            areas = [np.log10(x.area) for x in specimen.splinters if x.area > 0]
            # ascending sort, smallest to largest
            areas.sort()

            hist, edges = bin_data(areas, binrange)

            hist = np.array(hist)/np.max(hist)

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
    axs.set_xlabel("Splinter Area [mm²]")
    if not y_stress:
        axs.set_ylabel("Strain Energy [J/m²]")
    else:
        axs.set_ylabel("Surface Stress [MPa]")


    str_mod = 5 if len(stress) > 15 else 1
    x_ticks = np.arange(0, n_bins, 5)
    siz_mod = 2 if len(x_ticks) > 10 else 1
    axs.set_xticks(x_ticks, [f'{10**edges[x]:.2f}' if i % siz_mod == 0 else "" for i,x in enumerate(x_ticks)])
    axs.set_yticks(np.arange(0, len(stress), 1), [f'{np.abs(x):.2f}' if i % str_mod == 0 else "" for i,x in enumerate(stress)])

    axy = axs.secondary_yaxis('right')
    axy.set_yticks(axs.get_yticks(), [x  for i,x in enumerate(names)])

    dt = np.array(data)
    axim = axs.imshow(dt, cmap=modified_turbo, aspect='auto', interpolation='none')
    cbar = fig.colorbar(axim, ax=axs, orientation='vertical', label='Relative Intensity', pad=0.2)
    # fig2 = plot_histograms((0,2), specimens, plot_mean=True)
    # plt.show()

    axy.set_yticks(np.linspace(axy.get_yticks()[0], axy.get_yticks()[-1], len(axs.get_yticks())))
    fig.tight_layout()

    if sigmas is not None:
        out_name = general.get_output_file(f"loghist2d_{sigmas[0]}_{sigmas[1]}.{general.plot_extension}")
    elif names is not None:
        out_name = general.get_output_file( f"loghist2d_{names[0]}.{general.plot_extension}")
    fig.savefig(out_name)
    finalize(out_name)

def create_filter_function(name_filter,
                   sigmas = None,
                   sigma_delta = 10,
                   exclude: str = None,
                   needs_scalp = True,
                   needs_splinters = True
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
        name_filter = name_filter.replace(".","\.").replace("*", ".*").replace('!', '|')
        name_filter_function = in_names_wildcard
    elif name_filter is None:
        name_filter = ".*"
        print("[green]All[/green] specimen names included!")
        name_filter_function = all_names

    if sigmas is not None:
        if "-" in sigmas:
            sigmas = [float(s) for s in sigmas.split("-")]
        elif sigmas == "all":
            sigmas = [0,1000]
        else:
            sigmas = [float(sigmas), float(sigmas)]
            sigmas[0] = max(0, sigmas[0] - sigma_delta)
            sigmas[1] += sigma_delta

        print(f"Searching for splinters with stress in range {sigmas[0]} - {sigmas[1]}")

    def filter_specimens(specimen: Specimen):
        if needs_scalp and not specimen.has_scalp:
            return False
        elif needs_splinters and not specimen.has_splinters:
            return False
        elif exclude is not None and re.match(exclude, specimen.name):
            return False
        elif not name_filter_function(specimen, name_filter):
            return False
        elif sigmas is not None:
            return sigmas[0] <= abs(specimen.scalp.sig_h) <= sigmas[1]

        return True

    return filter_specimens

@app.command()
def log_histograms(specimen_names: Annotated[list[str], typer.Argument(help='Names of specimens to load')],
                   xlim: Annotated[tuple[float,float], typer.Option(help='X-Limits for plot')] = (0, 2),
                   more_data: Annotated[bool, typer.Option(help='Write specimens sig_h and thickness into legend.')] = False,
                   nolegend: Annotated[bool, typer.Option(help='Dont display the legend on the plot.')] = False,
                   n_bins: Annotated[int, typer.Option(help='Number of bins for histogram.')] = 50):
    """Plot logaritmic histograms of splinter sizes for specimens."""
    path = general.base_path
    specimens = Specimen.get_all(specimen_names)

    if len(specimens)==0:
        print("[red]No specimens loaded.[/red]")
        return


    def legend_none(x: Specimen):
        return f'{x.name}'

    legend = legend_none

    if more_data:
        def legend_f(x: Specimen):
            return f'{x.name}_{x.scalp.measured_thickness:.2f}_{abs(x.scalp.sig_h):.2f}'
        legend = legend_f

    fig = plot_histograms(xlim, specimens, legend=legend, n=n_bins, has_legend=not nolegend)
    out_name = f"{specimens[0].name.replace('.','_')}_log_histograms"
    c = len([x for x in os.listdir(path) if x.startswith(out_name)])
    out_name = os.path.join(path, f"{out_name}_{c}.png")
    fig.savefig(out_name)

    disp_mean_sizes(specimens)

    finalize(out_name)


def disp_mean_sizes(specimens: list[Specimen]):
    """Displays mean splinter sizes.

    Args:
        specimens (list[Specimen]): Specimens to display.
    """
    print("* Mean splinter sizes:")
    for specimen in specimens:
        print(f"\t '{specimen.name}' ({specimen.scalp.sig_h:.2f}): {np.mean([x.area for x in specimen.splinters]):.2f}")


def plot_histograms(xlim: tuple[float,float],
                    specimens: list[Specimen],
                    legend = None,
                    plot_mean = False,
                    n: int = 50,
                    has_legend: bool = True) -> Figure:
    fig, axs = datahist_plot(xlim=xlim, has_legend=has_legend)

    if legend is None:
        def legend(x):
            return f'{x.name}'

    for specimen in specimens:
        areas = [x.area for x in specimen.splinters]
        datahist_to_ax(axs[0], areas, n_bins=n, plot_mean=plot_mean, label=legend(specimen))

    fig.tight_layout()
    return fig

def plot_stress_size(specimens: list[Specimen]):
    fig,ax = plt.subplots()

    sizes = [x.splinters.get_mean_splinter_size() for x in specimens]
    stresses = [x.scalp.sig_h for x in specimens]

    # sort by stress
    sorted_x, sorted_y = sort_two_arrays(stresses, sizes)
    ax.plot(sorted_x, sorted_y, 'b-')
    ax.set_xlabel("Stress [MPa]")
    ax.set_ylabel("Mean splinter size [mm²]")
    fig.tight_layout()

    # count amount of files whose name is stress_vs_size
    count = 0
    for file in os.listdir(general.base_path):
        if file.startswith("stress_vs_size"):
            count += 1

    out_name = general.get_output_file( f"stress_vs_size.png")
    fig.savefig(out_name)
    finalize(out_name)

@app.command(name='loghist_sigma')
def loghist_sigma(sigmas: Annotated[str, typer.Argument(help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120").')],
                  delta: Annotated[float, typer.Option(help='Additional range for sigmas.')] = 10,
                  maxspecimen: Annotated[int, typer.Option(help='Maximum amount of specimens.')] = 10,
                  nolegend: Annotated[bool, typer.Option(help='Dont display the legend on the plot.')] = False,
                  xlim: Annotated[tuple[float,float], typer.Option(help='X-Limits for plot')] = (0,2)):
    """
    Plots histograms of splinter sizes for specimens with stress in a given range.


    Args:
        sigmas (str): The stress range. Either a single value or a range separated by a dash (i.e. '100-110' or '120').
        delta (float, optional): Additional margin for the stress range. Defaults to 10.
        xlim (tuple[float,float], optional): Plot limits on x axis. Defaults to None.
    """
    if "-" in sigmas:
        sigmas = [float(s) for s in sigmas.split("-")]
    else:
        sigmas = [float(sigmas), float(sigmas)]
        sigmas[0] = max(0, sigmas[0] - delta)
        sigmas[1] += delta

    print(f"Searching for splinters with stress in range {sigmas[0]} - {sigmas[1]}")

    def in_sigma_range(specimen: Specimen):
        if not specimen.has_scalp:
            return False
        if not specimen.has_splinters:
            return False

        return sigmas[0] <= abs(specimen.scalp.sig_h) <= sigmas[1]

    specimens: list[Specimen] = Specimen.get_all_by(in_sigma_range, max_n=maxspecimen)


    out_path = general.get_output_file( f"{sigmas[0]}-{sigmas[1]}_log_histograms.png")
    fig = plot_histograms(xlim, specimens, lambda x: f"{x.name}_{abs(x.scalp.sig_h):.2f}", has_legend=not nolegend)
    fig.savefig(out_path)
    plt.close(fig)

    disp_mean_sizes(specimens)

    plot_stress_size(specimens)

    finalize(out_path)

# @app.command(name='accumulation_all')
# def plot_all_accumulations():
#     """Plot histograms for all specimens in the base path."""
#     for dir in (pbar := track(os.listdir(general.base_path))):
#         spec_path = general.get_output_file( dir)
#         if not os.path.isdir(spec_path):
#             continue

#         spec = Specimen(spec_path)
#         if spec.splinters is None or spec.scalp is None:
#             continue
#         pbar.set_description(f"Processing {spec.name}...")
#         spec.splinters.plot_splintersize_accumulation()


@app.command()
def splinter_orientation(specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')]):
    """Plot the orientation of splinters."""
    specimen = Specimen.get(specimen_name)
    out_name = general.get_output_file(f"{specimen_name}.png")

    impact_pos = specimen.get_impact_position()
    splinters = specimen.splinters
    size_fac = specimen.get_size_factor()

    # analyze splinter orientations
    orientation_image = np.zeros_like(specimen.get_fracture_image(), dtype=np.uint8)
    orients = []
    for s in track(splinters):
        orientation = s.measure_orientation(impact_pos)
        orients.append(orientation)
        color = get_color(orientation)
        cv2.drawContours(orientation_image, [s.contour], -1, color, -1)

    # draw splinter contour lines
    cv2.drawContours(orientation_image, [x.contour for x in splinters], -1, (0,0,0), 1)

    cv2.circle(orientation_image,
                (np.array(impact_pos) / size_fac).astype(np.uint32),
                np.min(orientation_image.shape[:2]) // 50,
                (255,0,0),
                -1)

    cv2.circle(orientation_image,
                (np.array(impact_pos) / size_fac).astype(np.uint32),
                np.min(orientation_image.shape[:2]) // 50,
                (255,255,255),
                5)

    orientation_image = annotate_image(
        orientation_image,
        cbar_title='Orientation Strength',
        min_value=0,
        max_value=1
    )

    general.save_image(out_name, orientation_image)
    finalize(out_name)

@app.command()
def fracture_intensity_img(
    specimen_name: str,
    kernel_width: Annotated[int, typer.Option(help='Kernel width.')] = 100,
    skip_edges: Annotated[bool, typer.Option(help='Skip 10% of the edges when calculating intensities.')] = False,
):
    """
    Plot the intensity of the fracture image.

    Basically the same as fracture-intensity, but performs operations on the image
    instead of the splinters.

    Intensity here is the mean value of the image part (defined by kernel_width).
    Higher Intensity => Darker image part (more cracks)
    Lower Intensity => Brighter image part (less crack)

    Args:
        specimen_name (str): Name of specimen to load.
        kernel_width (int, optional): Grid size. Defaults to 100.
    """
    specimen = Specimen.get(specimen_name)

    def mean_img_value(img_part):
        return 255-np.mean(img_part)

    img = specimen.get_fracture_image()
    img = preprocess_image(img, specimen.splinter_config)

    fig = plot_image_kernel_contours(img, kernel_width,
                                     mean_img_value,
                                     clr_label="Amnt Black",
                                     fig_title="Fracture Intensity (Based on image mean values)",
                                     skip_edge=skip_edges)

    out_path = os.path.join(specimen.splinters_path, f"fig_img_intensity.{general.image_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    finalize(out_path)

@app.command()
def fracture_intensity(
        specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
        kernel_width: Annotated[int, typer.Option(help='Kernel width.')] = 200,
        plot_vertices: Annotated[bool, typer.Option(help='Plot the kernel points.')] = False,
        skip_edges: Annotated[bool, typer.Option(help='Skip 10% of the edges when calculating intensities.')] = False,
        ):
    """Plot the intensity of the fracture morphology."""

    specimen = Specimen.get(specimen_name)

    original_image = specimen.get_fracture_image()

    # this counts the splinters in the kernel by default
    fig = plot_splinter_kernel_contours(original_image,
                                        specimen.splinters,
                                        kernel_width,
                                        plot_vertices=plot_vertices,
                                        skip_edge=skip_edges)

    out_name = specimen.get_splinter_outfile(f"fig_fracture_intensity.{general.plot_extension}")
    fig.savefig(out_name, dpi=500)
    finalize(out_name)

@app.command()
def create_voronoi(specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],):
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

def get_detection_rate(splinters: list[Splinter], real_size: tuple[float,float]) -> float:
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
    size: Annotated[tuple[int,int], typer.Option(help='Image size.', metavar='Y X')] = general.default_image_size_px,
    rotate_only: Annotated[bool, typer.Option('--rotate-only', help='Only rotate image by 90°, skip cropping.')] = False,
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
    from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE
    if all:
        specimens = Specimen.get_all()
    elif isinstance(specimen_name, Specimen):
        specimens = [specimen_name]
    else:
        specimens = Specimen.get_all(specimen_name)


    for specimen in track(specimens):
        path = specimen.fracture_morph_dir
        if not  os.path.exists(path):
            continue

        imgs = [(x,cv2.imread(x, cv2.IMREAD_GRAYSCALE)) for x in find_files(path, '*.bmp')]

        if len(imgs) == 0:
            continue

        img0 = [y for x,y in imgs if "Transmission" in x][0]
        _, M0 = crop_perspective(img0, size, False, True)

        for file,img in imgs:
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
            os.chmod(os.path.join(path, file), S_IREAD|S_IRGRP|S_IROTH)

        # elif file.endswith(".bmp") and not os.access(os.path.join(path, file), os.W_OK):
        #     os.chmod(os.path.join(path, file), S_IWRITE)


@app.command()
def watershed(
    name: Annotated[str, typer.Argument(help='Name of the specimen.', metavar='*.*.*.*')],
    debug: Annotated[bool, typer.Option(help='Show debug plots.')] = False,
):
    #TODO: Check the individual steps of the watershed algorithm
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

    size_factor = specimen.splinters_data['size_factor']
    splinters = Splinter.analyze_image(image, debug=debug, px_per_mm=size_factor)

    # ORIGINAL output
    sp_img = cv2.imread(specimen.get_splinter_outfile("img_filled.png"))
    size_img_file = find_file(specimen.get_splinter_outfile(""), "img_splintersizes")
    if not size_img_file:
        size_img_file = specimen.get_splinter_outfile("img_splintersizes.png")
        create_splinter_sizes_image(specimen.splinters, image.shape, size_img_file)
    sz_img = cv2.imread(size_img_file, cv2.IMREAD_COLOR)


    m_img = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(m_img, [x.contour for x in splinters], -1, (255,255,255), 1)
    cmp_image = cv2.addWeighted(image, 1.0, sp_img, 0.2, 0)
    cmp_image = cv2.addWeighted(cmp_image, 1, m_img, 1, 0)
    plotImages((("Original", image), ("Comparison", cmp_image), ("Splinter Sizes", sz_img)))
    cv2.imwrite(general.get_output_file("legacy_vs_watershed_contours", is_image=True), cmp_image)

    ## create splinter size image
    sz_image2 = create_splinter_sizes_image(
        splinters,
        image.shape,
        out_file=general.get_output_file(f"{specimen.name}_img_splintersizes_watershed.png"),
        annotate = True,
        annotate_title="Watershed",
        with_contours=True)

    rnd_splinters = create_colored_splinter_image(
        splinters,
        image.shape,
        specimen.get_splinter_outfile("img_filled_watershed.png"),
    )

    plotImages([("Splinter Sizes", sz_img), ("Splinter Sizes Watershed", sz_image2)])


    # overlay contours over rnd splinters
    rnd_splinters = cv2.addWeighted(rnd_splinters, 1.0, to_rgb(m_img), 1.0, 0)
    plotImage(rnd_splinters, "Splinters Watershed")

    # plot splinter histograms
    fig, axs = datahist_plot(1,2)
    fig.set_size_inches(12,6)
    fig.suptitle("Splinter Sizes")
    axs[0].set_title("Original")
    axs[1].set_title("Watershed")
    axs[0].set_xlabel("Splinter Size [mm²]")
    axs[1].set_xlabel("Splinter Size [mm²]")
    axs[0].set_ylabel("PDF [-]")
    axs[1].set_ylabel("PDF [-]")
    datahist_to_ax(axs[0], [x.area for x in specimen.splinters], 20, as_log=True)
    datahist_to_ax(axs[1], [x.area for x in splinters], 20, as_log=True)

    fig.tight_layout()
    plt.show()

    fig.savefig(general.get_output_file("legacy_vs_watershed_histograms", is_plot=True))
