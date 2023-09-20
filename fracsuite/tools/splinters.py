import os
from itertools import groupby
import re
from typing import Annotated, List

import altair as alt
import cv2
from matplotlib.ticker import FuncFormatter, LinearLocator
import numpy as np
import typer
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rich import print
from rich.progress import Progress, SpinnerColumn, track
from scipy.optimize import curve_fit

from fracsuite.core.plotting import plot_image_kernel_contours, plot_splinter_kernel_contours
from fracsuite.core.image import to_gray, to_rgb
from fracsuite.core.progress import get_progress
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from fracsuite.splinters.processing import preprocess_image
from fracsuite.splinters.splinter import Splinter
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import annotate_image, bin_data, get_color, write_image
from fracsuite.tools.specimen import Specimen

app = typer.Typer()

general = GeneralSettings.get()

def finalize(out_name: str):
    print(f"Saved to '{out_name}'.")
    os.system(f"start {out_name}")

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

    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"fig_roundintensity.{general.plot_extension}")
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
    out_img = annotate_image(out_img, "Roughness", min_value=min_r, max_value=max_r)
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"roughness.{general.image_extension}")

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
                                  "Roundness",
                                  cv2.COLORMAP_TURBO,
                                  min_value=min_r,
                                  max_value=max_r,
                                  unit="[-]",
                                  background='white')
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"roundness.{general.image_extension}")
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
        return (spec.boundary, np.mean([x.area for x in spec.splinters]), np.abs(spec.get_energy()))

    specimens = Specimen.get_all_by(decider, specimen_value, sortby=lambda x: x[0])

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

        # params = curve_fit(func, stresses, sizes, bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))

        # s_inter = np.linspace(min_sig, max_sig, 50)
        # ax.plot(s_inter, func(s_inter, *params[0]), '--', color=ps.get_facecolor()[0], linewidth=0.5)

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
    out_name = os.path.join(general.base_path, f"stress_vs_size.png")
    fig.savefig(out_name)

    finalize(out_name)


@app.command(name="log2dhist")
def log_2d_histograms(
    names: Annotated[str, typer.Option(help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120" or "all").', metavar='s, s1-s2, all')] = None,
    sigmas: Annotated[str, typer.Option(help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120" or "all").', metavar='s, s1-s2, all')] = None,
    boundary: Annotated[str, typer.Option(help='Allowed boundaries.')] = ["ABZ"],
    exclude: Annotated[str, typer.Option(help='Exclude specimen names matching this.')] = None,
    delta: Annotated[float, typer.Option(help='Additional range for sigmas.')] = 10,
    maxspecimen: Annotated[int, typer.Option(help='Maximum amount of specimens.')] = 50,
    n_bins: Annotated[int, typer.Option(help='Number of bins for histogram.')] = 60):
    """Plot a 2D histogram of splinter sizes and stress."""
    if names is None:
        assert sigmas is not None, "Either names or sigmas must be specified."
    elif "," in names:
        names = names.split(",")
    elif " " in names:
        names = names.split(" ")
    else:
        names = [names]

    if sigmas is None:
        assert names is not None, "Either names or sigmas must be specified."

    if sigmas is not None:
        if "-" in sigmas:
            sigmas = [float(s) for s in sigmas.split("-")]
        elif sigmas == "all":
            sigmas = [0,1000]
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
            if specimen.boundary not in boundary:
                return False
            if exclude is not None and re.match(exclude.replace(".","\.").replace("*",".*"), specimen.name):
                return False

            return sigmas[0] <= abs(specimen.scalp.sig_h) <= sigmas[1]

        specimens: list[Specimen] = Specimen.get_all_by(in_sigma_range, max_n=maxspecimen)
    elif names is not None:
        specimens: list[Specimen] = Specimen.get_all(names)


    if len(specimens) == 0 :
        print("[red]No specimens loaded.[/red]")
        return
    elif any([x.splinters is None for x in specimens]):
        print("[red]Some specimens have no splinters.[/red]")
        specimens = [x for x in specimens if x.splinters is not None]

    binrange = np.linspace(0,2,n_bins)
    fig, axs = plt.subplots(figsize=(8, 5))

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
            stress.append(specimen.get_energy())
            progress.update(an_task, advance=1)

    # sort data and names for ascending stress
    stress, data = sort_two_arrays(stress, data, True)
    names = [x[1] for x in data]
    data = [x[0] for x in data]
    axs.set_xlabel("Splinter Area [mm²]")
    axs.set_ylabel("Strain Energy [J/m²]")
    axs.set_xticks(np.arange(0, n_bins, 5), [f'{10**edges[x]:.2f}' for x in np.arange(1, n_bins + 1, 5)])
    axs.set_yticks(np.arange(0, len(stress), 1), [f'{np.abs(x):.2f}' if i % 5 == 0 else "" for i,x in enumerate(stress)])

    axy = axs.secondary_yaxis('right')
    axy.set_yticks(axs.get_yticks(), [x  for i,x in enumerate(names)])

    dt = np.array(data)
    axs.imshow(dt, cmap='Blues', aspect='auto', interpolation='none')

    fig2 = plot_histograms((0,2), specimens, plot_mean=True)
    plt.show()

    axy.set_yticks(np.linspace(axy.get_yticks()[0], axy.get_yticks()[-1], len(axs.get_yticks())))
    fig.tight_layout()


    disp_mean_sizes(specimens)

    if sigmas is not None:
        out_name = os.path.join(general.base_path, f"loghist2d_{sigmas[0]}_{sigmas[1]}.{general.plot_extension}")
    elif names is not None:
        out_name = os.path.join(general.base_path, f"loghist2d_{names[0]}.{general.plot_extension}")
    fig.savefig(out_name)
    finalize(out_name)

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
    fig, ax = plt.subplots()

    if legend is None:
        def legend(x):
            return f'{x.name}'

    for specimen in specimens:
        # fetch areas from splinters
        areas = [np.log10(x.area) for x in specimen.splinters if x.area > 0]
        # ascending sort, smallest to largest
        areas.sort()

        # density: normalize the bins data count to the total amount of data
        _,_,container = ax.hist(areas, bins=int(n),
                density=True, label=legend(specimen),
                alpha=0.5)

        if plot_mean:
            mean = np.mean([x.area for x in specimen.splinters])
            ax.axvline(np.log10(mean), linestyle='--', label=f"Ø={mean:.2f}mm²")

    if xlim is not None:
        ax.set_xlim(xlim)


    if has_legend:
        ax.legend(loc='best')

    # ax.set_xscale('log')
    ticks = FuncFormatter(lambda x, pos: '{0:.00f}'.format(10**x))
    ticksy = FuncFormatter(lambda x, pos: '{0:.2f}'.format(x))
    ax.xaxis.set_major_formatter(ticks)
    ax.yaxis.set_major_formatter(ticksy)

    # ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Splinter Area [mm²]')
    ax.set_ylabel('Probability (Area) [-]')
    ax.grid(True, which='both', axis='both')
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

    out_name = os.path.join(general.base_path, f"stress_vs_size.png")
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


    out_path = os.path.join(general.base_path, f"{sigmas[0]}-{sigmas[1]}_log_histograms.png")
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
#         spec_path = os.path.join(general.base_path, dir)
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

    cfg = specimen.splinter_config
    cfg.impact_position = (50,50)
    out_name = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"splinter_orientation.{general.plot_extension}")

    def plot_impact_influence(size, splinters: list[Splinter], out_file, impact_pos, size_fac,  updater = None):
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
            orientation = s.measure_orientation(impact_pos)
            orients.append(orientation)
            color = get_color(orientation, colormap_name='turbo')
            cv2.drawContours(orientation_image, [s.contour], -1, color, -1)
            # p2 = (s.centroid_px + s.angle_vector * 15).astype(np.int32)
            # cv2.line(orientation_image, s.centroid_px, p2, (255,255,255), 3)
        cv2.circle(orientation_image, (np.array(impact_pos) / size_fac).astype(np.uint32),
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

        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)

    plot_impact_influence(general.default_image_size_px,
                          specimen.splinters,
                          out_name,
                          specimen.get_impact_position(),
                          cfg.size_factor)
    finalize(out_name)

@app.command()
def fracture_intensity_img(specimen_name: str,
                           kernel_width: Annotated[int, typer.Option(help='Kernel width.')] = 100):
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
                                     fig_title="Fracture Intensity (Based on image mean values)")

    out_path = os.path.join(specimen.splinters_path, f"fig_img_intensity.{general.image_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    finalize(out_path)

@app.command()
def fracture_intensity(
        specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
        kernel_width: Annotated[int, typer.Option(help='Kernel width.')] = 200,
        plot_vertices: Annotated[bool, typer.Option(help='Plot the kernel points.')] = False):
    """Plot the intensity of the fracture image."""

    specimen = Specimen.get(specimen_name)

    original_image = specimen.get_fracture_image()

    # this counts the splinters in the kernel by default
    fig = plot_splinter_kernel_contours(original_image,
                                        specimen.splinters,
                                        kernel_width,
                                        plot_vertices=plot_vertices,)

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