import os
from itertools import groupby
from typing import Annotated, List

import altair as alt
import cv2
from matplotlib.ticker import FuncFormatter
import numpy as np
import typer
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rich import print
from rich.progress import Progress, SpinnerColumn, track
from scipy.optimize import curve_fit

from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from fracsuite.splinters.splinter import Splinter
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import annotate_image_cbar, get_color, write_image
from fracsuite.tools.plotting import plot_impact_influence
from fracsuite.tools.specimen import Specimen, fetch_specimens, fetch_specimens_by

app = typer.Typer()

general = GeneralSettings.create()

def finalize(out_name: str):
    print(f"Saved to '{out_name}'.")
    os.system(f"start {out_name}")


@app.command()
def roughness_f(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
                regionsize: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 200,):
    """Create a contour plot of the roughness on the specimen.

    Args:
        specimen_name (str, optional): Name of the specimens to load.
    """
    def roughness_function(splinters: list[Splinter]):
        return np.mean([splinter.calculate_roughness() for splinter in splinters])
            
    # create contour plot of roughness
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    fig = specimen.splinters.plot_intensity(regionsize, roughness_function, clr_label='Mean roughness')
    
    # with Progress(SpinnerColumn("arc", ), transient=False, ) as progress:
    #     task = progress.add_task("Create intensity plots", total=1, )
    #     # Start your operation here
    #     # Mark the task as complete
    #     progress.update(task, completed=1)
    
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"fig_roughintensity.{general.plot_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    finalize(out_path)
    
@app.command()
def roundness_f(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
                regionsize: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 200,):
    """Create a contour plot of the roundness on the specimen.

    Args:
        specimen_name (str, optional): Name of the specimens to load.
    """
    def roundness_function(splinters: list[Splinter]):
        return np.mean([splinter.calculate_roundness() for splinter in splinters])
            
    # create contour plot of roughness
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    fig = specimen.splinters.plot_intensity(regionsize, roundness_function, clr_label='', fig_title='Mean roundness ')
    
    # with Progress(SpinnerColumn("arc", ), transient=False, ) as progress:
    #     task = progress.add_task("Create intensity plots", total=1, )
    #     # Start your operation here
    #     # Mark the task as complete
    #     progress.update(task, completed=1)
    
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"fig_roundintensity.{general.plot_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    finalize(out_path)

    
@app.command()
def intensity(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
              regionsize: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 200,):
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    with Progress(SpinnerColumn("arc", ), transient=False, ) as progress:
        task = progress.add_task("Create intensity plots", total=1, )
        # Start your operation here
        fig = specimen.splinters.create_intensity_plot(regionsize)
        # Mark the task as complete
        progress.update(task, completed=1)
        
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"fig_fracintensity.{general.plot_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    finalize(out_path)


    

@app.command()
def roughness(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')]):    
    """Plot the roughness of a specimen."""
       
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    
    out_img = specimen.splinters.original_image.copy()
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    
    rough = [splinter.calculate_roughness() for splinter in specimen.splinters.splinters]    
    max_r = np.max(rough)
    min_r = np.min(rough)
    
    print(f"Max roughness: {max_r}")
    print(f"Min roughness: {min_r}")
    print(f"Mean roughness: {np.mean(rough)}")
    print(f"Median roughness: {np.median(rough)}")
    
    # d = np.median(rough) - min_r
    # max_r = np.median(rough) + d
    
    for splinter in track(specimen.splinters.splinters, description="Calculating roughness", transient=True):
        roughness = splinter.calculate_roughness()
        clr = get_color(roughness, min_r, max_r)
        
        cv2.drawContours(out_img, [splinter.contour], 0, clr, -1)
        
    # plot an overlay of the colorbar into the image on the right side
    # colorbar = np.zeros((out_img.shape[0], 50, 3), dtype=np.uint8)
    # for i in range(out_img.shape[0]):
    #     clr = get_color(i, 0, out_img.shape[0])
    #     colorbar[i] = clr
    # out_img = np.concatenate((out_img, colorbar), axis=1)
    out_img = annotate_image_cbar(out_img, "Roughness", min_value=min_r, max_value=max_r)    
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"roughness.{general.plot_extension}")
    
    write_image(out_img, out_path)
    
    finalize(out_path)

    
@app.command()
def roundness(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')]):    
    """Plot the roundness of a specimen."""
       
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    
    out_img = specimen.splinters.original_image.copy()
    
    rounds = [splinter.calculate_roundness() for splinter in specimen.splinters.splinters]    
    max_r = np.max(rounds)
    min_r = np.min(rounds)
    
    print(f"Max roundness: {max_r}")
    print(f"Min roundness: {min_r}")
    print(f"Mean roundness: {np.mean(rounds)}")
    
    # scale max and min roundness to +- 60% around mean
    max_r = np.mean(rounds) + np.mean(rounds) * 0.6
    min_r = np.mean(rounds) - np.mean(rounds) * 0.6
    
    
    for splinter in track(specimen.splinters.splinters):
        r = splinter.calculate_roundness()
        clr = get_color(r, min_r, max_r)
        
        cv2.drawContours(out_img, [splinter.contour], 0, clr, -1)
        
    out_img = annotate_image_cbar(out_img, "Roundness", min_value=min_r, max_value=max_r)
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"roundness.{general.plot_extension}")
    cv2.imwrite(out_path, out_img)
    
    finalize(out_path)

    
def str_to_intlist(input: str) -> list[int]:
    if isinstance(input, int):
        return [input    ]
    
    return [int(x) for x in input.split(",")]
def specimen_parser(input: str):
    return input

def sort_two_arrays(array1, array2) -> tuple[list, list]:
    # Combine x and y into pairs
    pairs = list(zip(array1, array2))
    # Sort the pairs based on the values in x
    sorted_pairs = sorted(pairs, key=lambda pair: pair[0])
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
        return (spec.boundary, spec.splinters.get_mean_splinter_size(), np.abs(spec.scalp.sig_h))
    
    specimens = fetch_specimens_by(decider, general.base_path, value=specimen_value, sortby=lambda x: x[0])
    
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
            
        params = curve_fit(func, stresses, sizes, bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
        
        s_inter = np.linspace(min_sig, max_sig, 50)
        ax.plot(s_inter, func(s_inter, *params[0]), '--', color=ps.get_facecolor()[0], linewidth=0.5)
        
        # p = np.polyfit(stresses, sizes, 4)
        # s_inter = np.linspace(stresses[0], stresses[-1], 100)
        # plt.plot(s_inter, np.polyval(p, s_inter), '--', color=ps.get_facecolor()[0])
    ax.set_xlabel("Stress [MPa]")
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
def log_2d_histograms(specimen_names: Annotated[list[str], typer.Argument(help='Names of specimens to load', parser=specimen_parser)], 
                   xlim: Annotated[tuple[float,float], typer.Option(help='X-Limits for plot')] = (0, 2),
                   more_data: Annotated[bool, typer.Option(help='Write specimens sig_h and thickness into legend.')] = False,
                   nolegend: Annotated[bool, typer.Option(help='Dont display the legend on the plot.')] = False,
                   n_bins: Annotated[int, typer.Option(help='Number of bins for histogram.')] = 20):
    
    path = general.base_path
    specimens = fetch_specimens(specimen_names, path)
    
    
    if len(specimens) == 0 or any([x.splinters is None for x in specimens]):
        print("[red]No specimens loaded.[/red]")
        return
    
    def legend_none(x: Specimen):
        return f'{x.name}'
    
    legend = legend_none
    
    if more_data:        
        def legend_f(x: Specimen):
            return f'{x.name}_{x.scalp.measured_thickness:.2f}_{abs(x.scalp.sig_h):.2f}'
        legend = legend_f
    
    alt.Chart(specimens).mark_rect().encode(
        alt.X('IMDB_Rating:Q').bin(maxbins=60),
        alt.Y('Rotten_Tomatoes_Rating:Q').bin(maxbins=40),
        alt.Color('count():Q').scale(scheme='greenblue')
    )
        
    fig = plot_histograms(xlim, specimens, legend=legend, n=n_bins, has_legend=not nolegend)
    out_name = f"{specimens[0].name.replace('.','_')}_log_histograms"
    c = len([x for x in os.listdir(path) if x.startswith(out_name)])
    out_name = os.path.join(path, f"{out_name}_{c}.png")
    fig.savefig(out_name)
    
    disp_mean_sizes(specimens)
    
    finalize(out_name)

@app.command(name="loghist")
def log_histograms(specimen_names: Annotated[list[str], typer.Argument(help='Names of specimens to load', parser=specimen_parser)], 
                   xlim: Annotated[tuple[float,float], typer.Option(help='X-Limits for plot')] = (0, 2),
                   more_data: Annotated[bool, typer.Option(help='Write specimens sig_h and thickness into legend.')] = False,
                   nolegend: Annotated[bool, typer.Option(help='Dont display the legend on the plot.')] = False,
                   n_bins: Annotated[int, typer.Option(help='Number of bins for histogram.')] = 20):
    
    path = general.base_path
    specimens = fetch_specimens(specimen_names, path)
    
    
    if specimens is None or (isinstance(specimens, list) and len(specimens)==0):
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
    
    for specimen in specimens:
        # fetch areas from splinters
        areas = [np.log10(x.area) for x in specimen.splinters if x.area > 0]
        # ascending sort, smallest to largest
        areas.sort()
            
        # density: normalize the bins data count to the total amount of data
        ax.hist(areas, bins=int(n),
                density=True, label=legend(specimen),
                alpha=0.5)

        if plot_mean:
            mean = np.mean([x.area for x in specimen.splinters])
            ax.axvline(np.log10(mean), color='r', linestyle='--', label=f"Ø={mean:.2f}mm²")
    
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
    
    specimens: list[Specimen] = fetch_specimens_by(in_sigma_range, general.base_path, max_n=maxspecimen)
     
     
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
def splinter_orientation(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')]):
    """Plot the orientation of splinters."""
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    cfg = specimen.splinter_config
    cfg.impact_position = (50,50)
    out_name = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"splinter_orientation.{general.plot_extension}")
    plot_impact_influence((4000,4000), specimen.splinters, out_name, cfg)
    finalize(out_name)