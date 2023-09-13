import os
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

import typer
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rich import print
from typing_extensions import Annotated

from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.specimen import Specimen, fetch_specimens

from fracsuite.tools.plot import app as plt_app
from fracsuite.tools.config import app as config_app

plt.rcParams['figure.figsize'] = (6, 4)
plt.rc('axes', axisbelow=True) # to get grid into background
plt.rc('grid', linestyle="--") # line style
plt.rcParams.update({'font.size': 12}) # font size

general = GeneralSettings()

app = typer.Typer(pretty_exceptions_short=False)
app.add_typer(plt_app, name="plot")
app.add_typer(config_app, name="config")

def specimen_parser(input: str):
    return input

def sort_two_arrays(array1, array2) -> tuple[list, list]:
    # Combine x and y into pairs
    pairs = list(zip(array1, array2))
    # Sort the pairs based on the values in x
    sorted_pairs = sorted(pairs, key=lambda pair: pair[0])
    # Separate the sorted pairs back into separate arrays
    return zip(*sorted_pairs)    

@app.command(name="loghist")
def log_histograms(specimen_names: Annotated[List[str], typer.Argument(help='Names of specimens to load', parser=specimen_parser)], 
                   xlim: Annotated[Tuple[float,float], typer.Option(help='X-Limits for plot')] = (0, 2),
                   more_data: Annotated[bool, typer.Option(help='Write specimens sig_h and thickness into legend.')] = False,
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
        
    fig = plot_histograms(xlim, specimens, legend=legend, n=n_bins)
    out_name = f"{specimens[0].name.replace('.','_')}_log_histograms"
    c = len([x for x in os.listdir(path) if x.startswith(out_name)])
    out_name = os.path.join(path, f"{out_name}_{c}.png")
    fig.savefig(out_name)
    print(f"Saved to '{out_name}'.")
    os.system(f"start {out_name}")
    disp_mean_sizes(specimens)
    
def disp_mean_sizes(specimens: list[Specimen]):
    """Displays mean splinter sizes.

    Args:
        specimens (list[Specimen]): Specimens to display.
    """
    print("* Mean splinter sizes:")
    for specimen in specimens:
        print(f"\t '{specimen.name}' ({specimen.scalp.sig_h:.2f}): {specimen.splinters.get_mean_splinter_size():.2f}")
    
    
def plot_histograms(xlim: tuple[float,float], 
                    specimens: list[Specimen], 
                    legend = None, 
                    plot_mean = False,
                    n: int = 50) -> Figure:
    cfg = AnalyzerConfig()
    
    cfg.probabilitybins = n
    
    fig, ax = plt.subplots()
    
    for specimen in specimens:
        specimen.splinters.plot_logarithmic_to_axes(ax, cfg, label=legend(specimen))
        if plot_mean:
            mean = specimen.splinters.get_mean_splinter_size()
            ax.axvline(np.log10(mean), color='r', linestyle='--', label=f"Ø={mean:.2f}mm²")
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
            
    
    ax.legend(loc='best')
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
    
    fig.savefig(os.path.join(general.base_path, f"stress_vs_size.png"))
    
@app.command(name='loghist_sigma')
def loghist_sigma(path: Annotated[str, typer.Argument(help='Base path for specimens')], 
                  sigmas: Annotated[str, typer.Argument(help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120").')], 
                  delta: Annotated[float, typer.Option(help='Additional range for sigmas.')] = 10, 
                  xlim: Annotated[tuple[float,float], typer.Option(help='X-Limits for plot')] = (0,2)):
    """
    Plots histograms of splinter sizes for specimens with stress in a given range.


    Args:
        path (str): The base path for the specimens.
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
    
    specimens: list[Specimen] = []
    
    for dir in os.listdir(path):
        spec_path = os.path.join(path, dir)
        
        specimen = Specimen(spec_path)
        
        if specimen.splinters is None or specimen.scalp is None:
            continue
                        
        stress = specimen.scalp.sig_h        
        if stress >= sigmas[0] and stress <= sigmas[1]:
            specimens.append(specimen.splinters)
            print(f"Loaded '{dir}' ({stress:.2f}).")
                
    fig = plot_histograms(xlim, specimens, lambda x: f"{x.name}_{abs(x.scalp.sig_h):.2f}")
    fig.savefig(os.path.join(path, f"{sigmas[0]}-{sigmas[1]}_log_histograms.png"))
    plt.close(fig)
    
    disp_mean_sizes(specimens)
    
@app.command(name='loghist_all')
def plot_all_histos():
    """Plot histograms for all specimens in the base path."""
    for dir in (pbar := tqdm(os.listdir(general.base_path))):
        spec_path = os.path.join(general.base_path, dir)
        if not os.path.isdir(spec_path):
            continue
        
        spec = Specimen(spec_path)
        if spec.splinters is None or spec.scalp is None:
            continue
        pbar.set_description(f"Processing {spec.name}...")
        fig = plot_histograms(None, [spec], lambda x: f"{x.name}_{abs(x.scalp.sig_h):.2f}", plot_mean=True)
        fig.savefig(os.path.join(spec.splinters_path, f"{spec.name}_log_histogram.png"))
        plt.close(fig)
        del fig
@app.command(name='accumulation_all')
def plot_all_accumulations():
    """Plot histograms for all specimens in the base path."""
    for dir in (pbar := tqdm(os.listdir(general.base_path))):
        spec_path = os.path.join(general.base_path, dir)
        if not os.path.isdir(spec_path):
            continue
        
        spec = Specimen(spec_path)
        if spec.splinters is None or spec.scalp is None:
            continue
        pbar.set_description(f"Processing {spec.name}...")
        spec.splinters.plot_splintersize_accumulation()
   
@app.command()
def plot_roughness():
    pass
    
@app.command()
def marina_organize(path: str):
    # find all folders in path that contain three dots
    for dir in os.listdir(path):
        if dir.count(".") != 3:
            continue
        
        dirpath = os.path.join(path, dir)
        
        # create subdirectors
        os.makedirs(os.path.join(dirpath, "scalp"), exist_ok=True)
        os.makedirs(os.path.join(dirpath, "fracture"), exist_ok=True)
        acc_path = os.path.join(dirpath, "fracture", "acceleration")
        os.makedirs(acc_path, exist_ok=True)
        morph_path = os.path.join(dirpath, "fracture", "morphology")
        os.makedirs(morph_path, exist_ok=True)
        
        # put all .bmp files into morphology folder
        for file in os.listdir(dirpath):
            if file.endswith(".bmp") or file.endswith(".zip"):
                os.rename(os.path.join(dirpath, file), os.path.join(morph_path, file))
                
        # put all .tsx, .tst, .xlsx and .bin files into acceleration folder
        for file in os.listdir(dirpath):
            if file.lower().endswith(".tsx") or file.lower().endswith(".tst") or file.lower().endswith(".xlsx") or file.lower().endswith(".bin"):
                os.rename(os.path.join(dirpath, file), os.path.join(acc_path, file))
                
        # in morphology folder, search for filename that starts with a 4digit number and prepend it to the file 
        # that contains "Transmission" in its name, if it does not start with the same 4digit num
        for file in os.listdir(morph_path):
            if "Transmission" in file:
                continue
            
            # find 4digit number in filename
            num = None
            if file[0].isdigit() and file[0+1].isdigit() and file[0+2].isdigit() and file[0+3].isdigit():
                num = file[:4]
                        
            if num is None:
                continue
            
            # find file with "Transmission" in its name
            for file2 in os.listdir(morph_path):
                if "Transmission" in file2 and num not in file2:
                    os.rename(os.path.join(morph_path, file2), os.path.join(morph_path, num + " " + file2))
                    break
     
    
app()


