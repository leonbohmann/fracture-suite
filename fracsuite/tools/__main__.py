import argparse
import os
import pickle

import typer
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rich import print
from typing_extensions import Annotated

from fracsuite.splinters.analyzer import Analyzer
from fracsuite.splinters.analyzerConfig import AnalyzerConfig

plt.rcParams['figure.figsize'] = (6, 4)
plt.rc('axes', axisbelow=True) # to get grid into background
plt.rc('grid', linestyle="--") # line style
plt.rcParams.update({'font.size': 12}) # font size


app = typer.Typer()

def sort_two_arrays(array1, array2) -> tuple[list, list]:
    # Combine x and y into pairs
    pairs = list(zip(array1, array2))
    # Sort the pairs based on the values in x
    sorted_pairs = sorted(pairs, key=lambda pair: pair[0])
    # Separate the sorted pairs back into separate arrays
    return zip(*sorted_pairs)    

@app.command(name="loghist")
def log_histograms(path: Annotated[str, typer.Argument(help='Base path for specimens')], 
                   specimen_names: Annotated[list[str], typer.Argument(help='Names of specimens to load')], 
                   xlim: Annotated[tuple[float,float], typer.Option(help='X-Limits for plot')] = None):
    analyzers: list[Analyzer] = []
    for name in specimen_names:
        spec_path = os.path.join(path, name)
        basepath = os.path.join(spec_path, "fracture", "splinter/splinters.pkl")
        
        if not os.path.exists(basepath):
            print(f"Could not find splinter file for '{name}'. Create it using:\n [green]py -m fracsuite.splinters '{spec_path}' -cropsize 4000 -realsize 500[/green]")
            continue
        
        with open(basepath, "rb") as f:
            analyzer = pickle.load(f)
            analyzers.append(analyzer)
            
            print(f"Loaded '{name}'.")
    
    fig = plot_histograms(xlim, analyzers)
    fig.savefig(os.path.join(path, f"{analyzers[0].config.specimen_name.replace('.','_')}_log_histograms.png"))

def plot_histograms(xlim: tuple[float,float], analyzers:  list[Analyzer], legend = None) -> Figure:
    cfg = AnalyzerConfig()
    
    fig, ax = plt.subplots()
    
    for analyzer in analyzers:        
        analyzer.plot_logarithmic_to_axes(ax, cfg, label=legend)
    
    if xlim is not None:
        ax.set_xlim(xlim)
        
    ax.legend(loc='best')
    ax.grid(True, which='both', axis='both')
    fig.tight_layout()
    return fig
    
    
    
@app.command(name='loghist_sigma')
def loghist_sigma(path: Annotated[str, typer.Argument(help='Base path for specimens')], 
                  sigmas: Annotated[str, typer.Argument(help='Stress range. Either a single value or a range separated by a dash (i.e. "100-110" or "120").')], 
                  delta: Annotated[float, typer.Option(help='Additional range for sigmas.')] = 10, 
                  xlim: Annotated[tuple[float,float], typer.Option(help='X-Limits for plot')] = None):
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
    
    analyzers: list[Analyzer] = []
    stresses: dict[str, float] = {}
    
    for dir in os.listdir(path):
        spec_path = os.path.join(path, dir)
        # if dir is a directory
        if os.path.isdir(spec_path):
            basepath = os.path.join(spec_path, "scalp")
            
            if not os.path.exists(basepath):
                print(f"Could not find scalp folder for {dir}.")
                continue
            
            # find "*_stress.txt" file
            for file in os.listdir(basepath):
                if file.endswith("_stress.txt"):
                    # extract stress from filename
                    with open(os.path.join(basepath, file), "r") as f:
                        lines = f.readlines()
                        stress = abs(float(lines[1].split(" ")[0]))
                        
                        stresses[dir] = stress
                        
                        if stress >= sigmas[0] and stress <= sigmas[1]:
                            # load splinters
                            spl_file = os.path.join(path, dir, "fracture", "splinter/splinters.pkl")
                            
                            if not os.path.exists(spl_file):
                                print(f"Could not find splinter file for '{dir}'. Create it using:\n [green]py -m fracsuite.splinters '{spec_path}' -cropsize 4000 -realsize 500[/green]")
                                continue
                            
                            with open(spl_file, "rb") as f:
                                analyzer = pickle.load(f)
                                analyzers.append(analyzer)
                                print(f"Loaded '{dir}'.")
                            break
                
    fig = plot_histograms(xlim, analyzers, lambda x: f"{x.config.specimen_name}_{stresses[x.config.specimen_name]:.2f}")
    fig.savefig(os.path.join(path, f"{sigmas[0]}-{sigmas[1]}_log_histograms.png"))
    plt.close(fig)
    
    print("* Mean splinter sizes:")
    for analyzer in analyzers:
        print(f"\t '{analyzer.config.specimen_name}' ({stresses[analyzer.config.specimen_name]:.2f}): {analyzer.get_mean_splinter_size():.2f}")
    
    fig,ax = plt.subplots()
    
    sizes = [analyzer.get_mean_splinter_size() for analyzer in analyzers]
    stresses = [stresses[analyzer.config.specimen_name] for analyzer in analyzers]
    
    # sort by stress
    sorted_x, sorted_y = sort_two_arrays(stresses, sizes)
    ax.plot(sorted_x, sorted_y, 'b-')
    ax.set_xlabel("Stress [MPa]")
    ax.set_ylabel("Mean splinter size [mm²]")
    fig.tight_layout()
    fig.savefig(os.path.join(path, f"{sigmas[0]}-{sigmas[1]}_stress_vs_size.png"))

    
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

