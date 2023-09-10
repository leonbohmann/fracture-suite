import pickle
import os
import argparse
import typer
from rich import print
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

plt.rcParams['figure.figsize'] = (6, 4)
plt.rc('axes', axisbelow=True) # to get grid into background
plt.rc('grid', linestyle="--") # line style
plt.rcParams.update({'font.size': 12}) # font size


app = typer.Typer()


@app.command(name="loghist")
def log_histograms(path:str, specimen_names: list[str], xlim: tuple[float,float] = None):
    analyzers: list[Analyzer] = []
    for name in specimen_names:
        basepath = os.path.join(path, name, "fracture", "splinter/splinters.pkl")
        
        if not os.path.exists(basepath):
            print(f"Could not find splinter file for {name}. Create it using:\n [green]py -m fracsuite.splinters '{basepath}' -cropsize 4000 -realsize 500[/green]")
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
    fig.tight_layout()
    return fig
    
@app.command(name='loghist_sigma')
def loghist_sigma(path: str, sigmas: str, delta: float = 10, xlim: tuple[float,float] = None):
    """Find all splinters from the path, whose stress is within a range sigmas.
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
        # if dir is a directory
        if os.path.isdir(os.path.join(path, dir)):
            basepath = os.path.join(path, dir, "scalp")
            
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
                                print(f"Could not find splinter file for {dir}. Create it using:\n [green]py -m fracsuite.splinters '{spl_file}' -cropsize 4000 -realsize 500[/green]")
                                continue
                            
                            with open(spl_file, "rb") as f:
                                analyzer = pickle.load(f)
                                analyzers.append(analyzer)
                                print(f"Loaded '{dir}'.")
                            break
                
    fig = plot_histograms(xlim, analyzers, lambda x: f"{x.config.specimen_name}_{stresses[x.config.specimen_name]:.2f}")
    fig.savefig(os.path.join(path, f"{sigmas[0]}-{sigmas[1]}_log_histograms.png"))
    
app()

