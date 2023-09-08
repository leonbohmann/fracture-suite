import pickle
import os
import argparse
import typer
from rich import print
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (6, 4)
plt.rc('axes', axisbelow=True) # to get grid into background
plt.rc('grid', linestyle="--") # line style
plt.rcParams.update({'font.size': 12}) # font size


app = typer.Typer()


@app.command(name="loghist")
def log_histograms(path:str, specimen_names: list[str]):
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
    
    cfg = AnalyzerConfig()
    
    fig, ax = plt.subplots()
    
    for analyzer in analyzers:        
        analyzer.plot_logarithmic_to_axes(ax, cfg)
    
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(path, "log_histograms.png"))
    
@app.command()
def test():
    pass
app()

