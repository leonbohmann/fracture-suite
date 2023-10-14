import os
import sys
import time
import numpy as np

import typer
from matplotlib import pyplot as plt
from rich import inspect, print
from rich.theme import Theme

from fracsuite.core.progress import get_progress

from fracsuite.state import State
from fracsuite.config import app as config_app
from fracsuite.splinters import app as splinter_app
from fracsuite.acc import app as acc_app
from fracsuite.general import GeneralSettings
from fracsuite.specimen import app as specimen_app
from fracsuite.test_prep import test_prep_app
from fracsuite.nominals import nominals_app
from fracsuite.scalp import scalp_app
from fracsuite.tester import tester_app
from cycler import cycler

# used for redirection of pickling
import fracsuite.core.splinter as splt

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

general = GeneralSettings.get()
    # # Use LaTeX to write all text
    # "text.usetex": True,
    # "font.family": "serif",
    # # Use 10pt font in plots, to match 10pt font in document
    # "axes.labelsize": 10,
    # "font.size": 10,
    # # Make the legend/label fonts a little smaller
    # "legend.fontsize": 8,
    # "xtick.labelsize": 8,
    # "ytick.labelsize": 8
params = {
    'text.latex.preamble': r'\usepackage{gensymb}\usepackage{amsmath}\usepackage{xfrac}\usepackage{mathpazo}',
    'text.usetex': True,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 9, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 10, # was 10
    'legend.fontsize': 8,
    'xtick.labelsize': 8, # was 8
    'ytick.labelsize': 8, # was 8
    # 'pdf.fonttype': 42,
    # 'ps.fonttype': 42,
    'font.family': 'serif',
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
}
plt.rcParams.update(params)
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{xfrac}')
# plt.style.use('fast')

# cmap = plt.get_cmap('turbo')
# num_colors = 10
# color_cycle = [cmap(i) for i in np.linspace(0, 1, num_colors)]

# import to redirect pickle import
sys.modules['fracsuite.splinters.splinter'] = splt

def main_callback(ctx: typer.Context, debug: bool = None):
    """Fracsuite tools"""
    # print(Panel.fit(f"# Running [bold]{ctx.invoked_subcommand}[/bold]", title="Fracsuite tools", border_style="green"))
    # print(ctx.protected_args)
    State.start_time = time.time()
    State.debug = debug
    State.progress = get_progress()
    State.sub_outpath = ctx.invoked_subcommand


    os.makedirs(os.path.join(general.out_path, GeneralSettings.sub_outpath), exist_ok=True)

def end_callback(*args, **kwargs):
    # for i in args:
    #     inspect(i)
    # for k in kwargs:
    #     inspect(k)

    d = time.time() - State.start_time
    print(f"Finished in {d:.2f}s.")

app = typer.Typer(pretty_exceptions_short=False, result_callback=end_callback, callback=main_callback)
app.add_typer(splinter_app, name="splinters")
app.add_typer(config_app, name="config")
app.add_typer(specimen_app, name="specimen")
app.add_typer(acc_app, name="acc")
app.add_typer(scalp_app, name="scalp")
app.add_typer(test_prep_app, name="test-prep")
app.add_typer(nominals_app, name="nominals")
app.add_typer(tester_app, name="tester")


# @app.command()
# def marina_organize(path: str):
#     # find all folders in path that contain three dots
#     for dir in os.listdir(path):
#         if dir.count(".") != 3:
#             continue

#         dirpath = os.path.join(path, dir)

#         # create subdirectors
#         os.makedirs(os.path.join(dirpath, "scalp"), exist_ok=True)
#         os.makedirs(os.path.join(dirpath, "fracture"), exist_ok=True)
#         acc_path = os.path.join(dirpath, "fracture", "acceleration")
#         os.makedirs(acc_path, exist_ok=True)
#         morph_path = os.path.join(dirpath, "fracture", "morphology")
#         os.makedirs(morph_path, exist_ok=True)

#         # put all .bmp files into morphology folder
#         for file in os.listdir(dirpath):
#             if file.endswith(".bmp") or file.endswith(".zip"):
#                 os.rename(os.path.join(dirpath, file), os.path.join(morph_path, file))

#         # put all .tsx, .tst, .xlsx and .bin files into acceleration folder
#         for file in os.listdir(dirpath):
#             if file.lower().endswith(".tsx") or file.lower().endswith(".tst") or file.lower().endswith(".xlsx") or file.lower().endswith(".bin"):
#                 os.rename(os.path.join(dirpath, file), os.path.join(acc_path, file))

#         # in morphology folder, search for filename that starts with a 4digit number and prepend it to the file
#         # that contains "Transmission" in its name, if it does not start with the same 4digit num
#         for file in os.listdir(morph_path):
#             if "Transmission" in file:
#                 continue

#             # find 4digit number in filename
#             num = None
#             if file[0].isdigit() and file[0+1].isdigit() and file[0+2].isdigit() and file[0+3].isdigit():
#                 num = file[:4]

#             if num is None:
#                 continue

#             # find file with "Transmission" in its name
#             for file2 in os.listdir(morph_path):
#                 if "Transmission" in file2 and num not in file2:
#                     os.rename(os.path.join(morph_path, file2), os.path.join(morph_path, num + " " + file2))
#                     break

@app.command()
def test(input: list[str]):
    print(input)

app()
