import os
import subprocess
import sys
import time

import typer
from matplotlib import pyplot as plt
from rich import print
from rich.progress import Progress, TextColumn
from rich.theme import Theme

# used for redirection of pickling
import fracsuite.core.splinter as splt
from fracsuite.acc import app as acc_app
from fracsuite.config import app as config_app
from fracsuite.core.coloring import norm_color
from fracsuite.core.progress import get_progress
from fracsuite.general import GeneralSettings
from fracsuite.layer import layer_app
from fracsuite.nominals import nominals_app
from fracsuite.over_nrg import over_nrg
from fracsuite.scalp import scalp_app
from fracsuite.simulate import sim_app
from fracsuite.specimen import app as specimen_app
from fracsuite.splinters import app as splinter_app
from fracsuite.state import State
from fracsuite.test_prep import test_prep_app
from fracsuite.tester import tester_app

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

general = GeneralSettings.get()

cmap1 = [
    norm_color((38, 70, 83)),
    norm_color((42, 157, 143)),
    norm_color((233, 196, 106)),
    norm_color((244, 162, 97)),
    norm_color((231, 111, 81))
]

cmap2 = [
    norm_color((255, 190, 11)),
    norm_color((251, 86, 7)),
    norm_color((255, 0, 110)),
    norm_color((131, 56, 236)),
    norm_color((58, 134, 255))
]

cmap3 = [
    norm_color('#2E4057'),
    norm_color('#FFA07A'),
    norm_color('#5F9EA0'),
    norm_color('#4682B4'),
    norm_color('#FAD02E'),
]
cmap4 = [
    norm_color('#4E79A7'),  # Blue
    norm_color('#F28E2B'),  # Orange
    norm_color('#E15759'),  # Red
    norm_color('#76B7B2'),  # Teal
    norm_color('#59A14F')   # Green
]
# get colormap
cmap=plt.cm.Dark2
# build cycler with 5 equally spaced colors from that colormap
# c = cycler(color=cmap.colors)
# c = cycler(color=cmap1)

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
    'lines.markersize': 3,
    # 'axes.grid.which': 'both',
    # 'xtick.minor.visible': True,
    # 'ytick.minor.visible': True,
}
plt.rcParams.update(params)
# plt.rc('axes', prop_cycle=c)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{xfrac}')
# plt.style.use('bmh')

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

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_show_locals=False, result_callback=end_callback, callback=main_callback)
app.add_typer(splinter_app, name="splinters")
app.add_typer(config_app, name="config")
app.add_typer(specimen_app, name="specimen")
app.add_typer(acc_app, name="acc")
app.add_typer(scalp_app, name="scalp")
app.add_typer(test_prep_app, name="test-prep")
app.add_typer(nominals_app, name="nominals")
app.add_typer(tester_app, name="tester")
app.add_typer(over_nrg, name="over-nrg")
app.add_typer(sim_app, name="simulate")
app.add_typer(layer_app, name="model")
class PrintWrapper():

    def __init__(self, func, index):
        """Setup the object with a logger and a loglevel
        and start the thread
        """
        self.func = func
        self.index = index
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)

    def fileno(self):
        """Return the write file descriptor of the pipe
        """
        return self.fdWrite

    def run(self):
        """Run the thread, logging everything.
        """
        for line in iter(self.pipeReader.readline, ''):
            print(f'[green]{self.index}[/green]> {line}')

        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)

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
def replot(
    tex_file: str,
    dry: bool = False
):
    plot_command = "%pltcmd:"

    if os.path.isdir(tex_file) and not tex_file.endswith(".tex"):
        # find all tex files in folder
        tex_files = []
        for file in os.listdir(tex_file):
            if file.endswith(".tex"):
                tex_files.append(os.path.join(tex_file, file))
    else:
        tex_files = [tex_file]

    all_lines = []
    for f in tex_files:
        with open(f, "r") as f_io:
            lines = f_io.readlines()
        all_lines.append((os.path.basename(f), lines))

        print(f'Read {len(lines)} lines from {os.path.basename(f)}.')
    commands = []

    for file, lines in all_lines:
        for li, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(plot_command):
                command = stripped[len(plot_command)+1:]
                commands.append((f'[cyan]{file}[/cyan] ([dim white]L{li}[/dim white])',command))

    with Progress(
            TextColumn("[progress.description]{task.description:<50}"),
        ) as progress:

        tasks = []
        for file, command in commands:
            cmd_task_descr = f"{file} {command}"
            cmd_task = progress.add_task(cmd_task_descr)
            tasks.append((file, command, cmd_task))

        for file, command, task in tasks:
            if not dry:
                progress.update(task, description=f"{file} [green]> [/green] {command}")
                proc = subprocess.Popen(["cmd", "/c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out,err = proc.communicate()
                progress.update(task, description=f"{file} [green]{command}")
            else:
                print(f"DRY: Running {command}")
                time.sleep(1)



@app.command()
def test(input: list[str]):
    print(input)

try:
    app()
finally:
    State.checkpoint_save()
