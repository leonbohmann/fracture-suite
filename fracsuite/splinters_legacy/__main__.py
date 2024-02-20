"""
███████╗██████╗ ██╗     ██╗███╗   ██╗████████╗███████╗██████╗ ███████╗
██╔════╝██╔══██╗██║     ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔════╝
███████╗██████╔╝██║     ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝███████╗
╚════██║██╔═══╝ ██║     ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗╚════██║
███████║██║     ███████╗██║██║ ╚████║   ██║   ███████╗██║  ██║███████║
╚══════╝╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

Leon Bohmann     TUD - ISMD - GCC        www.tu-darmstadt.de/glass-cc


Description:
-------------------------
This module helps with the automated detection of fragments on fractured glass plys. It
performs some preprocessing actions to improve the quality of the input image and then
uses that to search for contours on the image. After filtering and further improvement
the found contours are converted into `Splinter` objects, which allow further investigation
of the fragments (size, roughness, roundness, ...).

Used packages:
-------------------------
- opencv-python
- matplotlib
- numpy

- tqdm

Usage:
-------------------------

Command line usage is shown below. For further information visit:
https://github.com/leonbohmann/fracture-suite
"""


import os
import shutil
import time
from matplotlib import pyplot as plt
# import module
from fracsuite.core.progress import get_progress
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from rich import print

import matplotlib

from fracsuite.general import GeneralSettings
from fracsuite.helpers import print_exc_to_log, print_to_log
from fracsuite.specimen import Specimen

matplotlib.rcParams['figure.figsize'] = (6, 4)
matplotlib.rc('axes', axisbelow=True) # to get grid into background
matplotlib.rc('grid', linestyle="--") # line style
matplotlib.rcParams.update({'font.size': 12}) # font size

general_settings = GeneralSettings.get()

parser = AnalyzerConfig.get_parser(__doc__)

parser.add_argument("--all", default=False,
                    help="Instruct the analyzer to run the analysis on every subfolder.",
                    action='store_true')
parser.add_argument("--all-exclude", default=[], nargs="+")
parser.add_argument("--clear-splinters", action='store_true', default=False)
parser.add_argument("--update-plots", action='store_true', default=False)
# parser.add_argument("--open", action='store_true', default=False)

args = parser.parse_args()


config = AnalyzerConfig.from_args(args)

if args.all or config.path[1] != ":":
    print(f"[bold][green]Using base path '{general_settings.base_path}'.[/green][/bold]")
    config.path = os.path.join(general_settings.base_path, config.path) + "\\"


if args.all:
    print(f"Running analysis on all subfolders of '{config.path}'.")
    project_dir = config.path
    matplotlib.use('Agg')

    if os.path.exists("log.txt"):
        os.remove("log.txt")

    with get_progress() as progress:

        all_task = progress.add_task("[green]Analyzing...", total=len(os.listdir(project_dir)))
        for file in os.listdir(project_dir):
            progress.update(all_task, description=f"'{file}'...", total=len(os.listdir(project_dir)))
            file_task = progress.add_task(f"Analyzing '{file}'...", total=1.0)

            def update_file_task(val, title, total = 1.0):
                progress.update(file_task, completed=val, description=title, total=total)

            if any([x in file for x in args.all_exclude]):
                progress.remove_task(file_task)
                continue

            project_path = os.path.join(project_dir, file) + "\\"

            if os.path.exists(project_path) and os.path.isdir(project_path):
                spec = Specimen(project_path, log_missing=False)


                if spec.settings['break_pos'] == "center":
                    config.impact_position = (250,250)
                else:
                    config.impact_position = (50,50)

                if args.clear_splinters:
                    shutil.rmtree(spec.splinters_folder, ignore_errors=True)

                if not spec.has_fracture_scans:
                    progress.remove_task(file_task)
                    continue

                try:
                    config.path = project_path
                    analyzer = Analyzer(config, progress, file_task)
                    plt.close()
                except Exception as e:
                    print_to_log(f'[bold red]Error[/bold red] while analyzing specimen: {file}')
                    print_to_log(e.__dict__)
                    print_exc_to_log()
                    progress.remove_task(file_task)
                    continue

            progress.remove_task(file_task)
            progress.update(all_task, advance=1, refresh=True)
            progress.refresh()
            time.sleep(0.1)

        print("[green]Finished.")
else:
    analyzer = Analyzer(config, clear_splinters = args.clear_splinters)
