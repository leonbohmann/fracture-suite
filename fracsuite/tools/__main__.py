import os
import time

import typer
from matplotlib import pyplot as plt
from rich import inspect, print

from fracsuite.core.progress import get_progress

from fracsuite.tools.GlobalState import GlobalState
from fracsuite.tools.config import app as config_app
from fracsuite.tools.splinters import app as splinter_app
from fracsuite.tools.acc import app as acc_app
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.specimen import app as specimen_app
from fracsuite.tools.test_prep import test_prep_app
from fracsuite.tools.nominals import nominals_app
from fracsuite.tools.scalp import scalp_app


general = GeneralSettings.get()

plt.rcParams['figure.figsize'] = general.figure_size
plt.rc('axes', axisbelow=True) # to get grid into background
plt.rc('grid', linestyle="--") # line style
plt.rcParams.update({'font.size': 12}) # font size

def main_callback(ctx: typer.Context, debug: bool = None):
    """Fracsuite tools"""
    # print(Panel.fit(f"# Running [bold]{ctx.invoked_subcommand}[/bold]", title="Fracsuite tools", border_style="green"))
    # print(ctx.protected_args)
    GlobalState.start_time = time.time()
    GlobalState.debug = debug
    GlobalState.progress = get_progress()
    GlobalState.sub_outpath = ctx.invoked_subcommand


    os.makedirs(os.path.join(general.out_path, GeneralSettings.sub_outpath), exist_ok=True)

def end_callback(*args, **kwargs):
    # for i in args:
    #     inspect(i)
    # for k in kwargs:
    #     inspect(k)

    d = time.time() - GlobalState.start_time
    print(f"Finished in {d:.2f}s.")

app = typer.Typer(pretty_exceptions_short=False, result_callback=end_callback, callback=main_callback)
app.add_typer(splinter_app, name="splinters")
app.add_typer(config_app, name="config")
app.add_typer(specimen_app, name="specimen")
app.add_typer(acc_app, name="acc")
app.add_typer(scalp_app, name="scalp")
app.add_typer(test_prep_app, name="test-prep")
app.add_typer(nominals_app, name="nominals")


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

app()
