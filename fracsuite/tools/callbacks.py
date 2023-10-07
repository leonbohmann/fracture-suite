import os
from typing import Any
import typer
from rich import print
from fracsuite.tools.state import State

from fracsuite.tools.general import GeneralSettings

general = GeneralSettings.get()


def main_callback(ctx: typer.Context, set_path: str = None, out: str = None, clear_output: bool = False, debug: bool = False):
    """Splinter analyzation tools."""
    cmd = os.path.basename(State.sub_outpath) + "/" + ctx.invoked_subcommand

    if set_path is not None:
        general.output_paths[cmd] = set_path
        general.save()

    if cmd in general.output_paths:
        State.sub_outpath = general.output_paths[cmd]
    else:
        State.sub_outpath = os.path.join(State.sub_outpath, ctx.invoked_subcommand)

    State.current_subcommand = ctx.invoked_subcommand

    State.clear_output = clear_output
    if clear_output:
        print("[yellow]Similar files will be deleted when finalizing.")

    os.makedirs(os.path.join(general.out_path, State.sub_outpath), exist_ok=True)


    State.debug = debug

#TODO: Implement
def specimen_callback(name_or_names: Any):
    pass