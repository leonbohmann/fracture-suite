import os
import typer
from rich import print, inspect
from fracsuite.tools.GlobalState import GlobalState

from fracsuite.tools.general import GeneralSettings

general = GeneralSettings.get()


def main_callback(ctx: typer.Context, set_path: str = None, out: str = None, clear_output: bool = False):
    """Splinter analyzation tools."""
    cmd = os.path.basename(GlobalState.sub_outpath) + "/" + ctx.invoked_subcommand

    if set_path is not None:
        general.output_paths[cmd] = set_path
        general.save()

    if cmd in general.output_paths:
        GlobalState.sub_outpath = general.output_paths[cmd]
    else:
        GlobalState.sub_outpath = os.path.join(GlobalState.sub_outpath, ctx.invoked_subcommand)

    GlobalState.current_subcommand = ctx.invoked_subcommand

    GlobalState.clear_output = clear_output
    if clear_output:
        print("[yellow]Similar files will be deleted when finalizing.")

    os.makedirs(os.path.join(general.out_path, GlobalState.sub_outpath), exist_ok=True)
