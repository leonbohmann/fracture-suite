import os
import typer

from fracsuite.tools.general import GeneralSettings

general = GeneralSettings.get()


def main_callback(ctx: typer.Context, debug: bool = None, set_path: str = None):
    """Splinter analyzation tools."""
    cmd = os.path.basename(GeneralSettings.sub_outpath) + "/" + ctx.invoked_subcommand

    if set_path is not None:
        general.output_paths[cmd] = set_path
        general.save()

    if cmd in general.output_paths:
        GeneralSettings.sub_outpath = general.output_paths[cmd]
    else:
        GeneralSettings.sub_outpath = os.path.join(GeneralSettings.sub_outpath, ctx.invoked_subcommand)

    os.makedirs(os.path.join(general.out_path, GeneralSettings.sub_outpath), exist_ok=True)
