from fracsuite.core.logging import info
import logging
import os
import re
from typing import Annotated, Any, Callable
import typer
from rich import print
from fracsuite.core.logging import start
from fracsuite.state import State

from fracsuite.general import GeneralSettings

general = GeneralSettings.get()


def main_callback(
        ctx: typer.Context,
        set_additional_path: Annotated[str, typer.Option(help='Set an output path for the subcommand.')] = None,
        clear_path: Annotated[bool, typer.Option('--clear-path', help='Remove the additional output path.')] = None,
        clear_output: Annotated[bool, typer.Option(help='Clears similar files when generating output.')] = False,
        no_additional: Annotated[bool, typer.Option(help='Do not use an additional output path.')] = False,
        to_temp: Annotated[bool, typer.Option(help='Redirect all output to temp folder.')] = False,
        save_plots: Annotated[bool, typer.Option(help='Save plots to output folder.')] = False,
        figasimgonly: Annotated[bool, typer.Option(help='Save plots as images only.')] = False,
        mod: Annotated[str, typer.Option(help='Modifies the output name.')] = '',
        subfolder: Annotated[str, typer.Option(help='Puts all output in a subfolder.')] = None,
        max_spec: Annotated[int, typer.Option(help='Maximum number of specimens to process.')] = 1000,
        no_open: Annotated[bool, typer.Option(help='Do not open any outputs.')] = False,
        no_out: Annotated[bool, typer.Option(help='Do not generate any output.')] = False,
    ):
    """Splinter analyzation tools."""
    cmd = os.path.basename(State.sub_outpath) + "/" + ctx.invoked_subcommand

    if set_additional_path is not None:
        general.output_paths[cmd] = set_additional_path
        general.save()

    if clear_path:
        general.output_paths.pop(cmd, None)
        general.save()

    if cmd in general.output_paths and not no_additional:
        State.additional_output_path = general.output_paths[cmd]

    State.sub_outpath = os.path.join(State.sub_outpath, ctx.invoked_subcommand)

    State.current_subcommand = ctx.invoked_subcommand

    State.clear_output = clear_output
    if clear_output:
        info("[cyan]Similar files will be deleted when finalizing.")

    os.makedirs(os.path.join(general.out_path, State.sub_outpath), exist_ok=True)

    State.to_temp = to_temp
    State.output_name_mod = mod
    State.save_plots = save_plots
    State.figasimgonly = figasimgonly
    State.maximum_specimen = max_spec
    State.no_open = no_open
    State.no_out = no_out

    State.pointoutput(subfolder)

    if mod != "":
        info(f"[cyan]Output name will be modified with: {mod}")

    if to_temp:
        info("[cyan]Output will be written to temp folder.")

#TODO: In the future this can be used to make the commands more modular
def specimen_callback(name_or_names_with_sigma: list[str]):
    """Creates a filter function for specimens.

    Args:
        names (str): String wildcard to match specimen names.
        sigmas (str): String with sigma range.
        sigma_delta (int, optional): If a single sigma value is passed, this range is added around the value. Defaults to 10.
        exclude (str, optional): Name filter to exclude. Defaults to None.
        needs_scalp (bool, optional): The specimen needs valid scalp data. Defaults to True.
        needs_splinters (bool, optional): The specimen needs valid splinter data. Defaults to True.

    Returns:
        Callable[[Specimen], bool]: Modified names, sigmas and filter function.
    """

    from fracsuite.core.specimen import Specimen

    def in_names_wildcard(s: Specimen, filter: str) -> bool:
        return re.match(filter, s.name) is not None
    def in_names_list(s: Specimen, filter: list[str]) -> bool:
        return s.name in filter
    def all_names(s, filter) -> bool:
        return True

    name_filter_function: Callable[[Specimen, Any], bool] = None

    # split input
    if len(name_or_names_with_sigma) == 1:
        name_filter, sigmas = name_or_names_with_sigma[0], None
    elif len(name_or_names_with_sigma) == 2:
        name_filter, sigmas = name_or_names_with_sigma

    if "," in sigmas:
        sigmas, sigma_delta = sigmas.split(",")
    else:
        sigma_delta = 10

    # create name_filter_function based on name_filter
    if name_filter is not None and "," in name_filter:
        name_filter = name_filter.split(",")
        print(f"Searching for specimen whose name is in: '{name_filter}'")
        name_filter_function = in_names_list
    elif name_filter is not None and " " in name_filter:
        name_filter = name_filter.split(" ")
        print(f"Searching for specimen whose name is in: '{name_filter}'")
        name_filter_function = in_names_list
    elif name_filter is not None and "*" not in name_filter:
        name_filter = [name_filter]
        print(f"Searching for specimen whose name is in: '{name_filter}'")
        name_filter_function = in_names_list
    elif name_filter is not None and "*" in name_filter:
        print(f"Searching for specimen whose name matches: '{name_filter}'")
        name_filter = name_filter.replace(".","\.").replace("*", ".*").replace('!', '|')
        name_filter_function = in_names_wildcard
    elif name_filter is None:
        name_filter = ".*"
        print("[green]All[/green] specimen names included!")
        name_filter_function = all_names

    if sigmas is not None:
        if "-" in sigmas:
            sigmas = [float(s) for s in sigmas.split("-")]
        elif sigmas == "all":
            sigmas = [0,1000]
        else:
            sigmas = [float(sigmas), float(sigmas)]
            sigmas[0] = max(0, sigmas[0] - sigma_delta)
            sigmas[1] += sigma_delta

        print(f"Searching for splinters with stress in range {sigmas[0]} - {sigmas[1]}")

    exclude = name_or_names_with_sigma[2] if len(name_or_names_with_sigma) > 2 else None
    needs_scalp = name_or_names_with_sigma[3] if len(name_or_names_with_sigma) > 3 else True
    needs_splinters = name_or_names_with_sigma[4] if len(name_or_names_with_sigma) > 4 else True

    def filter_specimens(specimen: Specimen):
        if needs_scalp and not specimen.has_scalp:
            return False
        elif needs_splinters and not specimen.has_splinters:
            return False
        elif exclude is not None and re.match(exclude, specimen.name):
            return False
        elif not name_filter_function(specimen, name_filter):
            return False
        elif sigmas is not None:
            return sigmas[0] <= abs(specimen.scalp.sig_h) <= sigmas[1]

        return True

    return filter_specimens