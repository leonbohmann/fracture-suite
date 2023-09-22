import re
import numpy as np
import typer
from rich import print, inspect

from fracsuite.tools.general import GeneralSettings

app = typer.Typer()
general = GeneralSettings.get()


@app.command()
def list():
    inspect(general, title= "General Settings")

@app.command()
def clear():
    general.clear()

@app.command()
def set(key, value):
    general = GeneralSettings.get()

    # match input value for tuple format
    if re.match(r"^\(.*\)$", value):
        value = tuple(map(int, value[1:-1].split(',')))
    elif re.match(r"^\[.*\]$", value):
        value = list(map(int, value[1:-1].split(',')))

    if not hasattr(general, key):
        print(f"Setting '{key}' does not exist.")
        return

    general.update_setting(key, value)

    print(f"Updated setting '{key}' to '{value}'.")
