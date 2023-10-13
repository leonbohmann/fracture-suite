"""
Configuration of fracsuite.
"""
import os
import re
import typer
from rich import print, inspect

from fracsuite.general import GeneralSettings

app = typer.Typer(help=__doc__)
general = GeneralSettings.get()


@app.command()
def ls():
    inspect(general, title= "General Settings")

@app.command()
def clear():
    general.clear()

@app.command()
def set(key, value):
    general = GeneralSettings.get()

    print(f"Setting '{key}' to '{value}'.")
    # match input value for tuple format
    if (is_tuple := re.match(r"^\(.*\)$", value)) or (is_list := re.match(r"^\[.*\]$", value)):
        nums= value[1:-1].split(',')
        for i in range(len(nums)):
            if 'cm' in nums[i]:
                f = 2.54
                nums[i] = nums[i].replace('cm', '').strip()
            else:
                f = 1

            nums[i] = float(nums[i]) / f

        if is_tuple:
            value = tuple(nums)
        elif is_list:
            value = list(nums)

    if not hasattr(general, key):
        print(f"Setting '{key}' does not exist.")
        return

    general.update_setting(key, value)

    print(f"Updated setting '{key}' to '{value}'.")


@app.command()
def init():
    """Initialize the fracsuite configuration and necessary system settings."""
    import sys
    app_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
    if not app_path in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + app_path
        print(f"Added '{app_path}' to PATH environment variable. You may need to restart the computer.")
    if not app_path in sys.path:
        sys.path.append(app_path)