import typer
from rich import print, inspect

from fracsuite.tools.general import GeneralSettings

app = typer.Typer()
general = GeneralSettings()


@app.command()
def list():
    inspect(general, title= "General Settings")

@app.command()
def clear():
    general.clear()             

@app.command()
def set(key, value):
    general = GeneralSettings()
    
    if not hasattr(general, key):
        print(f"Setting '{key}' does not exist.")
        return
    
    general.update_setting(key, value)
    
    print(f"Updated setting '{key}' to '{value}'.")

    