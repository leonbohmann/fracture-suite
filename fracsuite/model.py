from enum import Enum
from typing import Annotated
from matplotlib import pyplot as plt
import typer
import os
import numpy as np

from fracsuite.callbacks import main_callback
from fracsuite.core.plotting import FigureSize, get_fig_width, renew_ticks_cb
from fracsuite.general import GeneralSettings
from scipy.interpolate import griddata
from fracsuite.core.lbreak import ModeChoices, load_model, ModelBoundary, plt_model
from fracsuite.state import State, StateOutput

model_app = typer.Typer(callback=main_callback, help="Model related commands.")
general = GeneralSettings.get()

@model_app.command()
def plot(
    mode: Annotated[ModeChoices, typer.Argument(help="The mode to display")],
    boundary: Annotated[ModelBoundary, "Boundary condition"],
    filter_nan: Annotated[bool, typer.Option(True, help="Filter NaN values")] = False,
):
    model_name = f'interpolate_{mode}_{boundary}_corner.npy'
    R,U,V = load_model(model_name)

    xlabel = 'Radius [mm]'
    ylabel = 'Elastic Strain Energy U [J/m²]'
    clabel = 'Aspect ratio $L/L_p$'

    if mode == ModeChoices.AREA:
        clabel = "Splinter Area $A_S$ [mm²]"
    elif mode == ModeChoices.ORIENTATION:
        clabel = "Splinter Orientation $\Delta$ [-]"
    elif mode == ModeChoices.ROUNDNESS:
        clabel = "Roundness $\lambda_c$ [-]"
    elif mode == ModeChoices.ROUGHNESS:
        clabel = "Roughness $\lambda_r$ [-]"
    elif mode == ModeChoices.ASP:
        clabel = "Aspect Ratio $L/L_p$ [-]"
    elif mode == ModeChoices.ASP0:
        clabel = "Aspect Ratio $L_1/L_2$ [-]"
    elif mode == ModeChoices.L1:
        clabel = "Splinter Height $L_1$ [mm]"
    elif mode == ModeChoices.L2:
        clabel = "Splinter Width $L_2$ [mm]"

    fig = plt_model(R,U,V,filter_nan=filter_nan, xlabel=xlabel, ylabel=ylabel, clabel=clabel)
    State.output(StateOutput(fig, FigureSize.ROW2), f"model_{mode}_{boundary}.png")


    fig.show()
    plt.waitforbuttonpress()