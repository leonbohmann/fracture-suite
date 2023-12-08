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
from fracsuite.core.lbreak import load_model, ModelBoundary, plt_model

model_app = typer.Typer(callback=main_callback, help="Model related commands.")
general = GeneralSettings.get()

@model_app.command()
def display_asp(
    boundary: Annotated[ModelBoundary, "Boundary condition"],
):
    model_name = f'interpolate_asp_{boundary}_corner.npy'
    R,U,V = load_model(model_name)

    xlabel = 'Radius [mm]'
    ylabel = 'Elastic Strain Energy U [J/m²]'
    clabel = 'Aspect ratio $L/L_p$'

    fig = plt_model(R,U,V,filter_nan=False, xlabel=xlabel, ylabel=ylabel, clabel=clabel)
    fig.show()
    plt.waitforbuttonpress()