from typing import Annotated
from matplotlib import pyplot as plt
import typer

from fracsuite.callbacks import main_callback
from fracsuite.core.plotting import FigureSize
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.general import GeneralSettings
from fracsuite.core.lbreak import load_layer, ModelBoundary, plt_layer
from fracsuite.state import State, StateOutput

layer_app = typer.Typer(callback=main_callback, help="Model related commands.")
general = GeneralSettings.get()

@layer_app.command()
def plot(
    layer: Annotated[str, typer.Argument(help="The layer to display")],
    mode: Annotated[SplinterProp, typer.Argument(help="The mode to display")],
    boundary: Annotated[ModelBoundary, typer.Argument(help="Boundary condition")],
    ignore_nan_plot: Annotated[bool, typer.Option(help="Filter NaN values")] = True,
):
    model_name = f'{layer}-layer_{mode}_{boundary}_corner.npy'
    R,U,V = load_layer(model_name)

    xlabel = 'Distance $R$ from Impact [mm]'
    ylabel = 'Elastic Strain Energy U [J/m²]'
    clabel = Splinter.get_mode_labels(mode)

    fig = plt_layer(R,U,V,ignore_nan=ignore_nan_plot, xlabel=xlabel, ylabel=ylabel, clabel=clabel,
                    interpolate=False)
    State.output(StateOutput(fig, FigureSize.ROW2), f"impact-layer-2d_{mode}_{boundary}")
    plt.close(fig)