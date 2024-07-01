from enum import Enum
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import numpy as np
import cv2

from fracsuite.general import GeneralSettings

general = GeneralSettings.get()

class DataHistMode(str, Enum):
    PDF = 'pdf'
    "Probability density function."
    CDF = 'cdf'
    "Cumulative density function."

class DataHistPlotMode(str, Enum):
    KDE = 'kde'
    "Kernel density estimation."
    HIST = 'hist'
    "Histogram."

class FigureSize(str, Enum):
    """ Figwidth factor for different figure configurations! """
    ROW1 = 'row1'
    "The width of a figure in a row with one figure."
    ROW2 = 'row2'
    "The width of a figure in a row with two figures."
    ROW3 = 'row3'
    "The width of a figure in a row with three figures."
    ROW1H = 'row1h'
    "The width of a figure in one row in landscape."

    @classmethod
    def has_value(cls, value):
        return value in cls.values()

    @classmethod
    def values(cls):
        return set(item.value for item in cls)


def get_fig_width(w: FigureSize, hf=None, dimf=1.0) -> float:
    """
    Calculates the figure width and height in inches based on the given width factor, height factor and dimension factor.

    The dimension factor dimf is used so that the figures are rendered slightly larger than the actual size,
    because in Latex the textwidth is not the same as the actual width of the environment.

    Args:
        w (FigWidth): The width factor of the figure.
        hf (Optional[float]): The height factor of the figure. Defaults to None.
        dimf (float): The dimension factor of the figure. Defaults to 1.1.

    Returns:
        Tuple[float, float]: The figure width and height in inches.
    """
    is_landscape = False
    if w.endswith('h'):
        is_landscape = True

    assert FigureSize.has_value(w), f"FigWidth must be one of {FigureSize.values()}."

    mm_per_inch = 1 / 25.4

    w_mm,h_mm = general.figure_sizes_mm[w]
    w_inch = mm_per_inch * w_mm
    h_inch = mm_per_inch * h_mm

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    wfac = 1 if not is_landscape else golden_ratio

    fig_width_in = w_inch * dimf
    fig_height_in = h_inch * dimf * wfac

    return (fig_width_in, fig_height_in)


class DataPlotter():
    _figure: Figure
    _axes: list[Axes]

    @property
    def figure(self):
        return self._figure
    @property
    def axes(self):
        return self._axes

    def __init__(
        self,
        ncols:int = 1,
        nrows:int = 1,
        xlim: tuple[float,float] = None,
        x_format: str = "{0:.0f}",
        x_label: str = 'Splinter Area $A_S$ [mm²]',
        y_format: str = "{0:.2f}",
        y_label: str = None,
        data_mode : DataHistMode = DataHistMode.PDF,
        figwidth = FigureSize.ROW1,
        log_axes: bool = False,
        log_data: bool = False,
    ) -> None:
        """Create a figure and axes for a data histogram."""
        self.figsize = get_fig_width(figwidth)
        self._figure, self._axes = plt.subplots(ncols, nrows, figsize=self.figsize, sharex=True, sharey=True)

        self.logarithmic_axes = log_axes
        "Wether the axes should be logarithmic."
        self.logarithmic_data = log_data
        "Wether the data should be logarithmic."


        if nrows == 1 and ncols == 1:
            self._axes = [self._axes]

        if xlim is not None:
            for ax in self._axes:
                ax.set_xlim(xlim)
        else:
            for ax in self._axes:
                ax.set_xlim((0, 2))

        ticks = FuncFormatter(lambda x, pos: x_format.format(10**x))
        ticksy = FuncFormatter(lambda x, pos: y_format.format(x))

        for ax in self._axes:
            ax.xaxis.set_major_formatter(ticks)
            ax.yaxis.set_major_formatter(ticksy)

            ax.set_xlabel(x_label)
            if y_label is not None:
                ax.set_ylabel(y_label)
            else:
                if data_mode == DataHistMode.PDF:
                    ax.set_ylabel('Probability density $p(A_S)$')
                elif data_mode == DataHistMode.CDF:
                    ax.set_ylabel('Cumulative Distr. Func. $C(A_S)$')
            ax.grid(True, which='both', axis='both')

    def add_data(
        self,
        data: list[float],
        n_bins: int = None,
        binrange: list[float] = None,
        plot_mean: bool = True,
        label: str = None,
        color = None,
        as_log:bool = True,
        alpha: float = 0.75,
        data_mode: DataHistMode = DataHistMode.PDF,
        as_density = True,
        plot_mode: DataHistPlotMode = DataHistPlotMode.HIST,
        unit: str = "mm²",
        mean_format: str = ".2f",
        new_ax: bool = False,
    ):
        pass

    def plot_data():
        pass

    def save_plot():
        pass
