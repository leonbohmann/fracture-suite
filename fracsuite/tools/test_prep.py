"""
TESTS for the preprocessing unit of this module.
"""

import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import numpy as np
from fracsuite.core.image import to_rgb
from fracsuite.core.plotting import plot_values
from fracsuite.core.progress import get_progress
from fracsuite.splinters.analyzerConfig import AnalyzerConfig, PreprocessorConfig
from fracsuite.tools.helpers import bin_data, img_part
from fracsuite.tools.specimen import Specimen

import typer

from fracsuite.splinters.analyzerConfig import defaultPrepConfig, softgaussPrepConfig, softmeanPrepConfig, aggressivegaussPrepConfig, aggressivemeanPrepConfig

test_prep_app = typer.Typer()

@test_prep_app.command()
def test_configs(specimen_name: str):

    specimen = Specimen.get(specimen_name)

    configs = [defaultPrepConfig, softgaussPrepConfig, softmeanPrepConfig, aggressivegaussPrepConfig, aggressivemeanPrepConfig]

    with get_progress() as progress:
        an_task = progress.add_task("Testing configs...", total=len(configs))

        def plot_config(cfg: PreprocessorConfig, axs: Axes):
            progress.update(an_task, description=f"Testing {cfg.name}...")
            ancfg = specimen.splinter_config
            ancfg.prep = cfg

            c_task = progress.add_task(f"Testing {cfg.name}...", total=1, parent=an_task)
            analyzer = specimen.get_analyzer(ancfg, progress=progress, task=c_task)
            im = to_rgb(analyzer.preprocessed_image)
            cv2.drawContours(im, [x.contour for x in analyzer.splinters], -1, (255,0,0), 1)
            im = img_part(im, 2000,1800, 100, 100)

            axs.imshow(im)
            axs.set_title(cfg.name)
            progress.remove_task(c_task)
            progress.update(an_task, advance=1)


        plot_values(configs, plot_config)


@test_prep_app.command()
def test_configs_hists(specimen_name: str):

    specimen = Specimen.get(specimen_name)

    configs = [defaultPrepConfig, softgaussPrepConfig, softmeanPrepConfig, aggressivegaussPrepConfig, aggressivemeanPrepConfig]

    with get_progress() as progress:
        an_task = progress.add_task("Testing configs...", total=len(configs))

        fig,axs = plt.subplots()

        def plot_config(cfg: PreprocessorConfig, axs: Axes):
            progress.update(an_task, description=f"Testing {cfg.name}...")
            ancfg = specimen.splinter_config
            ancfg.prep = cfg

            c_task = progress.add_task(f"Testing {cfg.name}...", total=1, parent=an_task)
            analyzer = specimen.get_analyzer(ancfg, progress=progress, task=c_task)


            areas = [np.log10(x.area) for x in analyzer.splinters if x.area > 0]
            # ascending sort, smallest to largest
            areas.sort()

            # density: normalize the bins data count to the total amount of data
            axs.hist(areas, bins=int(50),
                    density=True,
                    alpha=0.4)

            axs.set_title(cfg.name)
            progress.remove_task(c_task)
            progress.update(an_task, advance=1)

        for cfg in configs:
            plot_config(cfg, axs)

        axs.set_xlim((0, 2))

        ticks = FuncFormatter(lambda x, pos: '{0:.00f}'.format(10**x))
        ticksy = FuncFormatter(lambda x, pos: '{0:.2f}'.format(x))
        axs.xaxis.set_major_formatter(ticks)
        axs.yaxis.set_major_formatter(ticksy)

        # ax.xaxis.set_major_formatter(ScalarFormatter())
        axs.set_xlabel('Splinter Area [mm²]')
        axs.set_ylabel('Probability (Area) [-]')
        axs.grid(True, which='both', axis='both')
        fig.tight_layout()
        plt.show()


@test_prep_app.command()
def test_prep_2d(specimen_name: str):
    specimen = Specimen.get(specimen_name)

    configs = [defaultPrepConfig, softgaussPrepConfig, softmeanPrepConfig, aggressivegaussPrepConfig, aggressivemeanPrepConfig]
    # configs = [defaultPrepConfig, softgaussPrepConfig,]
    binrange = np.linspace(0,2,40)

    with get_progress() as progress:
        an_task = progress.add_task("Testing configs...", total=len(configs))
        data = []
        for cfg in configs:
            ancfg = specimen.splinter_config
            ancfg.prep = cfg

            c_task = progress.add_task(f"Testing {cfg.name}...", total=1, parent=an_task)
            analyzer = specimen.get_analyzer(ancfg, progress=progress, task=c_task)


            areas = [np.log10(x.area) for x in analyzer.splinters if x.area > 0]
            # ascending sort, smallest to largest
            areas.sort()

            data.append(bin_data(areas, binrange))

        dt = np.array(data)
        plt.xticks([10**x for x in np.arange(0, len(binrange), 5)], np.round(binrange[::5], 2))
        plt.imshow(dt, cmap='turbo', aspect='auto')
        plt.show()