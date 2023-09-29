"""
TESTS for the preprocessing unit of this module.
"""

import os
import time
import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import numpy as np
from fracsuite.core.image import to_gray, to_rgb
from fracsuite.core.plotting import plot_image_kernel_contours, plot_splinter_kernel_contours, plot_values
from fracsuite.core.progress import get_progress, get_spinner
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.splinters.analyzerConfig import AnalyzerConfig, PreprocessorConfig
from fracsuite.splinters.processing import preprocess_image
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import bin_data, img_part
from fracsuite.tools.specimen import Specimen
from pathos.multiprocessing import ProcessPool

import typer

from fracsuite.splinters.analyzerConfig import defaultPrepConfig, softgaussPrepConfig, softmeanPrepConfig, aggressivegaussPrepConfig, aggressivemeanPrepConfig
from fracsuite.tools.splinters import finalize

general = GeneralSettings.get()

test_prep_app = typer.Typer(help=__doc__)

@test_prep_app.command()
def test_configs(specimen_name: str):
    """Test the different preprocessing configs.

    Args:
        specimen_name (str): Name of specimen to perform tests on.
    """
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


        fig, axs = plot_values(configs, plot_config)
        plt.show()


@test_prep_app.command()
def test_configs_hists(specimen_name: str):
    """Test the different preprocessing configs and plot a histogram."""
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
    """Test the different preprocessing configs and plot a histogram in 2D."""
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

@test_prep_app.command()
def disp_mean(specimen_name: str,):
    """Display the mean value of a fracture image."""
    specimen = Specimen.get(specimen_name)
    mean_value = np.mean(specimen.get_fracture_image())
    print(mean_value)

    im = (to_gray(specimen.get_fracture_image())-mean_value).astype(np.uint8)
    im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    plt.imshow(im)
    plt.show()
    pass

@test_prep_app.command()
def test_splinter_count(specimen_name: str, load: bool = False, calibrated: int = 301):
    """Test the splinter count for different preprocessing configs.

    Args:
        specimen_name (str): Specimen name
        load (bool, optional): Loads previous results. Defaults to False.
        calibrated (int, optional): The calibrated splinter count value for the area. Defaults to 301.
    """
    specimen = Specimen.get(specimen_name)

    # mean_value = np.mean(specimen.get_fracture_image())
    # print(mean_value)

    # im = (to_gray(specimen.get_fracture_image())).astype(np.uint8)
    # im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 8)
    # im = cv2.floodFill(im, None, (0,0), 0)[1]
    # plt.imshow(im)
    # plt.show()

    steps = np.linspace(0, 8, 20)
    stepsy = np.array([3,5,7,11,13])

    p = ProcessPool(nodes=8)

    def get_count(steps: tuple[float, list[int]]):
        # steps = step, stepsy = None
        step, stepsy = steps

        if stepsy is None:
            cfg = specimen.splinter_config
            cfg.prep.thresh_c = step
            analyzer = Analyzer(cfg, silent=True, splinters_only=True)

            return len(analyzer.splinters)

        counts = []
        for stepy in stepsy:
            cfg = specimen.splinter_config
            cfg.prep.thresh_c = step
            cfg.prep.thresh_block_size = stepy
            analyzer = Analyzer(cfg, silent=True)
            counts.append(len(analyzer.splinters))

        return counts

    def plot_counts(counts, calibrated=276):
        print(counts)
        fig,axs = plt.subplots(figsize=GeneralSettings.get().figure_size)
        for i, s in enumerate(stepsy):
            c = [x[i] for x in counts]
            axs.plot(steps, c, label=f"{s}")
            axs.set_xlabel("Threshold C")
            axs.set_ylabel("Splinter Count")
        axs.axhline(calibrated, color='k', linestyle='--')

        axs.legend()
        fig.tight_layout()
        plt.show()

    if load:
        with open('counts.pkl', "rb") as f:
            import pickle
            counts = pickle.load(f)
        plot_counts(counts, calibrated)
        return

    counts = []
    with get_spinner("Testing configs...", True) as spinner:
        spinner.set_total(len(steps))

        # for cfg in configs:
        #     counts.append(get_count(cfg))
        # tasks = []
        # for cfg in configs:
        #     tasks.append(spinner.progress.add_task(f"Testing {cfg.name}..."))

        # for cfg, t_id in zip(configs, tasks):
        #     counts.append(get_count((cfg, t_id, spinner.progress)))
        #     spinner.advance()

        data = p.amap(get_count, [(x,stepsy) for x in steps])
        while not data.ready():
            time.sleep(0.5)
            spinner.set_completed(len(steps) - data._number_left)

    counts = data.get()

    with open('counts.pkl', "wb") as f:
        import pickle
        pickle.dump(counts, f)
