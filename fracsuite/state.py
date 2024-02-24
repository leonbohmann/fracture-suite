import pickle
import tempfile
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from fracsuite.core.ProgWrapper import ProgWrapper
from fracsuite.core.logging import debug, info, warning
from fracsuite.core.outputtable import Outputtable
from fracsuite.general import GeneralSettings


import cv2
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from rich import print
from rich.progress import Progress


import os
import subprocess
import time

general = GeneralSettings.get()

SAVE_FORMAT = "[cyan]SAVE[/cyan] [dim white]{0:<11}[/dim white] '{1}'"

def is_image(data):
    return type(data).__module__ == np.__name__

class StateOutput:
    Data: Figure | npt.ArrayLike
    "Data of the output, may be an image or a figure."
    FigWidth: Any
    "If generated using a plot command, the FigWidth of the figure."
    add_data: dict[str, Any]
    "Additional data to save."

    is_image: bool
    is_figure: bool

    img_ext: str

    def __init__(self, data, figwidth, img_ext = None, img_rsz = 1.0, fig_as_img_only=False, **additional_data):
        assert isinstance(data, Figure) or type(data).__module__ == np.__name__, f"Data must be a matplotlib figure or a numpy array. Is '{type(data)}'."

        self.Data = data
        self.FigWidth = figwidth

        self.add_data = additional_data if len(additional_data) > 0 else {}

        self.is_image = type(data).__module__ == np.__name__
        self.is_figure = not self.is_image

        self.has_detailed_image = 'img_detailed' in additional_data
        self.fig_as_img_only = fig_as_img_only

        self.img_ext = img_ext
        if self.is_image and img_rsz != 1.0:
            self.Data = cv2.resize(self.Data, (0, 0), fx=img_rsz, fy=img_rsz)
        # if self.is_figure:
        #     sz = get_fig_width(figwidth)
        #     self.Data.set_size_inches(sz[0], sz[1])

    def save(self, path, resize_factor:float =1) -> str:
        """Saves the output to a file."""
        from fracsuite.core.plotting import FigureSize
        saved = False
        while not saved:
            try:
                if self.is_image:
                    extension = self.img_ext if self.img_ext is not None else general.image_extension
                    data = cv2.resize(self.Data, (0, 0), fx=resize_factor, fy=resize_factor)
                    cv2.imwrite(
                        outfile := f'{path}.{extension}',
                        data
                    )
                elif self.is_figure:
                    figw = None
                    if FigureSize.has_value(self.FigWidth):
                        figw = self.FigWidth
                    else:
                        figw = "custom"

                    if 'override_legendpos' in State.kwargs:
                        # get legend from figure
                        origin = self.Data.axes[0]
                        legend: Legend = origin.get_legend()
                        if legend is not None:
                            handles = legend.legendHandles
                            labels = [t.get_text() for t in legend.texts]

                            # remove existing legend from plot
                            legend.remove()

                            # add new legend
                            origin.legend(handles, labels, loc=State.kwargs['override_legendpos'])

                    if not self.fig_as_img_only and not State.figasimgonly:
                        self.Data.savefig(
                            outfile := f'{path}_{figw}.{general.plot_extension}',
                        )
                    self.Data.savefig(
                        outfile := f'{path}_{figw}.png',
                    )
                saved = True

            except PermissionError:
                print("[red]Error: Cannot access file. Waiting for 1 second...[/red]")
                time.sleep(1)

        return outfile

    def cvtData(self):
        """If data is an image, converts BGR to RGB."""
        if self.is_image:
            self.Data = cv2.cvtColor(self.Data, cv2.COLOR_BGR2RGB)

    def overlayImpact(self, specimen):
        if self.is_figure:
            assert self.add_data is not None, "add_data must not be None."
            assert 'ax' in self.add_data, "add_data must contain 'axs' to overlay impact point."

        impact_pos = specimen.get_impact_position()
        size_fac = specimen.calculate_px_per_mm()

        # overlay impact point
        orientation_image = np.zeros_like(specimen.get_fracture_image(True), dtype=np.uint8)
        orientation_image = cv2.cvtColor(orientation_image, cv2.COLOR_BGR2RGBA)
        orientation_image[:, :, 3] = 0

        cv2.circle(orientation_image,
                (np.array(impact_pos) * size_fac).astype(np.uint32),
                np.min(orientation_image.shape[:2]) // 50,
                (255, 0, 0, 255),
                -1)

        cv2.circle(orientation_image,
                (np.array(impact_pos) * size_fac).astype(np.uint32),
                np.min(orientation_image.shape[:2]) // 50,
                (255, 255, 255, 255),
                5)

        if self.is_figure:
            ax = self.add_data['ax']
            ax.imshow(orientation_image, alpha=1)
        elif self.is_image:
            mask = orientation_image[:, :, 3] > 0
            # copy all data from the orientation image to the output image, skip alpha channel
            self.Data[mask] = orientation_image[mask][:, :-1]

    def clear(self):
        if self.is_figure:
            plt.close(self.Data)

class State:
    """Contains static variables that are set during execution of a command."""
    start_time: float = time.time()
    progress: ProgWrapper = None
    debug: bool = False

    sub_outpath: str = ""
    "Current sub-path for current command."
    additional_output_path: str = None
    "Current additional path for output."
    sub_specimen: str = ""
    "Current specimen, if any is analysed."

    current_subcommand: str = ""
    "The current subcommand."
    clear_output: bool = False
    "Clear the output directory of similar files when finalizing."
    to_temp: bool = False
    "Redirect all output to the temp folder."
    output_name_mod: str = ""
    "A modifier to append to all output file names."
    save_plots: bool = False
    "When doing image processing this will save the plots instead of just showing them."
    figasimgonly: bool = False
    "Save plots as images only."
    maximum_specimen: int = 1000
    "Globaly define the maximum amount of specimen that can be loaded."
    no_open: bool = False
    "Disables automatic opening from State.output."
    no_out: bool = False
    "Disables all output via State.output."
    kwargs: dict = {}

    __progress_started: bool = False

    __checkpoint_data: dict = None

    __suboutputfolder: str = None

    @classmethod
    def set_arg(cls, key, value):
        if hasattr(cls, key):
            setattr(cls, key, value)
        else:
            State.kwargs[key] = value

    def has_progress():
        return State.__progress_started

    def start_progress():
        State.progress.start()
        State.__progress_started = True

    def stop_progress():
        State.progress.stop()
        State.__progress_started = False

    def output_nopen(
        object: Figure | npt.ArrayLike,
        *names: str,
        spec: Any = None,
        force_delete_old=False,
        no_print=False,
        to_additional=False,
        cvt_rgb=False,
        figwidth='row1',
        mods=None, # modifiers
    ):
        State.output(
            object,
            *names,
            open=False,
            spec=spec,
            force_delete_old=force_delete_old,
            no_print=no_print,
            to_additional=to_additional,
            cvt_rgb=cvt_rgb,
            figwidth=figwidth,
            mods=mods
        )

    def output(
        object: Figure | npt.ArrayLike | StateOutput,
        *path_and_name: str,
        open=True,
        spec: Any = None,
        force_delete_old=False,
        no_print=False,
        to_additional=False,
        cvt_rgb=False,
        figwidth=None,
        mods: list[str]=None, # modifiers
        resize_factor: float = 1,
        close_fig: bool = True,
        force_open: bool = False,
        force_out: bool = False,
        **kwargs
    ):
        """
        Saves an object to a file and opens it.

        Args:
            object (Figure | numpy.ndarray): The object to save.
            path_and_name (str): Path parts to use to create output file.
            open (bool): Whether to open the output file after saving.
            spec (Outputtable): The specimen to save the object to.
            force_delete_old (bool): Whether to delete the old output file if it exists.
            no_print (bool): Whether to suppress printing of output file paths.
            to_additional (bool): Whether to save the output file to the additional output path.
            cvt_rgb (bool): Whether to convert the image data to RGB.
            figwidth (float): The width of the figure.
            mods (list[str]): A list of modifiers to append to the output file name.

        Remarks:
            The current subcommand will be appended to the last path part.
        """
        if State.no_out and not (force_out or force_open):
            print("[yellow]Warning: Output is disabled. Use 'force_out=True' to override.[/yellow]")
            return


        if 'override_name' in kwargs:
            warning("'override_name' is deprecated. Use 'names' instead.")

        if 'override_figwidth' in State.kwargs:
            info(f"Overriding figwidth with {State.kwargs['override_figwidth']}.")
            if isinstance(object, StateOutput):
                object.FigWidth = State.kwargs['override_figwidth']
            else:
                figwidth = State.kwargs['override_figwidth']


        assert not isinstance(object, tuple), "Object passed to State.output must not be a tuple."
        # make sure that all path parts are strings
        for x in path_and_name:
            assert type(x) == str, "Path parts must be strings."

        # warn, if figwidth is ignored
        if isinstance(object, StateOutput):
            if figwidth is not None:
                warning("'figwidth' is ignored when passing StateOutput")
        else:
            if not is_image(object):
                assert figwidth is not None, "figwidth must be set when passing a figure."
            if figwidth is None:
                figwidth = 'img'

            object = StateOutput(object, figwidth)

        # use the subcommand as file name if no name is passed
        if len(path_and_name) == 0:
            path_and_name = (State.current_subcommand,)

        # convert the image data to RGB, if needed
        if cvt_rgb:
            object.cvtData()

        # we need a list because we might change the last element
        path_and_name = list(path_and_name)

        # modify name with mods
        if mods is not None:
            for mod in mods:
                path_and_name[-1] += f'_{mod}'

        if State.output_name_mod != "":
            path_and_name[-1] += f'_{State.output_name_mod}'

        # If a spec is passed, use its output functions to save
        #   the object to the specimen path as well.
        # The name of this output file does not need the specimen ID,
        #   as it is already in the path.
        if isinstance(spec, Outputtable) and not State.to_temp:
            specimen_output_funcs = spec.get_output_funcs()
            for key, func in specimen_output_funcs.items():
                if key in State.sub_outpath:
                    spec_out = object.save(func(path_and_name[-1]), resize_factor=resize_factor)
                    if not no_print:
                        print(SAVE_FORMAT.format('SPECIMEN', spec_out))

        # From here, we need the specimen name to save the object.
        if spec is not None and hasattr(spec, 'name'):
            path_and_name[-1] = spec.name + "_" + path_and_name[-1]

        # save to COMMAND output
        out = State.get_output_file(*path_and_name)
        out = object.save(out, resize_factor=resize_factor)
        if not no_print:
            print(SAVE_FORMAT.format('COMMAND', os.path.join(State.sub_outpath, os.path.basename(out))))


        # save to ADDITIONAL output
        if (additional_path := State.additional_output_path) is not None \
            and to_additional and not State.to_temp:
            add_path = os.path.join(additional_path, path_and_name[-1])
            add_path = object.save(add_path, resize_factor=resize_factor)
            if not no_print:
                print(SAVE_FORMAT.format('ADDITIONAL', add_path))

        if close_fig:
            object.clear()

        # open file
        if (open and not State.no_open) or force_open:
            subprocess.Popen(['start', '', '/b', out], shell=True)

    def pointoutput(subfolder):
        """Sets the subfolder for output files."""
        if State.__suboutputfolder is not None:
            print(f"[yellow]Warning: Subfolder '{State.__suboutputfolder}' is already set. Overwriting with '{subfolder}'[/yellow]")
        State.__suboutputfolder = subfolder

    def get_input_dir():
        """Gets the input directory, which is subfolder tree resembling the command structure."""
        # sub_outpath might be set to custom output path, join will take the last valid path start
        p = os.path.join(general.out_path, State.sub_outpath)

        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))

        return p

    def get_output_dir():
        """Gets the output directory, which is subfolder tree resembling the command structure."""
        # sub_outpath might be set to custom output path, join will take the last valid path start
        if State.to_temp:
            p = os.path.join(tempfile.gettempdir(), State.sub_outpath)
        else:
            p = os.path.join(general.out_path, State.sub_outpath)

        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))

        return p

    def get_general_outputfile(*names: str, **kwargs):
        """Gets an output file path.

        Returns:
            str: path
        """
        p = os.path.join(general.out_path, *names)

        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))

        return p

    def get_output_file(*names: str, **kwargs):
        """Gets an output file path.

        Returns:
            str: path
        """
        if State.__suboutputfolder is not None:
            p = os.path.join(State.get_output_dir(), State.__suboutputfolder, *names)
        else:
            p = os.path.join(State.get_output_dir(), *names)

        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))


        # force_delete_old = 'force_delete_old' in kwargs and kwargs['force_delete_old']

        # fname = os.path.splitext(os.path.basename(p))[0]
        # ext = os.path.splitext(p)[1]
        # if os.path.exists(p):
        #     if State.clear_output or force_delete_old:
        #         for file in os.listdir(os.path.dirname(p)):
        #             if file.startswith(fname):
        #                 os.remove(os.path.join(os.path.dirname(p), file))

        #     # count files with same name
        #     count = 1
        #     while os.path.exists(p):
        #         count += 1
        #         p = os.path.join(State.get_output_dir(), *names[:-1], f'{fname} ({count}){ext}')


        return p

    def checkpoint(**kwargs):
        """Saves data to a checkpoint file."""
        if State.__checkpoint_data is None:
            State.__checkpoint_data = {}

        for k in kwargs:
            State.__checkpoint_data[k] = kwargs[k]
    def checkpoint_save():
        """Saves data to a checkpoint file. Is called from __main__ at the end of the program."""
        tmpFile = os.path.join(tempfile.gettempdir(), State.current_subcommand + ".checkpoint")
        with open(tmpFile, 'wb') as f:
            pickle.dump(State.__checkpoint_data, f)
    def checkpoint_clear():
        tmpFile = os.path.join(tempfile.gettempdir(), State.current_subcommand + ".checkpoint")
        if os.path.exists(tmpFile):
            os.remove(tmpFile)

    def from_checkpoint(key: str, default):
        """Loads data from a checkpoint file."""
        if State.__checkpoint_data is None:
            tmpFile = os.path.join(tempfile.gettempdir(), State.current_subcommand + ".checkpoint")
            if os.path.exists(tmpFile):
                with open(tmpFile, 'rb') as f:
                    print("[yellow]Loaded checkpoint data from previous run.[/yellow]")
                    State.__checkpoint_data = pickle.load(f)
            else:
                State.__checkpoint_data = {}
                return default

        if State.__checkpoint_data is not None and key in State.__checkpoint_data:
            return State.__checkpoint_data[key]
        else:
            return default