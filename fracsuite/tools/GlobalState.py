import os
import subprocess
import time
import cv2
from matplotlib.figure import Figure
import numpy as np
from rich import print
from rich.progress  import Progress
import numpy.typing as npt

from fracsuite.core.progress import get_progress
from fracsuite.tools.general import GeneralSettings

general = GeneralSettings.get()

class GlobalState:
    """Contains static variables that are set during execution of a command."""
    start_time: float = time.time()
    progress: Progress = get_progress()
    debug: bool = False

    sub_outpath: str = ""
    "Current sub-path for current command."
    sub_specimen: str = ""
    "Current specimen, if any is analysed."

    current_subcommand: str = ""
    "The current subcommand."
    clear_output: bool = False
    "Clear the output directory of similar files when finalizing."

    __progress_started: bool = False

    def has_progress():
        return GlobalState.__progress_started

    def start_progress():
        GlobalState.progress.start()
        GlobalState.__progress_started = True

    def stop_progress():
        GlobalState.progress.stop()
        GlobalState.__progress_started = False

    def finalize(object: Figure | npt.ArrayLike, *names: str, override_name: str = None):
        """
        Saves an object to a file and opens it.

        Args:
            object (Figure | numpy.ndarray): The object to save.
            names (str | Specimen): Path parts to use to create output file.

        Remarks:
            The current subcommand will be appended to the last path part.
        """
        # append last subcommand to output path
        sep = "_" if len(names) > 0 else ""
        if len(names) == 0:
            names = [""]
            sep = ""

        names = list(names)

        if 'splinter' in GlobalState.sub_outpath:
            if callable(b := getattr(names[-1], 'put_splinter_output', None)):
                b(object)
        elif 'acc' in GlobalState.sub_outpath:
            if callable(b := getattr(names[-1], 'put_acc_output', None)):
                b(object)
        if hasattr(names[-1], 'name'):
            names[-1] = names[-1].name

        names = names[:-1] + [names[-1]+sep+GlobalState.current_subcommand]

        if override_name is not None:
            names[-1] = override_name

        # check how to save object
        if isinstance(object, tuple):
            if isinstance(object[0], Figure):
                out_name = GlobalState.get_output_file(*names, is_plot=True)
                object[0].savefig(out_name, bbox_inches='tight')
        elif isinstance(object, Figure):
            out_name = GlobalState.get_output_file(*names, is_plot=True)
            object.savefig(out_name, bbox_inches='tight')
        elif type(object).__module__ == np.__name__:
            out_name = GlobalState.get_output_file(*names, is_image=True)
            image = object
            f = np.max(image.shape[:2]) / general.output_image_maxsize

            h,w = image.shape[:2] / f
            w = int(w)
            h = int(h)

            image = cv2.resize(image, (w,h))
            cv2.imwrite(out_name, image)
            cv2.imwrite(out_name, object)
        else:
            raise Exception("Object must be a matplotlib figure or a numpy array.")

        # success, start process
        print(f"Saved to '{out_name}'.")
        subprocess.Popen(['start', '', '/b', out_name], shell=True)

    def get_output_file(*names, **kwargs):
        """Gets an output file path.

        Kwargs:
            is_plot (bool): If true, the plot extension is appended.
            is_image (bool): If true, the image extension is appended.
        Returns:
            str: path
        """
        names = list(names)
        if 'is_plot' in kwargs and kwargs['is_plot']:
            names[-1] = f'{GlobalState.sub_specimen}{names[-1]}.{general.plot_extension}'
        if 'is_image' in kwargs and kwargs['is_image']:
            names[-1] = f'{GlobalState.sub_specimen}{names[-1]}.{general.image_extension}'

        # sub_outpath might be set to custom output path, join will take the last valid path start
        p = os.path.join(general.out_path, GlobalState.sub_outpath, *names)
        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))

        fname = os.path.splitext(os.path.basename(p))[0]
        ext = os.path.splitext(p)[1]
        if os.path.exists(p):

            if GlobalState.clear_output:
                for file in os.listdir(os.path.dirname(p)):
                    if file.startswith(fname):
                        os.remove(os.path.join(os.path.dirname(p), file))

            # count files with same name
            count = 1
            while os.path.exists(p):
                count += 1
                p = os.path.join(general.out_path, GlobalState.sub_outpath, f'{fname} ({count}){ext}')



        return p