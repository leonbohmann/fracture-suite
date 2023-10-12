import tempfile
from fracsuite.core.progress import get_progress
from fracsuite.tools.general import GeneralSettings


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


class State:
    """Contains static variables that are set during execution of a command."""
    start_time: float = time.time()
    progress: Progress = get_progress()
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
    __progress_started: bool = False

    def has_progress():
        return State.__progress_started

    def start_progress():
        State.progress.start()
        State.__progress_started = True

    def stop_progress():
        State.progress.stop()
        State.__progress_started = False



    def __save_object(object, dir, *sub_path):
        saved = False
        while not saved:
            try:
                # check how to save object
                if isinstance(object, tuple):
                    if isinstance(object[0], Figure):
                        out_path = os.path.join(dir, *sub_path) + f'.{general.plot_extension}'
                        object[0].savefig(out_path, dpi=300, bbox_inches='tight')
                elif isinstance(object, Figure):
                    out_path = os.path.join(dir, *sub_path) + f'.{general.plot_extension}'
                    object.savefig(out_path, dpi=300, bbox_inches='tight')
                elif type(object).__module__ == np.__name__:
                    out_path = os.path.join(dir, *sub_path) + f'.{general.image_extension}'
                    image = object
                    # f = np.max(image.shape[:2]) / general.output_image_maxsize

                    # h,w = image.shape[:2] / f
                    # w = int(w)
                    # h = int(h)

                    # image = cv2.resize(image, (w,h))
                    cv2.imwrite(out_path, image)
                else:
                    raise Exception("Object must be a matplotlib figure or a numpy array.")

                saved = True
                return out_path
            except Exception as e:
                print(e)
                print("[red]Error while saving. Waiting for 1 second...[/red]")
                time.sleep(1)
                continue

    def output_nopen(
        object: Figure | npt.ArrayLike,
        *names: str,
        force_delete_old=False,
        no_print=False,
        to_additional=False,
    ):
        State.output(
            object,
            *names,
            open=False,
            force_delete_old=force_delete_old,
            no_print=no_print,
            to_additional=to_additional
        )

    def output(
        object: Figure | npt.ArrayLike,
        *names: str,
        open=True,
        force_delete_old=False,
        no_print=False,
        to_additional=False,
        **kwargs
    ):
        """
        Saves an object to a file and opens it.

        Args:
            object (Figure | numpy.ndarray): The object to save.
            names (str | Specimen): Path parts to use to create output file.

        Remarks:
            The current subcommand will be appended to the last path part.
        """
        if 'override_name' in kwargs:
            print("[yellow]Warning: 'override_name' is deprecated. Use 'names' instead.[/yellow]")

        assert len(names) != 0, "No output names given."

        names = list(names)
        # file_name might be the specimen itself!
        file_name = names[-1]
        first = names[0]
        if hasattr(first, 'name'):
            names[0] = first.name
            file_name = first.name + "_" + State.current_subcommand
        if 'splinter' in State.sub_outpath:
            if callable(b := getattr(first, 'put_splinter_output', None)):
                b(object, State.current_subcommand)
        elif 'acc' in State.sub_outpath:
            if callable(b := getattr(first, 'put_acc_output', None)):
                b(object, State.current_subcommand)


        out = State.get_output_file(*names, force_delete_old=force_delete_old)
        out = State.__save_object(object, ".", out)
        # success, start process
        if not no_print:
            n = State.sub_outpath + '\\' + '\\'.join(names) + os.path.splitext(out)[1]
            print(f"Saved to '{n}'.")

        if (additional_path := State.additional_output_path) is not None \
            and to_additional and not State.to_temp:
            add_path = State.__save_object(object, additional_path, file_name)
            if not no_print:
                print(f" > Additional file to '{add_path}'.")



        if open:
            subprocess.Popen(['start', '', '/b', out], shell=True)

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

    def get_output_file(*names, **kwargs):
        """Gets an output file path.

        Kwargs:
            is_plot (bool): If true, the plot extension is appended.
            is_image (bool): If true, the image extension is appended.
            force_delete_old (bool): If true, all files with the same name will be deleted.
        Returns:
            str: path
        """
        names = list(names)
        if 'is_plot' in kwargs and kwargs['is_plot']:
            names[-1] = f'{State.sub_specimen}{names[-1]}.{general.plot_extension}'
        if 'is_image' in kwargs and kwargs['is_image']:
            names[-1] = f'{State.sub_specimen}{names[-1]}.{general.image_extension}'


        p = os.path.join(State.get_output_dir(), *names)

        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))


        force_delete_old = 'force_delete_old' in kwargs and kwargs['force_delete_old']

        fname = os.path.splitext(os.path.basename(p))[0]
        ext = os.path.splitext(p)[1]
        if os.path.exists(p):
            if State.clear_output or force_delete_old:
                for file in os.listdir(os.path.dirname(p)):
                    if file.startswith(fname):
                        os.remove(os.path.join(os.path.dirname(p), file))

            # count files with same name
            count = 1
            while os.path.exists(p):
                count += 1
                p = os.path.join(State.get_output_dir(), *names[:-1], f'{fname} ({count}){ext}')


        return p