from __future__ import annotations

import json
import os
import pickle
import re
import time
from typing import Any, Callable, List, TypeVar

import cv2
import numpy as np
import typer
from pathos.pools import ProcessPool
from rich import print
from rich.progress import Progress, track, SpinnerColumn, TextColumn, TimeElapsedColumn
from fracsuite.core.progress import get_specimen_loader, ProgSpinner

from fracsuite.scalper.scalpSpecimen import ScalpSpecimen
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from fracsuite.splinters.splinter import Splinter
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import find_file

app = typer.Typer()

general = GeneralSettings.get()

# def fetch_specimens(specimen_names: list[str] | str | Specimen, path: str) -> list[Specimen] | Specimen | None:
#     """Fetch a list of specimens from a given path.

#     Args:
#         specimen_names (list[str]): The names of the specimens.
#         path (str): THe base path to the specimens.

#     Returns:
#         specimen_names is list[str]?
#             list[Specimen]: The list of specimens.
#         specimen_names is str?
#             Specimen: The specimen.
#         specimen_names is Specimen?
#             Specimen: The specimen.
#     """
#     if isinstance(specimen_names, Specimen):
#         return specimen_names

#     single_specimen = False
#     if not isinstance(specimen_names, list):
#         single_specimen = True
#         specimen_names = [specimen_names]

#     specimens: list[Specimen] = []
#     for name in specimen_names:
#         spec_path = os.path.join(path, name)

#         if not os.path.exists(spec_path):
#             continue

#         specimen = Specimen(spec_path, lazy=True)

#         specimen.lazy_load()
#         specimens.append(specimen)
#         print(f"Loaded '{name}'.")

#     if len(specimens) > 1 and not single_specimen:
#         return specimens
#     elif len(specimens) == 1 and single_specimen:
#         return specimens[0]
#     elif single_specimen:
#         return None

#     return []

class SpecimenException(Exception):
    """Exception for specimen related errors."""
    pass


class Specimen:
    """ Container class for a specimen. """

    splinters: list[Splinter] = None
    "Splinters on the glass ply."
    splinter_config: AnalyzerConfig = None
    "Splinter analysis configuration that can be used to rerun it."
    scalp: ScalpSpecimen = None
    "Scalp analysis."

    settings: dict[str, str] = \
    {
        "break_mode": "punch",
        "break_pos": "corner"
    }
    "Settings for the specimen."

    path: str
    "Specimen folder."
    name: str
    "Specimen name."

    loaded: bool = False
    "Whether the specimen is loaded or not."

    has_splinters: bool = False
    "Whether the specimen can load splinters or not."

    has_scalp: bool = False
    "Whether the specimen can load a scalp or not."

    boundary: str = ""
    "Boundary condition of the specimen."
    nom_stress: int = 0
    "Nominal stress of the specimen."
    thickness: float = 0
    "Thickness of the specimen."
    nbr: int = 0
    "Number of the specimen."
    comment: str = ""
    "Comment of the specimen."

    acc_file: str = ""
    "Path to the acceleration file."

    def load(self, log_missing_data: bool = False):
        """Load the specimen lazily."""

        if self.loaded:
            print("[red]Specimen already loaded.")

        if self.__splinters_file:
            self.__load_splinters()
        elif log_missing_data:
            print(f"Could not find splinter file for '{self.name}'. Create it using [green]fracsuite.splinters[/green].")

        if self.__config_file is not None:
            self.__load_splinter_config()
        elif log_missing_data:
            print(f"Could not find splinter config file for '{self.name}'. Create it using [green]fracsuite.splinters[/green].")

        if self.__scalp_file is not None:
            self.__load_scalp()
        elif log_missing_data:
            print(f"Could not find scalp file for '{self.name}'. Create it using the original scalper project and [green]fracsuite.scalper[/green].")

        self.loaded = True

    def __init__(self, path: str, log_missing = True, lazy = False):
        """Create a new specimen.

        Args:
            path (str): Path of the specimen.
        """
        self.path = path

        self.__cfg_path = os.path.join(path, "config.json")
        if not os.path.exists(self.__cfg_path):
            with open(self.__cfg_path, "w") as f:
                json.dump(self.settings, f, indent=4)
        else:
            with open(self.__cfg_path, "r") as f:
                self.settings = json.load(f)


        # get name from path
        self.name = os.path.basename(os.path.normpath(path))

        # get thickness from name
        if self.name.count(".") == 3:
            vars = self.name.split(".")
            self.thickness = float(vars[0])
            self.nom_stress = float(vars[1])

            if vars[2].isdigit():
                vars[2],vars[3] = vars[3], vars[2]

            self.boundary = vars[2]

            if "-" not in vars[3]:
                self.nbr = int(vars[3])
                self.comment = ""
            else:
                last_sec = vars[3].split("-")
                self.nbr = int(last_sec[0])
                self.comment = last_sec[1]
        # load acceleration
        acc_path = os.path.join(self.path, "fracture", "acceleration")
        self.acc_file = find_file(acc_path, "*.bin")

        # scalp requisites
        scalp_path = os.path.join(self.path, "scalp")
        self.__scalp_file = find_file(scalp_path, "*.pkl")
        self.has_scalp = self.__scalp_file is not None

        # splinters requisites
        self.fracture_morph_dir = os.path.join(self.path, "fracture", "morphology")
        self.has_fracture_scans = os.path.exists(self.fracture_morph_dir) \
            and find_file(self.fracture_morph_dir, "*.bmp") is not None
        self.splinters_path = os.path.join(self.path, "fracture", "splinter")
        self.__splinters_data_file = find_file(self.splinters_path, "splinters_data.json")
        self.__splinters_file = find_file(self.splinters_path, "splinters.pkl")
        self.__config_file = find_file(self.splinters_path, "config.pkl")
        self.has_splinters = self.__splinters_file is not None
        self.has_splinter_config = self.__config_file is not None
        self.has_config = self.__config_file is not None

        if self.__splinters_data_file is not None:
            with open(self.__splinters_data_file, "r") as f:
                self.splinters_data = json.load(f)
        else:
            self.splinters_data = {}

        if not lazy:
            self.load(log_missing)

    def set_setting(self, key, value):
        self.settings[key] = value
        self.__save_settings()

    def __save_settings(self):
        with open(self.__cfg_path, "w") as f:
            json.dump(self.settings, f, indent=4)

    def get_analyzer(self, cfg: AnalyzerConfig = None, progress: Progress = None, task = None):
        return Analyzer(cfg if cfg is not None else self.splinter_config, progress = progress, main_task=task)

    def get_filled_image(self):
        filled_file = find_file(self.splinters_path, "img_filled.png")
        if filled_file is not None:
            return cv2.imread(filled_file)

    def get_fracture_image(self):
        transmission_file = find_file(self.fracture_morph_dir, "*Transmission*")
        if transmission_file is not None:
            return cv2.imread(transmission_file)

    def get_splinter_outfile(self, name: str) -> str:
        return os.path.join(self.splinters_path, name)

    def get_impact_position(self):
        if self.settings['break_pos'] == "center":
            return (250,250)
        elif self.settings['break_pos'] == "corner":
            return (50,50)

        raise Exception("Invalid break position.")

    def __load_scalp(self, file = None):
        if not self.has_scalp:
            return
        if file is None:
            file = self.__scalp_file

        self.scalp = ScalpSpecimen.load(file)

    def __load_splinters(self, file = None):
        if not self.has_splinters:
            return

        if file is None:
            file = self.__splinters_file

        with open(file, "rb") as f:
            self.splinters = pickle.load(f)

    def __load_splinter_config(self, file = None):
        if not self.has_splinter_config:
            return

        if file is None:
            file = self.__config_file

        self.splinter_config = AnalyzerConfig.load(file)

    def get(name: str | Specimen, load: bool = True) -> Specimen:
        """Gets a specimen by name. Raises exception, if not found."""
        if isinstance(name, Specimen):
            return name

        path = os.path.join(general.base_path, name)
        if not os.path.isdir(path):
            raise SpecimenException(f"Specimen '{name}' not found.")

        return Specimen(path, lazy=not load)

    def get_all(names: list[str] = None, load: bool = True) -> List[Specimen]:
        """
        Get a list of specimens by name. Raises exception, if any is not found.

        If names=None and no name_filter, all specimens are returned.

        Args:
            names(list[str]): List of specimen names.
            load(bool): States, if the specimens should be loaded or not.
            name_filter(str): Filter for the specimen names.
        """
        specimens: list[Specimen] = []

        if names is None:
            return Specimen.get_all_by(lambda x: True, load=load)
        elif isinstance(names, str) or (names is not None and len(names) == 1 and "*" in names[0]):
            name_filter = names.replace(".", "\.").replace("*", ".*")
            filter = re.compile(name_filter)
            return Specimen.get_all_by(
                lambda x: filter.search(x.name) is not None,
                load=load
            )


        for name in track(names, description="Loading specimens...", transient=False):
            dir = os.path.join(general.base_path, name)
            specimen = Specimen.get(dir, load)
            specimens.append(specimen)

        if len(specimens) == 0:
            raise SpecimenException("No specimens found.")

        return specimens

    def __default_value(specimen: Specimen) -> Specimen:
        return specimen

    _T1 = TypeVar('_T1')
    def get_all_by( decider: Callable[[Specimen], bool],
                    value: Callable[[Specimen], _T1 | Specimen] = None,
                    max_n: int = 1000,
                    sortby: Callable[[Specimen], Any] = None,
                    load: bool = True) -> list[_T1]:
        """
        Loads specimens with a decider function.
        Iterates over all specimens in the base path.
        """

        def load_specimen(spec_path, load, decider, value) -> Specimen | None:
            """Load a single specimen.

            Args:
                spec_path (str):        Path of the specimen.
                decider (func(bool)):   Decides, if the specimen should be loaded.
                value (func(Specimen)->Specimen):
                                        Function that can convert the specimen.
                load (bool):            Do not load the specimen data.
                lazy_load (bool):       Instruct the function to load the specimen data afterwards.

            Returns:
                Specimen | None: Specimen or None.
            """
            if not os.path.isdir(spec_path):
                return None

            if value is None:
                value = Specimen.__default_value

            specimen = Specimen(spec_path, log_missing=False, lazy=not load)
            if not decider(specimen):
                return None

            return value(specimen)

        if value is None:
            def value(specimen: Specimen):
                return specimen

        directories = [os.path.join(general.base_path,x) for x in os.listdir(general.base_path)]
        directories = [x for x in directories if os.path.isdir(x)]

        data: list[Any] = []

        max_iter = len(directories)

        with get_specimen_loader('Loading specimens...') as p:
            p.set_total(max_iter)

            for dir in directories:
                spec = load_specimen(dir, load, decider, value)
                p.advance()

                if spec is not None:
                    data.append(spec)
                    p.set_description(f'Loaded {len(data)} specimens...')
                if len(data) >= max_n:
                    break

        print(f"Loaded {len(data)} specimens.")

        if len(data) == 0:
            raise SpecimenException("No specimens found.")

        if sortby is not None:
            data = sorted(data, key=sortby)

        return data

@app.command()
def sync():
    """Sync all specimen configs."""
    # iterate over all splinters
    for name in track(os.listdir(general.base_path), description="Syncing specimen configs...", transient=False):

        spec_path = os.path.join(general.base_path, name)
        if not os.path.isdir(spec_path):
            continue


        s = Specimen(spec_path, log_missing=False, lazy=True)


@app.command()
def export():
    """Export all specimen configs to a single excel file."""
    import xlsxwriter

    workbook_path = os.path.join(general.base_path, "summary1.xlsx")

    workbook = xlsxwriter.Workbook(workbook_path)

    # The workbook object is then used to add new
    # worksheet via the add_worksheet() method.
    worksheet = workbook.add_worksheet()


    worksheet.write(0, 0, "Boundary: A (allseitig), Z (zweiseitig), B (gebettet)")
    worksheet.write(1, 0, "Comment: B (Bohrung)")

    start_row = 10

    worksheet.write(start_row, 0, "Name")
    worksheet.write(start_row, 1, "Thickness")
    worksheet.write(start_row, 2, "Pre-Stress")
    worksheet.write(start_row, 3, "Boundary")
    worksheet.write(start_row, 4, "Nbr")
    worksheet.write(start_row, 5, "Comment")
    worksheet.write(start_row, 6, "Break-Mode")
    worksheet.write(start_row, 7, "Break-Position")
    worksheet.write(start_row, 8, "Real pre-stress")
    worksheet.write(start_row, 9, "(std-dev)")
    worksheet.write(start_row, 10, "Mean splinter size")


    row = start_row + 1
    for name in track(os.listdir(general.base_path), description="Syncing specimen configs...", transient=False):

        spec_path = os.path.join(general.base_path, name)
        if not os.path.isdir(spec_path):
            continue


        s = Specimen(spec_path, log_missing=False)
        # extract data
        worksheet.write(row, 0, s.name)
        worksheet.write(row, 1, s.thickness)
        worksheet.write(row, 2, s.nom_stress)
        worksheet.write(row, 3, s.boundary)
        worksheet.write(row, 4, s.nbr)
        worksheet.write(row, 5, s.comment)
        worksheet.write(row, 6, s.settings['break_mode'])
        worksheet.write(row, 7, s.settings['break_pos'])
        if s.has_scalp:
            worksheet.write(row, 8, s.scalp.sig_h)
            worksheet.write(row, 9, s.scalp.sig_h_dev)
        if s.has_splinters:
            worksheet.write(row, 10, s.splinters.get_mean_splinter_size())




        row += 1
        del s



    # Finally, close the Excel file
    # via the close() method.
    workbook.close()

    os.system(f'start {workbook_path}')


@app.command()
def list(setting: str = None, value: str = None):
    all = Specimen.get_all(load=False)
    print("Name\tSetting\tValue")
    for spec in all:

        if setting not in spec.settings:
            continue
        if value is not None and spec.settings[setting] != value:
            continue

        print(spec.name, end="")

        for s, k in spec.settings.items():

            print(f"\t{k}", end="")

        print()