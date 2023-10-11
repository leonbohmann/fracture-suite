from __future__ import annotations
import sys

from fracsuite.core.progress import get_spinner
from fracsuite.scalper.scalpSpecimen import ScalpSpecimen, ScalpStress
from fracsuite.core.splinter import Splinter
from fracsuite.tools.helpers import checkmark, find_file
from fracsuite.tools.state import State
from fracsuite.tools.general import GeneralSettings

import cv2
import numpy as np
from matplotlib.figure import Figure
from rich import print
from rich.progress import Progress, track


import json
import os
import pickle
import re
from typing import Any, Callable, ClassVar, List, TypeVar


general: GeneralSettings = GeneralSettings.get()

class SpecimenException(Exception):
    """Exception for specimen related errors."""
    pass

class Specimen:
    """ Container class for a specimen. """

    @property
    def splinters(self) -> list[Splinter]:
        "Splinters on the glass ply."
        assert self.__splinters is not None, "Splinters are empty. Specimen not loaded?"
        return self.__splinters

    @property
    def scalp(self) -> ScalpSpecimen:
        "Scalp analysis."
        assert self.__scalp is not None, "Scalp is empty. Specimen not loaded?"
        return self.__scalp

    @property
    def break_pos(self):
        "Break position of the specimen."
        assert "break_pos" in self.settings, "break_pos not in settings."
        return self.settings["break_pos"]

    @property
    def break_mode(self):
        "Break mode of the specimen."
        assert "break_mode" in self.settings, "break_mode not in settings."
        return self.settings["break_mode"]

    @property
    def fall_height_m(self):
        "Fall height in meters."
        assert "fall_height_m" in self.settings, "fall_height_m not in settings."
        return self.settings["fall_height_m"]

    @property
    def settings(self):
        return self.__settings


    path: str
    "Specimen folder."
    name: str
    "Specimen name."
    __settings: dict[str, Any]
    "Settings of the specimen."

    nue: ClassVar[float] = 0.23
    "Poisson's ratio of the specimen."
    E: ClassVar[float] = 70e9
    "Young's modulus of the specimen."

    @property
    def sig_h(self):
        "Measured pre-stress of the specimen."
        assert self.loaded, "Specimen not loaded."
        return self.__sigma_h

    @property
    def measured_thickness(self):
        "Measured thickness of the specimen."
        assert self.loaded, "Specimen not loaded."
        return self.__measured_thickness

    @property
    def U_d(self):
        "Strain energy density of the specimen."
        assert self.loaded, "Specimen not loaded."
        return self.__U_d

    @property
    def U(self):
        "Strain Energy of the specimen."
        assert self.loaded, "Specimen not loaded."
        return self.__U

    def load(self, log_missing_data: bool = False):
        """Load the specimen lazily."""

        if self.loaded:
            print("[red]Specimen already loaded.")

        if self.__splinters_file:
            self.__load_splinters()
        elif log_missing_data:
            print(f"Could not find splinter file for '{self.name}'. Create it using [green]fracsuite.splinters[/green].")

        if self.__scalp_file is not None:
            self.__load_scalp()
        elif log_missing_data:
            print(f"Could not find scalp file for '{self.name}'. Create it using the original scalper project and [green]fracsuite.scalper[/green].")

        self.loaded = True

    def print_loaded(self):
        name = f"'{self.name}'"
        print(f"Loaded {name:>30} ({checkmark(self.has_scalp)}, "
                f"{checkmark(self.has_splinters)}).")


    def __init__(self, path: str, log_missing = True, lazy = False):
        """Create a new specimen.

        Args:
            path (str): Path of the specimen.
        """

        self.__splinters: list[Splinter] = None
        self.__scalp: ScalpSpecimen = None
        self.__sigma_h: ScalpStress = ScalpStress.default()
        self.__measured_thickness: float = np.nan
        self.__U_d: float = np.nan
        self.__U: float = np.nan

        self.loaded: bool = False
        "Whether the specimen is loaded or not."
        self.has_splinters: bool = False
        "Whether the specimen can load splinters or not."
        self.has_scalp: bool = False
        "Whether the specimen can load a scalp or not."
        self.boundary: str = ""
        "Boundary condition of the specimen."
        self.nom_stress: int = 0
        "Nominal stress of the specimen."
        self.thickness: int = 0
        "Thickness of the specimen."
        self.nbr: int = 0
        "Number of the specimen."
        self.comment: str = ""
        "Comment of the specimen."

        self.acc_file: str = ""
        "Path to the acceleration file."

        self.path = path
        self.__settings = {
            "break_mode": "punch",
            "break_pos": "corner",
            "fall_height_m": 0.07,
            "real_size_mm": (500,500)
        }

        self.__cfg_path = os.path.join(path, "config.json")
        if not os.path.exists(self.__cfg_path):
            with open(self.__cfg_path, "w") as f:
                json.dump(self.__settings, f, indent=4)
        else:
            with open(self.__cfg_path, "r") as f:
                sets = json.load(f)
                for k,v in sets.items():
                    self.__settings[k] = v

        # get name from path
        self.name = os.path.basename(os.path.normpath(path))

        # get thickness from name
        if self.name.count(".") == 3:
            vars = self.name.split(".")
            self.thickness = int(vars[0])
            self.nom_stress = int(vars[1])

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
        self.__scalp_file = find_file(scalp_path, "scalp_data.pkl")
        self.has_scalp = self.__scalp_file is not None

        # splinters requisites
        self.fracture_morph_dir = os.path.join(self.path, "fracture", "morphology")
        self.has_fracture_scans = os.path.exists(self.fracture_morph_dir) \
            and find_file(self.fracture_morph_dir, "*.bmp") is not None
        self.splinters_path = os.path.join(self.path, "fracture", "splinter")
        self.__splinters_file = find_file(self.splinters_path, "splinters.pkl")
        self.__config_file = find_file(self.splinters_path, "config.pkl")
        self.has_splinters = self.__splinters_file is not None
        self.has_splinter_config = self.__config_file is not None
        self.has_config = self.__config_file is not None

        if not lazy:
            self.load(log_missing)

    def set_setting(self, key, value):
        self.settings[key] = value
        self.__save_settings()

    def __save_settings(self):
        with open(self.__cfg_path, "w") as f:
            json.dump(self.settings, f, indent=4)

    def put_acc_output(self, object, override_name = None):
        """Saves the object to the acc folder."""
        self.__put_output(object, self.get_acc_outfile, override_name)

    def put_splinter_output(self, object: Figure | np.ArrayLike, override_name = None):
        """Saves the object to the specimen folder."""
        self.__put_output(object, self.get_splinter_outfile, override_name)

    def __put_output(self, object: Figure | np.ArrayLike, name_func, override_name = None):
        name = override_name or State.current_subcommand

        # check how to save object
        if isinstance(object, tuple):
            if isinstance(object[0], Figure):
                out_name = name_func(name + "." + general.plot_extension)
                object[0].savefig(out_name, dpi=300)
        elif isinstance(object, Figure):
            out_name = name_func(name + "." + general.plot_extension)
            object.savefig(out_name, dpi=300)
        elif type(object).__module__ == np.__name__:
            out_name = name_func(name + "." + general.image_extension)
            cv2.imwrite(out_name, object)
        else:
            raise Exception("Object must be a matplotlib figure or a numpy array.")

        # success, start process
        print(f"Saved to '{out_name}'.")

    def get_filled_image(self):
        filled_file = find_file(self.splinters_path, "img_filled.png")
        if filled_file is not None:
            return cv2.imread(filled_file)

    def get_fracture_image(self):
        transmission_file = find_file(self.fracture_morph_dir, "*Transmission*")
        if transmission_file is not None:
            return cv2.imread(transmission_file)

    def get_acc_outfile(self, name: str) -> str:
        return os.path.join(self.path, 'fracture', 'acceleration', name)

    def get_splinter_outfile(self, name: str) -> str:
        return os.path.join(self.splinters_path, name)

    def get_impact_position(self):
        """
        Returns the impact position of the specimen in mm.
        Depends on the setting break_pos.
        """
        if self.settings['break_pos'] == "center":
            return np.array((250,250))
        elif self.settings['break_pos'] == "corner":
            return np.array((50,50))

        raise Exception("Invalid break position.")

    def get_size_factor(self):
        """Returns the size factor of the specimen. mm/px."""
        realsize = self.settings['real_size_mm']
        if realsize is None:
            return 1

        frac_img = self.get_fracture_image()
        assert frac_img is not None, "Fracture image not found."
        return realsize[0] / frac_img.shape[0]

    def __get_energy(self):
        t0 = self.scalp.measured_thickness
        return self.__get_energy_density() * t0

    def __get_energy_density(self):
        nue = 0.23
        E = 70e6
        return 1e6/5 * (1-nue)/E * self.scalp.sig_h ** 2

    def __load_scalp(self, file = None):
        """Loads scalp data. Make sure to access self.__scalp until the load method returns. """
        if not self.has_scalp:
            return
        if file is None:
            file = self.__scalp_file

        # load the scalp
        self.__scalp = ScalpSpecimen.load(file)

        # then perform calculations
        self.__measured_thickness = self.__scalp.measured_thickness
        self.__sigma_h = self.__scalp.sig_h
        self.__U_d = self.__get_energy_density()
        self.__U = self.__get_energy()

    def __load_splinters(self, file = None):
        if not self.has_splinters:
            return

        if file is None:
            file = self.__splinters_file

        with open(file, "rb") as f:
            self.__splinters = pickle.load(f)

    def get(name: str | Specimen, load: bool = True) -> Specimen:
        """Gets a specimen by name. Raises exception, if not found."""
        if isinstance(name, Specimen):
            return name

        path = os.path.join(general.base_path, name)
        if not os.path.isdir(path):
            raise SpecimenException(f"Specimen '{name}' not found.")

        return Specimen(path, lazy=not load)

    def get_all(names: list[str] | str | Specimen | list[Specimen] | None = None) \
        -> List[Specimen]:
        """
        Get a list of specimens by name. Raises exception, if any is not found.

        If names=None and no name_filter, all specimens are returned.

        Args:
            names(list[str]): List of specimen names.
            name_filter(str): Filter for the specimen names.
        """
        specimens: list[Specimen] = []

        if names is None:
            return Specimen.get_all_by(lambda x: True)
        elif isinstance(names, Specimen):
            return [names]
        elif isinstance(names, list) and len(names) > 0 and isinstance(names[0], Specimen):
            return names
        elif isinstance(names, str) and "*" in names:
            name_filter = names.replace(".", "\.").replace("*", ".*")
            return Specimen.get_all_by(
                lambda x: re.match(name_filter, x.name) is not None,
            )
        elif isinstance(names, str):
            names = [names]
        elif isinstance(names, list) and (len(names) == 1 and "*" in names[0]):
            name_filter = names[0].replace(".", "\.").replace("*", ".*")
            filter = re.compile(name_filter)
            return Specimen.get_all_by(
                lambda x: filter.search(x.name) is not None
            )

        # this is quicker than to call get_all_by because only the names are loaded
        for name in track(names, description="Loading specimens...", transient=True):
            dir = os.path.join(general.base_path, name)
            specimen = Specimen.get(dir, load=True)
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
                    lazyload: bool = False,
                    sortby: Callable[[Specimen], Any] = None) -> list[_T1]:
        """
        Loads specimens with a decider function.
        Iterates over all specimens in the base path.
        """

        def load_specimen(spec_path, decider, value) -> Specimen | None:
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

            specimen = Specimen(spec_path, log_missing=False, lazy=lazyload)

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

        with get_spinner('Loading specimens...') as prog:
            prog.set_total(max_iter)

            for dir in directories:
                spec = load_specimen(dir, decider, value)
                prog.advance()

                if spec is not None:
                    data.append(spec)
                    prog.set_description(f'Loaded {len(data)} specimens...')
                    spec.print_loaded()
                if len(data) >= max_n:
                    break

        print(f"Loaded {len(data)} specimens.")

        if len(data) == 0:
            raise SpecimenException("No specimens found.")

        if sortby is not None:
            data = sorted(data, key=sortby)

        return data