from __future__ import annotations

import json
from fracsuite.core.logging import debug, warning
import os
import pickle
import re
from stat import S_IREAD, S_IRGRP, S_IROTH
from typing import Any, Callable, List, TypeVar

import cv2
import numpy as np
from rich import print
from rich.progress import track
from fracsuite.core.accelerationdata import AccelerationData
from fracsuite.core.anisotropy_images import AnisotropyImages
from fracsuite.core.coloring import rand_col

from fracsuite.core.imageprocessing import crop_matrix, crop_perspective, simplify_contour
from fracsuite.core.kernels import KernelerData, ObjectKerneler, rhc_kernel
from fracsuite.core.model_layers import DEFAULT_ANGLE_DELTA, DEFAULT_RADIUS_DELTA
from fracsuite.core.outputtable import Outputtable
from fracsuite.core.plotting import get_log_range
from fracsuite.core.preps import PreprocessorConfig, defaultPrepConfig
from fracsuite.core.progress import get_spinner
from fracsuite.core.region import RectRegion
from fracsuite.core.specimenregion import SpecimenRegion
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stochastics import khat_xy, lhat_xy, lhatc_xy
from fracsuite.general import GeneralSettings
from fracsuite.helpers import checkmark, find_file, find_files
from fracsuite.scalper.scalpSpecimen import ScalpSpecimen, ScalpStress
from fracsuite.core.mechanics import U as calc_U, Ud as calc_Ud

from fracsuite.core.specimenprops import SpecimenBreakMode, SpecimenBreakPosition, SpecimenBoundary
from fracsuite.state import State


general: GeneralSettings = GeneralSettings.get()



sensor_positions = {
    "corner": {
        1: (450, 50),
        # 2: (-500, -500), # can't really be excluded
        3: (250, 250),
        4: (450, 450),
        5: (50, 450),
        # 6: (-500, -500), # can't really be excluded
    },
    "center": {
        1: None, # has to be implemented
    },
}

class SpecimenException(Exception):
    """Exception for specimen related errors."""
    pass

def default(x, _default):
    return x if x is not None else _default

# the maximum distance for the L-Function in polar calculations
#   set this to None to calculate it automatically
CALC_DMAX = None
DMAX_K = 50
DMAX_L = 50


class Specimen(Outputtable):
    """ Container class for a specimen. """

    DAT_HCRADIUS: str = "hc_radius"
    "Key for the hard core radius in the simdata file."
    DAT_ACCEPTANCE_PROB: str = "acceptance_prob"
    "Key for the acceptance probability in the simdata file."
    DAT_LAMBDA: str = "lambda"
    "Key for the fracture intensity parameter in the simdata file."
    DAT_NFIFTY: str = "nfifty"
    "Key for the nfifty value in the simdata file."
    DAT_CRACKSURFACE: str = "crack_surface"
    "Key for the crack surface in the simdata file."
    DAT_BROKEN_IMMEDIATELY: str = "broken_immediately"
    "Key for the broken immediately flag in the simdata file."

    SET_BREAKMODE: str = "break_mode"
    "Break mode of the specimen (PUNCH, LASER, DRILL)."
    SET_BREAKPOS: str = "break_pos"
    "Break position of the specimen (CORNER,CENTER,EDGE)."
    SET_ACTUALBREAKPOS: str = "actual_break_pos"
    "Custom break position in (mm,mm)."
    SET_CBREAKPOSEXCL: str = "custom_break_pos_exclusion_radius"
    "Radius in mm to exclude from the break position."
    SET_EDGEEXCL: str = "custom_edge_exclusion_distance"
    "Distance in mm to exclude from the edge."
    SET_FALLHEIGHT: str = "fall_height_m"
    "Fall height in meters."
    SET_REALSIZE: str = "real_size_mm"
    "Real size of the specimen in mm."
    SET_FALLREPEAT: str = "fall_repeat"
    "Number of times the fallweight is dropped."
    SET_EXCLUDED_SENSOR_POSITIONS: str = "excluded_sensor_positions"
    "Excluded sensor positions. This setting is a list of integers giving the sensor number."
    SET_EXCLUDE_ALL_SENSORS: str = "exclude_all_sensors"
    "Excludes all sensors from the calculation."
    SET_EXCLUDED_POSITIONS: str = "excluded_positions"
    "Excluded positions. This setting is a list of tuples (x,y) giving the position in mm."
    SET_EXCLUDED_POSITIONS_RADIUS: float = "excluded_positions_radius"
    "Radius in mm to exclude from the excluded positions."

    SPLINTER_FILE_NAME = "splinters_v2.pkl"
    SCALP_DATA_FILENAME = "scalp_data.pkl"

    @staticmethod
    def setting_keys():
        # find all members that start with "SET_"
        ret = []
        for k in dir(Specimen):
            if k.startswith("SET_"):
                ret.append(getattr(Specimen, k))
        return ret

    @property
    def splinters(self) -> list[Splinter]:
        "Splinters on the glass ply."
        if self.has_splinters and self.__splinters is None:
            self.load_splinters()
            self.has_adjacency = any([len(s.adjacent_splinter_ids) > 0 for s in self.splinters])

        assert self.__splinters is not None, f"Splinters are empty. Specimen {self.name} not loaded?"
        return self.__splinters

    @property
    def allsplinters(self) -> list[Splinter]:
        "All splinters on the glass ply."
        if self.has_splinters and self.__allsplinters is None:
            self.load_splinters()
            self.has_adjacency = any([len(s.adjacent_splinter_ids) > 0 for s in self.splinters])
        return self.__allsplinters

    @property
    def scalp(self) -> ScalpSpecimen:
        "Scalp analysis."
        assert self.__scalp is not None, "Scalp is empty. Specimen not loaded?"
        return self.__scalp

    @property
    def break_pos(self) -> SpecimenBreakPosition:
        "Break position of the specimen."
        assert "break_pos" in self.settings, "break_pos not in settings."
        return SpecimenBreakPosition(self.settings["break_pos"])

    @property
    def break_mode(self) -> SpecimenBreakMode:
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

    @property
    def break_lambda(self):
        "Fracture intensity parameter."
        return self.simdata.get(Specimen.DAT_LAMBDA, None)

    @property
    def break_rhc(self):
        "Hard core radius."
        return self.simdata.get(Specimen.DAT_HCRADIUS, None)
    @property
    def broken_immediately(self):
        "Checks if the specimen was broken immediately after impactor hit."
        return self.simdata.get(Specimen.DAT_BROKEN_IMMEDIATELY, False)

    @property
    def mean_splinter_area(self):
        "Mean splinter area in mm²."
        assert self.splinters is not None, "Splinters not loaded."
        return np.mean([s.area for s in self.splinters])

    @property
    def mean_splinter_area_px(self):
        "Mean splinter area in px²."
        assert self.splinters is not None, "Splinters not loaded."
        return np.mean([s.area for s in self.splinters]) * self.calculate_px_per_mm()**2

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value
    path: str
    "Specimen folder."
    __name_store: str
    "Specimen name."
    __settings: dict[str, Any]
    "Settings of the specimen."

    @property
    def sig_h(self) -> float:
        "Measured pre-stress of the specimen."
        assert self.loaded, "Specimen not loaded."
        return self.__sigma_h

    @property
    def measured_thickness(self) -> float:
        "Measured thickness of the specimen."
        assert self.loaded, "Specimen not loaded."
        return self.__measured_thickness

    @property
    def U_d(self) -> float:
        "Strain energy density of the specimen in J/m³."
        assert self.loaded, "Specimen not loaded."
        return self.__U_d

    @property
    def U(self) -> float:
        "Strain Energy of the specimen in J/m²."
        assert self.loaded, "Specimen not loaded."
        return self.__U

    @property
    def crack_surface(self):
        "Total crack surface in mm². Calculated using pixel algorithm."
        if not hasattr(self, '_crack_surface') or self._crack_surface is None:
            self._crack_surface = self.simdata.get(Specimen.DAT_CRACKSURFACE, None)
        return self._crack_surface

    def set_crack_surface(self, value):
        self._crack_surface = value
        self.set_data(Specimen.DAT_CRACKSURFACE, value)

    @property
    def splinter_area(self):
        if not hasattr(self, '_splinter_area') or self._splinter_area is None:
            self._splinter_area = np.sum([s.area for s in self.splinters])
        return self._splinter_area

    @splinter_area.setter
    def splinter_area(self, value):
        self._splinter_area = value
        self.set_data('splinter_area', value)

    @property
    def accdata(self) -> AccelerationData:
        "Acceleration data."
        assert self.loaded, "Specimen not loaded."
        return self.__acc

    @property
    def has_fracture_scans(self):
        return self.__has_fracture_scans


    def pdf(self, binrange = None, return_binrange = False):
        # calculate the probability density function of this specimens splinters
        if binrange is None:
            binrange = get_log_range([s.area for s in self.splinters], 30)
            return_binrange = True
        areas = [s.area for s in self.splinters]

        bins, _ = np.histogram(areas, bins=binrange, density=True)

        if return_binrange:
            return bins, binrange

        return bins

    def cdf(self, binrange = None, return_binrange = False):
        # calculate the cumulative density function of this specimens splinters
        # calculate the probability density function of this specimens splinters
        if binrange is None:
            binrange = get_log_range([s.area for s in self.splinters], 30)
            return_binrange = True
        areas = [s.area for s in self.splinters]

        bins = np.histogram(areas, bins=binrange, density=True)

        if return_binrange:
            return bins, binrange

        return bins

    def load(self, log_missing_data: bool = False):
        """Load the specimen lazily."""

        if self.loaded:
            print("[red]Specimen already loaded.")

        if self.has_scalp:
            self.load_scalp()
        elif log_missing_data:
            print(f"Could not find scalp file for '{self.name}'. Create it using the original scalper project and [green]fracsuite.scalper[/green].")

        self.load_acc()

        self.loaded = True

    def print_loaded(self):
        name = f"'{self.name}'"

        print(f"Loaded {name:>15} (Ani: {checkmark(self.anisotropy.available)} , Scalp: {checkmark(self.has_scalp)} , "
                f"Splinters: {checkmark(self.has_splinters)} ) "
                    f': t={self.measured_thickness:>5.2f}mm, U={self.U:>7.2f}J/m², U_d={self.U_d:>9.2f}J/m³, σ_s={self.sig_h:>7.2f}MPa')

    def put_scalp_data(self, scalp: ScalpSpecimen):
        """Puts the scalp data into the specimen folder."""
        with open(os.path.join(self.path, "scalp", Specimen.SCALP_DATA_FILENAME), "wb") as f:
            pickle.dump(scalp, f)

    def put_fracture_image(self, img: np.ndarray):
        """Puts the fracture image into the specimen folder."""
        cv2.imwrite(os.path.join(self.fracture_morph_folder, "Transmission.bmp"), img)
        self.__has_fracture_scans = True


    def __init__(self, path: str, log_missing = True, load = False):
        """Create a new specimen.

        Args:
            path (str): Path of the specimen.
        """

        self.__splinters: list[Splinter] = None
        self.__allsplinters: list[Splinter] = None
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
        self.boundary: SpecimenBoundary = SpecimenBoundary.Unknown
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

        # set default settings (overwritten in the next step)
        self.__settings = {
            Specimen.SET_BREAKMODE: "punch",
            Specimen.SET_BREAKPOS: "corner",
            Specimen.SET_ACTUALBREAKPOS: None,
            Specimen.SET_CBREAKPOSEXCL: 20,
            Specimen.SET_EDGEEXCL: 10,
            Specimen.SET_FALLHEIGHT: 0.07,
            Specimen.SET_REALSIZE: (500,500),
            Specimen.SET_FALLREPEAT: 1,
            Specimen.SET_EXCLUDED_SENSOR_POSITIONS: [],
            Specimen.SET_EXCLUDE_ALL_SENSORS: False,
            Specimen.SET_EXCLUDED_POSITIONS: [],
            Specimen.SET_EXCLUDED_POSITIONS_RADIUS: 100,
        }

        # load settings from config and overwrite defaults
        self.__cfg_path = os.path.join(path, "config.json")
        if not os.path.exists(self.__cfg_path):
            with open(self.__cfg_path, "w") as f:
                json.dump(self.__settings, f, indent=4)
        else:
            with open(self.__cfg_path, "r") as f:
                sets = json.load(f)
                for k,v in sets.items():
                    if k in self.__settings:
                        self.__settings[k] = v
            # update settings file
            with open(self.__cfg_path, "w") as f:
                json.dump(self.__settings, f, indent=4)

        # get name from path
        self.name = os.path.basename(os.path.normpath(path))

        # get thickness from name
        if self.name.count(".") == 3:
            vars = self.name.split(".")
            self.thickness = int(vars[0])
            self.nom_stress = int(vars[1])

            if vars[2].isdigit():
                vars[2],vars[3] = vars[3], vars[2]

            self.boundary = SpecimenBoundary(vars[2])

            if "-" not in vars[3]:
                self.nbr = int(vars[3])
                self.comment = ""
            else:
                last_sec = vars[3].split("-")
                self.nbr = int(last_sec[0])
                self.comment = last_sec[1]

        # load acceleration
        acc_path = os.path.join(self.path, "fracture", "acceleration")
        self.acc_file = find_file(acc_path, f"{self.name}.bin")

        # scalp requisites
        self.scalp_folder = os.path.join(self.path, "scalp")
        self.__scalp_file = find_file(self.scalp_folder, Specimen.SCALP_DATA_FILENAME)
        self.has_scalp = self.__scalp_file is not None

        # splinters requisites
        self.fracture_morph_folder = os.path.join(self.path, "fracture", "morphology")
        self.__has_fracture_scans = os.path.exists(self.fracture_morph_folder) \
            and find_file(self.fracture_morph_folder, "*.bmp") is not None
        self.splinters_folder = os.path.join(self.path, "fracture", "splinter")
        "Path to the splinter output folder."
        self.__splinters_file_legacy = find_file(self.splinters_folder, "splinters_v1.pkl")
        self.splinters_file = find_file(self.splinters_folder, Specimen.SPLINTER_FILE_NAME)
        "File that contains splinter information."
        self.has_splinters = self.splinters_file is not None
        "States wether there is a file with splinter information or not."
        self.has_adjacency = False
        "States wether adjacency information are present or not."


        self.anisotropy_folder = os.path.join(self.path, "anisotropy")
        "Path to anisotropy scans."
        self.anisotropy = AnisotropyImages(self.anisotropy_folder)

        self.simdata_path = self.get_splinter_outfile("simdata.json")
        "Path to the simulation data file."
        if os.path.exists(self.simdata_path):
            with open(self.simdata_path, "r") as f:
                self.simdata = json.load(f)
        else:
            self.simdata = {}
            self.__save_simdata()

        # init done, load data
        if load:
            self.load(log_missing)

        self.layer_region = SpecimenRegion(DEFAULT_RADIUS_DELTA, DEFAULT_ANGLE_DELTA, self.get_impact_position(), self.get_real_size())

    def set_setting(self, key, value):
        """Set an experimental setting of the specimen. Use Specimen.SET_* constants."""
        self.settings[key] = value
        self.__save_settings()

    def get_setting(self, key, default):
        """Get an experimental setting of the specimen. Use Specimen.SET_* constants."""
        ret = self.settings.get(key, default)
        if ret is None:
            debug(f"Could not find {key} for {self.name}. Using default value: {default}.")
            ret = default

        return ret


    def set_data(self, key: str, value):
        """Set evaluated data of the specimen. Use Specimen.DAT_* constants."""
        self.simdata[key] = value
        self.__save_simdata()

    def __save_simdata(self):
        backup = None
        if os.path.exists(self.simdata_path):
            with open(self.simdata_path, "r") as f:
                backup = f.readlines()

        try:
            with open(self.simdata_path, "w") as f:
                json.dump(self.simdata, f, indent=4)
        except Exception as e:
            if backup is not None:
                with open(self.simdata_path, "w") as f:
                    f.writelines(backup)

            raise e

    def __save_settings(self):
        with open(self.__cfg_path, "r") as f:
            backup = f.readlines()

        try:
            with open(self.__cfg_path, "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            warning(f"Could not save settings for {self.name}.")
            with open(self.__cfg_path, "w") as f:
                f.writelines(backup)

            raise e

    def get_output_funcs(self) -> dict[str, Callable[[str], str]]:
        paths = {
            'acc': self.get_acc_outfile,
            'splinter': self.get_splinter_outfile,
        }
        return paths

    def get_filled_image(self):
        filled_file = find_file(self.splinters_folder, "img_filled.png")
        if filled_file is not None:
            return cv2.imread(filled_file)

    def get_label_image(self, as_rgb=True):
        label_file = find_file(self.fracture_morph_folder, "label*")
        if label_file is not None:
            return cv2.imread(label_file, cv2.IMREAD_GRAYSCALE if not as_rgb else cv2.IMREAD_COLOR)

    def get_fracture_image(self, as_rgb = True):
        """Gets the fracture image. Default is RGB."""
        transmission_file = find_file(self.fracture_morph_folder, "*transmission*")
        if transmission_file is not None:
            return cv2.imread(transmission_file, cv2.IMREAD_GRAYSCALE if not as_rgb else cv2.IMREAD_COLOR)

    def get_transmission_image(self, as_rgb = True):
        transmission_file = find_file(os.path.join(self.path, "anisotropy"), "*transmission*")
        if transmission_file is not None:
            return cv2.imread(transmission_file, cv2.IMREAD_GRAYSCALE if not as_rgb else cv2.IMREAD_COLOR)

    def get_image_size(self):
        """Returns the size of the specimen in px."""
        img = self.get_fracture_image()
        assert img is not None, f"Fracture image not found in {self.name}."

        return img.shape[1], img.shape[0]

    def get_real_size(self):
        """Returns the real size of the specimen in mm."""
        return self.settings[Specimen.SET_REALSIZE]

    def get_acc_outfile(self, name: str) -> str:
        return os.path.join(self.path, 'fracture', 'acceleration', name)

    def get_splinter_outfile(self, name: str, create = True) -> str:
        path = os.path.join(self.splinters_folder, name)

        if create:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_fall_height_m(self):
        return self.settings[Specimen.SET_FALLHEIGHT]

    def get_impact_position_name(self):
        return self.settings[Specimen.SET_BREAKPOS]

    def get_impact_position(self, in_px = False, as_tuple = False):
        """
        Returns the impact position of the specimen in mm.
        Depends on the setting break_pos.

        Origin is top left corner!
        """
        factor = self.calculate_px_per_mm() if in_px else 1.0

        if (arr1 := self.settings.get(Specimen.SET_ACTUALBREAKPOS, None)) is not None:
            arr = np.asarray(arr1) * factor
        else:
            arr = self.break_pos.default_position() * factor


        if as_tuple:
            return tuple(arr.astype(int))
        else:
            return arr

    def get_prepconf(self, warn=True) -> PreprocessorConfig | None:
        """
        Returns a prepconfig object or none, if not found.
        Can be created using 'fracsuite tester threshhold 8.100.Z.01'.
        """
        prep_file = find_file(self.splinters_folder, "prep.json")
        if prep_file is not None:
            with open(prep_file, "r") as f:
                js = json.load(f)
                return PreprocessorConfig.from_json(js)

        if warn:
            print("[yellow]No prep.json found. Using default.")

        return defaultPrepConfig

    def get_splinters_asarray(self, simplify: float = -1) -> np.ndarray:
        """
        Returns two arrays.

        First array (id): Contains id connectors [id](splinter).
        Second array (p): Contains points (x,y).

        Example:
            To get the splinter for point p, use id[p] and p[p].
        """
        id_list: list[tuple[int,int]] = []
        p_list: list[tuple[int,int]] = []

        for i, s in enumerate(self.splinters):

            contour = s.contour if simplify > 0 else simplify_contour(s.contour, simplify)

            for p in contour:
                p_list.append(p[0])
                id_list.append(i)

        return np.array(id_list), np.array(p_list)

    def calculate_intensity(self, force_recalc: bool = False, D_mm: float = 50) -> int:
        """
        Calculates the fracture intensity by running a kde over the whole pane domain.

        The fracture intensity is
        `lambda = N / A`
        where N is the number of splinters and A is the area of the window. Using a KDE
        this value is averaged over the whole pane domain.

        Arguments:
            force_recalc (bool, optional): Recalculate. Defaults to False.
            D_mm (float, optional): Window width in mm. Defaults to 50.

        Returns:
            float: The mean fracture intensity in 1/mm².
        """

        f_intensity = self.simdata.get(Specimen.DAT_LAMBDA, None)
        if force_recalc or f_intensity is None:
            _,_,Z,_ = self.calculate_2d(SplinterProp.INTENSITY, D_mm, 25)

            f_intensity = np.nanmean(Z)
            self.set_data(Specimen.DAT_LAMBDA, f_intensity)

        return f_intensity

    def calculate_break_params(self, force_recalc: bool = False, D_mm: float = 50.0):
        """
        Calculate the fracture intensity parameter according to the BREAK approach.

        Args:
            force_recalc (bool, optional): Recalculate. Defaults to False.
            D_mm (float, optional): Window width in mm. Defaults to 50.

        Returns:
            lambda, rhc, acc: The fracture intensity parameter in 1/mm², the hard core radius in mm and the acceptance probability.
        """
        lam = self.calculate_break_lambda(force_recalc, D_mm)
        rhc, acc = self.calculate_break_rhc_acc(force_recalc)
        return lam, rhc, acc


    def calculate_break_lambda(self, force_recalc: bool = False, D_mm: float = 50.0) -> float:
        """
        Calculate the fracture intensity parameter according to the BREAK approach.

        Args:
            force_recalc (bool, optional): Recalculate. Defaults to False.

        Returns:
            float: The fracture intensity parameter in 1/mm².
        """
        lam = self.calculate_intensity(force_recalc, D_mm)
        self.set_data(Specimen.DAT_LAMBDA, lam)
        return lam

    def calculate_break_rhc_acc(self, force_recalc: bool = False) -> float:
        """
        Use ripleys K-Function and L-Function to estimate the hard core radius between centroids. The maximum distance to calculate is estimated using
        the biggest splinter and assuming a minimum splinter width of 2mm. Making the maximum distance 1/4 of the
        biggest splinter area root.

        The ratio between


        Arguments:
            force_recalc (bool, optional): Recalculate. Defaults to False.
            d_max (float, optional): Maximum distance in mm. Defaults to 50.

        Returns:
            float, float: The hard core radius in mm and the acceptance probability.
        """

        rhc = self.simdata.get(Specimen.DAT_HCRADIUS, None)
        acceptance = self.simdata.get(Specimen.DAT_ACCEPTANCE_PROB, None)

        if force_recalc or rhc is None or acceptance is None:
            rhc,_ = rhc_kernel(self.splinters)
            # be careful, if no minimum is found rhc will be the last value of Lhat-d
            pane_size = self.get_real_size()

            # acceptance parameter is simply the ratio of the first minimum to the maximum distance
            acceptance = rhc / (np.sqrt(pane_size[0]**2 + pane_size[1]**2))

            self.set_data(Specimen.DAT_ACCEPTANCE_PROB, float(acceptance))
            self.set_data(Specimen.DAT_HCRADIUS, float(rhc))

        return rhc,acceptance

    def calculate_px_per_mm(self, realsize_mm: None | tuple[float,float] = None):
        """Returns the size factor of the specimen. px/mm."""
        realsize = realsize_mm if realsize_mm is not None else self.settings[Specimen.SET_REALSIZE]
        assert realsize is not None, "Real size not found."
        assert realsize[0] > 0 and realsize[1] > 0, "Real size must be greater than zero."

        if realsize_mm is not None:
            self.set_setting(Specimen.SET_REALSIZE, realsize_mm)

        frac_img = self.get_fracture_image()
        assert frac_img is not None, "Fracture image not found."
        return frac_img.shape[1] / realsize[0]

    def kfun(self):
        """
        Calculate the K function for a range of distances. The maximum distance to calculate is {DMAX_K}mm.

        Returns:
            tuple[d,K(d)]: Tuple of distances and K(d) values.
        """
        max_d = DMAX_K
        all_centroids = np.array([s.centroid_mm for s in self.splinters])
        pane_size = self.get_real_size()
        x2,y2 = khat_xy(all_centroids, pane_size[0], pane_size[1], max_d)
        return np.asarray(x2),np.asarray(y2)

    def lfun(self):
        """
        Calculates the L function for a range of distances. The maximum distance to calculate is estimated using
        the biggest splinter and assuming a minimum splinter width of 2mm. Making the maximum distance 1/4 of the
        biggest splinter area root.

        Returns:
            tuple[d,L(d)]: Tuple of distances and L(d) values.
        """
        max_d = DMAX_L
        all_centroids = np.array([s.centroid_mm for s in self.splinters])
        pane_size = self.get_real_size()
        x2,y2 = lhat_xy(all_centroids, pane_size[0], pane_size[1], max_d)
        return x2,y2

    def lcfun(self):
        """
        Calculates the centered L function for a range of distances. The maximum distance to calculate is estimated using
        the biggest splinter and assuming a minimum splinter width of 2mm. Making the maximum distance 1/4 of the
        biggest splinter area root.

        Returns:
            tuple[d,L(d)]: Tuple of distances and L(d) values.
        """
        max_d = DMAX_L
        all_centroids = np.array([s.centroid_mm for s in self.splinters])
        pane_size = self.get_real_size()
        x2,y2 = lhatc_xy(all_centroids, pane_size[0], pane_size[1], max_d)
        return x2,y2


    def get_splinters_in_region(self, region: RectRegion) -> list[Splinter]:
        in_region = []
        for s in self.splinters:
            if region.is_point_in(s.centroid_px):
                in_region.append(s)

        return in_region

    def calculate_nfifty_kde(self, D = 50.0, force_recalc=False):
        """
        Calculate the nfifty value using a KDE.

        The mean intensity value is calculated using `calculate_intensity`.
        The area of the window is calculated using `D**2`. `N50 = l * D**2`.

        Args:
            D (float, optional): Window width in mm. Defaults to 50.0.
            force_recalc (bool, optional): Forces a recalculation of the intensity, even if it has been calculated before. Defaults to False.

        Returns:
            float: The average number of splinters in a window of size D.
        """
        intensity = self.calculate_intensity(force_recalc, D)
        return intensity * D**2

    def calculate_nfifty_count(self, centers = [], size = (50,50), force_recalc=False, simple = False):


        if centers == [] or simple:
            centers = [(400,400)]

        nfifty = self.simdata.get(Specimen.DAT_NFIFTY, None)
        if nfifty is None or force_recalc:
            # area = float(size[0] * size[1])
            nfifty = 0.0
            for center in centers:
                nfiftyi = self.calculate_esg_norm(center, size)[0]
                nfifty += nfiftyi

            nfifty = nfifty / len(centers)
            self.set_data(Specimen.DAT_NFIFTY, nfifty)
            return nfifty
        else:
            return nfifty

    def calculate_esg_norm(
        self: Specimen,
        norm_region_center: tuple[int, int] = (400, 400),
        norm_region_size: tuple[int, int] = (50, 50),
        edge_width: int = 25,
    ) -> tuple[float, list[Splinter]]:
        """
        Count the number of splinters in a specified norm region.

        Splinters, are counted as follows, where a percentage of contour points lie within the region:
            - 100%: 1
            - 50-99%: 0.5
            - <50%: 0

        Args:
            specimen (Specimen): The specimen to analyze.
            norm_region_center (tuple[int, int], optional): Center of the norm region in mm. Defaults to (400, 400).
            norm_region_size (tuple[int, int], optional): Size of the norm region in mm. Defaults to (50, 50).

        Returns:
            tuple[float, list[Splinter]]: The count of splinters in the norm region and a list of splinters in that region.
        """
        x, y = norm_region_center
        w, h = norm_region_size
        x1 = x - w // 2
        x2 = x + w // 2
        y1 = y - h // 2
        y2 = y + h // 2

        f = self.calculate_px_per_mm()
        # transform to real image size
        x1 = int(x1 * f)
        x2 = int(x2 * f)
        y1 = int(y1 * f)
        y2 = int(y2 * f)

        s_count = 0
        splinters_in_region: list[Splinter] = []
        rs = self.get_real_size()
        # count splinters in norm region
        for s in self.splinters:
            # check if splinter is close to edge, if so, omit
            p = s.centroid_mm
            if p[0] < edge_width or p[0] > rs[0] - edge_width or p[1] < edge_width or p[1] > rs[1] - edge_width:
                continue

            sc = s.in_region_exact((x1, y1, x2, y2))
            if sc == 1:
                s_count += 1
            if sc > 0.5:
                s_count += 0.5
                splinters_in_region.append(s)

        return s_count, splinters_in_region

    def calculate_energy(self):
        t0 = self.scalp.measured_thickness
        # print('Thickness: ', t0)
        return calc_U(self.scalp.sig_h, t0)

    def calculate_energy_density(self):
        return calc_Ud(self.scalp.sig_h)

    def calculate_tensile_energy(self):
        t0 = self.scalp.measured_thickness
        # print('Thickness: ', t0)
        return self.calculate_tensile_energy_density() * t0 * 1e-3 # thickness in mm

    def calculate_tensile_energy_density(self):
        nue = 0.23
        E = 70e3
        return 1e6 * (128/125) * (1-nue)/E * (self.scalp.sig_h ** 2)

    def calculate_2d(
        self,
        prop: SplinterProp,
        kw: int = 50,
        n_points: int = 25,
        quadrat_count: bool = False,
        include_all_splinters: bool = False
    ):
        """
        Calculate a value in 2D.

        Uses a constant density estimator with a kernel width of kw in a `n_points x n_points` grid.

        Arguments:
            prop (SplinterProp): Property to calculate.
            kw (int, optional): Kernel width. Defaults to 50.
            n_points (int, optional): Number of points in the grid. Defaults to 25.
            quadrat_count (bool, optional): Use quadrat counting. Defaults to False. When using this
                the `n_points` parameter is used as the number of quadrats.

        Returns:
            tuple[X(n), Y(m), Values(n,m), Stddev(n,m)]
        """
        impact_position = self.get_impact_position()
        sz = self.get_real_size()

        if quadrat_count:
            kw = sz[0] / n_points
            debug(f"Using quadrat counting. kw is now {kw:.2f}mm")

        # create kerneler
        kerneler = ObjectKerneler(
            sz,
            self.splinters if not include_all_splinters else self.allsplinters,
        )

        X,Y,Z,Zstd = kerneler.window(
            prop,
            kw,
            n_points,
            impact_position,
            self.calculate_px_per_mm()
        )

        return X,Y,Z,Zstd

    def calculate_2d_polar(
        self,
        prop: SplinterProp,
        r_range_mm = None,
        t_range_deg = None,
        return_data = False
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray, KernelerData]:
        """
        Calculate a value in polar 2D.

        Returns:
            tuple[Radii(n), Angles(m), Values(n,m), Stddev(n,m)]
        """
        impact_position = self.get_impact_position()
        size = self.get_real_size()
        # create kerneler
        kerneler = ObjectKerneler(
            size,
            self.splinters,
            None,
            False
        )

        if t_range_deg is None:
            t_range_deg = self.layer_region.theta
        if r_range_mm is None:
            r_range_mm = self.layer_region.radii

        R,T,Z,Zstd,rData = kerneler.polar(
            prop,
            r_range_mm,
            t_range_deg,
            impact_position,
            self.calculate_px_per_mm(),
            return_data = True
        )
        T = np.radians(T)

        # data contains more information about the calculation
        if return_data:
            return R,T,Z,Zstd,rData

        return R,T,Z,Zstd

    def load_acc(self):
        """Loads the acceleration data."""
        if self.acc_file is None:
            return

        self.__acc = AccelerationData(self.acc_file)
        self.set_data(Specimen.DAT_BROKEN_IMMEDIATELY, self.__acc.broken_immediately)

    def load_scalp(self, file = None):
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
        self.__U_d = self.calculate_energy_density()
        self.__U = self.calculate_energy()

    def load_splinters(self, file = None):
        assert self.has_splinters, f"Splinters not found in specimen {self.name}."

        if file is None:
            file = self.splinters_file

        with open(file, "rb") as f:
            self.__splinters = pickle.load(f)

        # print(self.get_impact_position())


        realsz = self.get_real_size()
        # remove all splinters whose centroid is closer than 1 cm to the edge
        delta_edge = self.get_setting(Specimen.SET_EDGEEXCL, 10)
        self.__allsplinters = self.__splinters
        self.__splinters = [s for s in self.__allsplinters
                            if  delta_edge < s.centroid_mm[0] < realsz[0] - delta_edge
                            and delta_edge < s.centroid_mm[1] < realsz[1] - delta_edge]

        # or within a 2cm radius to the impact point
        delta_impact = self.get_setting(Specimen.SET_CBREAKPOSEXCL, 20)
        self.__splinters = [s for s in self.__splinters if np.linalg.norm(np.array(s.centroid_mm) - np.array(self.get_impact_position())) > delta_impact]

        # remove all splinters within 1cm of sensor positions
        excl_sensor_positions = self.settings.get(Specimen.SET_EXCLUDED_SENSOR_POSITIONS, [])

        if self.settings.get(Specimen.SET_EXCLUDE_ALL_SENSORS, False):
            excl_sensor_positions = [1,3,4,5]

        for pos in excl_sensor_positions:
            sensor_position = sensor_positions[self.break_pos][pos]

            # filter splinters
            self.__splinters = [s for s in self.__splinters if np.linalg.norm(np.array(s.centroid_mm) - np.array(sensor_position)) > 10]


        radius = self.settings.get(Specimen.SET_EXCLUDED_POSITIONS_RADIUS, [])
        for pos in self.settings.get(Specimen.SET_EXCLUDED_POSITIONS, []):
            # filter splinters
            self.__splinters = [s for s in self.__splinters if np.linalg.norm(np.array(s.centroid_mm) - np.array(pos)) > radius]


        if State.debug:
            print(f"Loaded {len(self.__allsplinters)} splinters.")
            print(f" > Filtered {len(self.__allsplinters) - len(self.__splinters)}")

    def transform_fracture_images(
        self,
        resize_only: bool = False,
        rotate_only: bool = False,
        rotate: bool = True,
        crop: bool = True,
        size_px: tuple[int, int] = (4000, 4000)
    ):
        """
        Transforms the fracture morphology images. This will check, if the images are
        transformed and skips them if so.

        Args:
            resize_only (bool, optional): Only resize the images. Defaults to False.
            rotate_only (bool, optional): Only rotate the images. Defaults to False.
            rotate (bool, optional): Rotate the images. Defaults to True.
            crop (bool, optional): Crop the images. Defaults to True.
            size (tuple[int, int], optional): Size of the images. Defaults to (500, 500).
        """
        path = self.fracture_morph_folder
        if not os.path.exists(path):
            raise Exception("Fracture morphology folder not found.")

        frac_imgs = find_files(path, '*.bmp')
        imgs = [(x, cv2.imread(x, cv2.IMREAD_GRAYSCALE)) for x in frac_imgs]

        if len(imgs) == 0:
            raise Exception("No fracture morphology images found.")

        img0path, img0 = [(x,y) for x, y in imgs if "Transmission" in x][0]
        _, M0 = crop_perspective(img0, size_px, False, True)

        for file, img in imgs:
            if not os.access(file, os.W_OK):
                print(f"Skipping '{os.path.basename(file)}', no write access.")
                continue

            print(f"'{os.path.basename(file)}'...", end="")

            if resize_only:
                img = cv2.resize(img, size_px)

            if not rotate_only and crop and not resize_only:
                img = crop_matrix(img, M0, size_px)

            if (rotate or rotate_only) and not resize_only:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(file, img)
            os.chmod(os.path.join(path, file), S_IREAD | S_IRGRP | S_IROTH)

            print("[green]OK")

        return img0path, img0

    @staticmethod
    def get(name: str | Specimen, load: bool = True, panic: bool = True, printout:bool=True) -> Specimen:
        """Gets a specimen by name. Raises exception, if not found."""
        if isinstance(name, Specimen):
            return name

        path = os.path.join(general.base_path, name)
        if not os.path.isdir(path):
            if panic:
                raise SpecimenException(f"Specimen '{name}' not found.")
            else:
                return None

        spec = Specimen(path, load=load)

        if printout:
            spec.print_loaded()
        return spec

    @staticmethod
    def get_all(names: list[str] | str | Specimen | list[Specimen] | None = None, load = True, max_n = None) \
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
            return Specimen.get_all_by(lambda x: True, load=load)
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

        if max_n is None:
            max_n = State.maximum_specimen

        # this is quicker than to call get_all_by because only the names are loaded
        for name in track(names, description="Loading specimens...", transient=True):
            if name.startswith("."):
                print(f"Skipping {name}.")
                continue

            dir = os.path.join(general.base_path, name)
            specimen = Specimen.get(dir, load=load)
            specimens.append(specimen)

            if len(specimens) == max_n:
                break

        if len(specimens) == 0:
            raise SpecimenException("No specimens found.")

        return specimens

    def __default_value(specimen: Specimen) -> Specimen:
        return specimen

    _T1 = TypeVar('_T1')
    @staticmethod
    def get_all_by( decider: Callable[[Specimen], bool],
                    value: Callable[[Specimen], _T1 | Specimen] = None,
                    max_n: int = None,
                    load: bool = False,
                    sortby: Callable[[Specimen], Any] = None) -> list[_T1] | list[Specimen]:
        """
        Loads specimens with a decider function.
        Iterates over all specimens in the base path.
        """
        if max_n is None:
            max_n = State.maximum_specimen

        # show yellow hint when not loading
        if not load:
            print("[yellow]Not loading data of specimens.")

        def load_specimen(spec_path, decider, value) -> Specimen | None:
            """Load a single specimen.

            Args:
                spec_path (str):        Path of the specimen.
                decider (func(bool)):   Decides, if the specimen should be loaded.
                value (func(Specimen)->Specimen):
                                        Function that can convert the specimen.

            Returns:
                Specimen | None: Specimen or None.
            """
            if not os.path.isdir(spec_path):
                return None

            if value is None:
                value = Specimen.__default_value

            specimen = Specimen(spec_path, log_missing=False, load=load)

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
                    if spec.name.startswith("."):
                        print(f"Skipping {spec.name}.")
                        continue

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

    @staticmethod
    def create(name: str, force_create = False) -> Specimen:
        path = os.path.join(general.base_path, name)

        if os.path.exists(path) and not force_create:
            raise SpecimenException(f"Specimen '{name}' already exists.")

        necessary_folders = [
            os.path.join(path, "fracture", "acceleration"),
            os.path.join(path, "fracture", "morphology"),
            os.path.join(path, "scalp"),
            os.path.join(path, "anisotropy"),
        ]

        for folder in necessary_folders:
            os.makedirs(folder)

        specimen = Specimen(path, log_missing=True)

        return specimen

    def plot_region_count(self, region_center, region_size, splinters_in_region, nfifty, frac_img = None):
        # calculate points
        x, y = region_center
        w, h = region_size
        x1 = x - w // 2
        x2 = x + w // 2
        y1 = y - h // 2
        y2 = y + h // 2
        f = self.calculate_px_per_mm()
        # transform to real image size
        x1 = int(x1 * f)
        x2 = int(x2 * f)
        y1 = int(y1 * f)
        y2 = int(y2 * f)

        if frac_img is None:
            frac_img = self.get_fracture_image()

        norm_filled_img = np.zeros((frac_img.shape[0], frac_img.shape[1], 3), dtype=np.uint8)
        print(norm_filled_img.shape)
        for s in splinters_in_region:
            clr = rand_col()
            cv2.drawContours(norm_filled_img, [s.contour], -1, clr, -1)


        # # get norm region from original image (has to be grayscale for masking)
        # norm_region_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        # cv2.rectangle(norm_region_mask, (x1,y1), (x2,y2), 255, -1)
        # # create image parts
        # normed_image = cv2.bitwise_and(norm_filled_img, norm_filled_img, mask=norm_region_mask)
        # normed_image_surr = self.original_image #cv2.bitwise_and(self.original_image, self.original_image, mask=norm_region_inv)
        # # add images together
        normed_image = cv2.addWeighted(frac_img, 1, norm_filled_img, 0.5, 0)
        cv2.rectangle(normed_image, (x1,y1), (x2,y2), (255,0,0), 5)

        # extract image part
        norm_region_detail = normed_image[y1-50:y2+50, x1-50:x2+50].copy()

        # extract overview image
        cv2.putText(normed_image, f'{nfifty}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,0,255), 20)
        # draw a circle of 100mm around the impactpoint
        impact_pos = self.get_impact_position(True).astype(int)
        print(impact_pos)

        annotations = np.zeros((normed_image.shape[0], normed_image.shape[1], 3), dtype=np.uint8)
        cv2.circle(annotations, impact_pos, int(100 * f), (0,0,255), -1)
        # mark all edges with 25mm wide rectangles
        cv2.rectangle(annotations, (0,0), (int(25 * f), annotations.shape[0]), (0,0,255), -1)
        cv2.rectangle(annotations, (annotations.shape[1]-int(25 * f),0), (annotations.shape[1], annotations.shape[0]), (0,0,255), -1)
        cv2.rectangle(annotations, (0,0), (annotations.shape[1], int(25 * f)), (0,0,255), -1)
        cv2.rectangle(annotations, (0,annotations.shape[0]-int(25 * f)), (annotations.shape[1], annotations.shape[0]), (0,0,255), -1)

        norm_region_overview = cv2.addWeighted(normed_image, 1, annotations, 0.5, 0)

        return norm_region_detail, norm_region_overview