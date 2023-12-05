from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any, Callable, ClassVar, List, TypeVar

import cv2
import numpy as np
from rich import print
from rich.progress import track
from fracsuite.core.coloring import rand_col

from fracsuite.core.imageprocessing import simplify_contour
from fracsuite.core.kernels import ObjectKerneler
from fracsuite.core.outputtable import Outputtable
from fracsuite.core.preps import PreprocessorConfig, defaultPrepConfig
from fracsuite.core.progress import get_spinner
from fracsuite.core.region import RectRegion
from fracsuite.core.splinter import Splinter
from fracsuite.general import GeneralSettings
from fracsuite.helpers import checkmark, find_file
from fracsuite.scalper.scalpSpecimen import ScalpSpecimen, ScalpStress

from spazial import k_test, l_test

general: GeneralSettings = GeneralSettings.get()

class SpecimenException(Exception):
    """Exception for specimen related errors."""
    pass

class Specimen(Outputtable):
    """ Container class for a specimen. """
    adjacency_file: str = "adjacency.pkl"
    "Name of the adjacency info file."
    KEY_FRACINTENSITY: str = "frac_intensity"
    "Key for the fracture intensity in the simdata file."
    KEY_HCRADIUS: str = "hc_radius"
    "Key for the hard core radius in the simdata file."
    KEY_LAMBDA: str = "lambda"
    "Key for the fracture intensity parameter in the simdata file."

    @property
    def splinters(self) -> list[Splinter]:
        "Splinters on the glass ply."
        assert self.__splinters is not None, f"Splinters are empty. Specimen {self.name} not loaded?"
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

    @property
    def break_lambda(self):
        "Fracture intensity parameter."
        return self.simdata.get(Specimen.KEY_LAMBDA, None)

    @property
    def break_rhc(self):
        "Hard core radius."
        return self.simdata.get(Specimen.KEY_HCRADIUS, None)

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

        if self.has_splinters:
            self.load_splinters()
            self.has_adjacency = any([len(s.adjacent_splinter_ids) > 0 for s in self.splinters])
        elif log_missing_data:
            print(f"Could not find splinter file for '{self.name}'. Create it using [green]fracsuite splinters gen[/green].")


        if self.has_scalp:
            self.load_scalp()
        elif log_missing_data:
            print(f"Could not find scalp file for '{self.name}'. Create it using the original scalper project and [green]fracsuite.scalper[/green].")

        self.loaded = True

    def print_loaded(self):
        name = f"'{self.name}'"
        print(f"Loaded {name:>30} (Scalp: {checkmark(self.has_scalp)}, "
                f"Splinters: {checkmark(self.has_splinters)}).")


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
        "Path to the splinter output folder."
        self.__splinters_file_legacy = find_file(self.splinters_path, "splinters_v1.pkl")
        self.splinters_file = find_file(self.splinters_path, "splinters_v2.pkl")
        "File that contains splinter information."
        self.has_splinters = self.splinters_file is not None
        "States wether there is a file with splinter information or not."
        self.has_adjacency = False
        "States wether adjacency information are present or not."


        self.simdata_path = self.get_splinter_outfile("simdata.json")
        "Path to the simulation data file."
        if os.path.exists(self.simdata_path):
            with open(self.simdata_path, "r") as f:
                self.simdata = json.load(f)
        else:
            self.simdata = {}
            self.__save_simdata()

        # init done, load data
        if not lazy:
            self.load(log_missing)

    def set_setting(self, key, value):
        self.settings[key] = value
        self.__save_settings()

    def update_simdata(self, key: str, value):
        self.simdata[key] = value
        self.__save_simdata()

    def __save_simdata(self):
        with open(self.simdata_path, "w") as f:
            json.dump(self.simdata, f, indent=4)

    def __save_settings(self):
        with open(self.__cfg_path, "w") as f:
            json.dump(self.settings, f, indent=4)

    def get_output_funcs(self) -> dict[str, Callable[[str], str]]:
        paths = {
            'acc': self.get_acc_outfile,
            'splinter': self.get_splinter_outfile,
        }
        return paths

    def get_filled_image(self):
        filled_file = find_file(self.splinters_path, "img_filled.png")
        if filled_file is not None:
            return cv2.imread(filled_file)

    def get_label_image(self, as_rgb=True):
        label_file = find_file(self.fracture_morph_dir, "label*")
        if label_file is not None:
            return cv2.imread(label_file, cv2.IMREAD_GRAYSCALE if not as_rgb else cv2.IMREAD_COLOR)

    def get_fracture_image(self, as_rgb = True):
        """Gets the grayscale fracture image."""
        transmission_file = find_file(self.fracture_morph_dir, "*Transmission*")
        if transmission_file is not None:
            return cv2.imread(transmission_file, cv2.IMREAD_GRAYSCALE if not as_rgb else cv2.IMREAD_COLOR)
    def get_transmission_image(self, as_rgb = True):
        transmission_file = find_file(os.path.join(self.path, "anisotropy"), "*Transmission*")
        if transmission_file is not None:
            return cv2.imread(transmission_file, cv2.IMREAD_GRAYSCALE if not as_rgb else cv2.IMREAD_COLOR)

    def get_image_size(self):
        """Returns the size of the specimen in px."""
        img = self.get_fracture_image()
        return img.shape[0], img.shape[1]

    def get_real_size(self):
        """Returns the real size of the specimen in mm."""
        return self.settings['real_size_mm']

    def get_acc_outfile(self, name: str) -> str:
        return os.path.join(self.path, 'fracture', 'acceleration', name)

    def get_splinter_outfile(self, name: str, create = True) -> str:
        path = os.path.join(self.splinters_path, name)

        if create:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_fall_height_m(self):
        return self.settings['fall_height_m']

    def get_impact_position_name(self):
        return self.settings['break_pos']
    def get_impact_position(self, in_px = False, as_tuple = False):
        """
        Returns the impact position of the specimen in mm.
        Depends on the setting break_pos.

        Origin is top left corner!
        """
        factor = self.calculate_px_per_mm() if in_px else 1

        if self.settings['break_pos'] == "center":
            arr =  np.array((250,250)) * factor
        elif self.settings['break_pos'] == "corner":
            arr = np.array((50,50)) * factor
        else:
            raise Exception("Invalid break position.")

        if as_tuple:
            return tuple(arr.astype(int))
        else:
            return arr

    def get_prepconf(self) -> PreprocessorConfig | None:
        """
        Returns a prepconfig object or none, if not found.
        Can be created using 'fracsuite tester threshhold 8.100.Z.01'.
        """
        prep_file = find_file(self.splinters_path, "prep.json")
        if prep_file is not None:
            with open(prep_file, "r") as f:
                js = json.load(f)
                return PreprocessorConfig.from_json(js)

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

    def calculate_NperWindow(self, force_recalc: bool = False, D_mm: float = 50) -> int:
        """
        Calculates the mean splinter count in a observation window of D_mm*D_mm on the fracture image.

        Arguments:
            force_recalc (bool, optional): Recalculate. Defaults to False.
            D (float, optional): Window width in mm. Defaults to 50.

        Returns:
            int: The mean fracture intensity, Amount of Splinters in observation field N.
        """

        f_intensity = self.simdata.get(Specimen.KEY_FRACINTENSITY, None)
        if force_recalc or f_intensity is None:
            region = self.settings['real_size_mm']
            kernel = ObjectKerneler(
                region,
                self.splinters,
                collector=lambda x,r: x.in_region(r),
                skip_edge=True,
            )

            _, _, Z = kernel.run(
                lambda x: len(x),
                D_mm,
                50,
                mode="area",
                exclude_points=[self.get_impact_position()],
                fill_skipped_with_mean=True
            )
            f_intensity = int(np.mean(Z))
            self.update_simdata(Specimen.KEY_FRACINTENSITY, f_intensity)

        return f_intensity

    def calculate_break_lambda(self, force_recalc: bool = False, D_mm: float = 50.0) -> float:
        """
        Calculate the fracture intensity parameter according to the BREAK approach.

        Args:
            force_recalc (bool, optional): Recalculate. Defaults to False.

        Returns:
            float: The fracture intensity parameter in 1/mm².
        """
        lam = self.calculate_NperWindow(force_recalc=force_recalc, D_mm=D_mm) / D_mm**2
        self.update_simdata(Specimen.KEY_LAMBDA, lam)
        return lam

    def calculate_break_rhc(self, force_recalc: bool = False, d_max: float = 50) -> float:
        """
        Use ripleys K-Function and L-Function to estimate the hard core radius between centroids.

        Arguments:
            force_recalc (bool, optional): Recalculate. Defaults to False.
            d_max (float, optional): Maximum distance in mm. Defaults to 50.

        Returns:
            float: The hard core radius in mm.
        """

        r1 = self.simdata.get(Specimen.KEY_HCRADIUS, None)

        if force_recalc or r1 is None:
            all_centroids = np.array([s.centroid_px for s in self.splinters])
            pane_size = self.get_image_size()
            x2,y2 = l_test(all_centroids, pane_size[0]*pane_size[1], d_max)
            min_idx = np.argmin(y2)
            r1 = x2[min_idx]

            self.update_simdata(Specimen.KEY_HCRADIUS, r1)

        return r1

    def calculate_px_per_mm(self, realsize_mm: None | tuple[float,float] = None):
        """Returns the size factor of the specimen. px/mm."""
        realsize = realsize_mm if realsize_mm is not None else self.settings['real_size_mm']
        assert realsize is not None, "Real size not found."
        assert realsize[0] > 0 and realsize[1] > 0, "Real size must be greater than zero."

        if realsize_mm is not None:
            self.set_setting('real_size_mm', realsize_mm)


        frac_img = self.get_fracture_image()
        assert frac_img is not None, "Fracture image not found."
        return frac_img.shape[0] / realsize[0]

    def kfun(self, max_d = 50):
        """
        Calculate the K function for a range of distances.

        Arguments:
            max_d: Maximum distance in mm.

        Returns:
            tuple[d,K(d)]: Tuple of distances and K(d) values.
        """
        all_centroids = np.array([s.centroid_px for s in self.splinters])
        pane_size = self.get_image_size()
        x2,y2 = k_test(all_centroids, pane_size[0]*pane_size[1], max_d)
        return x2,y2

    def lfun(self, max_d = 50):
        """
        Calculates the L function for a range of distances.

        Arguments:
            max_d: Maximum distance in mm.

        Returns:
            tuple[d,L(d)]: Tuple of distances and L(d) values.
        """

        all_centroids = np.array([s.centroid_px for s in self.splinters])
        pane_size = self.get_image_size()
        x2,y2 = l_test(all_centroids, pane_size[0]*pane_size[1], max_d)
        return x2,y2

    def get_splinters_in_region(self, region: RectRegion) -> list[Splinter]:
        in_region = []
        for s in self.splinters:
            if region.is_point_in(s.centroid_px):
                in_region.append(s)

        return in_region

    def calculate_nfifty(self, centers, size):
        area = float(size[0] * size[1])
        nfifty = 0.0
        for center in centers:
            nfiftyi = self.calculate_esg_norm(center, size)[0]
            nfifty += nfiftyi

        return nfifty / len(centers)

    def calculate_esg_norm(
        self: Specimen,
        norm_region_center: tuple[int, int] = (400, 400),
        norm_region_size: tuple[int, int] = (50, 50),
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
        # count splinters in norm region
        for s in self.splinters:
            sc = s.in_region_exact((x1, y1, x2, y2))
            if sc == 1:
                s_count += 1
            if sc > 0.5:
                s_count += 0.5
                splinters_in_region.append(s)

        return s_count, splinters_in_region

    def __get_energy(self):
        t0 = self.scalp.measured_thickness
        return self.__get_energy_density() * t0 * 1e-3 # thickness in mm

    def __get_energy_density(self):
        nue = 0.23
        E = 70e6
        return 1e6/5 * (1-nue)/E * self.scalp.sig_h ** 2

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
        self.__U_d = self.__get_energy_density()
        self.__U = self.__get_energy()

    def load_splinters(self, file = None):
        assert self.has_splinters, f"Splinters not found in specimen {self.name}."

        if file is None:
            file = self.splinters_file

        with open(file, "rb") as f:
            self.__splinters = pickle.load(f)

    @staticmethod
    def get(name: str | Specimen, load: bool = True) -> Specimen:
        """Gets a specimen by name. Raises exception, if not found."""
        if isinstance(name, Specimen):
            return name

        path = os.path.join(general.base_path, name)
        if not os.path.isdir(path):
            raise SpecimenException(f"Specimen '{name}' not found.")

        return Specimen(path, lazy=not load)

    @staticmethod
    def get_all(names: list[str] | str | Specimen | list[Specimen] | None = None, load = False) \
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
            return Specimen.get_all_by(lambda x: True, lazyload=not load)
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
    @staticmethod
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