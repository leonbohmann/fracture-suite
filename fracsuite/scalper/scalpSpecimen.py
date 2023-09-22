from __future__ import annotations


import base64
from collections import defaultdict
import os
import pickle
from typing import Tuple
import numpy as np
from rich import print
import xml.etree.ElementTree as ET

from fracsuite.scalper.scalp_stress import calculate_simple


SCALP_ANGLE = 71.5                          # scalp laser angle
SCALP_C = 2.72e-6                           # fracturing index
SWITCH_REMOVE_INVALID_LOCATIONS = False     # delete invalid measurement locations
                                            #   (invalid measurements are not used for stress calculation)
SWITCH_DISPLAY_MOHR_CIRCLE = False
class ScalpProject:
    """
    Main object that is used to wrap a scalp project file (.scp).

    Contains a list of all measurements and a list of specimens.
    """

    filepath: str
    measurements: list[Measurement]
    specimens: list[ScalpSpecimen]

    def __init__(self, path: os.PathLike):

        # print(f"Creating project from file: {path}")
        self.filepath: str = path
        self.measurements: list[Measurement] = []

        # read the scalp xml file
        with open(path, "r") as file:
            tree = ET.parse(file)
            root = tree.getroot()
            # each measurement has to be parsed into a measurement object
            for measurement_node in root.findall(".//Measurements/Measurement"):
                self.measurements.append(Measurement.from_node(measurement_node))


        # group measurements to specimens
        specimen_measurements: dict[str, list[Measurement]] = defaultdict(list)
        for ms in self.measurements:
            specimen_measurements[ms.specimen].append(ms)

        # create a list of all specimens for easy access
        self.specimens = [ScalpSpecimen(name, measurements) for name, measurements in specimen_measurements.items()]


class ScalpStress(float):
    """ Object, that contains information about the stress state of a specimen. """
    __value: float
    __deviation: float
    __n_points: int

    @property
    def value(self):
        return self.__value

    @property
    def deviation(self):
        return self.__deviation

    @property
    def n_points(self):
        return self.__n_points

    def __new__(self, value, *args):
        return float.__new__(self, value)

    def __init__(self, value, deviation, n_points):
        self.__value = value
        self.__deviation = deviation
        self.__n_points = n_points

    @classmethod
    def default(cls):
        return cls(np.nan, np.nan, np.nan)

class ScalpSpecimen:
    """ Object, that contains information about a specimen. """
    name: str
    "Name of the specimen."
    measurementlocations: list[MeasurementLocation]
    "List of all measurement locations on this specimen."
    measurements: list[Measurement]
    "List of all measurements on this specimen."
    invalid: bool = False
    "If true, the specimen is invalid and will not be used for calculation."

    measured_thickness: float
    "Measured thickness of the specimen."
    sig_h: ScalpStress
    "Stress state of the specimen."

    nue: float = 0.23
    "Poisson's ratio of the glass."
    E: float = 72.0e3
    "Young's modulus of the glass."

    def __init__(self, name: str, measurements: list[Measurement]):
        self.invalid = False
        self.measurements = measurements

        self.name = name

        # print(f"Extracting info for specimen {name}")

        # group measurements to locations
        locations_measurements: dict[str, list[Measurement]] = defaultdict(list)
        for ms in measurements:
            locations_measurements[ms.location].append(ms)

        self.measurementlocations = \
            [MeasurementLocation(loc, name, x) \
                for loc, x in locations_measurements.items()]

        # invalid locations can be removed, if the constant is set to TRUE
        #   even if the locations are not removed, they will not be used for calculation
        if SWITCH_REMOVE_INVALID_LOCATIONS:
            # remove invalids
            for loc in self.measurementlocations:
                if loc.invalid:
                    self.measurementlocations.remove(loc)


        # get homogenous stress level of the specimen
        self.__calc_homogenous_princ_stress()

        # get thickness data from measurements
        thickness = 0.0
        for meas in measurements:
            if thickness != 0.0 and thickness != meas.measured_thickness:
                print(f"WARNING: Different thickness measurements in specimen {name}.")

            thickness = meas.measured_thickness

        self.measured_thickness = thickness


    def __calc_homogenous_princ_stress(self):
        """
        Calculates the homogenous principal stress by taking the
        mean value of all measured locations. Invalid locations (those
        with missing directions like ur, m or ol) will not contribute
        to the mean value.
        """
        stresses1 = []
        stresses2 = []
        for location in self.measurementlocations:
            # dont use invalid locations for stress measurement
            if location.invalid:
                print(f"[yellow]WARNING[/yellow]: Invalid location {location.location_name} in specimen {self.name}.")
                continue

            stresses1.append(location.stress[0])
            stresses2.append(location.stress[1])

        stresses = stresses1 + stresses2

        # if no valid location is found, the specimen is invalid
        if len(stresses) == 0:
            print(f"[yellow]WARNING[/yellow]: No valid locations in specimen {self.name}.")
            self.sig_h = ScalpStress.default()
            self.invalid = True
            return

        sig_h = np.mean(stresses1 + stresses2)
        sig_h_dev = np.std(stresses1 + stresses2)
        n_points = len(self.measurementlocations)

        self.sig_h = ScalpStress(sig_h, sig_h_dev, n_points)

    def save(self, dir):
        """Saves the specimen to a directory.

        Args:
            dir (str): Directory to save to.
        """
        with open(os.path.join(dir, "scalp_data.pkl"), "wb") as file:
            pickle.dump(self, file)

    def load(file_path) -> ScalpSpecimen:
        with open(file_path, "rb") as file_path:
            return pickle.load(file_path)

    def to_file(self, file):
        file.write(f'{self.measured_thickness:<35}\t# thickness [mm]\n')
        file.write(f'{self.sig_h:<35} ({self.sig_h.n_points})\t# homogenous stress [MPa]\n')
        file.write(f'{self.sig_h.deviation:<35}\t# hom stress std-dev [-]\n')
        file.write('\n')

        file.write(f'# {"name":18}\t{"sig_1":<20}\t{"sig_2":<20}\n')

        for loc in self.measurementlocations:
            file.write(f'{loc.location_name:20}\t{loc.stress[0]:<20}\t{loc.stress[1]:<20}\n')


    def write_measurements(self, output_directory: str | None, output_extension):
        """Writes the measurements to an output directory.

        Args:
            output_directory (str): Output path
            output_extension (str): Output extension
        """


        for measurement in self.measurements:

            # build output path and create it if necessary
            output_path = os.path.join(output_directory, measurement.specimen, 'scalp')
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, f"{measurement.name}.{output_extension}")
            with open(output_path, "w") as file:
                measurement.to_file(file)


        output_path = os.path.join(output_directory, self.name, 'scalp')
        output_fpath = os.path.join(output_path, f"{self.name}_stress.txt")
        with open(output_fpath, "w") as file:
            self.to_file(file)

        self.save(output_path)

    def __str__(self):
        return self.name

class MeasurementLocation:
    """
    This class bundles all measurements (in different directions) at a location
    on the glass ply.
    Typically, this class contains the X, Y and XY direction. If any of the directions
    is missing, the location is marked as "invalid" and the measurement will not
    contribute to the calculation of the stresses on the ply.
    """
    location_name: str
    specimen_name: str
    measurements: list[Measurement]
    directions_stress: dict[str, Measurement]
    stress: Tuple[float,float]
    invalid: bool

    def __init__(self, location_name: str, specimen_name: str, measurements: list[Measurement]):
        self.location_name = location_name
        self.specimen_name = specimen_name
        self.measurements = measurements

        # save measurements in dictionary
        self.directions_stress = {}
        self.directions_stress["X"] = None
        self.directions_stress["Xy"] = None
        self.directions_stress["Y"] = None

        for meas in self.measurements:
            self.directions_stress[meas.orientation] = meas.stress


        # All 3 directions must be present in order to calculate stresses
        #   For every missing direction, stress state can be entered manually.
        #   If the stress state is not known, the stress of this location can
        #   can not contribute to the stress calculation of the specimen.
        if not self.__has_direction("X") \
            or not self.__has_direction("XY") \
                or not self.__has_direction("Y"):
            print(f"> WARNING: {specimen_name}-{location_name} is missing one or more directions!")

            for missing_dir in self.__get_missing_directions():
                sig = input(f"Enter stress for {specimen_name}-{location_name}, Direction {missing_dir} [sigma]: ")

                # if no stress is applied, the location is invalid
                if sig == "":
                    print("Measurement location invalid!")
                    self.invalid = True
                    self.stress = (np.nan, np.nan)
                    return

                sig = float(sig)
                self.directions_stress[missing_dir] = [sig]
                self.invalid = False
        else:
            self.invalid = False

        # this is only reached, if all directions are present as measurements
        # or if missing measurements are replaced by manual measurement inputs
        self.stress = calculate_simple(self.directions_stress['X'], \
            self.directions_stress['Y'], self.directions_stress['Xy'], \
                np.deg2rad(SCALP_ANGLE), \
                    SCALP_C, SWITCH_DISPLAY_MOHR_CIRCLE)

    def __has_direction(self, dir: str) -> bool:
        for meas in self.measurements:
            if meas.orientation.lower() == dir.lower():
                return True

        return False

    def __get_direction(self, dir: str) -> Measurement | None:
        for meas in self.measurements:
            if meas.orientation.lower() == dir.lower():
                return meas

        return None

    def __get_missing_directions(self) -> list[str]:
        dirs = ["X", "Xy", "Y"]
        missing = []

        for dir in dirs:
            if not self.__has_direction(dir):
                missing.append(dir)

        return missing

class Measurement:
    """
    Contains all raw data from a SCALP Measurement.
    """

    ## Attributes
    name: str
    specimen: str                   # z.B. 4.70.Z.1
    location: str                   # ul, m, or
    orientation: str                # x, y, xy
    measured_thickness: float

    ## Data
    data: list[tuple[str,float,float]] # Stress Value Name, Value, Sigma
    depth: list[float]
    stress: list[float]
    active: list[bool]
    fit: list[bool]
    retardation: list[list[float]] # list of all measured retardations in a Measurement

    def __init__(self, specimen: str,  loc: str, orientation: str, spec_thickness:float, data: list[tuple[str,float,float]], \
        depth: list[float], stress: list[float], active: list[float], fit: list[float], \
            ret: list[list[float]]):


        if specimen.count(".") == 3:
            vals = specimen.split(".")
            vals1 = vals[3].split("-")
            if len(vals1) > 1:
                vals[3] = f'{int(vals1[0]):02d}-{vals1[1]}'
            else:
                vals[3] = f'{int(vals[3]):02d}'

            specimen = f"{vals[0]}.{vals[1]}.{vals[2]}.{vals[3]}"

        self.name = f"{specimen}_{loc}-{orientation}"
        self.specimen = specimen
        self.orientation = orientation
        self.measured_thickness = spec_thickness

        self.data = data
        self.depth = depth
        self.stress = stress
        self.active = active
        self.fit = fit
        self.retardation = ret
        self.location = loc

    def from_node(measurement_node) -> Measurement:
        """Creates a Measurement object from a xml Node.

        Args:
            measurement_node (Node): XML Node retrieved from node.find("x").

        Returns:
            Measurement: A measurement object.
        """
        specimen = measurement_node.find("Specimen")
        if specimen is None:
            return None

        specimen_name = specimen.attrib.get("Name")
        specimen_note = specimen.attrib.get("Note")
        specimen_dir = specimen.attrib.get("Direction").replace("-axis", "")
        specimen_thickness = float(specimen.attrib.get("GlassThickness"))


        stress_distribution = measurement_node.find("StressDistribution")
        if stress_distribution is None:
            return None

        curve_depth = np.frombuffer(base64.b64decode(stress_distribution.find("Depth").text), dtype=np.float32)
        curve_stress = np.frombuffer(base64.b64decode(stress_distribution.find("Stress").text), dtype=np.float32)
        curve_active = np.frombuffer(base64.b64decode(stress_distribution.find("Active").text), dtype=np.bool_)
        curve_fit = np.frombuffer(base64.b64decode(stress_distribution.find("Fit").text), dtype=np.int8)

        # retardation is not fitted and averaged over the measured data
        #   So, every measured retardation is read and saved as a list of curves, so that
        #   we can later decide how to use it
        retardation_data = measurement_node.findall(".//MeasuredData/Data/StressDistribution/Retardation")
        # print(retardation_data)
        retardation_data_list: list[list[float]] = []

        for data in retardation_data:

            # retardation_node = data.find("StressDistribution").find("Retardation")
            # if retardation_node is None:
            #     print(data)
            #     continue

            retardation_text = data.text

            curve_retardation = np.frombuffer(\
                base64.b64decode(retardation_text),\
                    dtype=np.float32)
            retardation_data_list.append(curve_retardation)

        stress_data = []

        stress_values = measurement_node.find("StressValues")
        if stress_values is not None:
            for stress_value in stress_values.iter():
                sv_name = stress_value.tag
                avg = float(stress_value.attrib.get("Col1", 0))
                stddev = float(stress_value.attrib.get("Col2", 0))
                stress_data.append((sv_name, avg, stddev))

        return Measurement(specimen_name,  specimen_note, specimen_dir, specimen_thickness, stress_data, curve_depth, curve_stress, \
            curve_active, curve_fit, retardation_data_list)

    def to_file(self, file):
        file.write(f"# Measurement: {self.name}\n")
        for data in self.data:
            file.write(f"# {data[0]:<20} = {data[1]:<10.4f} (+/- {data[2]:<10.4f})\n")
        file.write("# Depth                Stress               Active               Fit\n")
        for i in range(len(self.depth)):
            file.write(f"{self.depth[i]:<20.4f}{self.stress[i]:<20.4f}{self.active[i]:<20}{self.fit[i]:<20}\n")
