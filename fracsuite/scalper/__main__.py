from __future__ import annotations

import argparse
import base64
import glob
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Tuple, TypeVar

import numpy as np

from fracsuite.scalper.scalp_stress import calculate_simple

descr = """
███████╗ ██████╗ █████╗ ██╗     ██████╗ ███████╗██████╗ 
██╔════╝██╔════╝██╔══██╗██║     ██╔══██╗██╔════╝██╔══██╗
███████╗██║     ███████║██║     ██████╔╝█████╗  ██████╔╝
╚════██║██║     ██╔══██║██║     ██╔═══╝ ██╔══╝  ██╔══██╗
███████║╚██████╗██║  ██║███████╗██║     ███████╗██║  ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
Leon Bohmann     TUD - ISMD - GCC        www.tu-darmstadt.de/glass-cc
                                                      
                                                        
ScalpProject:           Create with project file. This will split all measurements into 
                        specimens and then into locations.
Specimen:               All measurements on a single specimen.
MeasurementLocation:    One point on the ply that has been measured (3+ Measurements)
Measurement:            A measurement at a location in one specific direction.


This script iterates either one single project or recursively searches for projects in a folder.
For every .scp File, a ScapProject is created, which will handle the reading of data
and also the grouping into Specimens and MeasurementLocations and Measurement respectively.

At the end of the script, all specimens are extracted for better organisation, so that
individual specimen tasks can be performed. For example the calculation of prinicipal stresses
and the mean value of all measurements.

"""

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
    specimens: list[Specimen]
    
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
        self.specimens = [Specimen(name, measurements) for name, measurements in specimen_measurements.items()]
        
        
    def write_measurements(self, output_directory: str | None, output_extension):
        """Writes the measurements to an output directory.

        Args:
            output_directory (str): Output path
            output_extension (str): Output extension
        """
        for measurement in self.measurements:
            output_path = os.path.join(output_directory, measurement.specimen, 'scalp')
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, f"{measurement.name}.{output_extension}")
            with open(output_path, "w") as file:
                measurement.to_file(file)

        for specimen in self.specimens:
            output_path = os.path.join(output_directory, specimen.name, 'scalp')
            output_path = os.path.join(output_path, f"{specimen.name}_stress.txt")
            with open(output_path, "w") as file:
                specimen.to_file(file)
            
        
class Specimen:
    """ Object, that contains information about a specimen. """
    name: str
    measurementlocations: list[MeasurementLocation]
    
    measured_thickness: float   # measured thickness
    sig_h: float                # homogenous pre-stress value
    sig_h_dev: float            # standard deviation of pre-stress
    
    def __init__(self, name: str, measurements: dict[str,list[Measurement]]):
        self.name = name
        self.sig_h = 0.0
                        
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
                continue
                   
            stresses1.append(location.stress[0])
            stresses2.append(location.stress[1])
        
        self.sig_h = np.mean(stresses1 + stresses2) 
        self.sig_h_dev = np.std(stresses1 + stresses2)       
    
    def to_file(self, file):
        file.write(f'{self.measured_thickness:<35}\t# thickness\n')   
        file.write(f'{self.sig_h:<35}\t# homogenous stress\n')   
        file.write(f'{self.sig_h_dev:<35}\t# hom stress std-dev\n')
        file.write('\n')
        
        file.write(f'# {"name":18}\t{"sig_1":<20}\t{"sig_2":<20}\n')
        
        for loc in self.measurementlocations:
            file.write(f'{loc.location_name:20}\t{loc.stress[0]:<20}\t{loc.stress[1]:<20}\n')
    
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
                if sig == '':
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
    
T = TypeVar("T")
def flatten_and_sort(l: list[list[T]], sort_mtd) -> list[T]:
    ret = [item for sublist in l for item in sublist]
    ret.sort(key = sort_mtd)
    return ret

def remove_duplicates(my_list, my_lambda):
    new_list = []
    for item in my_list:
        exists = False
        for ex_item in new_list:
            if my_lambda(ex_item) == my_lambda(item):
                exists = True
                print(f'Found and removed duplicate specimen: {item}')

        if exists:
            continue
                
        new_list.append(item)
        
    return new_list


def get_specimens_from_projects(projects) -> list[Specimen]:
    """Extract all specimens in a distinct list from a list of projects.

    Args:
        projects (list[ScalpProject]): List of scalp projects.

    Returns:
        list[Specimen]: Ordered and distinct list of all specimens.
    """
    specimens = flatten_and_sort([x.specimens for x in projects], lambda s: s.name)
    specimens = remove_duplicates(specimens, lambda t: t.name)
    
    return specimens

if __name__ == "__main__":

    # implement parse to make this script callable from outside
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-project', nargs="?", help='The project to be processed.', default=None)  
    parser.add_argument('-output', nargs="?", help='Output directory.')  
    parser.add_argument('-folder', nargs="?", help='If this is passed, analyzes all scalp projects in a folder.', default=None)  
    parser.add_argument('--mohr', action='store_true', help='Display mohr"sche circle for stress states.')
    args = parser.parse_args()
    
    # overwrite global vars
    SWITCH_DISPLAY_MOHR_CIRCLE = args.mohr
    
    # input vars
    project_file = args.project
    output_directory = args.output
    
    output_extension = "txt"
    open_output_directory = True
    
    # default project file
    if project_file is None and args.folder is None:
        raise Exception("No project file or folder supplied!")
    # default project file
    if output_directory is None:
        if project_file is not None:
            output_directory = os.path.join(os.path.dirname(project_file), "out")
        elif args.folder is not None:
            output_directory = os.path.join(os.path.dirname(args.folder), "out")
    
    # create the output folder if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    projects: list[ScalpProject] = []
    
    # analyze all projects in the folder
    if args.folder is not None:
        project_folder = args.folder
        extension = '.scp'
        files = glob.glob(os.path.join(project_folder, '**', f'*{extension}'), recursive=True)
        
        for file in files:
            print(f'Analyze file: {file.replace(project_folder, "")}')
            project = ScalpProject(file)
            project.write_measurements(output_directory, output_extension)
            projects.append(project)
    # analyze one project
    else:
        project = ScalpProject(project_file)
        project.write_measurements(output_directory, output_extension)    

        projects.append(project)
        
        
    # write a summary    
    project_len = len(projects)
    measurements_len = np.sum([len(x.measurements) for x in projects])
    
    print(f'Projects       : {project_len}')
    print(f'Measurements   : {measurements_len}')
    
    # output all specimens independently from their project    
    specimens = get_specimens_from_projects(projects)
    for specimen in specimens:
        
        pre = ""
        if not all(not t.invalid for t in specimen.measurementlocations):
            pre = "!!!"
            
        print(f'\t{pre}\t{specimen.name:10}: {len(specimen.measurementlocations)} Locations ({[(x.location_name, [l.orientation for l in x.measurements]) for x in specimen.measurementlocations]})')
        
      
    # calculate stresses for specimens
    for specimen in specimens:

        print(f'\t{specimen.name:10}: sig_h={specimen.sig_h:.2f} (+- {specimen.sig_h_dev:.2f})MPa')
        print('\t\t', end="")
        
        for loc in specimen.measurementlocations:
            print(f'\t{loc.location_name}={loc.stress[0]:.2f}/{loc.stress[1]:.2f}', end = "")
            
        print()