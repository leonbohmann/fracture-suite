from __future__ import annotations
import json
import os

from rich import print

from fracsuite.scalper.scalpSpecimen import ScalpSpecimen
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import find_file

general = GeneralSettings()

def fetch_specimens(specimen_names: list[str], path: str) -> list[Specimen]:
    """Fetch a list of specimens from a given path.

    Args:
        specimen_names (list[str]): The names of the specimens.
        path (str): THe base path to the specimens.
    """
    
    specimens: list[Specimen] = []
    for name in specimen_names:
        spec_path = os.path.join(path, name)
        specimen = Specimen(spec_path)
        
        if specimen.splinters is None:
            continue
        
        specimens.append(specimen)
        print(f"Loaded '{name}'.")
        
    return specimens


class Specimen:
    """ Container class for a specimen. """
    
    splinters: Analyzer = None
    "Splinter analysis."
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
    
    def __init__(self, path: str, log_missing = True):
        """Create a new specimen.

        Args:
            path (str): Path of the specimen.
        """
        
        self.path = path
        
        cfg_path = os.path.join(path, "config.json")
        if not os.path.exists(cfg_path):            
            with open(cfg_path, "w") as f:
                json.dump(self.settings, f, indent=4)
        else:
            with open(cfg_path, "r") as f:
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
            
            if not "-" in vars[3]:
                self.nbr = int(vars[3])
                self.comment = ""
            else:
                last_sec = vars[3].split("-")
                self.nbr = int(last_sec[0])
                self.comment = last_sec[1]
        
        # load scalp
        scalp_path = os.path.join(self.path, "scalp")
        scalp_file = find_file(scalp_path, "pkl")
        if scalp_file is not None:
            self.scalp = ScalpSpecimen.load(scalp_file)
        elif log_missing:            
            print(f"Could not find scalp file for '{path}'. Create it using the original scalper project and [green]fracsuite.scalper[/green].")


        # load splinters
        self.has_fracture_scans = os.path.exists(os.path.join(self.path, "fracture", "morphology")) \
            and find_file(os.path.join(self.path, "fracture", "morphology"), ".bmp") is not None
        self.splinters_path = os.path.join(self.path, "fracture", "splinter")
        splinters_file = find_file(self.splinters_path, "pkl")
        if splinters_file is not None:
            self.splinters = Analyzer.load(splinters_file)  
            self.splinters.config.specimen_name = self.name          
        elif log_missing:            
            print(f"Could not find splinter file for '{path}'. Create it using [green]fracsuite.splinters[/green].")        