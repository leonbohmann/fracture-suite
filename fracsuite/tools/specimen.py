import json
import os

from rich import print

from fracsuite.scalper.scalpSpecimen import ScalpSpecimen
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import find_file


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
    
    def __init__(self, path: str):
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
        
        # load scalp
        scalp_path = os.path.join(self.path, "scalp")
        scalp_file = find_file(scalp_path, "*.pkl")
        if scalp_file is not None:
            self.scalp = ScalpSpecimen.load(scalp_file)
        else:            
            print(f"Could not find scalp file for '{path}'. Create it using the original scalper project and [green]fracsuite.scalper[/green].")


        # load splinters
        splinters_path = os.path.join(self.path, "fracture", "splinter")
        splinters_file = find_file(splinters_path, "*.pkl")
        if splinters_file is not None:
            self.splinters = Analyzer.load(splinters_file)  
            self.splinters.config.specimen_name = self.name          
        else:
            print(f"Could not find splinter file for '{path}'. Create it using [green]fracsuite.splinters[/green].")        