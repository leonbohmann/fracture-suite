from __future__ import annotations
import json
import os
from typing import Any, Callable, TypeVar
import numpy as np

from rich import print
from rich.progress import track
import typer

from fracsuite.scalper.scalpSpecimen import ScalpSpecimen
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import find_file

app = typer.Typer()

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
        specimen = Specimen(spec_path, lazy=True)
        
        specimens.append(specimen)
        print(f"Loaded '{name}'.")
        
    return specimens

def default_value(specimen: Specimen) -> Specimen:
    return specimen

_T1 = TypeVar('_T1')
def fetch_specimens_by(decider: Callable[[Specimen], bool], 
                       path: str, 
                       max_n: int = 1000, 
                       sortby: Callable[[Specimen], Any] = None,
                       value: Callable[[Specimen], _T1 | Specimen] = default_value) -> list[_T1]:
    """Fetch a list of specimens from a given path.

    Args:
        decider (func(Specimen)): Decider if the specimen should be selected.
        path (str): THe base path to the specimens.
    """
    if value is None:
        def value(specimen: Specimen):
            return specimen
    
    data: list[Any] = []
    
    for name in track(os.listdir(path), description="Loading specimens...", transient=True):
        
        spec_path = os.path.join(path, name)
        if not os.path.isdir(spec_path):
            continue
        
        
        specimen = Specimen(spec_path, log_missing=False, lazy=True)
        
        if not decider(specimen):
            continue
        
        specimen.lazy_load()
        data.append(value(specimen))
        # print(f"Loaded '{name}'.")
        
        if len(data) >= max_n:
            break
        
    print(f"Loaded {len(data)} specimens.")
    
    if sortby is not None:
        data = sorted(data, key=sortby)
                
    return data

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
    
    def lazy_load(self):
        """Load the specimen lazily."""
        
        if self.loaded:
            print("[red]Specimen already loaded.")
        
        # self.scalp = ScalpSpecimen.load(self.__scalp_file)
        # self.splinters = Analyzer.load(self.__splinters_file)        
        # self.splinters.config.specimen_name = self.name
        
        self.loaded = True
    
    def __init__(self, path: str, log_missing = True, lazy = False):
        """Create a new specimen.

        Args:
            path (str): Path of the specimen.
        """
        
        self.loaded = not lazy
        
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
            
            if "-" not in vars[3]:
                self.nbr = int(vars[3])
                self.comment = ""
            else:
                last_sec = vars[3].split("-")
                self.nbr = int(last_sec[0])
                self.comment = last_sec[1]
        
        # load scalp
        scalp_path = os.path.join(self.path, "scalp")
        self.__scalp_file = find_file(scalp_path, "pkl")
        self.has_scalp = self.__scalp_file is not None
        if self.__scalp_file is not None and not lazy:
            self.scalp = ScalpSpecimen.load(self.__scalp_file)
        elif log_missing:            
            print(f"Could not find scalp file for '{path}'. Create it using the original scalper project and [green]fracsuite.scalper[/green].")


        # load splinters
        self.fracture_morph_dir = os.path.join(self.path, "fracture", "morphology")
        self.has_fracture_scans = os.path.exists(self.fracture_morph_dir) \
            and find_file(self.fracture_morph_dir, ".bmp") is not None
        self.splinters_path = os.path.join(self.path, "fracture", "splinter")
        self.__splinters_file = find_file(self.splinters_path, "pkl")
        
        self.has_splinters = self.__splinters_file is not None
        
        if self.__splinters_file is not None and not lazy:
            # self.splinters = Analyzer.load(self.__splinters_file)  
            self.splinters.config.specimen_name = self.name          
        elif log_missing:            
            print(f"Could not find splinter file for '{path}'. Create it using [green]fracsuite.splinters[/green].")        

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