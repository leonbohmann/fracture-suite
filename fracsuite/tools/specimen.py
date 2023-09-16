from __future__ import annotations
import time 
import json
from pathos.pools import ProcessPool
import os
import pickle
from typing import Any, Callable, TypeVar
import numpy as np

from rich import inspect, print
from rich.progress import track, Progress
from fracsuite.scalper.scalpSpecimen import ScalpSpecimen
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from fracsuite.splinters.splinter import Splinter
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import find_file
import typer
import functools

app = typer.Typer()

general = GeneralSettings.get()

def fetch_specimens(specimen_names: list[str] | str, path: str) -> list[Specimen]:
    """Fetch a list of specimens from a given path.

    Args:
        specimen_names (list[str]): The names of the specimens.
        path (str): THe base path to the specimens.
    """
    if not isinstance(specimen_names, list):
        specimen_names = [specimen_names]
    
    specimens: list[Specimen] = []
    for name in specimen_names:
        spec_path = os.path.join(path, name)
        specimen = Specimen(spec_path, lazy=True)
        
        specimen.lazy_load()
        specimens.append(specimen)
        print(f"Loaded '{name}'.")
        
    return specimens

def default_value(specimen: Specimen) -> Specimen:
    return specimen

def load_specimen(args) -> Specimen | None:    
    spec_path, lazy_load = args[0], args[1]
    if not os.path.isdir(spec_path):
        return None
    
    global shared_decider
    global shared_value_converter
    
    specimen = Specimen(spec_path, log_missing=False, lazy=lazy_load)
    if not shared_decider(specimen):
        return None
    
    return shared_value_converter(specimen)

_T1 = TypeVar('_T1')
def fetch_specimens_by(decider: Callable[[Specimen], bool], 
                       path: str, 
                       max_n: int = 1000, 
                       sortby: Callable[[Specimen], Any] = None,
                       value: Callable[[Specimen], _T1 | Specimen] = default_value,
                       lazy_load: bool = False,
                       parallel_load: bool = False) -> list[_T1]:
    """Fetch a list of specimens from a given path.

    Args:
        decider (func(Specimen)): Decider if the specimen should be selected.
        path (str): THe base path to the specimens.
    """
    time0 = time.time()
    def init_pool(decider, value):
        global shared_decider
        global shared_value_converter
        shared_decider = decider
        shared_value_converter = value
        
    if parallel_load:
        p = ProcessPool(initializer=init_pool, initargs=(decider, value))    
    else:
        init_pool(decider, value)
    
    if value is None:
        def value(specimen: Specimen):
            return specimen
    
    directories = [os.path.join(path,x) for x in os.listdir(path) 
                   if os.path.isdir(os.path.join(path,x))]
    
    
    data: list[Any] = []
    if parallel_load:
        data = p.amap(load_specimen, 
                    [(x, lazy_load) for x in directories])
        with Progress() as progress:
            task = progress.add_task("[green]Loading specimens...", total=len(directories))
            # inspect(data, methods=True, private=True)
            while not data.ready():
                progress.update(task, completed=len(directories)-data._number_left, total=len(directories))
                pass
        data = [x for x in data.get() if x is not None]
    else:
        for dir in track(directories, description="Loading specimens...", transient=True):
            spec = load_specimen((dir, lazy_load))
            if spec is not None:
                data.append(spec)
        
    
    # for name in track(os.listdir(path), description="Loading specimens...", transient=True):
        
    #     spec_path = os.path.join(path, name)
    #     if not os.path.isdir(spec_path):
    #         continue
        
        
    #     specimen = Specimen(spec_path, log_missing=False, lazy=lazy_load)
        
    #     if not decider(specimen):
    #         continue
        
    #     if lazy_load:
    #         specimen.lazy_load()
            
    #     data.append(value(specimen))
    #     # print(f"Loaded '{name}'.")
        
    #     if len(data) >= max_n:
    #         break
        
    print(f"Loaded {len(data)} specimens.")
    
    if sortby is not None:
        data = sorted(data, key=sortby)

    if parallel_load:
        p.close()   
        
    time1 = time.time()
    print(f"Loading took {time1-time0}.")             
    return data

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
    
    def lazy_load(self):
        """Load the specimen lazily."""
        
        if self.loaded:
            print("[red]Specimen already loaded.")
        
        self.__load_scalp()        
        self.__load_splinters()
        self.__load_splinter_config()
        
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
        # load acceleration
        acc_path = os.path.join(self.path, "fracture", "acceleration")
        self.acc_file = find_file(acc_path, "*.bin")
        
        # load scalp
        scalp_path = os.path.join(self.path, "scalp")
        self.__scalp_file = find_file(scalp_path, "*.pkl")
        self.has_scalp = self.__scalp_file is not None
        if self.__scalp_file is not None and not lazy:
            self.scalp = ScalpSpecimen.load(self.__scalp_file)
        elif log_missing:            
            print(f"Could not find scalp file for '{path}'. Create it using the original scalper project and [green]fracsuite.scalper[/green].")


        # load splinters
        self.fracture_morph_dir = os.path.join(self.path, "fracture", "morphology")
        self.has_fracture_scans = os.path.exists(self.fracture_morph_dir) \
            and find_file(self.fracture_morph_dir, "*.bmp") is not None
        self.splinters_path = os.path.join(self.path, "fracture", "splinter")
        self.__splinters_file = find_file(self.splinters_path, "splinters.pkl")        
        self.__config_file = find_file(self.splinters_path, "config.pkl")        
        self.has_splinters = self.__splinters_file is not None
        self.has_splinter_config = self.__config_file is not None
        self.has_config = self.__config_file is not None
        
        if self.__splinters_file is not None and not lazy:
            self.__load_splinters()  
        elif log_missing:            
            print(f"Could not find splinter file for '{self.name}'. Create it using [green]fracsuite.splinters[/green].")        

        if self.__config_file is not None and not lazy:
            self.__load_splinter_config()
        elif log_missing:
            print(f"Could not find splinter config file for '{self.name}'. Create it using [green]fracsuite.splinters[/green].")
     
        
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
    
