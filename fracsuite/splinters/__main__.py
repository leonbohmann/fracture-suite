"""
███████╗██████╗ ██╗     ██╗███╗   ██╗████████╗███████╗██████╗ ███████╗
██╔════╝██╔══██╗██║     ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔════╝
███████╗██████╔╝██║     ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝███████╗
╚════██║██╔═══╝ ██║     ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗╚════██║
███████║██║     ███████╗██║██║ ╚████║   ██║   ███████╗██║  ██║███████║
╚══════╝╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝
                                                                      
Leon Bohmann     TUD - ISMD - GCC        www.tu-darmstadt.de/glass-cc

Description:
-------------------------
This module helps with the automated detection of fragments on fractured glass plys. It
performs some preprocessing actions to improve the quality of the input image and then
uses that to search for contours on the image. After filtering and further improvement
the found contours are converted into `Splinter` objects, which allow further investigation
of the fragments (size, roughness, roundness, ...).

Used packages:
-------------------------
- opencv-python
- matplotlib
- numpy
- tqdm    

Usage:
-------------------------                       
Command line usage is shown below. For further information visit:
https://github.com/leonbohmann/fracture-suite
"""

import os
import sys

from tqdm import tqdm
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.splinters.analyzerConfig import AnalyzerConfig

import matplotlib

from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.specimen import Specimen

matplotlib.rcParams['figure.figsize'] = (6, 4)
matplotlib.rc('axes', axisbelow=True) # to get grid into background
matplotlib.rc('grid', linestyle="--") # line style
matplotlib.rcParams.update({'font.size': 12}) # font size

general_settings = GeneralSettings()

parser = AnalyzerConfig.get_parser(__doc__)

parser.add_argument("--all", default=False, 
                    help="Instruct the analyzer to run the analysis on every subfolder.",
                    action='store_true')
parser.add_argument("--all-exclude", default=[], nargs="+")

args = parser.parse_args()

config = AnalyzerConfig.from_args(args)

if config.path[1] != ":":
    config.path = os.path.join(general_settings.base_path, config.path) + "\\"
    

if args.all:
    print(f"Running analysis on all subfolders of '{config.path}'.")
    project_dir = config.path
    
    for file in (pbar := tqdm(os.listdir(project_dir))):
        
        if any([x in file for x in args.all_exclude]):
            continue
        
        project_path = os.path.join(project_dir, file) + "\\"
        
        if os.path.exists(project_path) and os.path.isdir(project_path):
            spec = Specimen(project_path)
            
            pbar.set_description(f"Processing {spec.name}...")
            try:
                config.path = project_path
                stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                analyzer = Analyzer(config)
                sys.stdout = stdout                
            except:
                print(f"> [red]Failed to run analysis on '{project_path}'.[/red]")
                continue
        
        
        
else:
    analyzer = Analyzer(config)
    

