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
from fracsuite.splinters.analyzer import Analyzer
from fracsuite.splinters.analyzerConfig import AnalyzerConfig

parser = AnalyzerConfig.get_parser(__doc__)

parser.add_argument("--all", default=False, 
                    help="Instruct the analyzer to run the analysis on every subfolder.",
                    action='store_true')

args = parser.parse_args()
config = AnalyzerConfig.from_args(args)

if args.all:
    project_dir = config.path
    for file in os.listdir(project_dir):
        project_path = os.path.join(project_dir, file, '\\')
        
        if os.path.exists(project_path):
            config.path = project_path
            analyzer = Analyzer(config)
        
        
        
else:
    analyzer = Analyzer(config)
    

