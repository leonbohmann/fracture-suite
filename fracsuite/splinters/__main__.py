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
from fracsuite.splinters.analyzer import Analyzer, AnalyzerConfig

from rich import print

args, config = AnalyzerConfig.parse(__doc__)


if args.path.endswith('\\'):
    search_path = os.path.join(args.path, 'fracture', 'morph')
    for file in os.listdir(search_path):
        if 'Transmission' in file and file.endswith('.bmp'):
            print(f"[green]UPDATED analyzer path.[/green]")
            config.path = os.path.join(search_path, file)
            args.path = config.path
            break

print(f'Analyzing: {args.path}')


analyzer = Analyzer(args.path, config)

analyzer.plot(display=config.displayplots, region=config.display_region)
analyzer.plot_area(display=config.displayplots)
analyzer.plot_area_2(display=config.displayplots)

analyzer.save_images(extension=config.ext_imgs)
analyzer.save_plots(extension=config.ext_plots)