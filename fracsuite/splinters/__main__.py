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

args = AnalyzerConfig.get_parser(__doc__)    

if args.debug is True:
    args.displayplots = True

do_crop = args.cropsize is not None

if args.realsize is not None and len(args.realsize) > 1:
    args.realsize = tuple(args.realsize)
elif args.realsize is not None and len(args.realsize) == 1:
    args.realsize = (args.realsize[0], args.realsize[0])

if args.cropsize is not None and len(args.cropsize) > 1:
    args.cropsize = tuple(args.cropsize)
elif args.cropsize is not None and len(args.cropsize) == 1:
    args.cropsize = (args.cropsize[0], args.cropsize[0])
    
    
config = AnalyzerConfig(gauss_sz=args.gauss_size, gauss_sig=args.gauss_sigma, \
    fragment_min_area_px=args.min_area, fragment_max_area_px=args.max_area, \
        real_img_size=args.realsize, crop=do_crop, thresh_block_size=args.thresh_block,\
            thresh_sensitivity=args.thresh_sens, rsz_fac=args.resize_fac, cropped_img_size=args.cropsize,\
            debug=args.debug, display_region=args.display_region, skel_close_sz=args.skelclose_sz,\
                skel_close_amnt=args.skelclose_amnt, debug2=args.exp_debug)

config.ext_plots = args.plot_ext
config.ext_imgs = args.image_ext
config.skip_darkspot_removal = args.skip_spot_elim
config.intensity_h = args.intensity_width

if args.path.endswith('\\'):
    search_path = os.path.join(args.path, 'fracture', 'morph')
    for file in os.listdir(search_path):
        if 'Transmission' in file:
            args.path = os.path.join(search_path, file)
            break

print(f'Analyzing: {args.path}')

config.print()

analyzer = Analyzer(args.path, config)

analyzer.plot(display=args.displayplots, region=config.display_region)
analyzer.plot_area(display=args.displayplots)
analyzer.plot_area_2(display=args.displayplots)

analyzer.save_images(extension=args.image_ext)
analyzer.save_plots(extension=args.plot_ext)