
import argparse
from argparse import RawDescriptionHelpFormatter
import os
from fracsuite.splinters.analyzer import Analyzer, AnalyzerConfig
descr=\
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

# implement parse to make this script callable from outside
parser = argparse.ArgumentParser(description=descr, formatter_class=RawDescriptionHelpFormatter)    

gnrl_group = parser.add_argument_group("General")
gnrl_group.add_argument('--displayplots', action='store_true', \
    help='Instruct the analyzer to display output plots.', default=False)
gnrl_group.add_argument('--debug', action='store_true', \
    help='Sets a debug flag to display verbose output.', default=False)
gnrl_group.add_argument('--exp-debug', action='store_true', \
    help='Sets an experimental debug flag to display verbose output.', default=False)
gnrl_group.add_argument('-display-region', nargs=4, help='Region to display in debug outputs.',\
    type=int, default=None)

imgroup = parser.add_argument_group("Image operations")
imgroup.add_argument('path', nargs="?", help='The path of the image to be processed or a folder that contains a file in subfolder "[path]/fracture/morph/...Transmission.bmp".')
imgroup.add_argument('-realsize', nargs="*", help='Real size of the input image. If only one dim is provided, a square geometry is used.',\
    type=int, default=None, metavar=('WIDTH', 'HEIGHT'))
imgroup.add_argument('-cropsize', nargs="*", help='Crop image size in pixels. If only one dim is provided, a square geometry is used.',\
    type=int, default=None, metavar=('WIDTH', 'HEIGHT'))

prep = parser.add_argument_group("Preprocessor")
# image preprocessing arguments
prep.add_argument('-gauss-size', help='Gaussian filter size',\
    type=int, default=5)
prep.add_argument('-gauss-sigma', help='Gaussian filter sigma',\
    type=float, default=5)
prep.add_argument('-min-area', help='Minimum fragment area threshold [px²]',\
    type=float, default=20)
prep.add_argument('-max-area', help='Maximum fragment area threshold [px²]',\
    type=float, default=25000)
prep.add_argument('-thresh-sens', help='Adaptive threshold sensitivity',\
    type=float, default=6)
prep.add_argument('-thresh-block', help='Adaptive threshold block size',\
    type=int, default=11, choices=[1,3,5,7,9,11,13,15,17,19,21])
prep.add_argument('-resize-fac', help='Image resize factor before adaptive th.',\
    type=float, default=1.0)

post = parser.add_argument_group("Postprocessor")
post.add_argument('-skelclose-sz', help='Size for final skeleton close kernel.',\
    type=int, default=3)
post.add_argument('-skelclose-amnt', help='Iterations for final skeleton close kernel.',\
    type=int, default=5)
post.add_argument('--skip-spot-elim', help='Instruct the postprocessor to skip "dark-spot" removal.',\
    action="store_true", default=False)
post.add_argument('-intensity-width', help='Pixel width for intensity calculation.',\
    type=int, default=500)

output_group = parser.add_argument_group("Output")
output_group.add_argument('-out', nargs="?", help='Output directory path.', \
    default="fracsuite-output")
output_group.add_argument('-plot-ext', nargs="?", help='Plot file extension. Default: png.', \
    default="png", choices=['png', 'pdf', 'jpg', 'bmp'])
output_group.add_argument('-image-ext', nargs="?", help='Image file extension. Default: png.',\
    default="png", choices=['png', 'jpg', 'bmp'])

args = parser.parse_args()    

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

if not args.image.endswith('.bmp'):
    search_path = os.path.join(args.image, 'fracture', 'morph')
    for file in os.listdir(search_path):
        if 'Transmission' in file:
            args.image = os.path.join(search_path, file)
            break

print(f'Analyzing image: {args.image}')

config.print()

analyzer = Analyzer(args.image, config)

analyzer.plot(display=args.displayplots, region=config.display_region)
analyzer.plot_area(display=args.displayplots)
analyzer.plot_area_2(display=args.displayplots)

analyzer.save_images(extension=args.image_ext)
analyzer.save_plots(extension=args.plot_ext)