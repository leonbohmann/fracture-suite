import argparse
from argparse import RawDescriptionHelpFormatter
from fracsuite.splinters.analyzer import Analyzer, AnalyzerConfig

descr=\
"""
███████╗██████╗  █████╗  ██████╗████████╗██╗   ██╗██████╗ ███████╗      ███████╗██╗   ██╗██╗████████╗███████╗
██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║   ██║██╔══██╗██╔════╝      ██╔════╝██║   ██║██║╚══██╔══╝██╔════╝
█████╗  ██████╔╝███████║██║        ██║   ██║   ██║██████╔╝█████╗  █████╗███████╗██║   ██║██║   ██║   █████╗  
██╔══╝  ██╔══██╗██╔══██║██║        ██║   ██║   ██║██╔══██╗██╔══╝  ╚════╝╚════██║██║   ██║██║   ██║   ██╔══╝  
██║     ██║  ██║██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗      ███████║╚██████╔╝██║   ██║   ███████╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝      ╚══════╝ ╚═════╝ ╚═╝   ╚═╝   ╚══════╝
Leon Bohmann            Technical University Darmstadt - ISMD - GCC              www.tu-darmstadt.de/glass-cc

                                  
"""

# implement parse to make this script callable from outside
parser = argparse.ArgumentParser(description=descr, formatter_class=RawDescriptionHelpFormatter)    
imgroup = parser.add_argument_group("Image operations")
imgroup.add_argument('image', nargs="?", help='The image to be processed.')
imgroup.add_argument('-realsize', nargs=2, help='Real size of the input image.',\
    type=int, default=[500,500])
imgroup.add_argument('-imsize', nargs=2, help='Resized original image size in pixels.',\
    type=int, default=None)
imgroup.add_argument('--crop', action='store_true', \
    help='Instruct the analyzer to crop the input image.', default=False)

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
prep.add_argument('-resize-fac', help='Image resize factor between gauss and adaptive th.',\
    type=float, default=1.0)

output_group = parser.add_argument_group("Output")
output_group.add_argument('-out', nargs="?", help='Output directory path.', \
    default="fracsuite-output")
output_group.add_argument('-plot-ext', nargs="?", help='Plot file extension. Default: png.', \
    default="png", choices=['png', 'pdf', 'jpg', 'bmp'])
output_group.add_argument('-image-ext', nargs="?", help='Image file extension. Default: png.',\
    default="png", choices=['png', 'jpg', 'bmp'])
output_group.add_argument('--displayplots', action='store_true', \
    help='Instruct the analyzer to display output plots.', default=False)

args = parser.parse_args()    



if args.realsize is not None:
    args.realsize = tuple(args.realsize)
    
config = AnalyzerConfig(gauss_sz=args.gauss_size, gauss_sig=args.gauss_sigma, \
    fragment_min_area_px=args.min_area, fragment_max_area_px=args.max_area, \
        real_img_size=args.realsize, crop=args.crop, thresh_block_size=args.thresh_block,\
            thresh_sensitivity=args.thresh_sens, rsz_fac=args.resize_fac, img_size=args.imsize)
config.print()

analyzer = Analyzer(args.image, config)

analyzer.plot(display=args.displayplots)
analyzer.plot_area(display=args.displayplots)
analyzer.plot_area_2(display=args.displayplots)

analyzer.save_images(extension=args.image_ext)
analyzer.save_plots(extension=args.plot_ext)