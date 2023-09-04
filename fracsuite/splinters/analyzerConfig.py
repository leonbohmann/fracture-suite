from __future__ import annotations
from rich import print
import argparse


class AnalyzerConfig:
    gauss_size: tuple[int,int] = (5,5)
    "gaussian filter size before adaptive thold"
    gauss_sigma: float = 5.0
    "gaussian sigma before adaptive thold"

    thresh_block_size: int = 11
    "adaptive threshold block size"
    thresh_sensitivity: float = 5.0
    "adaptive threshold sensitivity"

    skelclose_size: int = 3
    "size of final closing kernel for skeleton"
    skelclose_amnt: int = 5
    "iteration count of final closing kernel for skel"

    fragment_min_area_px: int = 20
    "minimum fragment area"
    fragment_max_area_px: int = 25000
    "maximum fragment area"
    
    size_factor: float = 1.0
    "factor to scale fragment size in mm/px"
    real_image_size: tuple[int,int] = None
    "real image size in mm"
    cropped_image_size: tuple[int,int] = None
    "real image size in mm"
    crop: bool = False
    "crop input image"
    impact_position: tuple[float, float] = None
    "impact position in mm [X Y]"


    debug: bool = False
    "enable debug output"
    debug_experimental: bool = False
    "enable debug output"
    display_region: tuple[int,int,int,int]  = None
    "region to display in output plots (x1,y1,x2,y2)"
    resize_factor: float = 1.0
    "factor to resize input image in preprocess"
    displayplots: bool = False
    "Display plots during creation"
    printconfig: bool = False
    "Print the configuration before starting the script"

    out_name: str = ""
    "name of the output directory"
    ext_plots: str = "png"
    "output extension for plots"
    ext_imgs: str = "png"
    "output extension for images"

    skip_darkspot_removal: bool  = False
    "skip dark spot removal"
    intensity_h: int = 500
    "intensity kernel width in px"

    path: str = ""
    "Path to data"

    norm_region_center: tuple[float, float] = None
    "Center for evaluation region according to DIN in mm [X Y]"

    norm_region_size: tuple[float, float] = (50,50)
    "Size for evaluation region according to DIN in mm [W H]"



    def get_parser(descr) -> argparse.ArgumentParser:
        """
        Create and return an argumentParser, that can be used to initialize a new 
        AnalyzerConfig with `AnalyzerConfig.from_args(args)` method.

        This can be used, if the argumentparser should be extended. I.e. if highspeed
        module wants to add a specific argument group.

        Args:
            descr (string): The description, the argparse object should display on `-h`.

        Returns:
            argparse.ArgumentParser: Can be used to create AnalyzerConfig.
        """
        parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)

        gnrl_group = parser.add_argument_group("General")
        gnrl_group.add_argument('--displayplots', action='store_true', \
            help='Instruct the analyzer to display output plots.', default=False)
        gnrl_group.add_argument('--debug', action='store_true', \
            help='Sets a debug flag to display verbose output.', default=False)
        gnrl_group.add_argument('--exp-debug', action='store_true', \
            help='Sets an experimental debug flag to display verbose output.', default=False)
        gnrl_group.add_argument('--printconfig', action='store_true', \
            help='Print the config before starting the script.', default=False)
        gnrl_group.add_argument('-display-region', nargs=4, help='Region to display in debug outputs. [Pixels]',\
            type=int, default=None, metavar=('X1', 'Y1', 'X2', 'Y2'))

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
        post.add_argument('-impactposition', nargs=2, metavar=('X', 'Y'), type=float, 
            help='Impact position in mm [X Y]', default=(50,50))
        post.add_argument('-normregioncenter', nargs=2, metavar=('X', 'Y'), type=float, default=None,
            help='Center for evaluation region according to DIN in mm [X Y]')
        post.add_argument('-normregionsize', nargs=2, metavar=('W', 'H'), type=float, default=(50,50),
            help='Size for evaluation region according to DIN in mm.')


        output_group = parser.add_argument_group("Output")
        output_group.add_argument('-out', nargs="?", help='Output directory path.', \
            default="fracsuite-output")
        output_group.add_argument('-plot-ext', nargs="?", help='Plot file extension. Default: png.', \
            default="png", choices=['png', 'pdf', 'jpg', 'bmp'])
        output_group.add_argument('-image-ext', nargs="?", help='Image file extension. Default: png.',\
            default="png", choices=['png', 'jpg', 'bmp'])

        return parser

    def from_args(args: argparse.Namespace) -> AnalyzerConfig:
        """
        Create AnalyzerConfig from command line arguments.

        Args:
            args (argparse.Namespace): The arguments parsed from argparse.

        Returns:
            AnalyzerConfig: The configuration.
        """
        cfg = AnalyzerConfig()
        cfg.debug = args.debug
        cfg.debug_experimental = args.exp_debug
        cfg.printconfig = args.printconfig
        cfg.displayplots = args.displayplots
        
        cfg.gauss_size = (args.gauss_size,args.gauss_size)
        cfg.gauss_sigma = args.gauss_sigma
        cfg.fragment_min_area_px = args.min_area
        cfg.fragment_max_area_px = args.max_area


        cfg.thresh_block_size = args.thresh_block
        cfg.thresh_sensitivity = args.thresh_sens
        cfg.skelclose_size = args.skelclose_sz
        cfg.skelclose_amnt = args.skelclose_amnt
        cfg.crop = args.cropsize is not None
        cfg.cropped_image_size = args.cropsize
        cfg.real_image_size = args.realsize
        cfg.resize_factor = args.resize_fac
        cfg.impact_position = args.impactposition

        cfg.display_region = args.display_region
        cfg.intensity_h = args.intensity_width
        cfg.ext_plots = args.plot_ext
        cfg.ext_imgs = args.image_ext
        cfg.skip_darkspot_removal = args.skip_spot_elim

        cfg.norm_region_center = args.normregioncenter
        cfg.norm_region_size = args.normregionsize

        if args.debug is True:
            cfg.displayplots = True

        if args.realsize is not None and len(args.realsize) > 1:
            cfg.real_image_size = tuple(args.realsize)
        elif args.realsize is not None and len(args.realsize) == 1:
            cfg.real_image_size = (args.realsize[0], args.realsize[0])

        if args.cropsize is not None and len(args.cropsize) > 1:
            cfg.cropped_image_size = tuple(args.cropsize)
        elif args.cropsize is not None and len(args.cropsize) == 1:
            cfg.cropped_image_size = (args.cropsize[0], args.cropsize[0])

        cfg.path = args.path

        if cfg.printconfig:
            cfg.print()

        return cfg

    def parse(descr: str) -> tuple[argparse.Namespace, AnalyzerConfig]:
        """Directly parse and return a Namespace and AnalyzerConfig.

        Args:
            descr (str): Display description on `-h`.

        Returns:
            tuple[argparse.Namespace, AnalyzerConfig]: Results.
        """
        args = AnalyzerConfig.get_parser(descr).parse_args()
        return args, AnalyzerConfig.from_args(args)

    def __init__(self):
        pass

    def print(self):
        print(self.__dict__)