from __future__ import annotations
import pickle
import re
from rich import print
import argparse

from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import get_specimenname_from_path

general = GeneralSettings.get()

class AnalyzerConfig:
    resize_factor: float = 1.0
    "factor to resize input image in preprocess"

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
    fragment_max_area_px: int = 2500
    "maximum fragment area"

    size_factor: float = 1.0
    "factor to scale fragment size [mm/px]"
    real_image_size: tuple[int,int] = None
    "real image size in mm"
    cropped_image_size: tuple[int,int] = None
    "image size in px"
    crop: bool = False
    "crop input image"

    debug: bool = False
    "enable debug output"
    displayplots: bool = False
    "Display plots during creation"
    printconfig: bool = False
    "Print the configuration before starting the script"

    out_name: str = ""
    "name of the output directory"

    skip_darkspot_removal: bool  = False
    "skip dark spot removal"

    path: str = ""
    "Path to data"

    specimen_name: str = ""
    "Name of the specimen"

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
        parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter,
                                         argument_default=argparse.SUPPRESS)

        parser.add_argument('path', nargs="?",
                             help='The path of the image to be processed or a folder that contains' \
                                'a file in subfolder "[path]/fracture/morph/...Transmission.bmp". Instead'
                                ' of a folder, you can also specify the base_path in tools.settings and then use the specimen ID only "1.1.A.1".'
                                f' Current base_path: "{general.base_path}"')

        gnrl_group = parser.add_argument_group("General")
        gnrl_group.add_argument('--displayplots', action='store_true', \
            help='Instruct the analyzer to display output plots.', default=False)
        gnrl_group.add_argument('--debug', action='store_true', \
            help='Sets a debug flag to display verbose output.', default=False)
        gnrl_group.add_argument('--printconfig', action='store_true', \
            help='Print the config before starting the script.', default=False)

        imgroup = parser.add_argument_group("Image operations")
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
            type=float, default=5)
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

        output_group = parser.add_argument_group("Output")
        output_group.add_argument('-out', nargs="?", help='Output directory path.', \
            default="fracsuite-output")

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

        cfg.skip_darkspot_removal = args.skip_spot_elim

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

        if not hasattr(args, "all") or (hasattr(args, "all") and not args.all):
            cfg.path = args.path

        # find specimen pattern
        cfg.specimen_name = get_specimenname_from_path(cfg.path)

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

    def load(path) -> AnalyzerConfig:
        """Load a configuration from a file.

        Args:
            path (str): Path to the file.

        Returns:
            AnalyzerConfig: The loaded configuration.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __init__(self):
        pass

    def print(self):
        print(self.__dict__)