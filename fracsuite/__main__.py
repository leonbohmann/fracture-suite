import argparse
from fracsuite.analyzer import Analyzer


# implement parse to make this script callable from outside
parser = argparse.ArgumentParser()    
parser.add_argument('-image', nargs="?", help='The image to be processed.')
parser.add_argument('--crop', action='store_true', \
    help='Instruct the analyzer to crop the input image.')
args = parser.parse_args()    

args.image = r"d:\Forschung\Glasbruch\Versuche.Reihe\Proben\4.70.22.A\fracture\LS\2923 2023-06-15 17h11 pre (0,308-4323,4963) [Transmission].bmp"
args.crop = True

print(args)

analyzer = Analyzer(args.image, args.crop)

analyzer.plot()
analyzer.plot_area()
analyzer.plot_area_2()