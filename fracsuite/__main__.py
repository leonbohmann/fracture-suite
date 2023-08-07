import argparse
from fracsuite.analyzer import Analyzer


# implement parse to make this script callable from outside
parser = argparse.ArgumentParser()    
parser.add_argument('-image', nargs="?", help='The image to be processed.')
parser.add_argument('-realsize', nargs=2, help='Real size of the input image. Use: -realsize W H')
parser.add_argument('--crop', action='store_true', \
    help='Instruct the analyzer to crop the input image.')
args = parser.parse_args()    

print(args)

analyzer = Analyzer(args.image, args.crop, img_real_size=tuple(args.realsize))

analyzer.plot()
analyzer.plot_area()
analyzer.plot_area_2()