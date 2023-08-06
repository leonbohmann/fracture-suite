import sys
sys.path.append('../src')

import argparse
from fracture_suite.analyzer import Analyzer


if __name__ == '__main__':
    # implement parse to make this script callable from outside
    parser = argparse.ArgumentParser()    
    parser.add_argument('-image', nargs="?", help='The image to be processed.')
    args = parser.parse_args()    
    
    analyzer = Analyzer(args.image)
    
    analyzer.plot()
    analyzer.plot_area()