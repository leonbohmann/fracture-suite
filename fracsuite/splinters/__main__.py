import argparse
import os
import pickle

import cv2

from fracsuite.core.splinter import Splinter

parser = argparse.ArgumentParser(description='Splinter a file into multiple files.')

parser.add_argument('input', metavar='input', type=str)

args = parser.parse_args()

image = cv2.imread(args.input)

splinters = Splinter.analyze_image_legacy(image)

file_dir = os.path.dirname(args.input)

with open(os.path.join(file_dir, 'splinters.pkl'), 'wb') as f:
    pickle.dump(splinters, f)