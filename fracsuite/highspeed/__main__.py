import os
from fracsuite.splinters.analyzer import preprocess_image
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
import numpy as np

from rich import print
import cv2

parser = AnalyzerConfig.get_parser(__doc__)

hsp_group = parser.add_argument_group('High-Speed-Images Arguments')
# hsp_group.add_argument("--dummy", action='store_true')

args = parser.parse_args()

config = AnalyzerConfig.from_args(args)

# get folder
folder = config.path

# get all tiff files from folder and save them
img_files = [os.path.join(folder,x) for x in os.listdir(folder) if x.lower().endswith('tiff')]
print(img_files)

img0 = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
result_images = []
for i, img in enumerate(img_files[2:]):
    # get two images up to i
    img1 = cv2.imread(img_files[i-1], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
    
    img2_o = cv2.cvtColor(img2.copy(), cv2.COLOR_GRAY2BGR)
    
    # transform them first
    # img1 = cv2.subtract(img1, img0)    
    # img2 = cv2.subtract(img2, img0)
    
    # get difference from images
    img_result = cv2.subtract(img1, img2)
    img_result = 255-preprocess_image(img_result, config)
    
    zero = np.zeros_like(img_result)
    img_result = cv2.merge([zero, zero, img_result])
    img_r = cv2.addWeighted(img2_o, 1.0, img_result, 1, 0)
    result_images.append(img_r)
    
    
output_folder = os.path.join(folder, 'out')
os.makedirs(output_folder, exist_ok=True)

for i, im in enumerate(result_images):
    cv2.imwrite(os.path.join(output_folder, f'out_{i:0000}.png'), im)