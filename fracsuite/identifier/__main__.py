import os
import cv2

from argparse import ArgumentParser
from itertools import groupby

from fracsuite.splinters.analyzer import crop_perspective
from fracsuite.identifier import read_barcode
from fracsuite.splinters.analyzerConfig import AnalyzerConfig

def get_unique_scan_id(file) -> str:
    """Get the first 4 numbers of cullet scanner file name.
    Format is: [ID] [Date] ([img_type]).[ext]

    Args:
        file (str): Full file path.

    Returns:
        str: File identifier.
    """
    return os.path.basename(file)[:4]

parser = ArgumentParser()
parser.add_argument('directory', type=str, help='Directory with scans.')
args = parser.parse_args()

config = AnalyzerConfig()


root_dir = args.directory


# read input folder and group files
extensions = [".bmp", ".zip"]   
files: list[str] = [os.path.join(root_dir, f) \
        for f in os.listdir(root_dir) \
            if os.path.isfile(os.path.join(root_dir, f)) \
            and any(f.endswith(ext) for ext in extensions)]

sorted_files = sorted(files, key=get_unique_scan_id)
# group files for file id
grouped_files = {k: list(g) for k, g in groupby(sorted_files, key=get_unique_scan_id)}


for key, group in grouped_files.items():
    
    if not any(file.endswith(".bmp") and "Transmission" in file for file in group):
        print(f"Couldn't find Transmission-Scan for scan ID {key}!")
        continue
    
    # get bitmaps from group
    img0_path = next(file for file in group if file.endswith(".bmp") and "Transmission" in file)
    # load image and perform OCR to find specimen Identifier ([thickness].[residual_stress].[boundary].[ID])
    img0 = cv2.imread(img0_path)
    img0 = crop_perspective(img0, (4000,4000), False)    
    img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    series = read_barcode(img0)
    
    print(series)
    
    # # iterate over each file in group
    # for file in group:
    #     # load image
    #     img = cv2.imread(file)
    #     # crop image
    #     img = crop_perspective(img, 4000, True, False)
    #     # rotate image
    #     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     # save image
    #     cv2.imwrite(file, img)
        
    #     # rename image
    #     new_name = f"{series}_{os.path.basename(file)}"
    #     os.rename(file, os.path.join(root_dir, new_name))