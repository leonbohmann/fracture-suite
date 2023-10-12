import os
import cv2
import re
import shutil

from argparse import ArgumentParser
from itertools import groupby
from fracsuite.identifier.barcode import dispImage

from fracsuite.splinters.analyzer import crop_perspective
from fracsuite.identifier import read_barcode
from fracsuite.splinters.analyzerConfig import AnalyzerConfig

from rich import print

from fracsuite.general import GeneralSettings

general = GeneralSettings.get()

def get_unique_scan_id(file) -> str:
    """Get the first 4 numbers of cullet scanner file name.
    Format is: [ID] [Date] ([img_type]).[ext]

    Args:
        file (str): Full file path.

    Returns:
        str: File identifier.
    """
    return os.path.basename(file)[:4]

def get_scan_type(input: str) -> str:
    """
    Function to extract content between the last two brackets in a filename.
    Args:
        filename (str): The filename from which content is to be extracted.
    Returns:
        str: The content between the last two brackets.
    """
    # Find all substrings that match the pattern
    matches = re.findall(r'\[.*?\]', input)

    # If there are matches, return the content of the last match
    # without the brackets. Otherwise, return an empty string.
    if matches:
        return matches[-1][1:-1]
    else:
        return ''

parser = ArgumentParser()
parser.add_argument('directory', type=str, help='Directory with scans.')
parser.add_argument('--dry', action="store_true", help='Perform dry run that only reads codes.')
parser.add_argument('-debug', choices=['label', 'code', 'barcode', 'error'], nargs="*", help='Show debug images.')

args = parser.parse_args()

if args.debug:
    print("[yellow]Debug mode enabled![/yellow]")

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

# create archive folder
archive_folder = os.path.join(root_dir, ".archive")
os.makedirs(archive_folder, exist_ok=True)

for key, group in grouped_files.items():
    if not any(file.endswith(".bmp") and "Transmission" in file for file in group):
        print(f"Couldn't find Transmission-Scan for scan ID {key}!")
        continue

    # get bitmaps from group
    img0_path = next(file for file in group if file.endswith(".bmp") and "Transmission" in file)
    # load image and perform OCR to find specimen Identifier ([thickness].[residual_stress].[boundary].[ID])
    img0 = cv2.imread(img0_path)
    img0 = crop_perspective(img0, None, False)
    img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    dispImage(img0, "Original", "Main", args.debug)

    series = read_barcode(img0, args.debug)

    # if None, Datamatrix could not be read
    if series is None:
        print(f"Couldn't find series for scan ID {key}!")
        continue

    print(f"Found {series} for ID {key}...")

    if args.dry:
        continue

    series = os.path.join(root_dir, series)
    subfolder = os.path.join(series, "anisotropy")
    if os.path.exists(subfolder):
        print(f"Scan ID {key} already processed!")
        continue

    # create a subfolder for the scans called "anisotropy"
    os.makedirs(subfolder, exist_ok=True)

    # iterate over each file in group, run perspective transform on bitmaps only and copy all into the subfolder
    for file in group:
        if file.endswith(".bmp"):
            img = cv2.imread(file)
            img = crop_perspective(img, general.default_image_size_px, False)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(subfolder, f'{get_scan_type(file)}.bmp'), img)

            # move source image to archive
            os.rename(file, os.path.join(archive_folder, os.path.basename(file)))
        else:
            # copy file to archive
            shutil.copy(file, os.path.join(archive_folder, os.path.basename(file)))

            os.rename(file, os.path.join(subfolder, os.path.basename(file)))
