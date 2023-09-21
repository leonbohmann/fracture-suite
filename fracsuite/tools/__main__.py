import os
import time

import cv2
import typer
from matplotlib import pyplot as plt
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from fracsuite.splinters.processing import crop_matrix, crop_perspective

from fracsuite.tools.config import app as config_app
from fracsuite.tools.splinters import app as splinter_app
from fracsuite.tools.acc import app as acc_app
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import find_files
from fracsuite.tools.specimen import Specimen, app as specimen_app
from fracsuite.tools.test_prep import test_prep_app
from fracsuite.tools.nominals import nominals_app
from fracsuite.tools.scalp import scalp_app

plt.rcParams['figure.figsize'] = (6, 4)
plt.rc('axes', axisbelow=True) # to get grid into background
plt.rc('grid', linestyle="--") # line style
plt.rcParams.update({'font.size': 12}) # font size

general = GeneralSettings.get()

app = typer.Typer(pretty_exceptions_short=False)
app.add_typer(splinter_app, name="splinters")
app.add_typer(config_app, name="config")
app.add_typer(specimen_app, name="specimen")
app.add_typer(acc_app, name="acc")
app.add_typer(scalp_app, name="scalp")
app.add_typer(test_prep_app, name="test-prep")
app.add_typer(nominals_app, name="nominals")

@app.command()
def test(parallel:bool = False):
    time0 = time.time()
    all = Specimen.get_all_by(lambda x: "SCHOTT" not in x.name)
    time1 = time.time()
    print(f"Loading all specimens took {time1-time0:.2f}s.")

@app.command()
def test_find_by_filter(filter: str):
    all = Specimen.get_all(filter)

    for specimen in all:
        print(specimen.name)

@app.command()
def marina_organize(path: str):
    # find all folders in path that contain three dots
    for dir in os.listdir(path):
        if dir.count(".") != 3:
            continue

        dirpath = os.path.join(path, dir)

        # create subdirectors
        os.makedirs(os.path.join(dirpath, "scalp"), exist_ok=True)
        os.makedirs(os.path.join(dirpath, "fracture"), exist_ok=True)
        acc_path = os.path.join(dirpath, "fracture", "acceleration")
        os.makedirs(acc_path, exist_ok=True)
        morph_path = os.path.join(dirpath, "fracture", "morphology")
        os.makedirs(morph_path, exist_ok=True)

        # put all .bmp files into morphology folder
        for file in os.listdir(dirpath):
            if file.endswith(".bmp") or file.endswith(".zip"):
                os.rename(os.path.join(dirpath, file), os.path.join(morph_path, file))

        # put all .tsx, .tst, .xlsx and .bin files into acceleration folder
        for file in os.listdir(dirpath):
            if file.lower().endswith(".tsx") or file.lower().endswith(".tst") or file.lower().endswith(".xlsx") or file.lower().endswith(".bin"):
                os.rename(os.path.join(dirpath, file), os.path.join(acc_path, file))

        # in morphology folder, search for filename that starts with a 4digit number and prepend it to the file
        # that contains "Transmission" in its name, if it does not start with the same 4digit num
        for file in os.listdir(morph_path):
            if "Transmission" in file:
                continue

            # find 4digit number in filename
            num = None
            if file[0].isdigit() and file[0+1].isdigit() and file[0+2].isdigit() and file[0+3].isdigit():
                num = file[:4]

            if num is None:
                continue

            # find file with "Transmission" in its name
            for file2 in os.listdir(morph_path):
                if "Transmission" in file2 and num not in file2:
                    os.rename(os.path.join(morph_path, file2), os.path.join(morph_path, num + " " + file2))
                    break

@app.command(name='crop-frac')
def crop_fracture_morph(
    specimen_name: Annotated[str, typer.Option(help='Name of specimen to load')] = "",
    all: Annotated[bool, typer.Option('--all', help='Perform this action on all specimen.')] = False,
    rotate: Annotated[bool, typer.Option('--rotate', help='Rotate image by 90°.')] = False,
    crop: Annotated[bool, typer.Option('--crop', help='Crop the image.')] = True,
    size: Annotated[tuple[int,int], typer.Option(help='Image size.', metavar='Y X')] = general.default_image_size_px,
    rotate_only: Annotated[bool, typer.Option('--rotate-only', help='Only rotate image by 90°, skip cropping.')] = False,
    resize_only: Annotated[bool, typer.Option('--resize_only', help='Only resize the image to 4000px².')] = False,
):
    f"""Crop and resize fracture morphology images. Can run on all specimens, several or just one single one.

    Args:
        specimen_name (Annotated[str, typer.Option, optional): The specimen names. Defaults to 'Name of specimen to load')]
        all (bool, optional): Run the method on all specimens. Defaults to False.
        rotate (bool, optional): Rotate the input image 90° CCW. Defaults to False.
        crop (bool, optional): Crop the input image to ply bounds. Defaults to True.
        size (tuple[int,int], optional): Size of the image. Defaults to {general.default_image_size_px}.
        rotate_only (bool, optional): Only rotate the images. Defaults to False.
        resize_only (bool, optional): Only resizes the images. Defaults to False.
    """
    from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE
    if all:
        specimens = Specimen.get_all()
    elif isinstance(specimen_name, Specimen):
        specimens = [specimen_name]
    else:
        specimens = Specimen.get_all(specimen_name)


    for specimen in track(specimens):
        path = specimen.fracture_morph_dir
        if not  os.path.exists(path):
            continue

        imgs = [(x,cv2.imread(x, cv2.IMREAD_GRAYSCALE)) for x in find_files(path, '*.bmp')]

        if len(imgs) == 0:
            continue

        img0 = [y for x,y in imgs if "Transmission" in x][0]
        _, M0 = crop_perspective(img0, size, False, True)

        for file,img in imgs:
            if not os.access(file, os.W_OK):
                print(f"Skipping '{os.path.basename(file)}', no write access.")
                continue

            if resize_only:
                img = cv2.resize(img, size)

            if not rotate_only and crop and not resize_only:
                img = crop_matrix(img, M0, size)

            if (rotate or rotate_only) and not resize_only:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(file, img)
            os.chmod(os.path.join(path, file), S_IREAD|S_IRGRP|S_IROTH)

        # elif file.endswith(".bmp") and not os.access(os.path.join(path, file), os.W_OK):
        #     os.chmod(os.path.join(path, file), S_IWRITE)

app()
