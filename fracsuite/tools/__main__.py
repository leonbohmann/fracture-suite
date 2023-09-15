import os

import cv2
import typer
from matplotlib import pyplot as plt
from rich.progress import track
from typing_extensions import Annotated

from fracsuite.splinters.analyzer import crop_perspective
from fracsuite.tools.config import app as config_app
from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.plot import app as plt_app
from fracsuite.tools.specimen import fetch_specimens, app as specimen_app

plt.rcParams['figure.figsize'] = (6, 4)
plt.rc('axes', axisbelow=True) # to get grid into background
plt.rc('grid', linestyle="--") # line style
plt.rcParams.update({'font.size': 12}) # font size

general = GeneralSettings()

app = typer.Typer(pretty_exceptions_short=False)
app.add_typer(plt_app, name="plot")
app.add_typer(config_app, name="config")
app.add_typer(specimen_app, name="specimen")

    
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
  
@app.command(name='crop_frac')
def crop_fracture_morph(
    specimen_name: Annotated[str, typer.Argument(help='Name of specimen to load')],
    rotate: Annotated[bool, typer.Option(help='Rotate image by 90°.')] = False,
    rotate_only: Annotated[bool, typer.Option(help='Only rotate image by 90°, skip cropping.')] = True,
):
    from stat import S_IREAD, S_IRGRP, S_IROTH
    
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    path = specimen.fracture_morph_dir
    
    for file in track(os.listdir(path)):
        if file.endswith(".bmp"):
            img = cv2.imread(os.path.join(path, file))
            
            if not rotate_only:
                img = crop_perspective(img, (4000,4000), False)
            
            if rotate or rotate_only:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            cv2.imwrite(os.path.join(path, file), img)
            os.chmod(os.path.join(path, file), S_IREAD|S_IRGRP|S_IROTH)
       
    
app()


