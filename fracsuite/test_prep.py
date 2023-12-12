"""
TESTS for the preprocessing unit of this module.
"""

import time

import cv2
import numpy as np
import typer
from matplotlib import pyplot as plt
# from pathos.multiprocessing import ProcessPool

from fracsuite.core.image import to_gray
from fracsuite.core.progress import get_spinner
from fracsuite.core.specimen import Specimen
from fracsuite.core.splinter import Splinter
from fracsuite.callbacks import main_callback
from fracsuite.general import GeneralSettings
from fracsuite.helpers import img_part
from fracsuite.state import State

general = GeneralSettings.get()

test_prep_app = typer.Typer(help=__doc__, callback=main_callback)

@test_prep_app.command()
def disp_mean(specimen_name: str,):
    """Display the mean value of a fracture image."""
    specimen = Specimen.get(specimen_name)
    mean_value = np.mean(specimen.get_fracture_image())
    print(mean_value)

    im = (to_gray(specimen.get_fracture_image())-mean_value).astype(np.uint8)
    im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    plt.imshow(im)
    plt.show()

region = (200,100, 250, 250)

@test_prep_app.command()
def test_watershed_count():
    specimen = Specimen.get('.test01')

    splinters = Splinter.analyze_image(specimen.get_fracture_image())

    im0 = specimen.get_fracture_image()
    cv2.drawContours(im0, [x.contour for x in splinters], -1, (0,0,255), 2)

    im0 = img_part(im0, *region)
    State.output(im0, override_name='watershed_count')
    print(len(splinters))

@test_prep_app.command()
def test_legacy_count():
    specimen = Specimen.get('.test01')

    splinters = Splinter.analyze_image_legacy(specimen.get_fracture_image())

    im0 = specimen.get_fracture_image()
    cv2.drawContours(im0, [x.contour for x in splinters], -1, (0,255,255), 2)

    im0 = img_part(im0, *region)

    State.output(im0, override_name='legacy_count')
    print(len(splinters))