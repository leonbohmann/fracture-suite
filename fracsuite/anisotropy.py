

import os
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE
import cv2
import typer
from fracsuite.callbacks import main_callback
from fracsuite.core.imageprocessing import crop_perspective
from fracsuite.core.specimen import Specimen
from fracsuite.splinters import create_filter_function

from rich import print

ani_app = typer.Typer(callback=main_callback, help=__doc__)

@ani_app.command()
def image_transform(
    specimen_range: str,
    image_size: tuple[int,int] = typer.Option((4000,4000), help="Size of the images."),
    rotate: bool = typer.Option(True, help="Rotate images by 90 degrees."),
    overwrite: bool = typer.Option(False, help="Overwrite existing images."),
):
    """
    Transforms all anisotropy images of a specimen to the correct orientation.
    If the images have already been transformed, this command ignores them.

    By default, the images are rotated by 90 degrees. This can be disabled with
    the --no-rotate flag. Normally, when scanned on the CulletScanner, the panes
    are rotated 90 degrees clockwise.
    """
    # show confirmation if overwrite is true
    if overwrite:
        typer.confirm("Overwriting existing images?", abort=True)

    filter = create_filter_function(specimen_range, needs_scalp=False, needs_splinters=False)
    specimens: list[Specimen] = Specimen.get_all_by(filter, load=True)

    img_size = image_size

    for specimen in specimens:
        image_paths = specimen.anisotropy.all_paths

        for img_path in image_paths:
            if img_path is None:
                print(f'[green]{specimen.name}[/green]: Some file was missing...')
                continue

            if not os.access(img_path, os.W_OK) and not overwrite:
                print(f"Skipping '[dim]{os.path.basename(img_path)}[/dim]', no write access.")
                continue
            if not os.access(img_path, os.W_OK) and overwrite:
                os.chmod(img_path, S_IWRITE)
            print(f'[green]{specimen.name}[/green]: [dim]{os.path.basename(img_path)}[/dim]')

            img = cv2.imread(img_path)

            # rotate the original image by 90 degrees
            if rotate:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # crop the image to only fit the glass pane
            img1 = crop_perspective(img, img_size, False)

            # overwrite the original
            cv2.imwrite(img_path, img1)

            # make the image read-only, so future runs ignore this image
            os.chmod(img_path, S_IREAD | S_IRGRP | S_IROTH)
