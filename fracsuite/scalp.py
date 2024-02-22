"""
Tools for analyzing and importing scalp data.
"""

import glob
from fracsuite.core.logging import debug, info
import os
import shutil
from typing import TypeVar
import numpy as np
import typer
from rich import print
from rich.progress import track
from fracsuite.callbacks import main_callback
from fracsuite.general import GeneralSettings

from fracsuite.core.specimen import Specimen
from fracsuite.scalper.scalp_stress import calculate_simple
from fracsuite.scalper.scalpSpecimen import ScalpProject, ScalpSpecimen

scalp_app = typer.Typer(help=__doc__, callback=main_callback)
general = GeneralSettings.get()

@scalp_app.command()
def fill_sheet():
    """Fill scalp data into ubersicht excel sheet."""
    from openpyxl import load_workbook
    from openpyxl.worksheet.worksheet import Worksheet

    workbook_path = general.get_output_file("..", "Uebersicht.xlsx")
    workbook_path_backup = general.get_output_file("..", "Uebersicht_backup.xlsx")
    shutil.copy(workbook_path, workbook_path_backup)

    try:
        workbook = load_workbook(workbook_path)
    except ValueError:
        print(f"Could not load workbook at {workbook_path}. Maybe the filter is still applied?")
        return
    db: Worksheet = workbook['Datenbank']

    specimens = Specimen.get_all(load=True)

    sigmas = {
        4: {}, 8: {}, 12: {}
    }
    has_fracture = {
        4: {}, 8: {}, 12: {}
    }
    for spec in specimens:
        if spec.thickness not in sigmas:
            sigmas[spec.thickness] = {}
            has_fracture[spec.thickness] = {}
        if spec.nom_stress not in sigmas[spec.thickness]:
            sigmas[spec.thickness][spec.nom_stress] = {}
            has_fracture[spec.thickness][spec.nom_stress] = {}
        if spec.boundary not in sigmas[spec.thickness][spec.nom_stress]:
            sigmas[spec.thickness][spec.nom_stress][spec.boundary] = {}
            has_fracture[spec.thickness][spec.nom_stress][spec.boundary] = {}
        if spec.nbr not in sigmas[spec.thickness][spec.nom_stress][spec.boundary]:
            sigmas[spec.thickness][spec.nom_stress][spec.boundary][spec.nbr] = spec.sig_h
            has_fracture[spec.thickness][spec.nom_stress][spec.boundary][spec.nbr] = spec.has_splinters or spec.has_fracture_scans


    row = 2
    for i in track(range(row, 800)):
        try:
            t = int(db[f'A{row}'].value)
            s = int(db[f'B{row}'].value)
            l = str(db[f'C{row}'].value)
            i = int(db[f'D{row}'].value)

            db[f'H{row}'].value = sigmas[t][s][l][i]

            if has_fracture[t][s][l][i]:
                db[f'G{row}'].value = "Zerstört"
        except Exception:
            pass

        row += 1

    workbook.save(workbook_path)
    workbook.close()





T = TypeVar("T")
def flatten_and_sort(l: list[list[T]], sort_mtd) -> list[T]:
    ret = [item for sublist in l for item in sublist]
    ret.sort(key = sort_mtd)
    return ret

def remove_duplicates(my_list, my_lambda):
    new_list = []
    for item in my_list:
        exists = False
        for ex_item in new_list:
            if my_lambda(ex_item) == my_lambda(item):
                exists = True
                print(f'Found and removed duplicate specimen: {item}')

        if exists:
            continue

        new_list.append(item)

    return new_list


def get_specimens_from_projects(projects: list[ScalpProject]) -> list[ScalpSpecimen]:
    """Extract all specimens in a distinct list from a list of projects.

    Args:
        projects (list[ScalpProject]): List of scalp projects.

    Returns:
        list[Specimen]: Ordered and distinct list of all specimens.
    """
    specimens = flatten_and_sort([x.specimens for x in projects], lambda s: s.name)
    specimens = [x for x in specimens if not x.invalid]
    specimens = remove_duplicates(specimens, lambda t: t.name)

    return specimens


@scalp_app.command()
def transform(
    folder: str = typer.Argument(None, help="Folder to transform."),
    display_mohr: bool = typer.Option(False, "--mohr", help="Display mohr's circle."),
    load_to_db: bool = typer.Option(False, "--to-db", help="Load to database."),
):
    SWITCH_DISPLAY_MOHR_CIRCLE = display_mohr

    output_directory = os.path.join(folder, "out")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    projects: list[ScalpProject] = []
    extension = '.scp'
    files = glob.glob(os.path.join(folder, '**', f'*{extension}'), recursive=True)

    for file in files:
        print(f'[green]Analyze[/green] file: {file.replace(folder, "")}')
        project = ScalpProject(file)
        projects.append(project)
        for spec in project.specimens:
            print(f'\t{"!!!" if spec.invalid else ""}\t{spec.name:10}: {len(spec.measurementlocations)} Locations ({[f"{x.location_name} ({len(x.measurements)})" for x in spec.measurementlocations]})')

    # write a summary
    project_len = len(projects)
    measurements_len = np.sum([len(x.measurements) for x in projects])

    print(f'Projects       : {project_len}')
    print(f'Measurements   : {measurements_len}')

    # output all specimens independently from their project
    specimens = get_specimens_from_projects(projects)
    for specimen in specimens:

        pre = ""
        if not all(not t.invalid for t in specimen.measurementlocations):
            pre = "!!!"

        print(f'\t{pre}\t{specimen.name:10}: {len(specimen.measurementlocations)} Locations ({[(x.location_name, [l.orientation for l in x.measurements]) for x in specimen.measurementlocations]})')


    # calculate stresses for specimens
    for specimen in specimens:

        print(f'\t{specimen.name:10}: sig_h={specimen.sig_h:.2f} (+- {specimen.sig_h.deviation:.2f})MPa')
        print('\t\t', end="")

        for loc in specimen.measurementlocations:
            print(f'\t{loc.location_name}={loc.stress[0]:.2f}/{loc.stress[1]:.2f}', end = "")

        print()

    # write specimen
    for specimen in specimens:
        specimen.write_measurements(output_directory, "txt")


    if not load_to_db:
        return


    # copy all folder in the output_directory to the correct folders in the database
    for specimen in specimens:
        spec = Specimen.get(specimen.name, printout=False, load=False)
        debug(f'Loaded {specimen.name}')

        # copy all folder contents to spec
        scalp_folder = spec.scalp_folder
        debug(f'Scalp-Folder: {scalp_folder}')

        shutil.copytree(
            os.path.join(output_directory, specimen.name, 'scalp'),
            os.path.abspath(scalp_folder),
            dirs_exist_ok=True
        )

        info(f'Copy {specimen.name} to database.')