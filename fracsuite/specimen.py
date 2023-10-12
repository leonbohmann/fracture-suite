"""
Organisation module. Contains the Specimen class and some helpful tools to export specimens.
"""
from __future__ import annotations

import os

import numpy as np
import typer
from rich import print
from rich.progress import track

from fracsuite.core.specimen import Specimen
from fracsuite.general import GeneralSettings

app = typer.Typer(help=__doc__)

general = GeneralSettings.get()

@app.command()
def sync():
    """
    Sync all specimen configs.

    This function iterates over all splinters and syncs their specimen configs.
    """
    # iterate over all splinters
    for name in track(os.listdir(general.base_path), description="Syncing specimen configs...", transient=False):

        spec_path = general.get_output_file(name)
        if not os.path.isdir(spec_path):
            continue


        s = Specimen(spec_path, log_missing=False, lazy=True)


@app.command()
def export():
    """
    Export all specimen configs to a single excel file.

    This function exports all specimen configurations to a single Excel file. The exported file contains
    information about each specimen's name, thickness, pre-stress, boundary, number, comment, break mode,
    break position, real pre-stress, standard deviation, and mean splinter size.
    """
    import xlsxwriter

    workbook_path = general.get_output_file("summary1.xlsx")

    workbook = xlsxwriter.Workbook(workbook_path)

    # The workbook object is then used to add new
    # worksheet via the add_worksheet() method.
    worksheet = workbook.add_worksheet()


    worksheet.write(0, 0, "Boundary: A (allseitig), Z (zweiseitig), B (gebettet)")
    worksheet.write(1, 0, "Comment: B (Bohrung)")

    start_row = 10

    worksheet.write(start_row, 0, "Name")
    worksheet.write(start_row, 1, "Thickness")
    worksheet.write(start_row, 2, "Pre-Stress")
    worksheet.write(start_row, 3, "Boundary")
    worksheet.write(start_row, 4, "Nbr")
    worksheet.write(start_row, 5, "Comment")
    worksheet.write(start_row, 6, "Break-Mode")
    worksheet.write(start_row, 7, "Break-Position")
    worksheet.write(start_row, 8, "Real pre-stress")
    worksheet.write(start_row, 9, "(std-dev)")
    worksheet.write(start_row, 10, "Mean splinter size")


    row = start_row + 1
    for name in track(os.listdir(general.base_path), description="Syncing specimen configs...", transient=False):

        spec_path = os.path.join(general.base_path, name)
        if not os.path.isdir(spec_path):
            continue


        s = Specimen(spec_path, log_missing=False)
        # extract data
        worksheet.write(row, 0, s.name)
        worksheet.write(row, 1, s.thickness)
        worksheet.write(row, 2, s.nom_stress)
        worksheet.write(row, 3, s.boundary)
        worksheet.write(row, 4, s.nbr)
        worksheet.write(row, 5, s.comment)
        worksheet.write(row, 6, s.settings['break_mode'])
        worksheet.write(row, 7, s.settings['break_pos'])
        if s.has_scalp:
            worksheet.write(row, 8, s.scalp.sig_h)
            worksheet.write(row, 9, s.scalp.sig_h_dev)
        if s.has_splinters:
            worksheet.write(row, 10, np.mean([x.area for x in s.splinters]))




        row += 1
        del s



    # Finally, close the Excel file
    # via the close() method.
    workbook.close()

    os.system(f'start {workbook_path}')


@app.command()
def list_all(setting: str = None, value: str = None):
    all = Specimen.get_all(load=False)
    print("Name\tSetting\tValue")
    for spec in all:

        if setting not in spec.settings:
            continue
        if value is not None and spec.settings[setting] != value:
            continue

        print(spec.name, end="")

        for s, k in spec.settings.items():

            print(f"\t{k}", end="")

        print()

@app.command()
def disp(filter: str = None):
    data = Specimen.get_all(filter)

    for s in data:
        if s.has_scalp:
            print(f"{s.name}: {s.sig_h:.2f} (+- {s.sig_h.deviation:.2f})MPa")