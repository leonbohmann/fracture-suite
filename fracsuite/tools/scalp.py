"""
Scalp data tools.
"""

import typer
from rich import print
from rich.progress import track
from fracsuite.tools.general import GeneralSettings

from fracsuite.core.specimen import Specimen

scalp_app = typer.Typer(help=__doc__)
general = GeneralSettings.get()

@scalp_app.command()
def fill():
    from openpyxl import load_workbook
    from openpyxl.worksheet.worksheet import Worksheet

    workbook_path = general.get_output_file("..", "Uebersicht.xlsx")
    try:
        workbook = load_workbook(workbook_path)
    except ValueError:
        print(f"Could not load workbook at {workbook_path}. Maybe the filter is still applied?")
        return
    db: Worksheet = workbook['Datenbank']

    specimens = Specimen.get_all()

    sigmas = {
        4: {}, 8: {}, 12: {}
    }
    for spec in specimens:
        if spec.thickness not in sigmas:
            sigmas[spec.thickness] = {}
        if spec.nom_stress not in sigmas[spec.thickness]:
            sigmas[spec.thickness][spec.nom_stress] = {}
        if spec.boundary not in sigmas[spec.thickness][spec.nom_stress]:
            sigmas[spec.thickness][spec.nom_stress][spec.boundary] = {}
        if spec.nbr not in sigmas[spec.thickness][spec.nom_stress][spec.boundary]:
            sigmas[spec.thickness][spec.nom_stress][spec.boundary][spec.nbr] = spec.sig_h


    row = 2
    for i in track(range(row, 800)):
        try:
            t = int(db[f'A{row}'].value)
            s = int(db[f'B{row}'].value)
            l = str(db[f'C{row}'].value)
            i = int(db[f'D{row}'].value)

            db[f'H{row}'].value = sigmas[t][s][l][i]
        except:
            pass

        row += 1

    workbook.save(workbook_path)
    workbook.close()