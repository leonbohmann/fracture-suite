"""
Organisation module. Contains the Specimen class and some helpful tools to export specimens.
"""
from __future__ import annotations
from json import JSONEncoder
import json

import os
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
import typer
from rich import print
from rich.progress import track
from fracsuite.callbacks import main_callback
from fracsuite.core.detection import get_crack_surface, get_crack_surface_r
from fracsuite.core.imageprocessing import preprocess_image
from fracsuite.core.plotting import FigureSize, get_fig_width

from fracsuite.core.specimen import Specimen, SpecimenBoundary
from fracsuite.general import GeneralSettings
from fracsuite.splinters import create_filter_function
from fracsuite.state import State, StateOutput

app = typer.Typer(help=__doc__, callback=main_callback)

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


        s = Specimen(spec_path, log_missing=False, load=True)


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


@app.command()
def create(name):
    Specimen.create(name)


@app.command()
def nfifty(name):
    specimen = Specimen.get(name)


    locations = [
        [400,100],
        [400,400],
        [100,400],
    ]

    nfifties = []

    output_image = specimen.get_fracture_image()

    for loc in locations:
        nfifty_l,spl_l = specimen.calculate_esg_norm(tuple(loc))
        nfifties.append(nfifty_l)

        detail, output_image = specimen.plot_region_count(tuple(loc), (50,50), spl_l, nfifty_l, output_image)


    nfifty = np.mean(nfifties)

    print(f"NFifty: {nfifty:.2f} (+- {np.std(nfifties):.2f})")
    specimen.set_data("nfifty", nfifty)
    specimen.set_data("nfifty_dev", np.std(nfifties))

    State.output(output_image, spec=specimen, figwidth=FigureSize.ROW1)


@app.command()
def to_tex():
    """
    Retrieves all specimens and exports them to a latex file.

    The data is sorted in a table and contains several columns with information about the specimen.
    """
    all_specimens = Specimen.get_all(load=True)

    # sort by thickness, then nominal stress, then boundary and then number
    all_specimens.sort(key=lambda x: (x.thickness, x.nom_stress, x.boundary, x.nbr))

    # define some columns
    def t(s: Specimen):
        return s.thickness
    def stress(s: Specimen):
        return f"{-s.nom_stress:.0f}"
    def boundary(s: Specimen):
        return s.boundary
    def nbr(s: Specimen):
        return s.nbr
    def t_real(s: Specimen):
        if not s.has_scalp:
            return None
        return f"{s.measured_thickness:.2f}"
    def u(s: Specimen):
        if not s.has_scalp:
            return None
        return f"{s.U:.0f}"
    def ud(s: Specimen):
        if not s.has_scalp:
            return None
        return f"{s.U_d:.0f}"
    def stress_real(s: Specimen):
        if not s.has_scalp:
            return None
        return f"{s.sig_h:.2f}"
    def n50(s: Specimen):
        if s.has_fracture_scans and s.has_splinters:
            return f"{s.calculate_nfifty():.2f}"
        else:
            return None

    def farea(s: Specimen):
        return f"{s.crack_surface:.2f}" if s.crack_surface is not None else None

    columns = {
        "$\glsm{t}_{\\text{nom}}$": (t, "mm"),
        "$\glsm{sig_s}_{,\\text{nom}}$": (stress, "MPa"),
        "Lagerung": (boundary, "-"),
        "ID": (nbr, "-"),
        "$t_{\\text{real}}$": (t_real, "mm"),
        "$\glsm{sig_s}_{,\\text{real}}$": (stress_real, "MPa"),
        "$\glsm{fdens}$": (n50, "-"),
        "$\glsm{u}$": (u, "J/m²"),
        "$\glsm{ud}$": (ud, "J/m³"),
        "$\glsm{farea}$": (farea, "mm²")
    }

    # create the table
    table = []
    for spec in all_specimens:
        if spec.boundary == SpecimenBoundary.Unknown:
            continue

        row = []
        for name, (func, unit) in columns.items():
            row.append(func(spec))

        if any([x is None for x in row]):
            continue

        table.append(row)

    # create latex code
    latex_code = r"""
\appendix
\chapter{Daten aller Probekörper}

\begin{longtable}[l]{(COLUMN_DEF)}
	\caption{Zusammenfassung der Datensätze aller Probekörper}
	\label{tab:appendix_datensaetze}\\
	\toprule
	(HEADER) \\
    (HEADER_UNITS) \\
	\midrule
	\endfirsthead
 	\caption*{\cref{tab:appendix_datensaetze} (Fortsetzung)}\\
	\toprule
	(HEADER) \\
    (HEADER_UNITS) \\
	\midrule
	\endhead
    (CONTENT)
    \bottomrule
\end{longtable}
    """
    # column def
    column_def = ""
    for _ in columns:
        column_def += "c"

    latex_code = latex_code.replace("(COLUMN_DEF)", column_def)

    # add header
    header = ""
    header_unit = ""
    for name, (_,unit) in columns.items():
        header += f"\t{name} & "
        header_unit += "\\textcolor{gray}{\\small{{[{0}]}}} & ".replace("{0}", unit if unit != '' else '-')

    latex_code = latex_code.replace("(HEADER)", header[:-2])
    latex_code = latex_code.replace("(HEADER_UNITS)", header_unit[:-2])

    content = ""
    # add data
    for row in table:
        for col in row:
            content += f"\t{col} & "
        content = content[:-2] + "\\\\\n"

    latex_code = latex_code.replace("(CONTENT)", content)

    # save to output directory
    output_file = State.get_output_file("A1_Specimens.tex")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"Saved to '{output_file}'")


@app.command()
def import_fracture(
    specimen_name: str,
    imgsize: tuple[int, int] = (4000, 4000),
    realsize: tuple[float, float] = (500, 500),
):
    """
    Imports fracture images and generates splinters of a specific specimen.

    This function is safe to call because already transformed images are not overwritten and if
    there are already splinters, an overwrite has to be confirmed.
    """
    specimen = Specimen.get(specimen_name, load=True)

    assert specimen.has_fracture_scans, "Specimen has no fracture scans"

    if specimen.has_splinters:
        if not typer.confirm("Specimen already has splinters. Overwrite?"):
            return

    print('[yellow]> Transforming fracture images <')
    img0path, img0 = specimen.transform_fracture_images(size_px=imgsize)

    print('[yellow]> Running threshold tester <')
    from fracsuite.tester import threshold
    threshold(specimen.name)

    print('[yellow]> Generating splinters <')
    from fracsuite.splinters import gen
    gen(specimen.name, realsize=realsize)

    print('[yellow]> Drawing contours <')
    from fracsuite.splinters import draw_contours
    draw_contours(specimen.name)



@app.command()
def test_crack_surface(rust: bool = False, ud: bool = False, aslog: bool = False, overwrite: bool = False):
    filter_func = create_filter_function("*.*.*.*", needs_scalp=True, needs_splinters=True)

    specimens = Specimen.get_all_by(filter_func, load=True)

    surfaces = {}
    ############################
    # calculate crack surface
    for spec in tqdm(specimens):
        if spec.crack_surface is not None and not overwrite:
            print(f"Skipping {spec.name}, already calculated!")
            continue

        # transform splinters to json format
        splinters = spec.splinters
        frac_img = spec.get_fracture_image()
        thickness = spec.measured_thickness

        # preprocess frac_img
        frac_img = preprocess_image(frac_img, spec.get_prepconf(warn=False))
        # print('Image shape', frac_img.shape)
        # print('Image range', np.min(frac_img), np.max(frac_img))


        if not rust:
            crack_surface = get_crack_surface(splinters, frac_img, thickness)
        else:
            crack_surface = get_crack_surface_r(splinters, frac_img, thickness, spec.calculate_px_per_mm())

        surfaces[spec.name] = crack_surface

        spec.set_crack_surface(crack_surface)
        # print(f"{spec.name}: {crack_surface}")

    for name,surface in surfaces.items():
        print(f"{name}: {surface}")


    sz = FigureSize.ROW2
    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    b_marker = {
        'A': 'o',
        'B': 's',
        'Z': 'D'
    }
    t_color = {
        4: 'green',
        8: 'red',
        12: 'blue',
    }

    ##########################
    # plot data
    for spec in specimens:
        v = spec.U_d if ud else spec.U
        axs.scatter(v, spec.crack_surface, marker=b_marker[spec.boundary], color=t_color[spec.thickness])

    vs = np.array([spec.U_d if ud else spec.U for spec in specimens])
    surfs = np.array([spec.crack_surface for spec in specimens])

    ##########################
    # fit a line
    from scipy.optimize import curve_fit
    def func(x, a, b):
        return a * x + b
    popt, pcov = curve_fit(func, vs, surfs)

    # calculate R²
    residuals = surfs - func(vs, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((surfs-np.mean(surfs))**2)
    r_squared = 1 - (ss_res / ss_tot)


    x = np.linspace(np.min(vs), np.max(vs), 100)
    fitting = axs.plot(x, func(x, *popt), 'k-') #, label=f"Fit: {popt[0]:.2f}x + {popt[1]:.2f}"

    # annotate R² to fitting line
    axs.annotate(f"$R^2={r_squared:.2f}$", (x[20],func(x[20], *popt)), ha="left", va="top")


    ##########################
    # create legends
    if ud:
        axs.set_xlabel("Formänderungsenergiedichte $U_d$ (J/m³)")
    else:
        axs.set_xlabel("Formänderungsenergie $U$ (J/m²)")

    for t,c in t_color.items():
        axs.scatter([],[], marker='o', color=c, label=f"$t={t}$mm")

    axs.set_ylabel("Rissfläche $A_\\text{Riss}$ (mm²)")
    axs.legend()

    if aslog:
        axs.set_xscale("log")
        axs.set_yscale("log")

    State.output(StateOutput(fig,sz), "cracksurface_vs_energy" + "" if not ud else "_ud")