"""
Organisation module. Contains the Specimen class and some helpful tools to export specimens.
"""
from __future__ import annotations
from json import JSONEncoder
import json

import os
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect

import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import typer
from rich import inspect, print
from rich.progress import track
from fracsuite.callbacks import main_callback
from fracsuite.core.detection import get_crack_surface, get_crack_surface_r
from fracsuite.core.imageprocessing import preprocess_image
from fracsuite.core.plotting import FigureSize, get_fig_width

from fracsuite.core.specimen import Specimen, SpecimenBoundary
from fracsuite.core.stress import relative_remaining_stress
from fracsuite.general import GeneralSettings
from fracsuite.splinters import create_filter_function
from fracsuite.state import State, StateOutput

app = typer.Typer(help=__doc__, callback=main_callback)

general = GeneralSettings.get()

@app.command()
def checksettings(name):
    """Check the settings of a specimen."""
    spec = Specimen.get(name)

    inspect(spec.settings)

@app.command()
def put(name, setting, value):
    if setting not in Specimen.setting_keys():
        print(f"Setting '{setting}' not found!")
        return

    spec = Specimen.get(name)

    spec.set_setting(setting, value)

@app.command()
def check(areas: bool = False):
    """
    Sync all specimen configs.

    This function iterates over all splinters and syncs their specimen configs.
    """
    specs = Specimen.get_all()

    desired_img_size = (4000, 4000)
    desired_real_size = (500, 500)
    desired_splinter_area = 490**2 - np.pi * 20**2

    if areas:
        areas_values = []
        for specimen in track(specs, description="Calculating splinter areas...",transient=True):
            if not specimen.has_splinters:
                continue
            areas_values.append(specimen.splinter_area)

            discr = abs(1 - specimen.splinter_area / desired_splinter_area) * 100
            print(f"{specimen.name}: {discr:.2f}%")

    for specimen in track(specs, description='Checking specimens...', transient=True):
        if not specimen.has_fracture_scans:
            continue

        img_size = specimen.get_image_size()

        if img_size[0] != desired_img_size[0] or img_size[1] != desired_img_size[1]:
            print(f"Image size of {specimen.name} is {img_size}, resize to {desired_img_size}!")

        real_size = specimen.get_real_size()
        if real_size[0] != desired_real_size[0] or real_size[1] != desired_real_size[1]:
            print(f"Real size of {specimen.name} is {real_size}, resize to {desired_real_size}!")

        # print percentage of splinter area
        if specimen.has_splinters:
            discr = abs(1 - specimen.splinter_area / desired_splinter_area) * 100
            if discr > 5:
                print(f"[yellow]AREA WARNING[/yellow] '{specimen.name}': [red]{discr:.2f}[/red]%")

        if specimen.has_adjacency:
            print(f'[green]{specimen.name} has adjacency!')

        if specimen.broken_immediately:
            print(f'[green]{specimen.name} broke immediately!')
        else:
            print(f'[red]{specimen.name} did not break immediately!')


marked_pos = None
@app.command()
def mark_center(name):
    specimen = Specimen.get(name)

    if not specimen.has_fracture_scans:
        print("Specimen has no fracture scans!")
        return

    img = specimen.get_fracture_image()
    img = preprocess_image(img, specimen.get_prepconf(warn=False))

    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    axs.imshow(img, cmap='gray')
    axs.set_title(f"Mark center of {name}")


    px_p_mm = specimen.calculate_px_per_mm()

    # when clicking, mark the position but dont save yet
    def onclick(event):
        global marked_pos

        x = int(event.xdata)
        y = int(event.ydata)

        # display the marked point and remove the previous one
        if hasattr(onclick, 'mark'):
            onclick.mark.remove()

        onclick.mark = axs.plot(x, y, 'ro', markersize=8)[0]

        # save the marked position
        marked_pos = np.asarray([x,y]) / px_p_mm
        print(f"Marked center at {marked_pos}")
        fig.canvas.draw()


    fig.canvas.mpl_connect('button_press_event', onclick)
    # make cursor red
    fig.canvas.mpl_connect('motion_notify_event', lambda event: plt.gcf().canvas.set_cursor(1))

    plt.show(block=True)

    global marked_pos
    print(f"Marked position: {marked_pos}")
    specimen.set_setting(Specimen.SET_ACTUALBREAKPOS, tuple(marked_pos))


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
def to_tex(exclude_thickness: int = None):
    """
    Retrieves all specimens and exports them to a latex file.

    The data is sorted in a table and contains several columns with information about the specimen.
    """
    all_specimens = Specimen.get_all(load=True)

    # sort by thickness, then nominal stress, then boundary and then number
    all_specimens.sort(key=lambda x: (x.thickness, x.nom_stress, x.boundary, x.nbr))

    if exclude_thickness is not None:
        all_specimens = [x for x in all_specimens if x.thickness != exclude_thickness]

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

    def u0(s: Specimen):
        if not s.has_scalp:
            return None
        sz = s.get_real_size()
        A = sz[0]*1e-3 * sz[1]*1e-3
        return f"{s.U*A:.1f}"

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
        return f"{s.crack_surface*1e-6:.2f}" if s.crack_surface is not None else None

    columns = {
        "$\glsm{t}_{\\text{nom}}$": (t, "mm"),
        "$\glsm{sig_s}_{,\\text{nom}}$": (stress, "MPa"),
        "Lagerung": (boundary, None),
        "ID": (nbr, None),
        "$t_{\\text{real}}$": (t_real, "mm"),
        "$\glsm{sig_s}_{,\\text{real}}$": (stress_real, "MPa"),
        "$\glsm{fdens}$": (n50, None),
        "$\glsm{ut}$": (u0, "J"),
        "$\glsm{u}$": (u, "J/m²"),
        "$\glsm{ud}$": (ud, "J/m³"),
        "$\glsm{farea}$": (farea, "m²"),
    }

    # create the table
    table = []
    for spec in track(all_specimens, description='Creating rows...'):
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
        if unit is not None:
            header_unit += "\\textcolor{gray}{\\small{{({0})}}} & ".replace("{0}", unit)
        else:
            header_unit += "\t & "

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
def import_experimental_data(
    file: str
):
    # load csv file
    with open(file, 'r') as f:
        # skip first row
        data: list[str] = f.readlines()[1:]

    col = 7

    for r in track(data, total=len(data)):
        r = r.strip()
        v = r.split(';')
        d = np.array(v[1:]).astype(np.float64)

        name = str(v[0])
        print(name)
        specimen = Specimen.get(name, load=False, panic=False)

        if specimen is not None:
            new_height = float(d[7])*1e-2
            old_heiht = float(specimen.fall_height_m)
            if new_height != old_heiht:
                print(f"Fall height of {name} changed from {old_heiht} to {new_height}!")

                specimen.set_setting(Specimen.SET_FALLHEIGHT, new_height)
@app.command()
def import_fracture(
    specimen_name: str,
    imgsize: tuple[int, int] = (4000, 4000),
    realsize: tuple[float, float] = (500, 500),
    imsize_factor: float = None,
    no_rotate: bool = False,
    no_tester: bool = False,
    exclude_all_sensors: bool = False,
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

    if imsize_factor is not None:
        imgsize = (int(realsize[0] * imsize_factor), int(realsize[1] * imsize_factor))

    # set settings on specimen
    specimen.set_setting(Specimen.SET_EXCLUDE_ALL_SENSORS, exclude_all_sensors)

    print('[yellow]> Transforming fracture images <')
    img0path, img0 = specimen.transform_fracture_images(size_px=imgsize, rotate=not no_rotate)

    print('[yellow]> Running threshold tester <')
    if not no_tester:
        from fracsuite.tester import threshold
        threshold(specimen.name)

    print('[yellow]> Marking impact point <')
    mark_center(specimen.name)

    print('[yellow]> Generating splinters <')
    from fracsuite.splinters import gen
    gen(specimen.name, realsize=realsize)

    print('[yellow]> Drawing contours <')
    from fracsuite.splinters import draw_contours
    draw_contours(specimen.name)



@app.command()
def test_crack_surface(rust: bool = False, ud: bool = False, aslog: bool = False, overwrite: bool = False):
    filter_func = create_filter_function("*.*.B.*", needs_scalp=True, needs_splinters=True)

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


    sz = FigureSize.ROW1HL
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

    def rud(u):
        return u * 0.25

    ##########################
    # plot data
    for spec in specimens:
        v = spec.U_d if ud else rud(spec.U)
        axs.scatter(v, spec.crack_surface, marker=b_marker[spec.boundary], color=t_color[spec.thickness])


    ##########################
    # fit a line
    from scipy.optimize import curve_fit
    def func(x, a, b):
        return a * x + b
    for t in ([4,8,12] if ud else [None]):
        vs = np.array([spec.U_d if ud else rud(spec.U) for spec in specimens if t is None or spec.thickness == t])
        surfs = np.array([spec.crack_surface for spec in specimens if t is None or spec.thickness == t])

        popt, pcov = curve_fit(func, vs, surfs)

        # calculate R²
        residuals = surfs - func(vs, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((surfs-np.mean(surfs))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # plot fitted curve
        x = np.min(vs) + (np.max(vs) - np.min(vs))*0.6

        # create axline
        axs.axline((x,popt[1]+popt[0]*x), slope=popt[0], color=t_color[t] if t is not None else 'k', linestyle='--', linewidth=0.5)

        # annotate R² to fitting line
        axs.annotate(f"$R^2={r_squared:.2f}$ m={popt[0]:.2e}", (x,func(x, *popt)), ha="left", va="top")
        print(f"> t={t}mm: m={popt[0]:.2e}mm²/J, b={popt[1]:2e}, R²={r_squared:.2f}")
        print(f'\t{1/popt[0]:.2e}J/mm²')

    ##########################
    # create legends
    if ud:
        axs.set_xlabel("Formänderungsenergiedichte $U_d$ (J/m³)")
    else:
        axs.set_xlabel("Gesamtenergie $U_t$ (J)")

    for t,c in t_color.items():
        axs.scatter([],[], marker='o', color=c, label=f"{t}mm")

    axs.set_ylabel("Rissfläche $A_\\text{Riss}$ (mm²)")
    axs.legend()

    if aslog:
        axs.set_xscale("log")
        axs.set_yscale("log")

    State.output(StateOutput(fig,sz), "cracksurface_vs_energy" + ("" if not ud else "_ud"))

@app.command()
def energy_release_rate():
    """Calculate the energy release rate for a range of velocities."""
    def G(v: float):
        cs = 3300 #m/s
        cd = 5500 #m/s
        alphas = np.sqrt(1-v**2/cs**2)
        alphad = np.sqrt(1-v**2/cd**2)
        D = 4*alphas*alphad-(1+alphas**2)**2
        # schubmodul von glas
        mu = 26e3 # N/mm²

        K1 = 0.75 # MPa m^0.5

        # kJ/m²
        return 1e3 * (v**2 * alphad) / (2*cs**2*mu*D) * K1**2

    urr = relative_remaining_stress(22**2, 8)
    print('Relative remaining stress: ', urr)

    x = np.linspace(1, 2700, 100)
    y = G(x) * 1e3

    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    axs.plot(x,y)
    axs.set_xlabel("Geschwindigkeit $v$ (m/s)")
    axs.set_ylabel("Energie-Freisetzungsrate $\mathcal{G}$ (J/m²)")

    State.output(StateOutput(fig, FigureSize.ROW1), "energy_release_rate")