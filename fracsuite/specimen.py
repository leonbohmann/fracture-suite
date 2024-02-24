"""
Organisation module. Contains the Specimen class and some helpful tools to export specimens.
"""
from __future__ import annotations
from json import JSONEncoder
import json

from regex import F
from fracsuite.core.calculate import is_number
from fracsuite.core.geometry import delta_hcp
from fracsuite.core.logging import debug, error, info, warning

import os
from pickle import NONE
import re
import cv2
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
import rich
from rich.prompt import Prompt

from scipy.optimize import curve_fit
import numpy as np
from pygments import highlight
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import typer
from rich import inspect, print
from rich.progress import track
from fracsuite.callbacks import main_callback
from fracsuite.core.detection import get_crack_surface, get_crack_surface_r
from fracsuite.core.imageprocessing import preprocess_image
from fracsuite.core.mechanics import Ud2sigm
from fracsuite.core.navid_results import navid_nfifty, navid_nfifty_ud
from fracsuite.core.plotting import FigureSize, KernelContourMode, fit_curve, get_fig_width, legend_without_duplicate_labels, plot_kernel_results
from fracsuite.core.progress import tracker

from fracsuite.core.specimen import Specimen, SpecimenBoundary
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stochastics import quadrat_count, r_squared_f
from fracsuite.core.stress import relative_remaining_stress
from fracsuite.general import GeneralSettings
from fracsuite.splinters import create_filter_function
from fracsuite.state import State, StateOutput

app = typer.Typer(help=__doc__, callback=main_callback)

general = GeneralSettings.get()


b_markers = {
    'A': 'o',
    'B': 's',
    'Z': 'D'
}

t_colors = {
    4: 'green',
    8: 'red',
    12: 'blue'
}

scatter_args = {
        's': 10,
        'edgecolors': 'gray',
        'linewidth': 0.5
    }

def linfit(x, a, b):
    return a*x + b

def squarefit(x, a, b, c):
    return a*x**2 + b*x + c

def expfit(x, a, b):
    return a * np.exp(b * x)

def logfit(x, a, b):
    return a * np.log(x) + b

def cubicfit(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def quarticfit(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def quinticfit(x, a, b, c, d, e, f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f


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
def mark_impact(name):
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
    if marked_pos is not None:
        specimen.set_setting(Specimen.SET_ACTUALBREAKPOS, tuple(marked_pos))

@app.command()
def mark_excluded_points(
    name: str
):
    specimen = Specimen.get(name)

    if not specimen.has_fracture_scans:
        error("Specimen has no fracture scans!")
        return

    img = specimen.get_fracture_image()
    img = preprocess_image(img, specimen.get_prepconf(warn=False))



    try:
        radius = float(typer.prompt("Choose a radius in mm to exclude around the points [100mm]: ", default="100"))
    except Exception:
        warning("Invalid input, using default radius of 100mm.")
        radius = 100

    # default
    global marked_pos_exclpoint
    marked_pos_exclpoint = None

    specimen.set_setting(Specimen.SET_EXCLUDED_POSITIONS_RADIUS, radius)

    excluded_points = []
    no_more = False
    px_p_mm = specimen.calculate_px_per_mm()
    while not no_more:
        fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
        axs.imshow(img, cmap='gray')
        axs.set_title(f"Exclude point from {name}")
        # when clicking, mark the position but dont save yet
        def onclick(event):
            global marked_pos_exclpoint

            x = int(event.xdata)
            y = int(event.ydata)

            # display the marked point and remove the previous one
            if hasattr(onclick, 'mark'):
                onclick.mark.remove()

            onclick.mark = axs.plot(x, y, 'ro', markersize=radius, alpha=0.3)[0]

            # save the marked position
            marked_pos_exclpoint = np.asarray([x,y]) / px_p_mm
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        # make cursor red
        fig.canvas.mpl_connect('motion_notify_event', lambda event: plt.gcf().canvas.set_cursor(1))

        plt.show(block=True)

        print(f"Excluded position: {marked_pos_exclpoint}")
        if marked_pos_exclpoint is not None:
            excluded_points.append(tuple(marked_pos_exclpoint))

            if typer.prompt("Do you want to exclude more points?", default="Y").lower() != "y":
                no_more = True
        else:
            no_more = True

    if len(excluded_points) != 0:
        specimen.set_setting(Specimen.SET_EXCLUDED_POSITIONS, tuple(excluded_points))



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
            return f"{s.calculate_nfifty_count():.2f}"
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

@app.command('import')
def import_files(
    specimen_name: str = typer.Argument(help="Name of the specimen."),
    imgsize: tuple[int, int] = typer.Option((4000, 4000), help="Size of the image."),
    realsize: tuple[float, float] = typer.Option((-1, -1), help="Real size of the image. If any value is -1, the real size is not set."),
    imsize_factor: float = typer.Option(None, help="Image size factor."),
    no_rotate: bool = typer.Option(False, help="Option to disable rotation."),
    no_tester: bool = typer.Option(False, help="Option to disable tester."),
    exclude_all_sensors: bool = typer.Option(False, help="Option to exclude all sensors."),
    exclude_impact_radius: float = typer.Option(None, help="Radius to exclude impact."),
    exclude_points: bool = typer.Option(False, help="Shows helper windows to exclude points in the morphology."),
    fracture_image: str = typer.Option(None, help="Path to the fracture image.")
):
    """
    Imports fracture images and generates splinters of a specific specimen.

    This function is safe to call because already transformed images are not overwritten and if
    there are already splinters, an overwrite has to be confirmed.

    #### Specimen creation
    You can either create a new specimen using `fracsuite specimen create` and manually copy the images into
    the new folder, or you can use this command with the `--fracture-image` option to directly import the fracture image.

    #### Resulting image size
    There are two ways in which the resulting image size (px) can be set:
        - Use the `--imgsize` option to set the image size directly.
        - Use the `--imsize-factor` option to scale the image size from the real size. In that case, the real size has to be set.
    """

    assert not (imsize_factor is not None and imgsize[0] != 4000 and imgsize[1] != 4000), "Cannot set both imsize factor and imgsize!"
    # check that when passing an imsize factor the realsize has to be set
    assert not (imsize_factor is not None and (realsize[0] == -1 or realsize[1] == -1)), "Real size has to be set when setting imsize factor!"

    if "*" in specimen_name:
        info(f"Importing all specimens with filter '{specimen_name}'")
        # assume that the name contains a filter
        filterfunc = create_filter_function(specimen_name, needs_scalp=False, needs_splinters=False)
        specimens = Specimen.get_all_by(filterfunc, load=True)
        for spec in specimens:
            if not spec.has_fracture_scans:
                debug(f"Specimen '{spec.name}' has no fracture scans! Skipping...")
                continue

            import_files(spec.name, imgsize, realsize, imsize_factor, no_rotate, no_tester, exclude_all_sensors, exclude_impact_radius, fracture_image)

        return


    specimen = Specimen.get(specimen_name, load=True, panic=False)

    if specimen is None:
        info(f"Specimen '{specimen_name}' not found! Creating...")
        create(specimen_name)

        assert fracture_image is not None, "Fracture image is required for new specimen!"

        specimen = Specimen.get(specimen_name, load=True, panic=True, printout=False)
        # put fracture image into specimen!
        img = cv2.imread(fracture_image, cv2.IMREAD_GRAYSCALE)
        specimen.put_fracture_image(img)
    elif specimen is not None and fracture_image is not None:
        if specimen.has_fracture_scans:
            raise Exception("Specimen already has fracture scans! Overwrite not allowed! Delete the image at: " + specimen.fracture_morph_folder)

        info(f"Specimen '{specimen_name}' has no fracture scans! Adding...")
        img = cv2.imread(fracture_image, cv2.IMREAD_COLOR)
        specimen.put_fracture_image(img)

    assert specimen.has_fracture_scans, "Specimen has no fracture scans"

    if specimen.has_splinters:
        if not typer.confirm("Specimen already has splinters. Overwrite?"):
            return

    if imsize_factor is not None:
        imgsize = (int(realsize[0] * imsize_factor), int(realsize[1] * imsize_factor))

    # set settings on specimen
    specimen.set_setting(Specimen.SET_EXCLUDE_ALL_SENSORS, exclude_all_sensors)
    if exclude_impact_radius is not None:
        specimen.set_setting(Specimen.SET_CBREAKPOSEXCL, exclude_impact_radius)
    if realsize[0] != -1 and realsize[1] != -1:
        specimen.set_setting(Specimen.SET_REALSIZE, realsize)

    print('[yellow]> Transforming fracture images <')
    img0path, img0 = specimen.transform_fracture_images(size_px=imgsize, rotate=not no_rotate)

    print('[yellow]> Running threshold tester <')
    if not no_tester:
        from fracsuite.tester import threshold
        threshold(specimen.name)

    print('[yellow]> Marking impact point <')
    mark_impact(specimen.name)

    if exclude_points:
        print('[yellow]> Mark excluded points <')
        mark_excluded_points(specimen.name)
    else:
        info("Skipped excluding points. Enable with --exclude-points")

    print('[yellow]> Generating splinters <')
    from fracsuite.splinters import gen
    gen(specimen.name)

    print('[yellow]> Drawing contours <')
    from fracsuite.splinters import draw_contours
    draw_contours(specimen.name)

@app.command()
def compare_nfifty_estimation(
    boundary: str = None,
):
    """
    Compare the nfifty estimation of all specimens.

    This function compares the nfifty estimation of all specimens and calculates the mean and standard deviation.
    """
    if boundary is None:
        boundary = "*"

    filterfunc = create_filter_function(f"*.*.{boundary}.*", needs_splinters=True)

    all_specimens = Specimen.get_all_by(filterfunc, load=True)


    if boundary == "*":
        boundary = "all"

    nfifties: dict[Specimen,float] = {}
    sigh = []
    for spec in tqdm(all_specimens, desc="Calculating nfifty...", leave=False):
        if spec.has_fracture_scans and spec.has_splinters:
            nfifties[spec] = spec.calculate_nfifty_count(simple=True)
            sigh.append(-spec.sig_h/2.0)


    x_total = []
    y_total = []

    def n50_analytic(sigm):
        return (sigm/14.96)**4

    sz = FigureSize.ROW2
    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    n50_navid = navid_nfifty_ud()
    # convert ud to sigm
    n50_navid[:,1] = Ud2sigm(n50_navid[:,1])
    n50s_navid = n50_navid[:,0].flatten()
    sigm_navid = n50_navid[:,1].flatten()
# 1e6/5 * (1-nue)/E * (sigma_s ** 2)
    for i in range(len(n50s_navid)):
        axs.scatter(sigm_navid[i], n50s_navid[i], marker='x', color='gray', label='Literatur', alpha=0.75, **scatter_args)
        x_total.append(sigm_navid[i])
        y_total.append(n50s_navid[i])

    # plot data
    for spec,n50 in tqdm(nfifties.items(), desc="Plotting data...", leave=False):
        clr = t_colors[spec.thickness]
        marker = b_markers[spec.boundary]
        axs.scatter(-spec.sig_h/2, n50, marker=marker, color=clr, label=f'{spec.thickness}mm, {spec.boundary}', **scatter_args)

        x_total.append(-spec.sig_h/2)
        y_total.append(n50)

    axs.set_xlabel('Mittelzugspannung $\sigma_m$ (MPa)')
    axs.set_ylabel('$N_{50}$')

    nfifties = np.array(list(nfifties.values()) + list(n50s_navid))
    sigh = np.array(sigh)
    sigh = np.concatenate((sigh, np.asarray(sigm_navid)))
    sigh.sort()
    x = sigh
    y = n50_analytic(sigh)
    axs.plot(x, y, 'r--')

    fit_curve(axs, x_total, y_total, cubicfit, color='black', pltlabel='Fit')

    legend_without_duplicate_labels(axs, compact=True)

    State.output(StateOutput(fig,sz), "nfifty_estimation_comparison")



@app.command()
def crack_surface_simple(
    boundary: str = None,
):
    if boundary is None:
        boundary = "*"
    filter_func = create_filter_function(f"*.*.{boundary}.*", needs_scalp=True, needs_splinters=True)

    specimens = Specimen.get_all_by(filter_func, load=True)

    if boundary == "*":
        boundary = "all"

    # calculate total crack surfaces
    crack_surfaces = {}
    splinter_volumes = {}
    splinter_areas = {}
    splinter_circs = {}
    for spec in tqdm(specimens):
        t0 = spec.measured_thickness

        vols = np.zeros(len(spec.splinters))
        careas = np.zeros(len(spec.splinters))
        areas = np.zeros(len(spec.splinters))
        circs = np.zeros(len(spec.splinters))
        for i,splinter in enumerate(tqdm(spec.splinters, leave=False)):
            circ = splinter.circumfence
            A = splinter.area

            ms = A * t0
            carea = circ * t0

            vols[i] = ms
            careas[i] = carea
            areas[i] = A
            circs[i] = circ

        splinter_volumes[spec] = np.nanmean(vols)
        crack_surfaces[spec] = np.sum(careas)
        splinter_areas[spec] = np.nanmean(areas)
        splinter_circs[spec] = np.nanmean(circs)


    sz = FigureSize.ROW2


    ##########################
    # plot crack surface
    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    for spec,crack_area in crack_surfaces.items():
        clr = t_colors[spec.thickness]
        marker = b_markers[spec.boundary]
        axs.scatter(spec.U, crack_area, marker=marker, color=clr, label=f'{spec.thickness}mm, {spec.boundary}', **scatter_args)

    axs.set_xlabel("Formänderungsenergie $U$ (J/m²)")
    axs.set_ylabel("$U_\\text{S} \cdot t$ (mm²)")
    legend_without_duplicate_labels(axs)

    State.output(StateOutput(fig,sz), f"cracksurface_vs_energy_{boundary}")

    n50_navid = navid_nfifty_ud()
    # convert ud to sigm
    n50_navid[:,1] = n50_navid[:,1] * n50_navid[:,2] * 1e-3
    # calc volume in column 0
    navid_volumes = (2500 / n50_navid[:,0]) * n50_navid[:,2] * 1e-3

    ##########################
    # plot volume
    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    for spec,spl_volume in splinter_volumes.items():

        clr = t_colors[spec.thickness]
        marker = b_markers[spec.boundary]
        axs.scatter(spec.U, spl_volume, marker=marker, color=clr, label=f'{spec.thickness}mm, {spec.boundary}', **scatter_args)

    for t in ([4,8,12]):
        tspecs = [spec for spec in specimens if spec.thickness == t]

        if len(tspecs) == 0:
            continue

        clr = t_colors[t]
        x = np.array([spec.U for spec in tspecs])
        y = np.array([splinter_volumes[spec] for spec in tspecs])

        y_fit, popt = fit_curve(axs, x, y, squarefit, color=clr)

    axs.set_xlabel("Formänderungsenergie $U$ (J/m²)")
    axs.set_ylabel("$V_\\text{S}$ (mm³)")
    legend_without_duplicate_labels(axs, compact=True)
    State.output(StateOutput(fig,sz), f"volume_vs_energy_{boundary}")


    ##########################
    # plot mean splinter area
    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    for spec,spl_volume in splinter_areas.items():

        clr = t_colors[spec.thickness]
        marker = b_markers[spec.boundary]
        axs.scatter(spec.sig_h, spl_volume, marker=marker, color=clr, label=f'{spec.thickness}mm, {spec.boundary}', **scatter_args)

    axs.set_xlabel("Oberflächendruckspannung $-\sigma_\\text{S}$ (MPa)")
    axs.set_ylabel("$A_\\text{S}$ (mm²)")
    legend_without_duplicate_labels(axs, compact=True)

    State.output(StateOutput(fig,sz), f"area_vs_sig_{boundary}")


    ##########################
    # plot circumfences
    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    for spec,spl_volume in splinter_circs.items():

        clr = t_colors[spec.thickness]
        marker = b_markers[spec.boundary]
        axs.scatter(spec.U, spl_volume, marker=marker, color=clr, label=f'{spec.thickness}mm, {spec.boundary}', **scatter_args)


    axs.set_xlabel("Formänderungsenergie $U$ (J/m²)")
    axs.set_ylabel("$U_\\text{S}$ (mm³)")
    legend_without_duplicate_labels(axs, compact=True)

    State.output(StateOutput(fig,sz), f"circumfence_vs_energy_{boundary}")

@app.command()
def crack_surface(
    rust: bool = False, ud: bool = False, aslog: bool = False, overwrite: bool = False,
    highlight_filter: str = None
):
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

        if highlight_filter is not None and re.match(highlight_filter, spec.name):
            axs.scatter(v, spec.crack_surface, marker='x', label=spec.name, s=10)
        else:
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
        r2 = r_squared_f(vs,surfs,func,popt)


        # plot fitted curve
        x = np.min(vs) + (np.max(vs) - np.min(vs))*0.6

        # create axline
        axs.axline((x,popt[1]+popt[0]*x), slope=popt[0], color=t_color[t] if t is not None else 'k', linestyle='--', linewidth=0.5)

        # annotate R² to fitting line
        axs.annotate(f"$R^2={r2:.2f}$ m={popt[0]:.2e}", (x,func(x, *popt)), ha="left", va="top")
        print(f"> t={t}mm: m={popt[0]:.2e}mm²/J, b={popt[1]:2e}, R²={r2:.2f}")
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


@app.command()
def check_homogeneity():
    """Check the homogeneity of the specimens."""
    all_specimens = Specimen.get_all(load=True)

    for spec in all_specimens:
        info(f"Specimen {spec.name}")
        if not spec.has_splinters:
            continue

        d_hcp = delta_hcp(spec.calculate_intensity())
        d_hc,_ = spec.calculate_break_rhc_acc()

        info(f"> d_hcp={d_hcp}, d_hc={d_hc}")

        alpha = d_hc / d_hcp

        info(f"> alpha={alpha:.2f}")

        sz = spec.get_real_size()
        w = sz[0]
        h = sz[1]

        events = [x.centroid_mm for x in spec.allsplinters]

        # only analyze centroids that are more than 5 cm away from the border
        d = 50
        events = [x for x in events if x[0] > d and x[0] < w-d and x[1] > d and x[1] < h-d]

        events = np.array(events)

        X2, dof, c = quadrat_count(events, (w,h), 100)
        info(f"> X²={X2}, c={c}, dof={dof}")

        if X2 >= c:
            print(f"> Specimen {spec.name} is inhomogeneous! X²={X2}, c={c}")


@app.command('property')
def plot_property(
    specimen_name: str = typer.Argument(help='Name of specimens to load'),
    prop: SplinterProp = typer.Argument(help='Property to plot.'),
    n_points: str = typer.Option("25", help='Amount of points to evaluate. Pass as "n" or "nx,ny", including Anführungszeichen!'),
    w_mm: int = typer.Option(50, help='Size of the kernel window in mm. Defaults to 50mm.'),
    smooth: bool = typer.Option(True, help='Plot a linear interpolation of the results. Default is True.'),
    quadrat_count: bool = typer.Option(False, help='If used, modifies w_mm so that n_points are calculated in each dimension. This doesnt work yet for n_points="nx,ny"! Defaults to False.'),
):
    """
    Plot the property of the specimen using a KDE plot.

    n_points
    """
    specimen = Specimen.get(specimen_name)

    if is_number(n_points):
        n_points = int(n_points)
    elif "," in n_points:
        n_points = tuple(map(int, n_points.split(",")))

    X,Y,Z,Zstd = specimen.calculate_2d(prop, w_mm, n_points, quadrat_count=quadrat_count)

    output = plot_kernel_results(
        specimen.get_fracture_image(),
        Splinter.get_property_label(prop),
        True,
        False,
        KernelContourMode.FILLED,
        X, Y, Z,
        0,
        FigureSize.ROW2,
        clr_format=".2f",
        smooth=smooth,
        fill_skipped_with_mean=False,
    )
    output.overlayImpact(specimen)
    outputname = f'{prop}_2d' + ('_nosmooth' if not smooth else '') + ('_quadrat' if quadrat_count else '')
    State.output(output, outputname, spec=specimen, to_additional=True)