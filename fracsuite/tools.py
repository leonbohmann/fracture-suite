from collections import defaultdict
from glob import glob
import os
from matplotlib import pyplot as plt
import numpy as np

from spazial import csstraussproc2
from tqdm import tqdm

import typer
import cv2

from fracsuite.callbacks import main_callback
from fracsuite.core.coloring import rand_col
from fracsuite.core.model_layers import arrange_regions, get_layer_folder
from fracsuite.core.plotting import FigureSize, get_fig_width, renew_ticks_ax, renew_ticks_cb, voronoi_to_image
from fracsuite.core.signal import smooth_hanning
from fracsuite.core.specimenprops import SpecimenBreakPosition
from fracsuite.core.splinter import Splinter
from fracsuite.core.stochastics import khat, lhat, lhatc, pois, quadrat_count, rhc_minimum
from fracsuite.state import State, StateOutput

from rich import print

tools_app = typer.Typer(help=__doc__, callback=main_callback)


@tools_app.command()
def crop_images(
    image_folder: str,
    region: tuple[int,int,int,int] = typer.Option(None, help="Region to crop."),
):
    # read all images from the folder and crop them to the region
    for img_path in glob(os.path.join(image_folder, "*.png")):
        img = cv2.imread(img_path)
        img = img[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
        cv2.imwrite(img_path, img)

@tools_app.command()
def latex_img(
    image_folder: str,
    every_n: int = 2,
    basepath: str = "path/to/images",
    indices: tuple[int,int] = (0,-1),
):
    img_names = []
    image_folder = os.path.abspath(image_folder)


    for img_path in glob(os.path.join(image_folder, "*.png")):
        img_name = os.path.basename(img_path)
        img_names.append(img_name)

    img_names = sorted(img_names)
    img_names = img_names[indices[0]:indices[1]]
    img_names = img_names[::every_n]

    def imgtolatex(imgname,i):
        return r"""
    \begin{subfigure}[t]{0.1\textwidth}
        \centering
        \includegraphics[width=\linewidth]{[imgpath]}
        \caption{Frame [imgindex]}
        \label{fig:[imgname]}
    \end{subfigure}
    """.replace("[imgpath]", f"{basepath}/{imgname}") \
        .replace("[imgname]", f"{imgname}") \
        .replace("[imgindex]", str(i))

    latex_base = r"""
\begin{figure}[hbt]
    \centering
    [imglist]
    \caption{Caption for the Whole Figure}
    \label{fig:threefigures}
\end{figure}
"""
    image_list = r"\hfill%".join([imgtolatex(x,1+i*every_n) for i,x in enumerate(img_names)])

    latex_base = latex_base.replace("[imglist]", image_list)

    print(latex_base)




@tools_app.command()
def geometry():
    # choose any specimen and load a single splinter
    from fracsuite.core.specimen import Specimen
    from fracsuite.core.splinter import Splinter
    # import filter function
    from fracsuite.splinters import create_filter_function

    filter = create_filter_function("*.*.*.*", needs_scalp=False, needs_splinters=True)

    specimens = Specimen.get_all_by(filter, load=True)

    # generate 5 random indices
    import random
    random.seed(42)
    indices = random.sample(range(len(specimens)), 5)

    for i in indices:
        specimen = specimens[i]
        splinter = specimen.splinters[0]

        indsp = random.sample(range(len(specimen.splinters)), 5)

        for spl in indsp:
            splinter = specimen.splinters[spl]


            img_rect = specimen.get_fracture_image()
            img_ellipse = specimen.get_fracture_image()

            f0 = 2
            img_rect = cv2.resize(img_rect, (0,0), fx=f0, fy=f0)
            img_ellipse = cv2.resize(img_ellipse, (0,0), fx=f0, fy=f0)

            # apply scaling to contour
            contour = splinter.contour * f0


            # draw the contour
            cv2.drawContours(img_rect, [contour], 0, (0,0,255), 2)
            cv2.drawContours(img_ellipse, [contour], 0, (0,0,255), 2)

            # create an image that surrounds the splinter
            (x,y), (w,h), a = cv2.minAreaRect(contour)
            x,y,w,h = int(x), int(y), int(w), int(h)

            # draw the minarea rect
            box = cv2.boxPoints(((x,y), (w,h), a))
            box = np.int0(box)
            cv2.drawContours(img_rect, [box], 0, (0,255,0), 2)

            # draw ellipse
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(img_ellipse, ellipse, (0,255,0), 2)

            # find bounding box
            x,y,w,h = cv2.boundingRect(contour)

            padding = 5
            # expand the bounding box by 10%
            x -= w//padding
            y -= h//padding
            w += w//(padding//2)
            h += h//(padding//2)

            # crop
            img_rect = img_rect[y:y+h, x:x+w]
            img_ellipse = img_ellipse[y:y+h, x:x+w]

            # generate a name and save
            name = f"{specimen.name}_{spl}_rect.png"
            State.output(StateOutput(img_rect, 'img'), name, open=False)
            namee = f"{specimen.name}_{spl}_ellipse.png"
            State.output(StateOutput(img_ellipse, 'img'), namee, open=False)


@tools_app.command()
def plate_freq(
    a: float = 0.500,
    b: float = 0.500,
    h: float = 0.008,
    E: float = 70e9,
    rho: float = 2500,
    n_modes: int =5
):
    """
    Berechnet die Eigenfrequenzen einer allseitig gelagerten Platte.

    Parameter:
        a (float): Länge der Platte (m)
        b (float): Breite der Platte (m)
        h (float): Dicke der Platte (m)
        E (float): Elastizitätsmodul des Materials (Pa)
        rho (float): Dichte des Materials (kg/m^3)
        n_modes (int): Anzahl der zu berechnenden Eigenfrequenzen (Standardwert: 5)

    Returns:
        frequencies (list[float]): Liste der Eigenfrequenzen (Hz)
    """
    D = E*h**3 / (12*(1 - 0.3**2))  # Biegesteifigkeit der Platte
    frequencies = []

    for m in range(1, n_modes + 1):
        for n in range(1, n_modes + 1):
            freq = (np.pi**2) * np.sqrt(D / (rho * h)) * np.sqrt((m/a)**2 + (n/b)**2) / (2 * np.pi)
            frequencies.append(freq)

    frequencies.sort()
    frequencies = frequencies[:n_modes]
    print(frequencies)


# Hilfsfunktion zum Gruppieren der Layer-Props
def group_layer_props(layer_props):
    grouped = defaultdict(lambda: defaultdict(list))
    for boundary, thickness, prop in layer_props:
        grouped[prop][boundary].append(thickness)
    return grouped

@tools_app.command()
def layers_to_tex(base_path: str = ""):
    layer_folder = get_layer_folder()
    layer_folder = os.path.join(layer_folder, "create")

    # get all layers
    layers = glob(os.path.join(layer_folder, "impact-layer_*.pdf"))

    # settings
    plots_per_figure = 9
    rows_per_page = 4

    main_latex = r"""
    \chapter{LBREAK Layer}
    [PROPERTIES]
    """

    property_latex = r"""
    \section*{[PROPERTY]}
    [BOUNDARIES]
    """

    # create latex
    boundary_latex = r"""
    \subsection*{[BOUNDARY]}
    [THICKNESSES]
    """

    imagelist_latex = r"""
    \begin{figure}[H]
        \centering
        [IMAGELIST]
        \caption[]{[CAPTION]}
        \label{fig:[FIGURENAME]}
    \end{figure}
    """

    def imgtolatex(imgpath, imgname,i):
        return r"""
    \begin{subfigure}[t]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{[basepath]/[imgpath]}
        \caption[]{[thickness] mm}
        \label{fig:[imgname]}
    \end{subfigure}
    """.replace("[imgpath]", f"{imgname}") \
        .replace("[imgname]", f"{imgname}") \
        .replace("[thickness]", f"{i}") \
        .replace("[imgindex]", str(i)) \
        .replace("[basepath]", base_path)

    import re

    # create an image list for each page
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    layer_props = []
    for layer in layers:
        match = re.match(r"impact-layer_(.*?)_(.*?)_(.*?)_", os.path.basename(layer))
        if match:
            props = match.groups()
            layer_props.append((*props, layer))

            boundary, thickness, prop = props

            groups[prop][boundary][str(thickness)] = layer

    groups = {prop: {boundary: {thickness : groups[prop][boundary][thickness] for thickness in groups[prop][boundary]} for boundary in groups[prop]} for prop in groups}

    print(groups)

    prop_latexs = []
    for prop in groups:
        boundaries = groups[prop]
        b_latexs = []

        for boundary in boundaries:
            thicknesses = boundaries[boundary]
            image_list = r"\hfill%".join([imgtolatex(thicknesses[x], os.path.basename(thicknesses[x]), x) for i,x in enumerate(thicknesses) if x != "12"])
            imLatex = imagelist_latex.replace("[IMAGELIST]", image_list) \
                .replace("[FIGURENAME]", f"{prop}_{boundary}") \
                .replace("[CAPTION]", f"{Splinter.get_property_label(prop).strip()} in Abhängigkeit des Abstands zum Anschlagpunkt bei verschiedenen Glasdicken; Farbig markiert ist die Formänderungsenergie U.")

            b_latex = boundary_latex.replace("[THICKNESSES]", imLatex).replace("[BOUNDARY]", boundary)
            b_latexs.append(b_latex)

        prop_latex = property_latex.replace("[BOUNDARIES]", "\n".join(b_latexs)).replace("[PROPERTY]", Splinter.get_property_label(prop).strip())
        prop_latexs.append(prop_latex)


    main_latex = main_latex.replace("[PROPERTIES]", "\n".join(prop_latexs))

    # create an output latex file
    with open(os.path.join(layer_folder, "A_lbreak_LAYERS.tex"), "w", encoding='utf-8') as f:
        f.write(main_latex)


    # # create latex for each page
    # prop_latexs = []
    # for prop in grouped_layer_props:
    #     boundaries = grouped_layer_props[prop]
    #     b_latexs = []

    #     for boundary in boundaries:
    #         thicknesses = boundaries[boundary]
    #         image_list = r"\hfill%".join([imgtolatex(x, os.path.basename(x), i) for i,x in enumerate(thicknesses)])
    #         b_latex = boundary_latex.replace("[THICKNESSES]", image_list).replace("[BOUNDARY]", boundary)
    #         b_latexs.append(b_latex)

    #     prop_latex = property_latex.replace("[BOUNDARIES]", "\n".join(b_latexs)).replace("[PROPERTY]", prop)
    #     prop_latexs.append(prop_latex)

    # main_latex = main_latex.replace("[PROPERTIES]", "\n".join(prop_latexs))

    # # create an output latex file
    # with open(os.path.join(layer_folder, "layers.tex"), "w") as f:

    #     f.write(main_latex)


@tools_app.command()
def create_poisson():
    sz = FigureSize.ROW3

    w = 200
    n  = 300

    points = pois(w, w, n)

    # plot points
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.scatter(points[:,0], points[:,1], s=1)
    State.output(StateOutput(fig, sz), "poisson", open=False)

    # create K-Functions
    def LPois(r):
        # see Baddeley et al. S.206 K_pois
        return r
    def Kpois(r):
        # see Baddeley et al. S.206 K_pois
        return np.pi * r**2

    d_max = np.sqrt(200**2 + 200**2)
    lh = lhat(points, w*w, d_max)
    x = lh[:,0]
    y = lh[:,1]
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    axs.set_xlabel("d")
    axs.set_ylabel("$\hat{L}(d)$")
    axs.plot(x,y, label="Punkte")
    axs.plot(x, LPois(x), label="Poisson")
    axs.legend()
    State.output(StateOutput(fig, sz), "poisson_lhat", open=False)


@tools_app.command()
def gibbs_strauss(
    acc: str = "2e-3",
    rhc: str = "0.0000001, 10, 25, 50",
    intensity: float = 0.002,
    name: str = None,
):
    State.pointoutput(name)

    sz = FigureSize.ROW3

    # acceptance probabilities
    acc = np.array([float(x) for x in acc.split(",")])
    # hard core radii
    # rhc = np.array([0.00001, 10, 25, 50])
    rhc = np.array([float(x) for x in rhc.split(",")])


    d_max = rhc * 3
    if rhc[0] == 0:
        rhc[0] = 0.000000001
        d_max[0] = 10

    print('acc:', acc)
    print('rhc:', rhc)


    w = 500
    n = int(intensity * w**2)

    for id,d in enumerate(rhc):
        for a in acc:
            points = csstraussproc2(w,w, d, n, a, int(1e6))
            points = np.array(points)

            # check for homogenity
            X2, dof, c = quadrat_count(points, (w, w), w/10)
            print(f'Rhc: {d:>5.2f}, Acc: {a:>5.2f}, X2: {X2:>5.2f}, c: {c:>5.2f}, dof: {dof:>5.2f}, {("nicht homogen", "homogen")[X2 <= c]}')


            # plot points
            fig,axs = plt.subplots(figsize=get_fig_width(sz))
            axs.set_xlabel("x")
            axs.set_ylabel("y")
            axs.scatter(points[:,0], points[:,1], s=1)
            axs.set_aspect('equal', 'box')
            name = f"rhc_{d:.0f}_acc_{a}"
            renew_ticks_ax(axs, (0, w), (0, w), 0)
            axs.grid(False)
            axs.tick_params(axis='both', which='both', length=0)
            State.output(StateOutput(fig, sz), name, open=False)

            # create voronoi plot with the points
            from scipy.spatial import Voronoi
            vor = Voronoi(points)
            fig,axs = plt.subplots(figsize=get_fig_width(sz))
            img = np.zeros((w,w))
            voronoi_to_image(img, vor)
            axs.imshow(img, cmap='gray')
            axs.set_aspect('equal', 'box')
            State.output(StateOutput(fig, sz), f"{name}_voronoi", open=False)



            # create K-Functions
            def Kpois(r):
                # see Baddeley et al. S.206 K_pois
                return np.pi * r**2


            # L-Function
            lhat_v = lhat(points, w, w, d_max[id])
            x = lhat_v[:,0]
            y = lhat_v[:,1]
            fig,axs = plt.subplots(figsize=get_fig_width(sz))
            axs.set_xlabel("d (mm)")
            axs.set_ylabel("$\hat{L}(d)$")
            axs.plot(x,y, label="Messung")
            axs.plot(x, x, label="Poisson")
            axs.legend()
            State.output(StateOutput(fig, sz), f"{name}_lhat", open=False)

            # centered l-function
            lhatc_v = lhatc(points, w, w, d_max[id])
            x = lhatc_v[:,0]
            y = lhatc_v[:,1]
            fig,axs = plt.subplots(figsize=get_fig_width(sz))
            axs.set_xlabel("d (mm)")
            axs.set_ylabel("$\hat{L}(d) - d$")
            axs.plot(x,y, label="Messung")
            axs.plot(x, x-x, label="Poisson")
            axs.legend()
            State.output(StateOutput(fig, sz),  f"{name}_lhatc", open=False)

            # K-Function
            kh = khat(points, w, w, d_max[id])
            x = kh[:,0]
            y = kh[:,1]
            fig,axs = plt.subplots(figsize=get_fig_width(sz))
            axs.set_xlabel("d (mm)")
            axs.set_ylabel("$\hat{K}(d)$")
            axs.plot(x,y, label="Messung")
            axs.plot(x, Kpois(x), label="Poisson")
            axs.legend()
            State.output(StateOutput(fig, sz), f"{name}_khat", open=False)


def watershed_points(
    points,
    size
):
    # scaling factor (realsize > pixel size)
    size_f = 20

    # create output image store with marked points
    markers = np.zeros((int(size[1]*size_f),int(size[0]*size_f)), dtype=np.uint8)
    for point in points:
        markers[int(point[1]*size_f), int(point[0]*size_f)] = 255


    # perform watershedding on individual points
    markers = cv2.connectedComponents(np.uint8(markers))[1]
    shape = (int(size[1]*size_f),int(size[0]*size_f),3)
    blank_image = np.zeros(shape, dtype=np.uint8)
    markers = cv2.watershed(blank_image, markers)

    m_img = np.zeros(shape, dtype=np.uint8)
    m_img[markers == -1] = 255
    splinters = Splinter.analyze_contour_image(m_img, size_f, prep = None, areabounds = (0,1e6))

    # create contour image
    img = np.zeros(shape, dtype=np.uint8)
    for s in splinters:
        clr = rand_col()
        cv2.drawContours(img, [s.contour], 0, clr, -1)

    return img

@tools_app.command()
def test_bohmann(
    lam_max: float = 0.02,
    rhc_max: float = 10,
    c: float = 0,
    w: float = 500,
    sz: FigureSize = FigureSize.ROW3,
    point_sz: FigureSize = FigureSize.ROW2,
):
    """TEsting function for the bohmann process."""

    from spazial import bohmann_process

    h = w

    r_range, t_range = arrange_regions(30, 360, cx_mm=w/2, cy_mm=h/2, w_mm=w, h_mm=h)

    lam = r_range.copy()
    lam = np.column_stack((lam, lam))
    for i in range(len(lam)):
        lam[i,1] = lam_max * (1 - (i/len(lam)))

    rhc = r_range.copy()
    rhc = np.column_stack((rhc, rhc))
    for i in range(len(rhc)):
        rhc[i,1] = rhc_max* (1 - (i/len(rhc)))

    results = np.asarray(bohmann_process(
        w, h,  # width, height
        r_range,
        lam,
        rhc,
        (w/2,h/2),  # x,y
        c,  # acceptance probability
        int(1e6),  # max iterations
        True # no warnings
    ))

    lam_max_real = len(results) / (w*h)
    figsz = get_fig_width(sz)

    if not State.no_out:
        # plot points
        fig,axs = plt.subplots(figsize=get_fig_width(point_sz))
        axs.scatter(results[:,0], results[:,1], s=1)
        axs.set_aspect('equal', 'box')
        State.output(StateOutput(fig, point_sz), f"bohmann_{lam_max:0.3f}_{rhc_max:.3f}_process", open=True)

    if not State.no_out:
        # create voronoi plot with the points
        from scipy.spatial import Voronoi
        vor = Voronoi(results)
        fig,axs = plt.subplots(figsize=get_fig_width(point_sz))
        img = np.zeros((int(w),int(w)))
        voronoi_to_image(img, vor)
        axs.imshow(255-img, cmap='gray')
        axs.set_aspect('equal', 'box')
        State.output(StateOutput(fig, point_sz), f"bohmann_{lam_max:0.3f}_{rhc_max:.3f}_voronoi", open=True)


    # plot kfunction
    d_max = 50
    kh = khat(results, w, h, d_max)
    x = kh[:,0]
    y = kh[:,1]
    if not State.no_out:
        fig,axs = plt.subplots(figsize=figsz)
        axs.set_xlabel("d (mm)")
        axs.set_ylabel("$\hat{K}(d)$")
        axs.plot(x,y, label="Messung")
        axs.plot(x, np.pi * x**2, label="Poisson")
        axs.legend()
        State.output(StateOutput(fig, sz), f"bohmann_{lam_max:0.3f}_{rhc_max:.3f}_khat")

    # plot lfunction
    lh = lhat(results, w, h, d_max)
    x = lh[:,0]
    y = lh[:,1]
    if not State.no_out:
        fig,axs = plt.subplots(figsize=figsz)
        axs.set_xlabel("d (mm)")
        axs.set_ylabel("$\hat{L}(d)$")
        axs.plot(x,y, label="Messung")
        axs.plot(x, x, label="Poisson")
        axs.legend()
        State.output(StateOutput(fig, sz), f"bohmann_{lam_max:0.3f}_{rhc_max:.3f}_lhat")

    # plot lfunction
    lh = lhatc(results, w, h, d_max)
    x = lh[:,0]
    y = lh[:,1]
    if not State.no_out:
        fig,axs = plt.subplots(figsize=figsz)
        axs.set_xlabel("d (mm)")
        axs.set_ylabel("$\hat{L}(d) - d$")
        axs.plot(x,y, label="Messung")
        axs.plot(x, x-x, label="Poisson")
        # axs.axvline(x=1/lam_max, color='r', linestyle='--', label="1/$\lambda_\mathrm{max}$")
        # axs.axvline(x=1/lam_max_real, color='g', linestyle='--', label="1/$\lambda_\mathrm{max,real}$")
        axs.legend()
        State.output(StateOutput(fig, sz), f"bohmann_{lam_max:0.3f}_{rhc_max:.3f}_lhatc")

    # analyze lhatc function for maximum
    from scipy.signal import argrelextrema
    y = smooth_hanning(y, 20)
    maxima = argrelextrema(y, np.greater, order=3)[0]
    if len(maxima) == 0:
        maxima = [np.argmax(y)]

    maxi = x[maxima[0]]

    # find real rhc
    mini = rhc_minimum(y, x)
    mini = mini if mini != -1 else 0
    measures = {
        "i1": lam_max,
        "i2": lam_max_real,
        "i3": rhc_max,
        "i4": mini,
        "maximum": maxi,
        "measure1": 1/lam_max_real,
        "measure2": 1/lam_max,
    }

    if not State.no_out:
        # perform inverse watershedding here
        watershedded = watershed_points(results, (w,h))
        fix,axs = plt.subplots(figsize=get_fig_width(point_sz))
        axs.imshow(watershedded)
        axs.set_aspect('equal', 'box')
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        State.output(StateOutput(fix, point_sz), f"bohmann_{lam_max:0.3f}_{rhc_max:.3f}_watershed", open=True)

    return measures

@tools_app.command()
def test_bohmann_params(load_data: bool = False):
    """Testing function to find the influence of lambda and rhc on the maximum in the L-Function."""
    from rich.progress import Progress
    # disable subroutine output
    State.no_out = True

    if load_data:
        results_lam = State.from_checkpoint("results_lam", None)
        results_rhc = State.from_checkpoint("results_rhc", None)
        lam_max_values = State.from_checkpoint("lam_max_values", None)
        rhc_max_values = State.from_checkpoint("rhc_max_values", None)

    else:
        lam_max_values = np.linspace(0.01, 0.1, 10)  # Array of lam_max values to test
        rhc_max_values = np.linspace(1, 10, 10)       # Array of rhc_max values to test
        results_lam: dict[float, list[float]] = {}
        for i in rhc_max_values:
            results_lam[i] = []

        results_rhc: dict[float, list[float]] = {}
        for i in lam_max_values:
            results_rhc[i] = []


        with Progress() as progress:
            lamtask = progress.add_task("[cyan]Lambda calculation", total=len(lam_max_values), transient=False)
            for lam_max in lam_max_values:
                rhctask = progress.add_task("[cyan]Rhc calculation", total=len(rhc_max_values), transient=False)
                progress.update(lamtask, description=f'lambda={lam_max:.3f}')
                for rhc_max in rhc_max_values:
                    progress.update(rhctask, description=f'rhc={rhc_max:.3f}')

                    measures = test_bohmann(lam_max=lam_max, rhc_max=rhc_max, c=0)
                    results_lam[rhc_max].append(measures)


                    progress.update(rhctask, advance=1)
                progress.remove_task(rhctask)
                progress.update(lamtask, advance=1)

        # reorder results for plotting rhc
        for i,la in enumerate(lam_max_values):
            for j,rhc in enumerate(rhc_max_values):
                results_rhc[la].append(results_lam[rhc][i])

    State.checkpoint(results_lam=results_lam, results_rhc=results_rhc, lam_max_values=lam_max_values, rhc_max_values=rhc_max_values)  # Save results to disk

        # Plotting examples (Customize these!)
    fig, ax = plt.subplots()

    for rhc in rhc_max_values:
        lams = [x["i1"] for x in results_lam[rhc]]
        maximums = [x["maximum"] for x in results_lam[rhc]]
        ax.plot(lams, maximums, label=f"$r_\mathrm{{HC}}$= {rhc:.3f}")
    ax.set_xlabel('$\lambda_\mathrm{real}$')
    ax.set_ylabel('Observed maximum in L-Func')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    State.output(StateOutput(fig, FigureSize.ROW1),
                    "lam_vs_max",
                    force_open=True, force_out=True)  # Adapt filename as needed
    plt.close(fig)  # Close figure


    # create plot of maximums depending on rhc
    fig, ax = plt.subplots()
    for lam in lam_max_values:
        rhcs = [x["i3"] for x in results_rhc[lam]]
        maximums = [x["maximum"] for x in results_rhc[lam]]
        ax.plot(rhcs, maximums, label=f"$\lambda_\mathrm{{real}}$= {lam:.3f}")
    ax.set_xlabel('$r_\mathrm{HC}$')
    ax.set_ylabel('Observed maximum in L-Func')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    State.output(StateOutput(fig, FigureSize.ROW1),
                    "rhc_vs_max",
                    force_open=True, force_out=True)


    # transform results to X,Y,Z: X=lam, Y=rhc, Z=maximum
    X,Y = np.meshgrid(lam_max_values, rhc_max_values)
    Z = np.zeros((len(rhc_max_values),len(lam_max_values)))

    for i,la in enumerate(lam_max_values):
        for j,rhc in enumerate(rhc_max_values):
            Z[j,i] = results_rhc[la][j]["maximum"] / rhc




    fig, ax = plt.subplots()
    ctrs = ax.contourf(X,Y,Z, cmap='turbo')
    ax.set_xlabel('$\lambda_\mathrm{real}$')
    ax.set_ylabel('$r_\mathrm{HC}$')
    ax.set_title('Maximum in L-Function')
    cbar = fig.colorbar(ctrs, ax=ax)
    renew_ticks_cb(cbar)
    cbar.set_label('$r(\Hat{L}_\mathrm{max})$ / $r_\mathrm{HC}$')
    # mark the contours, where Z is 1 (rhc=lam_max)
    ax.contour(X,Y,Z, levels=[1], colors='r')

    State.output(StateOutput(fig, FigureSize.ROW1),
                    "contour_max",
                    force_open=True, force_out=True)
    plt.close(fig)



    import pickle

    picklefile = State.get_output_file("results_rhc.pickle")
    with open(picklefile, "wb") as f:
        pickle.dump(results_lam, f)

    return results_lam