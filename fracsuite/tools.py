from collections import defaultdict
from email.mime import image
from glob import glob
from itertools import groupby
import os
import numpy as np
from sklearn import base
import typer
import cv2

from fracsuite.callbacks import main_callback
from fracsuite.core.model_layers import get_layer_folder
from fracsuite.core.splinter import Splinter
from fracsuite.general import GeneralSettings
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
                .replace("[CAPTION]", f"{Splinter.get_mode_labels(prop).strip()} in Abhängigkeit des Abstands zum Anschlagpunkt bei verschiedenen Glasdicken; Farbig markiert ist die Formänderungsenergie U.")

            b_latex = boundary_latex.replace("[THICKNESSES]", imLatex).replace("[BOUNDARY]", boundary)
            b_latexs.append(b_latex)

        prop_latex = property_latex.replace("[BOUNDARIES]", "\n".join(b_latexs)).replace("[PROPERTY]", Splinter.get_mode_labels(prop).strip())
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
