from email.mime import image
from glob import glob
import os
import numpy as np
from sklearn import base
import typer
import cv2

from fracsuite.callbacks import main_callback
from fracsuite.state import State, StateOutput

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
