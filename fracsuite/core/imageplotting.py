from typing import Any
import cv2
from matplotlib import pyplot as plt
from fracsuite.core.image import is_rgb, to_rgb
from fracsuite.tools.GlobalState import GlobalState


def plotImage(img,title:str, cvt_to_rgb: bool = True, region: tuple[int,int,int,int] = None):
    if not GlobalState.debug:
        return

    if cvt_to_rgb:
        img = to_rgb(img)

    if is_rgb(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots()
    axs.imshow(img)
    axs.set_title(title)

    if region is not None:
        (x1, y1, x2, y2) = region
        axs.set_xlim((x1,x2))
        axs.set_ylim((y1,y2))

    plt.show()
    plt.close(fig)


def plotImages(imgs: list[(str, Any)], region = None ):
    """Plots several images side-by-side in a subplot.

    Args:
        imgs (list[tuple[str,Any]]): List of tuples containing the title and the image to plot.
        region (x,y,w,h, optional): A specific region to draw. Defaults to None.
    """
    fig,axs  = plt.subplots(1,len(imgs), sharex='all', sharey='all')
    for i, (title, img) in enumerate(imgs):
        if is_rgb(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].set_title(title)
        if region is not None:
            (x1, y1, w, h) = region
            axs[i].set_xlim((x1-w//2,x1+w//2))
            axs[i].set_ylim((y1-h//2,y1+h//2))
    plt.show()