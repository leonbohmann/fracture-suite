from typing import Any
from matplotlib import pyplot as plt
from fracsuite.core.image import to_rgb


def plotImage(img,title:str, cvt_to_rgb: bool = True, region: tuple[int,int,int,int] = None):
    if cvt_to_rgb:
        img = to_rgb(img)

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
        axs[i].imshow(img)
        axs[i].set_title(title)
        if region is not None:
            (x1, y1, w, h) = region
            axs[i].set_xlim((x1-w//2,x1+w//2))
            axs[i].set_ylim((y1-h//2,y1+h//2))
    plt.show()