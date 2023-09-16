"""
Plotting helper functions
"""

from matplotlib import pyplot as plt
from fracsuite.core.image import to_rgb


def plotImage(img,title:str, color: bool = True, region: tuple[int,int,int,int] = None):
    if color:
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