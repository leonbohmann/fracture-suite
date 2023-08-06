from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import numpy as np
import numpy.typing as nptyp

import cv2
from tqdm import tqdm
from skimage.morphology import skeletonize

from fracture_suite.splinter import Splinter
from matplotlib import pyplot as plt

def preprocess_image(image, gauss_sz = (3,3), gauss_sig = 5, \
    block_size=11, C=6, rsz=1) -> nptyp.ArrayLike:
    """Preprocess a raw image.

    Args:
        image (nd.array): The input image.
        gauss_sz (size-tuple, optional): The size of the gaussian filter. 
                                        Defaults to (5,5).
        gauss_sig (int, optional): The sigma for gaussian filter. Defaults to 3.
        C (int, optional): Sensitivity of adaptive threshold. Defaults to 6.
        rsz (int, optional): Resize factor for the image. Defaults to 1.

    Returns:
        np.array: Preprocessed image.
    """
    
    rsz_fac = rsz # x times smaller
    
    # Apply Gaussian blur to reduce noise and enhance edge detection
    image = cv2.GaussianBlur(image, gauss_sz, gauss_sig)
    image = cv2.resize(image, (image.shape[1]//rsz_fac, image.shape[0]//rsz_fac))

    # Use adaptive thresholding
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        cv2.THRESH_BINARY, block_size, C)
    
    return image



def filter_contours(contours, hierarchy) -> list[nptyp.ArrayLike]:
    """
    This function filters a list of contours.
    """
    len_0 = len(contours)
    # Sort the contours by area (desc)
    contours, hierarchy = zip(*sorted(zip(contours, hierarchy[0]), \
        key=lambda x: cv2.contourArea(x[0]), reverse=True))

    contours = list(contours)
    contours_areas = [cv2.contourArea(x) for x in contours]
    contour_area_avg = np.average(contours_areas)
    contour_area_med = np.average(contours_areas)
    contour_area_std = np.std(contours_areas)
    
    # Create a list to store the indices of the contours to be deleted
    to_delete = []
    As = []
    ks = []
    # Iterate over all contours
    for i in tqdm(range(len(contours)), desc = 'Eliminate small contours'):
        contour_area = contours_areas[i]
        contour_perim = cv2.arcLength(contours[i], False)
        
        
        if contour_area > 0:
            As.append(contour_area)
            ks.append(contour_perim / contour_area)
            
        if contour_area == 0:
            contour_area = 0.00001
            
        # small size
        if contour_perim < 30:
            to_delete.append(i)
        
    # def reject_outliers(data, m = 2.):
    #     return data[abs(data - np.mean(data)) < m * np.std(data)]
    
        # filter outliers
        elif contour_area > 25000:
            to_delete.append(i)
            
        # # more line shaped contour
        # elif contour_perim / contour_area > 1:
        #     contours[i] = cv2.convexHull(contours[i])
        
        # # Check if the contour has a parent
        # elif hierarchy[i][1] != -1:
        #     # If so, mark the contour for deletion
        #     to_delete.append(i)
        
        # # all other cases
        # else:    
        #     contours[i] = connect_closest_hard_angles(contours[i], 140)
        #     contours[i] = cv2.approxPolyDP(contours[i], 0.1, True)

    # Delete the marked contours
    for index in sorted(to_delete, reverse=True):
        del contours[index]
   
    len_1 = len(contours)
    
    print(f'Contours before: {len_0}, vs after: {len_1}')
    return contours

def detect_fragments(binary_image) -> list[nptyp.ArrayLike]:
    try:
        # Find contours of objects in the binary image
        contours, hierar = \
            cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours = filter_contours(contours, hierar)        
        return contours
    except Exception as e:
        raise ValueError(f"Error in fragment detection: {e}")

def closeImg(image, sz = 3, it=1):
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_closing, iterations=it)


class Analyzer(object):
    file_path: str
    
    original_image: nptyp.ArrayLike
    preprocessed_image: nptyp.ArrayLike
        
    contours: list[nptyp.ArrayLike]
    splinters: list[Splinter]
    
    axs: list[plt.axes]
    
    def __init__(self, file_path: str):
        # image operations
        self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        self.preprocessed_image = preprocess_image(self.original_image)
        
        # contour operations
        all_contours = detect_fragments(self.preprocessed_image)
        stencil = np.zeros((self.preprocessed_image.shape[1], \
            self.preprocessed_image.shape[0]), dtype=np.uint8)
        for c in tqdm(all_contours):
            cv2.drawContours(stencil, [c], -1, 255, thickness = -1) 
        
        # advanced image operations
        # first step is to skeletonize the stencil
        print('Skeletonize 1|2')
        skeleton = skeletonize(255-stencil)
        skeleton = skeleton.astype(np.uint8)
        skeleton = closeImg(skeleton, 3, 5)
        # second step is to skeletonize the closed skeleton from #1
        print('Skeletonize 2|2')
        skeleton = skeletonize(skeleton)
        skeleton = skeleton.astype(np.uint8)
        
        # detect fragments on the closed skeleton
        self.contours = detect_fragments(skeleton)        
        self.splinters = [Splinter(x,i) for i,x in enumerate(self.contours)]
    
    def __onselect(self,eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        selected_region = (x1, y1, x2, y2)
        
        if x1 > x2:
            x1,x2 = x2,x1
        if y1 > y2:
            y1,y2 = y2,y1
            
        self.ax1.set_xlim(x1, x2)
        self.ax1.set_ylim(y1, y2)
        self.ax2.set_xlim(x1, x2)
        self.ax2.set_ylim(y1, y2)
        self.ax3.set_xlim(x1, x2)
        self.ax3.set_ylim(y1, y2)
        plt.draw()
        # print(selected_region)
        return selected_region
    
    def plot(self, region = None):        
        fig, (self.ax3, self.ax1, self.ax2) = plt.subplots(1, 3, figsize=(12, 6))

        # Display the result from Canny edge detection
        self.ax3.imshow(self.original_image)
        self.ax3.set_title("Original")
        self.ax3.axis('off')

        # Display the result from Canny edge detection
        self.ax1.imshow(self.preprocessed_image, cmap='gray')
        self.ax1.set_title("Preprocessed image")
        self.ax1.axis('off')

        # Overlay found contours on the original image
        self.ax2.set_title("Detected Cracks")
        self.ax2.axis('off')
        image0 = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2RGB)
        
        # for c in contours:        
        #     cv2.drawContours(image0, [c], -1, (120,0,120), thickness = -1)        
        cv2.drawContours(image0, self.contours, -1, (255,0,0), thickness = 1)
        
        self.ax2.imshow(image0)
        # for contour in tqdm(contours):
        #     ax2.plot(contour[:, 0, 0], contour[:, 0, 1], 'r')

        if region is not None:
            (x1, y2, x2, y1) = region
            self.ax1.set_xlim(x1, x2)
            self.ax1.set_ylim(y1, y2)
            self.ax2.set_xlim(x1, x2)
            self.ax2.set_ylim(y1, y2)
            self.ax3.set_xlim(x1, x2)
            self.ax3.set_ylim(y1, y2)

        # Connect the rectangle selector to synchronize zooming
        rs = RectangleSelector(self.ax1, self.__onselect, useblit=True, \
            button=[1], spancoords='pixels', )
        rs1 = RectangleSelector(self.ax2, self.__onselect, useblit=True, \
            button=[1], spancoords='pixels', )
        rs2 = RectangleSelector(self.ax3, self.__onselect, useblit=True, \
            button=[1], spancoords='pixels', )    
        rs.add_state('square')

        rs1.add_state('square')
        rs2.add_state('square')
        plt.tight_layout()
        plt.show()
        
    def plot_area(self) -> Figure:
        areas = [x.area for x in self.splinters]
        total_area = np.sum(areas)
        
        # ascending sort, smallest to largest
        areas.sort()
            
        data = []
        
        area_i = 0
        for area in areas:
            area_i += area    
            
            p = area_i / total_area
            data.append((area, p))

        data_x, data_y = zip(*data)
        
        fig = plt.figure()
        plt.plot(data_x, data_y)    
        plt.title('Splinter Size Accumulation')
        plt.xlabel("$\sum Area_i [mm²]$")
        plt.ylabel(r"$\frac{\sum (Area_i)}{Area_t} [-]$")
        # plt.axvline(np.average(areas),c='b', label = "Area avg")
        # plt.axvline(np.sum(areas), c='g', label='Found Area')
        # plt.axvline(5000**2, c='r', label='Image Area')
        plt.show()
        return fig
