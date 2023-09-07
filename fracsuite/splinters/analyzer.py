from __future__ import annotations
import csv

import os
import random

import cv2
import numpy as np
import numpy.typing as nptyp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist
from skimage.morphology import skeletonize
from tqdm import tqdm
from rich import inspect, print
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from fracsuite.splinters.splinter import Splinter

from scipy.spatial import Delaunay

def isgray(img):
        if len(img.shape) < 3: 
            return True
        if img.shape[2] == 1: 
            return True
        # b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
        return False  
   
def plotImage(img,title:str, color: bool = True, region: tuple[int,int,int,int] = None):
    if isgray(img) and color:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    fig, axs = plt.subplots()
    axs.imshow(img)
    axs.set_title(title)
    
    if region is not None:
        (x1, y1, x2, y2) = region
        axs.set_xlim((x1,x2))
        axs.set_ylim((y1,y2))
        
    plt.show()
     

def crop_perspective(img, cropped_image_size: tuple[int,int], debug: bool):
    """
    Crops a given image to its containing pane bounds. Finds smallest pane countour with
    4 corner points and aligns, rotates and scales the pane to fit a resulting image.

    Args:
        img (Image): Input image with a clearly visible glass pane.
        size (int): Size of the resulting image after perspective transformation.
        dbg (bool): Displays debug image outputs during processing.
        
    Returns:
        img: A cropped image which only contains the glass pane. Size: 1000x1000.
            If no 4 point contour is found, the whole image is returned.
    """
    
    def fourCornersSort(pts):
        """ Sort corners: top-left, bot-left, bot-right, top-right """
        # Difference and sum of x and y value
        # Inspired by http://www.pyimagesearch.com
        diff = np.diff(pts, axis=1)
        summ = pts.sum(axis=1)
        
        # Top-left point has smallest sum...
        # np.argmin() returns INDEX of min
        return np.array([pts[np.argmin(summ)],
                        pts[np.argmax(diff)],
                        pts[np.argmax(summ)],
                        pts[np.argmin(diff)]])
    
    
    
    img_original = img.copy()
    
    im = img.copy()
    im0 = img.copy()
    
    if not isgray(im):
        im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    if isgray(im0):
        im0 = cv2.cvtColor(im0,cv2.COLOR_GRAY2BGR)
        
    
    # # apply gaussian blur to image to we can get rid of some noise
    im = cv2.GaussianBlur(im, (5,5), 5)
    # # restore original image by thresholding
    _,im = cv2.threshold(im,127,255,0)

    if debug:
        plotImage(im, 'CROP: Image for rectangle detection')
    
    # fetch contour information
    contour_info = []
    contours, _ = cv2.findContours(255-im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop through contours and find their properties
    for cnt in contours:
        contour_info.append((
            cnt,
            cv2.isContourConvex(cnt),
            cv2.contourArea(cnt)
        ))

    # os.makedirs('.debug', exist_ok=True)
    # # count files in debug folder
    # file_count = len(os.listdir('.debug'))

    # cv2.imwrite(os.path.join('.debug', f'contours_{file_count}.png'), 
    #             cv2.drawContours(im0, contours, -1, (0,0,255), 10))
    
    # sort contours after their area
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    # take the second largest contour (this has to be the outer bounds of pane)    
    if len(contour_info) > 0:
        max_contour = contour_info[0][0]
    else:
        return img_original
    
    cv2.drawContours(im0, contours, -1, (0,0,255), 10)
    if debug:
        plotImage(im0, 'CROP: Detected contours')
    
    # Simplify contour
    perimeter = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.03 * perimeter, True)

    # Page has 4 corners and it is convex
    # Page area must be bigger than maxAreaFound 
    if (len(approx) == 4 and
            cv2.isContourConvex(approx)):
        pageContour = fourCornersSort(approx[:, 0])
        
    else:
        rect = cv2.boundingRect(approx)
        x, y, w, h = rect

        # Compute the four corners of the rectangle using cv2.boxPoints
        corners = cv2.boxPoints(((x, y), (w, h), 0))

        # Convert the corners to integer values and print the result
        corners = corners.astype(int)
        #raise CropException("Pane boundary could not be found.")
        pageContour = corners        

    if debug:
        cv2.drawContours(im0, [pageContour],-1, (0,0,255), thickness=im.shape[0]//50)
        plotImage(im0, 'CROP: Found rectangle contour')
        
    
    
    im_h, im_w =im.shape[0],im.shape[1]
    
    if im_w > im_h:        
        f = im_w / 4000
        im_w = 4000
        im_h = int(im_w / f)
    else:
        f = im_h / 4000
        im_h = 4000
        im_w = int(im_h / f)
                
    # Create target points
    if cropped_image_size is not None:
        width, height=cropped_image_size
    else:
        height,width=im_w, im_h
        
    tPoints = np.array([[0, 0],
                    [0, height],
                    [width, height],
                    [width, 0]], np.float32)
    # source points are contour corners
    sPoints = pageContour

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)
    # Warping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints)     

    img_original = cv2.warpPerspective(img_original, M, (int(width), int(height)))
    # print(img_original.shape)
    # plotImage(img_original, 'CROP: Cropped image')
    # and return the transformed image
    return img_original



def filter_contours(contours, hierarchy, config: AnalyzerConfig) \
    -> list[nptyp.ArrayLike]:
    """
    This function filters a list of contours.
    """
    if config.debug: 
        len_0 = len(contours)
    # Sort the contours by area (desc)
    contours, hierarchy = zip(*sorted(zip(contours, hierarchy[0]), \
        key=lambda x: cv2.contourArea(x[0]), reverse=True))

    contours = list(contours)
    contours_areas = [cv2.contourArea(x) for x in contours]
    # contour_area_avg = np.average(contours_areas)
    # contour_area_med = np.average(contours_areas)
    # contour_area_std = np.std(contours_areas)
    
    # Create a list to store the indices of the contours to be deleted
    to_delete: list[int] = []
    As = []
    ks = []
    # Iterate over all contours
    for i in tqdm(range(len(contours)), desc = 'Eliminate small contours', leave=False):
        contour_area = contours_areas[i]
        contour_perim = cv2.arcLength(contours[i], False)
        
        
        if contour_area > 0:
            As.append(contour_area)
            ks.append(contour_perim / contour_area)
            
        if contour_area == 0:
            contour_area = 0.00001
            
        # small size
        if contour_perim < config.fragment_min_area_px:
            to_delete.append(i)
        
    # def reject_outliers(data, m = 2.):
    #     return data[abs(data - np.mean(data)) < m * np.std(data)]
    
        # filter outliers
        elif contour_area > config.fragment_max_area_px:
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
    
    if config.debug:    
        len_1 = len(contours)
        print(f'FILT: Contours before: {len_0}, vs after: {len_1}')
        
    return contours

def csintkern(events, region, h):
    n, d = events.shape
    
    # Get the ranges for x and y
    minx = np.min(region[:][0])
    maxx = np.max(region[:][0])
    miny = np.min(region[:][1])
    maxy = np.max(region[:][1])
    
    # Get 50 linearly spaced points
    xd = np.linspace(minx, maxx, 50)
    yd = np.linspace(miny, maxy, 50)
    X, Y = np.meshgrid(xd, yd)
    st = np.column_stack((X.ravel(), Y.ravel()))
    ns = len(st)
    xt = np.vstack(([0, 0], events))
    z = np.zeros(X.ravel().shape)
    
    for i in tqdm(range(ns), leave=False):
        # for each point location, s, find the distances
        # that are less than h.
        xt[0] = st[i]
        # find the distances. First n points in dist
        # are the distances between the point s and the
        # n event locations.
        dist = pdist(xt)
        ind = np.where(dist[:n] <= h)[0]
        t = (1 - dist[ind]**2 / h**2)**2
        z[i] = np.sum(t)
    
    z = z * 3 / (np.pi * h)
    Z = z.reshape(X.shape)
    
    return X, Y, Z

def csintkern_optimized(events, region, h=500):
    n, d = events.shape
    
    # Get the ranges for x and y
    minx = np.min(region[:][0])
    maxx = np.max(region[:][0])
    miny = np.min(region[:][1])
    maxy = np.max(region[:][1])
    
    # Get 50 linearly spaced points
    xd = np.linspace(minx, maxx, 50)
    yd = np.linspace(miny, maxy, 50)
    X, Y = np.meshgrid(xd, yd)
    st = np.column_stack((X.ravel(), Y.ravel()))
    ns = len(st)
    
    # Precompute distances
    dist_matrix = np.linalg.norm(events[:, np.newaxis, :] - st[np.newaxis, :, :], axis=-1)
    t = np.maximum(0, 1 - (dist_matrix / h) ** 2) ** 2

    # Calculate intensities
    z = np.sum(t, axis=0)
    z *= 3 / (np.pi * h)

    Z = z.reshape(X.shape)
    
    return X, Y, Z

def detect_fragments(binary_image, config: AnalyzerConfig) -> list[nptyp.ArrayLike]:
    try:
        # Find contours of objects in the binary image
        contours, hierar = \
            cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours = filter_contours(contours, hierar, config)        
        return contours
    except Exception as e:
        raise ValueError(f"Error in fragment detection: {e}")

def closeImg(image, sz = 3, it=1):
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_closing, iterations=it)

def rand_col():
    """Generate a random color.

    Returns:
        (r,g,b): A random color.
    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def get_color(value, min_value = 0, max_value = 1, colormap_name='viridis'):
    # Normalize the value to be in the range [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)

    # Choose the colormap
    colormap = plt.get_cmap(colormap_name)

    # Map the normalized value to a color
    color = colormap(normalized_value)

    # Convert the RGBA color to RGB
    rgb_color = colors.to_rgba(color)[:3]

    return tuple(int(255 * channel) for channel in rgb_color)

def preprocess_spot_detect(img):
    if not isgray(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.GaussianBlur(img, (5,5), 3)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return img
    

class Analyzer(object):
    """
    Analyzer class that can handle an input image.
    """
    
    file_path: str
    "Path of the input image."
    file_dir: str = None
    "Directory of the input image."
    out_dir: str
    "Directory of the output images."
    
    original_image: nptyp.ArrayLike
    "Original image, after cropping (if applied)."
    preprocessed_image: nptyp.ArrayLike
    "Preprocessed image, after cropping (if applied) and pre-processing."
        
    image_contours: nptyp.ArrayLike
    "Image with all contours drawn."
    image_filled: nptyp.ArrayLike
    "Image with all contours filled."
    
    contours: list[nptyp.ArrayLike]
    "List of all contours."
    splinters: list[Splinter]
    "List of all splinters."
    
    axs: list[plt.Axes]
    "List of all axes."
    fig_comparison: Figure
    fig_area_distr: Figure
    fig_area_sum: Figure
    
    config: AnalyzerConfig
    "Configuration of the analyzer."
    
    def __init__(self, config: AnalyzerConfig = None):
        """Create a new analyzer object.

        Args:
            file_path (str): Path to image.
            crop (bool, optional): Specify, if the input has to be cropped first. 
                Defaults to False.
            img_size (int, optional): Size of the cropped image. 
                Defaults to 4000, only needed if crop=True.
            img_real_size (tuple[int,int], optional):
                Actual size in mm of the input rectangular ply.
        """
        if config is None:
            config = AnalyzerConfig()
        
        self.config = config
        
        #############
        # folder operations
        if config.path.endswith('\\'):
            search_path = os.path.join(config.path, 'fracture', 'morphology')
            for file in os.listdir(search_path):
                if 'Transmission' in file and file.endswith('.bmp'):
                    print(f"[green]Found image in specimen folder.[/green]")
                    self.file_path = os.path.join(search_path, file)
                    self.file_dir = os.path.dirname(self.file_path)
                    break
            self.out_dir = os.path.join(config.path, 'fracture', 'splinter')

            if self.file_dir is None:
                raise Exception("Could not find a morphology file in the specified folder.")
        else:
            self.file_path = config.path
            self.file_dir = os.path.dirname(config.path)
            self.out_dir = os.path.join(self.file_dir, config.out_name)
            
            self.out_dir = os.path.join(self.out_dir, os.path.splitext(os.path.basename(config.path))[0])
        
        
        print(f"Input file: '[bold]{self.file_path}[/bold]'")
        print(f"Output directory: '[bold]{self.out_dir}[/bold]'")
        # create output directory if not exists
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        #############
        # image operations
        print('> Step 1: Preprocessing image...')
        self.original_image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        if config.crop:
            self.original_image = crop_perspective(self.original_image, config.cropped_image_size, config.debug)
            # this is the default for the input images from cullet scanner
            self.original_image = cv2.rotate(self.original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        
        self.preprocessed_image = self.preprocess_image(self.original_image, config)
        

        
        #############
        # calculate scale factors to make measurements in real units
        size_f = 1
        "size factor for mm/px"
        if config.real_image_size is not None and config.cropped_image_size is not None:
            # fx: mm/px
            fx = config.real_image_size[0] / config.cropped_image_size[0]
            fy = config.real_image_size[1] / config.cropped_image_size[1]
            
            # the factors must match because pixels are always squared
            # for landscape image, the img_real_size's aspect ratio must match
            if fx != fy:
                raise Exception("The scale factors for x and y must match!" + \
                    "Check the input image and also the img_real_size parameter.")
            
            # f: mm/px
            size_f = fx
            config.size_factor = size_f
            
        #############
        # initial contour operations
        print('> Step 2: Analyzing contours...')
        all_contours = detect_fragments(self.preprocessed_image, config)
        stencil = np.zeros((self.preprocessed_image.shape[0], \
            self.preprocessed_image.shape[1]), dtype=np.uint8)
        
        for c in all_contours:
            cv2.drawContours(stencil, [c], -1, 255, thickness = -1) 
        
        #############
        # advanced image operations
        # first step is to skeletonize the stencil        
        print('> Step 2.1: Skeletonize 1|2')
        skeleton = skeletonize(255-stencil)
        skeleton = skeleton.astype(np.uint8) * 255
        cv2.imwrite(self.__get_out_file('.debug/skeleton.png'), skeleton)
        
        if config.debug:
            plotImage(skeleton, 'SKEL1', False, region=config.display_region)
        skeleton = closeImg(skeleton, config.skelclose_size, config.skelclose_amnt)
        
        cv2.imwrite(self.__get_out_file('.debug/closed.png'), skeleton)
        
        if config.debug:
            plotImage(skeleton, 'SKEL1 - Closed', False, region=config.display_region)
        # second step is to skeletonize the closed skeleton from #1
        print('> Step 2.2: Skeletonize 2|2')
        skeleton = skeletonize(skeleton)
        skeleton = skeleton.astype(np.uint8) * 255
        cv2.imwrite(self.__get_out_file('.debug/closed_skeleton.png'), skeleton)
        if config.debug:
            plotImage(skeleton, 'SKEL2', False, region=config.display_region)
         
        self.image_skeleton = skeleton.copy()
        
        #############
        # detect fragments on the closed skeleton
        print('> Step 3: Preliminary contour analysis...')
        self.contours = detect_fragments(skeleton, config)        
        self.splinters = [Splinter(x,i,size_f) for i,x in enumerate(self.contours)]

        #############
        # filter dark spots and delete fragments
        if not config.skip_darkspot_removal:
            print('> Step 4: Filter spots...')
            self.__filter_dark_spots(config)
        else:
            print('> Step 4: Filter spots (SKIPPED)')
        
        #############
        # detect fragments on the closed skeleton
        print('> Step 5: Final contour analysis...')
        self.contours = detect_fragments(self.image_skeleton, config)        
        self.splinters = [Splinter(x,i,size_f, config) for i,x in enumerate(self.contours)]
        
        self.__save_data(config)
        
        self.image_skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        
        
        #############
        # create images
        self.image_contours = self.original_image.copy()
        self.image_filled = self.original_image.copy()
        for c in self.contours:
            cv2.drawContours(self.image_contours, [c], -1, rand_col(), 1)        
        for c in self.contours:
            cv2.drawContours(self.image_filled, [c], -1, rand_col(), -1)
        cv2.imwrite(self.__get_out_file(f"img_filled.{self.config.ext_imgs}"), self.image_filled)
        cv2.imwrite(self.__get_out_file(f"img_contours.{self.config.ext_imgs}"), self.image_contours)
        cv2.imwrite(self.__get_out_file(f"debug_preprocessed.{self.config.ext_imgs}"), self.preprocessed_image)
        cv2.imwrite(self.__get_out_file(f"debug_skeleton.{self.config.ext_imgs}"), self.image_skeleton_rgb)

        combined = cv2.addWeighted(self.image_contours, 1.0, self.image_filled, 0.3, 0.0)
        cv2.imwrite(self.__get_out_file(f"img_contours_filled_combined.{self.config.ext_imgs}"), combined)
        

        #############
        # Stochastic analysis
        print("> Step 6: Stochastic analysis...")    
        self.__create_voronoi(config)

        #############
        # Orientational analysis
        if config.impact_position is not None:
            print("> Step 7: Orientation analysis...") 
            self.__create_impact_influence(size_f, config)
        
        #############
        # count splinters in norm region
        if config.norm_region_center is not None:
            print("> Step 8: Count splinters in norm region...")
            self.__count_splinters_in_norm_region(config)
        else:
            print("> Step 8: (SKIPPED)")
            

        self.__create_splintersize_filled_image(config)

        print("> Create output plots...")
        self.__plot_backend(config.display_region, display=config.displayplots)
        self.__plot_splintersize_accumulation(display=config.displayplots)
        self.__plot_splintersize_distribution(display=config.displayplots)
        
        ############
        ############
        # Summary
        print('\n\n')
        print(f'Splinter count: {len(self.contours)}')
        self.__check_detection_ratio(config, doprint=True)
        print('\n')
        
    def preprocess_image(self, image, config: AnalyzerConfig) -> nptyp.ArrayLike:
        """Preprocess a raw image.

        Args:
            image (nd.array): The input image.
            gauss_sz (size-tuple, optional): The size of the gaussian filter. 
                                            Defaults to (5,5).
            gauss_sig (int, optional): The sigma for gaussian filter. Defaults to 3.
            block_size (int, optional): Block size of adaptive threshold. Defaults to 11.
            C (int, optional): Sensitivity of adaptive threshold. Defaults to 6.
            rsz (int, optional): Resize factor for the image. Defaults to 1.

        Returns:
            np.array: Preprocessed image.
        """
        
        rsz_fac = config.resize_factor # x times smaller

        if not isgray(image):
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and enhance edge detection
        image = cv2.GaussianBlur(image, config.gauss_size, config.gauss_sigma)
        image = cv2.resize(image,(int(image.shape[1]/rsz_fac), int(image.shape[0]/rsz_fac)))

        cv2.imwrite(self.__get_out_file('.debug/step1.png'), image)
        if config.debug:
            plotImage(image, 'PREP: GaussianBlur -> Resize', region=config.display_region)
        
        # Use adaptive thresholding
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
            cv2.THRESH_BINARY, config.thresh_block_size, config.thresh_sensitivity)
        
        cv2.imwrite(self.__get_out_file('.debug/step2.png'), image)
        if config.debug:
            plotImage(image, 'PREP: ... -> Adaptive Thresh', region=config.display_region)
        
        return image

    def __save_data(self, config: AnalyzerConfig) -> None:
        """Save splinter data to a csv file.

        Args:
            config (AnalyzerConfig): Configuration.
        """
        
        # save data to csv file
        with open(self.__get_out_file("splinter_data.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['id', 'area', 'circumfence', 'alignment_score', 'angle', 'c_x [px]', 'c_y [px]', 'c_x [mm]', 'c_y [mm]'])
            for i, s in tqdm(enumerate(self.splinters), desc='Saving data...', leave=False):
                writer.writerow([i, s.area, s.circumfence, s.alignment_score, s.angle, s.centroid_px[0], s.centroid_px[1], s.centroid_mm[0], s.centroid_mm[1]])

    def __count_splinters_in_norm_region(self, config: AnalyzerConfig) -> float:
        # create rectangle around args.normregioncenter with 5x5cm size
        # and count splinters in it
        x,y = config.norm_region_center
        w,h = config.norm_region_size
        x1 = x - w // 2
        x2 = x + w // 2
        y1 = y - h // 2
        y2 = y + h // 2

        s_count = 0
        # count splinters in norm region
        for s in self.splinters:
            if s.in_region((x1,y1,x2,y2)):
                s_count += 1

        print(f'Splinters in norm region: {s_count}')

        # transform to real image size
        x1 = int(x1 // config.size_factor)
        x2 = int(x2 // config.size_factor)
        y1 = int(y1 // config.size_factor)
        y2 = int(y2 // config.size_factor)

        # get norm region from original image (has to be grayscale for masking)
        norm_region_mask = np.zeros_like(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY))
        cv2.rectangle(norm_region_mask, (x1,y1), (x2,y2), 255, -1)
        # create image parts        
        normed_image = cv2.bitwise_and(self.image_filled, self.image_filled, mask=norm_region_mask)
        normed_image_surr = self.original_image #cv2.bitwise_and(self.original_image, self.original_image, mask=norm_region_inv)
        # add images together
        normed_image = cv2.addWeighted(normed_image, 0.3, normed_image_surr, 1.0, 0)
        cv2.rectangle(normed_image, (x1,y1), (x2,y2), (255,0,0), 5)        
        cv2.putText(normed_image, f'{s_count}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,0,255), 20)
        cv2.imwrite(self.__get_out_file(f"norm_count.{config.ext_imgs}"), cv2.resize(normed_image, (0,0), fx=0.5, fy=0.5))
        return s_count

    def __check_detection_ratio(self, config: AnalyzerConfig, doprint = False) -> float:
        #############
        # check percentage of detected splinters
        total_area = np.sum([x.area for x in self.splinters])
        
        if config.real_image_size is not None:
            total_img_size = config.real_image_size[0] * config.real_image_size[1]
        else:
            total_img_size = self.preprocessed_image.shape[0] * \
                self.preprocessed_image.shape[1]
            
        p = total_area / total_img_size * 100
        if doprint:
            print(f'Detection Ratio: {p:.2f}%')
        
        return p
        
    def __create_impact_influence(self, size_f: float, config: AnalyzerConfig):
        # analyze splinter orientations
        orientation_image = self.original_image.copy()
        orients = []
        for s in tqdm(self.splinters, leave=False, desc='Analyzing orientations...'):
            orientation = s.measure_orientation(config)
            orients.append(orientation)
            color = get_color(orientation, colormap_name='turbo')
            cv2.drawContours(orientation_image, [s.contour], -1, color, -1)
            # p2 = (s.centroid_px + s.angle_vector * 15).astype(np.int32)
            # cv2.line(orientation_image, s.centroid_px, p2, (255,255,255), 3)
        cv2.circle(orientation_image, (np.array(config.impact_position) / size_f).astype(np.uint32),
                   np.min(orientation_image.shape[:2]) // 50, (255,0,0), -1)

        # save plot
        fig, axs = plt.subplots()
        axim = axs.imshow(orientation_image, cmap='turbo', vmin=0, vmax=1)
        fig.colorbar(axim, label='Strength  [-]')
        axs.xaxis.tick_top()
        axs.xaxis.set_label_position('top')
        axs.set_xlabel('Pixels')
        axs.set_ylabel('Pixels')
        axs.set_title('Splinter orientation towards impact point')
        # create a contour plot of the orientations, that is overlayed onto the original image
        if config.debug:
            fig.show()
            fig.waitforbuttonpress()
        
        fig.tight_layout()
        fig.savefig(self.__get_out_file(f"fig_splinter_orientation.{config.ext_plots}"))


    def __create_voronoi(self, config: AnalyzerConfig):
        centroids = [x.centroid_px for x in self.splinters if x.has_centroid]
        voronoi = Voronoi(centroids)
        fig = voronoi_plot_2d(voronoi, show_points=True, point_size=5, show_vertices=False, line_colors='red')
        
        plt.imshow(self.original_image)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.title('Voronoi Plot Overlay on Image')
        if config.debug_experimental:
            plt.show()
        fig.savefig(self.__get_out_file(f"voronoi.{config.ext_plots}"))
        
        events = np.array(centroids)
        region = np.array([[0,self.original_image.shape[1]], [0, self.original_image.shape[0]]])
        
        # optimal_h = estimate_optimal_h(events, region)
        # print(f'Optimal h: {optimal_h}')
        
        #TODO: compare voronoi splinter size distribution to actual distribution
        # X,Y,Z = csintkern(events, region, 500)        
        X, Y, Z = csintkern_optimized(events, region, config.intensity_h)
        fig,axs = plt.subplots()
        axs.imshow(self.image_contours)
        axim = axs.contourf(X, Y, Z, cmap='turbo', alpha=0.5)
        fig.colorbar(axim, label='Intensity [Splinters / Area]')
        # plt.scatter([x[0] for x in centroids], [x[1] for x in centroids], color='red', alpha=0.5, label='Data Points')
        # plt.legend()
        axs.xaxis.tick_top()
        axs.xaxis.set_label_position('top')
        axs.set_xlabel('Pixels')
        axs.set_ylabel('Pixels')
        axs.set_title(f'Fracture Intensity (h={config.intensity_h:.2f})')
        
        if config.debug_experimental:
            plt.show()
        
        fig.tight_layout()
        fig.savefig(self.__get_out_file(f"fig_intensity.{config.ext_plots}"))
            
    def __filter_dark_spots(self, config: AnalyzerConfig):
        # create normal threshold of original image to get dark spots
        img = preprocess_spot_detect(self.original_image)
        if config.debug_experimental:
            cimg = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        i_del = []
        removed_splinters: list[Splinter] = []
        for i,s in enumerate(bar := tqdm(self.splinters, leave=False)):
            x, y, w, h = cv2.boundingRect(s.contour)
            roi_orig = img[y:y+h, x:x+w]
            
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [s.contour], -1, 255, thickness=cv2.FILLED)
            
            roi = mask[y:y+h, x:x+w]
            # Apply the mask to the original image
            result = cv2.bitwise_and(roi_orig, roi_orig, mask=roi)
            
            # Check if all pixels in the contour area are black
            if np.mean(result) < 0.25:
                bar.set_description(f'Removed {len(i_del)} Splinters')
                i_del.append(i)
            elif config.debug_experimental:
                cv2.drawContours(cimg, [s.contour], -1, rand_col(), 1)
                
        # remove splinters starting from the back                
        for i in sorted(i_del, reverse=True):
            removed_splinters.append(self.splinters[i])
            del self.splinters[i]
            
        if config.debug:
            print(f"Removed {len(removed_splinters)} Splinters")
            
        skel_mask = self.image_skeleton.copy()
        
        for s in tqdm(removed_splinters, leave=False):
            c = s.centroid_px
            
            # Remove the original contour from the mask
            cv2.drawContours(skel_mask, [s.contour], -1, 0, 1)
            cv2.drawContours(skel_mask, [s.contour], -1, 0, -1)
            
            # cv2.drawContours(skel_mask, [s.contour], -1, 0, -1)
            connections = []  
                      
            # Search for adjacent lines that were previously attached
            #   to the removed contour
            for p in s.contour:
                p = p[0]
                # Search the perimeter of the original pixel
                for i,j in [(-1,-1), (-1,0), (-1,1),\
                            (0,-1), (0,1),\
                            (1,-1), (1,0), (1,1) ]:
                    x = p[0] + j
                    y = p[1] + i
                    if x >= self.image_skeleton.shape[1] or x < 0:
                        continue
                    if y >= self.image_skeleton.shape[0] or y < 0:
                        continue
                                                                    
                    # Check if the pixel in the skeleton image is white
                    if skel_mask[y][x] != 0:
                        # Draw a line from the point to the centroid
                        connections.append((x,y))
                        
                        if config.debug_experimental:
                            cv2.drawMarker(cimg, (x,y), (0,255,0))
            
            # 2 connections -> connect them
            if len(connections) == 2:
                cv2.drawContours(self.image_skeleton, [s.contour], -1, (0), -1)
                x,y = connections[0]
                a,b = connections[1]
                
                cv2.line(self.image_skeleton, (int(x), int(y)), (int(a), int(b)), 255)
                
            # more than 2 -> connect each of them to centroid
            elif len(connections) > 2:
                cv2.drawContours(self.image_skeleton, [s.contour], -1, (0), -1)
                # cv2.drawContours(self.image_skeleton, [s.contour], -1, (0), 1)
                for x,y in connections:
                    cv2.line(self.image_skeleton, (int(x), int(y)), (int(c[0]), int(c[1])), 255)
                    
                    if config.debug_experimental:
                        cv2.line(cimg, (int(x), int(y)), (int(c[0]), int(c[1])), (255,0,0))
                        
        
        if config.debug_experimental:
            cv2.imwrite(self.__get_out_file("spots_filled.png"), cimg)
            plt.imshow(cimg)
            plt.show()
        
        
    def __get_out_file(self, file_name: str, file_ext: str = None) -> str:
        """Returns an absolute file path to the output directory.

        Args:
            file_name (str): The filename inside of the output directory.
            file_ext (str): The file extension.
        """
        if file_ext is not None:
            file_name = f'{file_name}.{file_ext}'
        
        path = os.path.join(self.out_dir, file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def __onselect(self,eclick, erelease):
        """ Private function, internal use only. """
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
    
    def __create_splintersize_filled_image(self, config: AnalyzerConfig):
        img = self.original_image.copy()
        areas = [x.area for x in self.splinters]
        
        min_area = np.min(areas)
        max_area = np.max(areas)
        
        for s in self.splinters:            
            clr = get_color(s.area, min_value=min_area, max_value=max_area, colormap_name='turbo')
            cv2.drawContours(img, [s.contour], -1, clr, -1)
            
        cv2.imwrite(self.__get_out_file("img_splintersizes", config.ext_imgs), img)
        combined = cv2.addWeighted(255-self.original_image, 1.0, img, 0.75, 0.0)
        cv2.imwrite(self.__get_out_file(f"img_splintersizes_combined.{self.config.ext_imgs}"), combined)
        combined = cv2.addWeighted(self.image_skeleton_rgb, 1.0, img, 0.6, 0.0)
        cv2.imwrite(self.__get_out_file(f"debug_skeleton_sizes_combined.{self.config.ext_imgs}"), combined)
        
    
    def __plot_backend(self, region = None, display = False) -> None:
        """
        Plots the analyzer backend.
        Displays the original img, preprocessed img, and an overlay of the found cracks
        side by side in a synchronized plot.
        """        
        self.fig_comparison, (self.ax3, self.ax1, self.ax2) = \
            plt.subplots(1, 3, figsize=(12, 6))

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
        self.ax2.imshow(self.image_contours)

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
        
        if display:
            plt.show()
            
        self.fig_comparison.savefig(self.__get_out_file(f"fig_comparison.{self.config.ext_plots}"))
        return self.fig_comparison
        
    def __plot_splintersize_accumulation(self, display = False) -> Figure:
        """Plots a graph of accumulated share of area for splinter sizes.

        Returns:
            Figure: The figure, that is displayed.
        """
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
        
        self.fig_area_sum, ax = plt.subplots()
        
        ax.plot(data_x, data_y, 'g-')    
        ax.plot(data_x, data_y, 'rx', markersize=2)    
        ax.set_title('Splinter Size Accumulation')
        ax.set_xlabel(f"$Area_i [{'mm' if self.config.cropped_image_size is not None else 'px'}²]$")
        ax.set_ylabel(r"$\frac{\sum (Area_i)}{Area_t} [-]$")
        # plt.axvline(np.average(areas),c='b', label = "Area avg")
        # plt.axvline(np.sum(areas), c='g', label='Found Area')
        # plt.axvline(5000**2, c='r', label='Image Area')
        if display:
            plt.show()

        self.fig_area_sum.savefig(self.__get_out_file(f"fig_sumofarea.{self.config.ext_plots}"))

        return self.fig_area_sum
    
    def __plot_splintersize_distribution(self, display = False) -> Figure:
        """Plots a graph of Splinter Size Distribution.

        Returns:
            Figure: The figure, that is displayed.
        """
        areas = [x.area for x in self.splinters]
        
        # ascending sort, smallest to largest
        areas.sort()
            
        data = []
        for area in np.linspace(np.min(areas),np.max(areas),50):
            index = next((i for i, value in enumerate(areas) if value > area), None)
            p = index
            data.append((area, p))

        data_x, data_y = zip(*data)
        
        self.fig_area_distr, ax = plt.subplots()
        ax.plot(data_x, data_y, 'g-')    
        ax.plot(data_x, data_y, 'ro', markersize=3)    
        ax.set_title('Splinter Size Distribution')
        ax.set_xlabel(f"Splinter Area [{'mm' if self.config.cropped_image_size is not None else 'px'}²]")
        ax.set_ylabel(r"Amount of Splinters [-]")
        # plt.axvline(np.average(areas),c='b', label = "Area avg")
        # plt.axvline(np.sum(areas), c='g', label='Found Area')
        # plt.axvline(5000**2, c='r', label='Image Area')
        if display:
            plt.show()
            
        self.fig_area_distr.savefig(self.__get_out_file(f"fig_distribution.{self.config.ext_plots}"))
        return self.fig_area_distr
