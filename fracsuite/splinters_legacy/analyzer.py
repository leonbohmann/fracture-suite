from __future__ import annotations

import csv
import json
import os
import pickle
import shutil

import cv2
import numpy as np
import numpy.typing as nptyp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rich import print
from rich.progress import Progress, Task
from skimage.morphology import skeletonize
from multiprocessing import Pool, shared_memory as sm

from fracsuite.core.coloring import rand_col
from fracsuite.core.image import to_rgb
from fracsuite.core.plotting import create_splinter_sizes_image
from fracsuite.core.imageplotting import plotImage, plotImages
from fracsuite.core.progress import get_progress
from fracsuite.core.detection import detect_fragments
from fracsuite.splinters.analyzerConfig import AnalyzerConfig
from fracsuite.core.imageprocessing import (
    closeImg,
    crop_perspective,
    erodeImg,
    preprocess_image,
    preprocess_spot_detect,
)
from fracsuite.core.splinter import Splinter
from fracsuite.general import GeneralSettings
from fracsuite.helpers import (
    get_specimen_path,
)

plt.rcParams['figure.figsize'] = (6, 4)
plt.rc('axes', axisbelow=True) # to get grid into background
plt.rc('grid', linestyle="--") # line style
plt.rcParams.update({'font.size': 12}) # font size

general = GeneralSettings.get()

SM_IMAGE = 'image_41jfnn1fh'

def check_splinter(kv):
    """Check if a splinter is valid.

    kv: tuple(int, splinter)
    """
    i,s,shp = kv

    shm = sm.SharedMemory(name=SM_IMAGE)
    img = np.ndarray(shp, dtype=np.uint8, buffer=shm.buf)
    x, y, w, h = cv2.boundingRect(s.contour)
    roi_orig = img[y:y+h, x:x+w]

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [s.contour], -1, 255, thickness=cv2.FILLED)

    roi = mask[y:y+h, x:x+w]
    # Apply the mask to the original image
    result = cv2.bitwise_and(roi_orig, roi_orig, mask=roi)

    # Check if all pixels in the contour area are black
    #TODO! this value is crucial to the quality of histograms!
    if np.mean(result) < 5:
        return i
    else:
        return -1

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

    "List of all axes."
    fig_comparison: Figure
    fig_area_distr: Figure
    fig_area_sum: Figure

    config: AnalyzerConfig
    "Configuration of the analyzer."
    detection_ratio: float
    "Ratio of detected fragments to all fragments."

    def __init__(self,
                 config: AnalyzerConfig = None,
                 progress: Progress = None,
                 main_task: Task = None,
                 clear_splinters: bool = False,
                 silent: bool = False,
                 no_save: bool = False,
                 splinters_only: bool = False):

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
            print("[orange]Warning[/orange]: No config specified. Using default config.")
            config = AnalyzerConfig()

        step_count = 8

        no_progress = progress is None
        if progress is None and not silent:
            progress = get_progress()
            progress.start()

        if main_task is None and not silent:
            main_task = progress.add_task("Initializing...", total=step_count)



        def update_main(value, str, total = None):
            """Updates the progress bar."""
            if silent:
                return

            progress.update(main_task,
                            completed=value,
                            description=str)

            if total is not None:
                progress.update(main_task, total=total)

        update_main(0, 'Initializing...', total=step_count)

        self.config = config
        self.specimen_config = {
            'break_pos': 'corner',

        }
        if clear_splinters:
            fracpath = os.path.join(get_specimen_path(config.specimen_name),
                                    'fracture',
                                    'splinter')
            shutil.rmtree(fracpath, ignore_errors=True)


        #############
        # folder operations
        if config.path.endswith('\\'):
            search_path = os.path.join(config.path, 'fracture', 'morphology')
            for file in os.listdir(search_path):
                if 'Transmission' in file and file.endswith('.bmp'):
                    update_main(0,
                                "[green]Found image in specimen folder.[/green]")
                    self.file_path = os.path.join(search_path, file)
                    self.file_dir = os.path.dirname(self.file_path)
                    break
            self.out_dir = os.path.join(config.path, 'fracture', 'splinter')

            if self.file_dir is None:
                raise Exception("Could not find a morphology file.")



            config.interest_region = general.interest_region
        else:
            self.file_path = config.path
            self.file_dir = os.path.dirname(config.path)
            self.out_dir = os.path.join(self.file_dir, config.out_name)

            self.out_dir = os.path.join(self.out_dir,
                                        os.path.splitext(
                                            os.path.basename(config.path))[0])


        update_main(0, f"Input file: '[bold]{self.file_path}[/bold]'")
        update_main(0, f"Output directory: '[bold]{self.out_dir}[/bold]'")
        # create output directory if not exists
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)


        #############
        # image operations
        update_main(1, 'Preprocessing image...')
        self.original_image = cv2.imread(self.file_path, cv2.IMREAD_COLOR)
        if config.crop:
            self.original_image = crop_perspective(self.original_image,
                                                   config.cropped_image_size,
                                                   config.debug)
            # this is the default for the input images from cullet scanner
            self.original_image = cv2.rotate(self.original_image,
                                             cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.original_image = to_rgb(self.original_image)
        # preprocessed image has black cracks

        self.preprocessed_image = preprocess_image(self.original_image)

        pipeline = []
        pipelined = []

        pipeline.append(closeImg)
        pipeline.append(erodeImg)
        pipeline.append(erodeImg)

        img = self.preprocessed_image
        for i in range(len(pipeline)):
            img = pipeline[i](img, sz=2, it=2)
            pipelined.append((f'Pipe{i}', img))


        if config.debug:
            plotImages([('Original', self.original_image),('Preprocessed', self.preprocessed_image)]
                       + pipelined, region=(500,500,100,100))

        #############
        # calculate scale factors to make measurements in real units
        size_f = 1
        "size factor for mm/px"
        if config.real_image_size is not None:
            if config.cropped_image_size is None:
                cmpSize = self.preprocessed_image.shape[:2]
            else:
                cmpSize = config.cropped_image_size

            # fx: mm/px
            fx = config.real_image_size[0] / cmpSize[0]
            fy = config.real_image_size[1] / cmpSize[1]


            # the factors must match because pixels are always squared
            # for landscape image, the img_real_size's aspect ratio must match
            if fx != fy:
                raise Exception("The scale factors for x and y must match!" + \
                    "Check the input image and also the img_real_size parameter."
                    f"Factors: {fx} and {fy}")

            # f: mm/px
            size_f = fx
            config.size_factor = size_f
        else:
            print("[orange]Warning[/orange]: No real image size specified.")

        #############
        # initial contour operations
        update_main(2, 'Analyzing contours...')
        all_contours = detect_fragments(self.preprocessed_image,
                                        min_area_px=config.fragment_min_area_px,
                                        max_area_px=config.fragment_max_area_px,
                                    )
        stencil = np.zeros((self.preprocessed_image.shape[0], \
            self.preprocessed_image.shape[1]), dtype=np.uint8)

        for c in all_contours:
            cv2.drawContours(stencil, [c], -1, 255, thickness = -1)
        cv2.imwrite(self.__get_out_file('.debug/stencil.png'), 255-stencil)


        # erode stencil
        er_stencil = erodeImg(stencil)
        er1_stencil = erodeImg(er_stencil, 2, 1)
        er1_stencil = erodeImg(er1_stencil, 1, 1)
        if config.debug:
            plotImages([('Original', self.original_image),('Preprocessed', self.preprocessed_image),
                    ('Stencil', stencil), ('Eroded Stencil', er_stencil), ('Eroded Stencil 1', er1_stencil)], region=(300,500,300,300))

        stencil = er1_stencil
        #############
        # advanced image operations
        # first step is to skeletonize the stencil
        update_main(2, 'Skeletonize 1|2')
        # skeleton = thinning(255-stencil)  * 255
        skeleton = skeletonize(255-stencil)
        skeleton = skeleton.astype(np.uint8) * 255
        cv2.imwrite(self.__get_out_file('.debug/skeleton.png'), skeleton)

        if config.debug:
            plotImage(skeleton, 'SKEL1', False, region=config.interest_region)
        skeleton = closeImg(skeleton, config.skelclose_size, config.skelclose_amnt)

        cv2.imwrite(self.__get_out_file('.debug/closed.png'), skeleton)

        if config.debug:
            plotImage(skeleton, 'SKEL1 - Closed', False, region=config.interest_region)
        # second step is to skeletonize the closed skeleton from #1
        update_main(2, 'Skeletonize 2|2')
        # skeleton = thinning(skeleton)* 255
        skeleton = skeletonize(skeleton)
        skeleton = skeleton.astype(np.uint8) * 255
        cv2.imwrite(self.__get_out_file('.debug/closed_skeleton.png'), skeleton)
        if config.debug:
            plotImage(skeleton, 'SKEL2', False, region=config.interest_region)

        self.image_skeleton = skeleton.copy()

        #############
        # detect fragments on the closed skeleton
        update_main(3, 'Preliminary contour analysis...')
        self.contours = detect_fragments(skeleton,
                                         min_area_px=config.fragment_min_area_px,
                                        max_area_px=config.fragment_max_area_px,)
        self.splinters = [Splinter(x,i,size_f) for i,x in enumerate(self.contours)]

        if splinters_only:
            return

        #############
        # filter dark spots and delete fragments
        if not config.skip_darkspot_removal:
            update_main(4, 'Filter spots...')
            dark_spot_task = progress.add_task("Filtering dark spots...") if not silent else None
            self.__filter_dark_spots(config, progress, dark_spot_task, silent)
        else:
            update_main(4, 'Filter spots... (SKIPPED)')

        #############
        # detect fragments on the closed and possibly filtered skeleton
        update_main(5, 'Final contour analysis')
        self.contours = detect_fragments(self.image_skeleton,
                                         min_area_px=config.fragment_min_area_px,
                                        max_area_px=config.fragment_max_area_px,)
        self.splinters = [Splinter(x,i,size_f) for i,x in enumerate(self.contours)]

        self.image_skeleton_rgb = to_rgb(self.image_skeleton)


        #############
        # create images
        update_main(6, 'Save images...')
        if not no_save:

            size_file = self.__get_out_file("img_splintersizes", general.image_extension)
            create_splinter_sizes_image(self.splinters, self.image_skeleton_rgb.shape, size_file)

            self.image_contours = self.original_image.copy()
            self.image_filled = self.original_image.copy()
            for c in self.contours:
                cv2.drawContours(self.image_contours, [c], -1, rand_col(), 1)
            for c in self.contours:
                cv2.drawContours(self.image_filled, [c], -1, rand_col(), -1)
            # filled splinters
            cv2.imwrite(self.__get_out_file(f"img_filled.{general.image_extension}"),
                        self.image_filled)
            # contoured splinters
            cv2.imwrite(self.__get_out_file(f"img_contours.{general.image_extension}"),
                        self.image_contours)
            # preprocessed image
            cv2.imwrite(self.__get_out_file(f"img_debug_preprocessed.{general.image_extension}"),
                        self.preprocessed_image)
            # skeleton
            cv2.imwrite(self.__get_out_file(f"img_debug_skeleton.{general.image_extension}"),
                        self.image_skeleton_rgb)
            # contours with filled combined
            combined = cv2.addWeighted(self.image_contours, 1.0, self.image_filled, 0.3, 0.0)
            cv2.imwrite(self.__get_out_file(f"img_contours_filled_combined.{general.image_extension}"),
                        combined)

        #############
        # Orientational analysis
        update_main(7, 'Orientation analysis')
        # position = \
        #     (50,50) if self.specimen_config['break_pos'] == "corner" \
        #     else (250,250)

        # for s in self.splinters:
        #     s.measure_orientation(position)

        update_main(8, 'Save data...')
        if not no_save:
            self.__save_data(config)
            self.save_object()


        self.__plot_backend(display=config.displayplots, region=config.interest_region)
        # #############
        # # Stochastic analysis
        # updater(6, 'Stochastic analysis')
        # self.__create_voronoi(config)


        data = {}

        data['detection_rate'] = self.__check_detection_ratio(config)
        data['realsize'] = config.real_image_size
        data['size_factor'] = config.size_factor
        data['cropsize'] = config.cropped_image_size \
            if config.cropped_image_size is not None else self.original_image.shape[:2]

        with open(self.__get_out_file('splinters_data.json'), 'w') as f:
            json.dump(data, f, indent=4)


        if no_progress and not silent:
            progress.stop()
        # self.__create_splintersize_filled_image(config)

        # update_main(9, 'Create output plots')
        # self.__plot_backend(config.display_region, display=config.displayplots)

        # ############
        # ############
        # # Summary
        # self.__check_detection_ratio(config, doprint=do_print)

        # update_main(10, 'Saving data')
        # self.save_object()

    def save_object(self):
        with open(self.__get_out_file("splinters.pkl"), 'wb') as f:
            pickle.dump(self.splinters, f)
        with open(self.__get_out_file("config.pkl"), 'wb') as f:
            pickle.dump(self.config, f)

    def __save_data(self, config: AnalyzerConfig) -> None:
        """Save splinter data to a csv file.

        Args:
            config (AnalyzerConfig): Configuration.
        """

        # save data to csv file
        with open(self.__get_out_file("splinter_data.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['id', 'area', 'circumfence', 'alignment_score', 'angle', 'c_x [px]', 'c_y [px]', 'c_x [mm]', 'c_y [mm]'])
            for i, s in enumerate(self.splinters):
                writer.writerow([i, s.area, s.circumfence, s.alignment_score, s.angle, s.centroid_px[0], s.centroid_px[1], s.centroid_mm[0], s.centroid_mm[1]])


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
            self.detection_ratio = p
            print(f'Detection Ratio: {p:.2f}%')

        return p

    def __filter_dark_spots(self,
                            config: AnalyzerConfig,
                            progress: Progress,
                            task: Task = None,
                            silent: bool = False
                        ):
        """Filter contours, that contain only dark spots in the original image.

        Args:
            config (AnalyzerConfig): Configuration.
            task (update-task, optional):
                Task that can be called to update some progress.
                Defaults to None.
        """
        # create normal threshold of original image to get dark spots
        img = preprocess_spot_detect(self.original_image)
        if config.debug:
            cimg = to_rgb(self.original_image)
        i_del = []
        removed_splinters: list[Splinter] = []

        if task is None and not silent:
            task = progress.add_task("Filtering dark spots...",
                                     total=len(self.splinters)+2)

        def update_task(task, advance=1, total=None, descr=None):
            if silent:
                return
            if total is not None:
                progress.update(task, total=total)
            if descr is not None:
                progress.update(task, description=descr)

            progress.update(task, advance=advance)

        print(img.shape)
        shm = sm.SharedMemory(create=True, size=img.nbytes, name=SM_IMAGE)
        shm_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
        shm_img[:] = img[:]
        i_del = []
        update_task(task, advance=1, descr='Finding dark spots...', total = len(self.splinters))
        with Pool(processes=4) as pool:
            for i,result in enumerate(pool.imap_unordered(check_splinter, [(i,s,img.shape) for i,s in enumerate(self.splinters)])):
                if result != -1:
                    i_del.append(result)
                update_task(task, advance=1)


        update_task(task, advance=1, total=len(i_del), descr="Remove splinters...")
        # remove splinters starting from the back
        for i in sorted(i_del, reverse=True):
            update_task(task, advance=1)
            removed_splinters.append(self.splinters[i])
            del self.splinters[i]

        if config.debug:
            print(f"Removed {len(removed_splinters)} Splinters")

        skel_mask = self.image_skeleton.copy()

        update_task(task, advance=1, total=len(removed_splinters), descr="Fill dark spots...")
        for s in removed_splinters:

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

                        if config.debug:
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

                    if config.debug:
                        cv2.line(cimg, (int(x), int(y)), (int(c[0]), int(c[1])), (255,0,0))

            update_task(task, advance=1)

        if config.debug:
            cv2.imwrite(self.__get_out_file("spots_filled.png"), cimg)
            plt.imshow(cimg)
            plt.show()

        del skel_mask

        if not silent:
            progress.remove_task(task)


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

    def __plot_backend(self, region = None, display = False) -> None:

        """
        Plots the analyzer backend.
        Displays the original img, preprocessed img, and an overlay of the found cracks
        side by side in a synchronized plot.
        """
        self.fig_comparison, (ax3, ax1, ax2) = \
            plt.subplots(1, 3, figsize=(12, 6), sharex='all', sharey='all')


        # Display the result from Canny edge detection
        ax3.imshow(self.original_image)
        ax3.set_title("Original")
        ax3.axis('off')

        # Display the result from Canny edge detection
        ax1.imshow(self.preprocessed_image, cmap='gray')
        ax1.set_title("Preprocessed image")
        ax1.axis('off')

        # Overlay found contours on the original image
        ax2.set_title("Detected Cracks")
        ax2.axis('off')
        img = self.image_contours
        for c in self.contours:
            cv2.drawContours(img, [c], -1, (255,0,0), 1)
        ax2.imshow(img)

        if region is not None:
            (x, y, w, h) = region
            ax1.set_xlim(x-w//2, x+w//2)
            ax1.set_ylim(y-h//2, y+h//2)
            ax2.set_xlim(x-w//2, x+w//2)
            ax2.set_ylim(y-h//2, y+h//2)
            ax3.set_xlim(x-w//2, x+w//2)
            ax3.set_ylim(y-h//2, y+h//2)
        else:
            # zoom into the image so that 25% of the image width is visible
            x0, x = self.original_image.shape[0] * 0.35, self.original_image.shape[0] * 0.40
            y0, h = self.original_image.shape[1] * 0.35, self.original_image.shape[1] * 0.40

            ax1.set_xlim(x0, x)
            ax1.set_ylim(y0, h)
            ax2.set_xlim(x0, x)
            ax2.set_ylim(y0, h)
            ax3.set_xlim(x0, x)
            ax3.set_ylim(y0, h)

        plt.tight_layout()

        if display:
            plt.show()

        self.fig_comparison.savefig(self.__get_out_file(f"fig_comparison.{general.plot_extension}"))
        plt.close(self.fig_comparison)


'''

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

        # print(f'Splinters in norm region: {s_count}')

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
        normed_image_surr = to_gray(self.original_image) #cv2.bitwise_and(self.original_image, self.original_image, mask=norm_region_inv)
        # add images together
        normed_image = cv2.addWeighted(normed_image, 0.3, normed_image_surr, 1.0, 0)
        cv2.rectangle(normed_image, (x1,y1), (x2,y2), (255,0,0), 5)
        cv2.putText(normed_image, f'{s_count}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,0,255), 20)
        cv2.imwrite(self.__get_out_file(f"norm_count.{config.ext_imgs}"), cv2.resize(normed_image, (0,0), fx=0.5, fy=0.5))
        return s_count

    def plot_splintersize_accumulation(self, display = False) -> Figure:

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

        fig, ax = plt.subplots()

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


        ax.grid(True, which='both', axis='both')
        fig.tight_layout()
        fig.savefig(self.__get_out_file(f"fig_sumofarea.{self.config.ext_plots}"))
        plt.close(fig)

    def __plot_splintersize_distribution(self, display = False) -> Figure:

        """Plots a graph of Splinter Size Distribution.

        Returns:
            Figure: The figure, that is displayed.
        """
        areas = [x.area for x in self.splinters]

        # ascending sort, smallest to largest
        areas.sort()

        for area in np.linspace(np.min(areas),np.max(areas),50):
            index = next((i for i, value in enumerate(areas) if value > area), None)
            p = index
            data.append((area, p))

        data_x, data_y = zip(*data)

        fig, ax = plt.subplots()
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

        ax.grid(True, which='both', axis='both')

        fig.tight_layout()
        fig.savefig(self.__get_out_file(f"fig_distribution.{self.config.ext_plots}"))
        plt.close(fig)

    def __create_voronoi(self, config: AnalyzerConfig):
        centroids = np.array([x.centroid_px for x in self.splinters if x.has_centroid])
        voronoi = Voronoi(centroids)

        voronoi_img = np.zeros_like(self.original_image, dtype=np.uint8)
        if not is_gray(voronoi_img):
            voronoi_img = cv2.cvtColor(voronoi_img, cv2.COLOR_BGR2GRAY)
        for i, r in enumerate(voronoi.regions):
            if -1 not in r and len(r) > 0:
                polygon = [voronoi.vertices[i] for i in r]
                polygon = np.array(polygon, dtype=int)
                cv2.polylines(voronoi_img, [polygon], isClosed=True, color=255, thickness=2)

        cv2.imwrite(self.__get_out_file(f"voronoi_img.{config.ext_imgs}"), voronoi_img)
        fig = voronoi_plot_2d(voronoi, show_points=True, point_size=5, show_vertices=False, line_colors='red')
        plt.imshow(self.original_image)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.title('Voronoi Plot Overlay on Image')
        if config.debug:
            plt.show()
        fig.savefig(self.__get_out_file(f"voronoi.{config.ext_plots}"))


        # optimal_h = estimate_optimal_h(events, region)
        # print(f'Optimal h: {optimal_h}')

        #TODO: compare voronoi splinter size distribution to actual distribution
        # X,Y,Z = csintkern(events, region, 500)
        self.create_intensity_plot(config.intensity_h, config)


    def plot_logarithmic_to_axes(self, axs, config: AnalyzerConfig, label: str = None):
        self.__plot_logarithmic_histograms(config, axes=axs, label=label)

    def __plot_logarithmic_histograms(self, config: AnalyzerConfig, display = False, axes = None, label: str = None) -> Figure:
        """Plots a graph of Splinter Size Distribution.

        Returns:
            Figure: The figure, that is displayed.
        """
        # fetch areas from splinters
        areas = [np.log10(x.area) for x in self.splinters if x.area > 0]
        # ascending sort, smallest to largest
        areas.sort()

        fig, ax = plt.subplots()
        if axes is not None:
            ax = axes

        if label is None:
            label = self.config.specimen_name

        # density: normalize the bins data count to the total amount of data
        ax.hist(areas, bins=int(config.probabilitybins),
                density=True, label=label,
                alpha=0.5)
        ax.set_xlim([0, np.max(areas)])
        # ax.set_xscale('log')
        ticks = FuncFormatter(lambda x, pos: '{0:.00f}'.format(10**x))
        ticksy = FuncFormatter(lambda x, pos: '{0:.2f}'.format(x))
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticksy)


        # ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Splinter Area [mm²]')
        ax.set_ylabel('Probability (Area) [-]')
        if display:
            plt.show()

        if axes is None:
            ax.grid(True, which='both', axis='both')
            fig.tight_layout()
            fig.savefig(self.__get_out_file(f"fig_log_probability.{self.config.ext_plots}"))

        plt.close(fig)
'''
