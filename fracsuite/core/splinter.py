from __future__ import annotations

import cv2
import numpy as np
from rich import print
from skimage.morphology import skeletonize

from fracsuite.core.coloring import rand_col
from fracsuite.core.detection import detect_fragments, remove_dark_spots
from fracsuite.core.image import to_gray, to_rgb
from fracsuite.core.imageplotting import plotImage, plotImages
from fracsuite.core.imageprocessing import (
    closeImg,
    dilateImg,
    erodeImg,
    preprocess_image,
)
from fracsuite.core.preps import PreprocessorConfig
from fracsuite.core.region import RectRegion


class Splinter:

    angle: float
    "Orientation of the splinter in degrees."
    angle_vector: tuple[float, float]
    "Normalized orientation vector"
    alignment_score: float = np.nan
    "Score indicating how much the splinter points into the direction of the impact point."


    roughness: float
    "Roughness of the splinter."
    roundness: float
    """
    Roundness of the splinter. Calculated using the relation between area and
    the corresponding circle area with the same circumfence.
    """

    area: float
    "Area of the splinter in mm²."
    circumfence: float
    "Circumfence of the splinter in mm."
    centroid_mm: tuple[float, float]
    "Centroid of the splinter in mm."
    centroid_px: tuple[float, float]
    "Centroid of the splinter in px."
    has_centroid: bool
    "True if the centroid could be calculated."


    def __init__(self, contour, index, px_per_mm: float):
        """Create a splinter from a contour.

        Args:
            contour (np.array): Input contour.
            index (int): Index of the splinter.
            px_per_mm (float): Scale factor for area. px/mm.
        """
        self.ID = index
        self.contour = contour

        self.area = cv2.contourArea(self.contour) / px_per_mm ** 2
        self.circumfence = cv2.arcLength(self.contour, True) / px_per_mm

        # roundness
        self.roundness = self.calculate_roundness()
        # roughness
        self.roughness = self.calculate_roughness()


        # centroid
        try:
            M = cv2.moments(self.contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            self.centroid_mm =  (cX / px_per_mm, cY / px_per_mm)
            self.centroid_px =  (cX, cY)
            self.has_centroid = True
        except:
            self.centroid_mm = (np.nan, np.nan)
            self.centroid_px = (np.nan, np.nan)

            self.has_centroid = False



        self.angle = self.__calculate_orientation()


        self.adjacent_splinter_ids: list[int] = []
        "List of adjacent splinters."
        self.contour_set = None
        "Set of contour points for faster membership testing."

    def calculate_roundness(self) -> float:
        """
        Calculate the roundness by comparing the actual contour area with the area
        of a circle with the same circumfence.

        Returns:
            float: A value indicating how round the contour is.
        """
        As = cv2.contourArea(self.contour)
        Us = cv2.arcLength(self.contour, True)

        Aeq = (Us ** 2) / (4 * np.pi)

        return As / Aeq

        # new and simpler approach
        # find enclosing circle circumfence
        (x,y),radius = cv2.minEnclosingCircle(self.contour)
        circumfence = 2 * np.pi * radius

        # find circumfence of contour
        contour_circumfence = cv2.arcLength(self.contour,True)

        min_circ = np.min([circumfence, contour_circumfence])
        max_circ = np.max([circumfence, contour_circumfence])

        return min_circ / max_circ

        #check if contour has at least 5 points
        if len(self.contour) < 5:
            rect = cv2.minAreaRect(self.contour)
            # find width and height of rect
            width = rect[1][0]
            height = rect[1][1]

            top = np.max([width,height])
            bot = np.min([width,height])


            if bot != 0:
                r = top/bot
                return np.abs(1-r)
            else:
                return 0

        # find enclosing ellipse radii
        ellipse = cv2.fitEllipse(self.contour)
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2

        return np.abs(1-a/b)

    def calculate_roughness(self) -> float:
        """Calculate the roughness of the contour by comparing the circumfence
        of the contour to the circumfence of its convex hull.

        Returns:
            float: A value indicating how rough the perimeter is.
        """
        contour = self.contour
        perimeter = cv2.arcLength(contour,True)
        hull = cv2.convexHull(contour)
        hullperimeter = cv2.arcLength(hull,True)

        return hullperimeter / perimeter
    def as_array(self) -> np.ndarray:
        """Return the contour as a numpy array."""
        return np.array(self.contour)

    def __calculate_orientation(self):
        """Calculate the orientation of the splinter in degrees."""
        M = cv2.moments(self.contour)

        # Calculate the angle
        angle = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])
        angle_degrees = np.degrees(angle)

        if angle_degrees < 0:
            angle_degrees += 180

        angle_radians = np.radians(angle_degrees)
        x = np.cos(angle_radians)
        y = np.sin(angle_radians)
        self.angle_vector = np.array((x, y))
        return angle_degrees

    def in_region(self, rect: tuple[float,float,float,float]) -> bool:
        """Check if the splinter is in the given region.

        Args:
            rect (tuple[float,float,float,float]): Region to check. (x1,y1,x2,y2)

        Returns:
            bool: True if the splinter is in the region.
        """
        x1,y1,x2,y2 = rect
        x,y = self.centroid_mm
        return x1 <= x <= x2 and y1 <= y <= y2

    def in_region_px(self, region: RectRegion) -> bool:
        return region.is_point_in(self.centroid_px)

    def in_rect_px(self, rect: tuple[int,int,int,int]) -> bool:
        """Check if the splinter is in the given region.

        Args:
            rect (tuple[int,int,int,int]): Region to check. (x1,y1,x2,y2)

        Returns:
            bool: True if the splinter is in the region.
        """
        x1,y1,x2,y2 = rect
        x,y = self.centroid_px
        return x1 <= x <= x2 and y1 <= y <= y2


    def touches(self, other: Splinter):
        # Calculate centroid distance
        # dist = np.linalg.norm(np.array(self.centroid_mm) - np.array(other.centroid_mm))
        # if dist > 100:
        #     return False

        # Convert contour points to sets for faster membership testing
        if self.contour_set is None:
            self.contour_set = set((int(p[0][1]), int(p[0][0])) for p in self.contour)
        if other.contour_set is None:
            other.contour_set = set((int(p[0][1]), int(p[0][0])) for p in other.contour)

        # Check if there are common points in the contours
        common_points = self.contour_set.intersection(other.contour_set)

        return len(common_points) > 0

    def __calculate_orientation_score(self, origin) -> float:
        """Calculate the alignment score of the splinter with the given vector.

        Returns:
            A value from [0,1] indicating, how much the splinter points into the direction of the given vector.
        """
        if len(self.contour) < 5:
            return np.nan

        centroid = self.centroid_mm
        Ax = origin[0] - centroid[0]
        Ay = origin[1] - centroid[1]

        A = np.array((Ax, Ay))

        #
        # Calculate the major axis vector
        ellipse = cv2.fitEllipse(self.contour)
        major_axis_angle = ellipse[2]
        major_axis_angle_rad = np.deg2rad(major_axis_angle)
        major_axis_vector = (np.cos(major_axis_angle_rad), np.sin(major_axis_angle_rad))

        B = np.array(major_axis_vector)


        dot = np.dot(A, B)
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)

        self.alignment_score = 1 - np.abs(dot / (magA * magB))
        return self.alignment_score


    def measure_orientation(self, impact_position: tuple[float,float]) -> float | np.nan:
        """Calculate, how much the splinters orientation points to the impactpoint of config.

        Args:
            impact_position (tuple[float,float]): Impact position in mm.

        Returns:
            float: Orientation in degrees.
        """
        if not self.has_centroid:
            return np.nan

        # calculate the angle between the centroid and the impact point

        return self.__calculate_orientation_score(impact_position)

    @staticmethod
    def analyze_marked_image(
        marked_image,
        original_image,
        px_per_mm=1
    ) -> list[Splinter]:
        """
        Analyzes a marked image to identify splinters.


        The marked image must have the following properties:
            - The image must be in RGB format.
            - The red channel is reserved for markings.
            - The red channel must be set to 255 for all marked pixels.

        The algorithm works as follows:
            1. The red channel is extracted and a mask is created.
            2. Using that mask, we proceed to modify the input image,
                so that marked areas are eroded from the background.
            3. Then, we use the watershed algorithm to identify the sure foreground.
                The markers that were found on the input are now as big as the
                surrounding area, bound by a threshold version of the input.
            4. Next, we use the new markers to perform a second watershed algorithm.
                This is done using Splinter.analyze_image(..).

        Args:
            marked_image (numpy.ndarray): The marked image to analyze.
            px_per_mm (int, optional): The number of pixels per millimeter in the image. Defaults to 1.

        Returns:
            list[Splinter]: A list of Splinter objects representing the identified splinters.
        """
        red_pixels = marked_image[:,:,2] == 255
        red_pixels = red_pixels.astype(np.uint8) * 255
        red_pixels = dilateImg(red_pixels, sz=3, it=1)

        # thresh = cv2.threshold(to_gray(original_image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # thresh[red_pixels == 255] = 255

        thresh = preprocess_image(original_image) # black cracks
        # thresh = cv2.GaussianBlur(thresh, (5, 5), 1)
        # thresh = cv2.threshold(to_gray(marked_image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh[red_pixels == 255] = 255
        thresh = erodeImg(thresh, sz = 5)

        plotImage(thresh, "MK: Thresholded Image")

        # combine the two binary masks using a bitwise AND operation
        red_mask = red_pixels # white splinter marks

        plotImage(red_mask, "MK: Red Mask")

        # step 4: create connectedcomponents and run watershed to identify sure foreground
        _, markers = cv2.connectedComponents(red_mask)
        markers = cv2.watershed(to_rgb(thresh),markers)

        # step 5: create binary stencil from the markers
        m_img = np.zeros_like(marked_image[:,:,0], dtype=np.uint8)
        m_img[markers == -1] = 255
        plotImage(m_img, "MK: WS Contours 1")

        contours, hierarchy = cv2.findContours(to_gray(m_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:]

        filtered_contours = []
        for i, h in enumerate(hierarchy[0]):
            first_child_idx = h[2]
            if first_child_idx == -1:
                filtered_contours.append(contours[i])

        filtered_contours = sorted(filtered_contours, key=cv2.contourArea)
        cimg = to_rgb(255-to_gray(m_img))
        for c in filtered_contours:
            clr = rand_col()
            cv2.drawContours(cimg, [c], -1, clr, 1)
        plotImage(255-cimg, "MK: Detected contours")
        # draw contours on new image
        step1_img = np.zeros_like(marked_image[:,:,0])
        # cv2.drawContours(step1_img, contours, -1, 255, -1)
        # cv2.drawContours(step1_img, contours, -1, 0, 1)

        for c in filtered_contours:
            mask = np.zeros_like(marked_image[:,:,0])
            cv2.drawContours(mask, [c], -1, 255, -1)
            cv2.drawContours(mask, [c], -1, 0, 3)
            step1_img = cv2.add(step1_img, mask)

        plotImage(step1_img, "MK: Step 1 Image") # still black cracks

        # create new set on markers, which are now the sure foreground
        _, markers = cv2.connectedComponents(step1_img)

        # perform second watershed to eliminate background
        markers = cv2.watershed(np.zeros_like(marked_image),markers)

        # create skeleton image from markers
        contour_image = np.zeros(marked_image.shape[:2], dtype=np.uint8)
        contour_image[markers == -1] = 255

        plotImage(contour_image, "MK: Final contours")

        # FINAL STEP: Analyze the resulting contour image
        splinters = Splinter.analyze_contour_image(contour_image, px_per_mm=px_per_mm)

        return splinters

    @staticmethod
    def analyze_label_image(
        label_image,
        px_per_mm: float = 1
    ):
        """
        Analyze a labeled image and return a list of splinters.

        The input label needs to have the following properties:
            - Marked regions describe the crack or any area, that is surely no splinter
            - Unmarked regions are either splinters or cracks, that are surely splinter

        Therefore one can mark everything that is surely no splinter as black, the
        algorithm will simply fill these spaces to find the exact contours.

        You can create a label easily by following these steps:

            1. Use fracsuite tester threshold "path/to/img.ext",
                with img being the input image
            2. Try out the best fitting threshold value, the GUI will save the current
                contours in the background to the output folder
            3. Use a tool like paint.net or GIMP to overlay the contours over the output
                image. Proceed to patch the detected contours, until you are
                satisfied with the result.
            4. Convert the contours to black and insert a white background.
            5. Export the label and use with this function.
        """
        if np.mean(to_gray(label_image)) < 127:
            print("[yellow]Image appears to show more cracks than splinters."
                  "Remember that cracks=black, splinters=white!")

        label_image = to_rgb(label_image)
        return Splinter.analyze_image(label_image, px_per_mm=px_per_mm)

    @staticmethod
    def analyze_contour_image(contour_image, px_per_mm: float = 1.0):
        """Analyze a contour image and return a list of splinters."""
        contour_image = to_gray(contour_image)
        contours = detect_fragments(contour_image, min_area_px=5, max_area_px=25000, filter=True)
        return [Splinter(c, i, px_per_mm) for i, c in enumerate(contours)]

    @staticmethod
    def analyze_image(
        image,
        px_per_mm: float = 1.0,
        skip_preprocessing: bool = False,
        prep: PreprocessorConfig = None,
    ):
        """
        Analyze an unprocessed image and return a list of splinters.

        Remarks:
            - A dilation of the threshold is not feasible, since that will eliminate a lot
            of the smaller splinters.
            - Input image should display cracks in black and splinters white


        Parameters:
            image: The input image to be analyzed. Must be in RGB format for watershed algorithm.
            px_per_mm: The number of pixels per millimeter in the image. Default is 1.0.
            skip_preprocessing: If True, the input image is assumed to be preprocessed already.
        Returns:
            (list[Splinter]): A list of Splinter objects representing the splinters detected in the input image.
        """
        if not skip_preprocessing:
            # here we need a rgb image!
            thresh = preprocess_image(image, prep)
        else:
            image = to_rgb(image)
            thresh = to_gray(image)

        plotImage(thresh, "WS: Preprocessed Image")

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        plotImage(opening, "WS: Opened Image")

        # sure background: white is splinter, black is crack
        sure_bg = cv2.dilate(opening,kernel,iterations=1)

        plotImage(sure_bg, "WS: Sure Background")

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0,)
        # cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        ret, sure_fg = cv2.threshold(dist_transform, 0, 255, 0)
        sure_fg = erodeImg(sure_fg, it=1)

        plotImages([("WS: Distance Transform", dist_transform),("WS: Sure Foreground", sure_fg)])

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[sure_fg!=255] = 0

        plotImages([
                ("WS: Sure Background", sure_bg),
                ("WS: Sure Foreground", sure_fg),
                ("WS: Back - Foreground", unknown),
                ("WS: Markers", np.abs(markers).astype(np.uint8)),
            ])

        markers = cv2.watershed(np.zeros_like(image),markers)

        m_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        m_img[markers == -1] = 255

        return Splinter.analyze_contour_image(m_img, px_per_mm=px_per_mm)

    @staticmethod
    def analyze_image_legacy(image, px_per_mm=1):
        preprocess = preprocess_image(image)
        plotImage(preprocess, "Preprocessed Image")

        prelim_contours = detect_fragments(preprocess, min_area_px=5, max_area_px=2000, filter=False)
        stencil = np.zeros((preprocess.shape[0], preprocess.shape[1]), dtype=np.uint8)
        cv2.drawContours(stencil, prelim_contours, -1, 255, -1)
        plotImage(stencil, "Preliminary Contours")

        stencil = erodeImg(stencil)
        stencil = erodeImg(stencil, 2, 1)
        stencil = erodeImg(stencil, 1, 1)
        plotImage(stencil, "Preliminary Contours")

        skeleton = skeletonize(255-stencil).astype(np.uint8) * 255
        skeleton = closeImg(skeleton, 3, 5)

        skeleton = skeletonize(skeleton).astype(np.uint8) * 255
        plotImage(skeleton, "Before detection")

        contours = detect_fragments(skeleton, min_area_px=5, max_area_px=2000, filter=False)

        filtered_img = remove_dark_spots(
            original_image=image,
            skeleton_image=skeleton,
            contours=contours,
        )
        plotImage(filtered_img, "After detection")

        contours = detect_fragments(filtered_img, min_area_px=5, max_area_px=2000, filter=False)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = [x for x in contours if cv2.contourArea(x) > 0 and len(x) > 5]

        splinters = [Splinter(c, i, px_per_mm) for i, c in enumerate(contours)]

        return splinters