import numpy as np
import cv2
from fracsuite.core.coloring import rand_col

from fracsuite.core.image import is_rgb, to_gray, to_rgb
from fracsuite.core.imageplotting import plotImage, plotImages
from fracsuite.core.detection import detect_fragments
from fracsuite.splinters.processing import erodeImg


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
    "Area of the splinter."
    circumfence: float
    "Circumfence of the splinter."
    centroid_mm: tuple[float, float]
    "Centroid of the splinter in mm."
    centroid_px: tuple[float, float]
    "Centroid of the splinter in px."
    has_centroid: bool
    "True if the centroid could be calculated."


    def __init__(self, contour, index, mm_px: float):
        """Create a splinter from a contour.

        Args:
            contour (np.array): Input contour.
            index (int): Index of the splinter.
            mm_px (float): Scale factor for area. px/mm.
        """
        self.ID = index
        self.contour = contour

        self.area = cv2.contourArea(self.contour) * mm_px ** 2
        self.circumfence = cv2.arcLength(self.contour, True) * mm_px

        # roundness
        self.roundness = self.calculate_roundness()
        # roughness
        self.roughness = self.calculate_roughness()


        # centroid
        try:
            M = cv2.moments(self.contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            self.centroid_mm =  (cX * mm_px, cY * mm_px)
            self.centroid_px =  (cX, cY)
            self.has_centroid = True
        except:
            self.centroid_mm = (np.nan, np.nan)
            self.centroid_px = (np.nan, np.nan)

            self.has_centroid = False



        self.angle = self.__calculate_orientation()




    def calculate_roundness(self) -> float:
        """Calculate the roundness of the contour by comparing the area of the
        contour to the area of its corresponding circle with the same circumfence.

        Returns:
            float: A value indicating how round the contour is.
        """
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

        return perimeter / hullperimeter - 1

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
    def in_region_px(self, rect: tuple[float,float,float,float]) -> bool:
        """Check if the splinter is in the given region.

        Args:
            rect (tuple[float,float,float,float]): Region to check. (x1,y1,x2,y2)

        Returns:
            bool: True if the splinter is in the region.
        """
        x1,y1,x2,y2 = rect
        x,y = self.centroid_px
        return x1 <= x <= x2 and y1 <= y <= y2

    def __calculate_orientation_score(self, origin) -> float:
        """Calculate the alignment score of the splinter with the given vector.

        Returns:
            A value from [0,1] indicating, how much the splinter points into the direction of the given vector.
        """
        centroid = self.centroid_mm
        Ax = origin[0] - centroid[0]
        Ay = origin[1] - centroid[1]
        A = np.array((Ax, Ay))

        ellipse = cv2.fitEllipse(self.contour)

        # Get the major axis angle in degrees
        major_axis_angle = ellipse[2]

        # Convert the angle to radians
        major_axis_angle_rad = np.deg2rad(major_axis_angle)

        # Calculate the major axis vector
        major_axis_vector = (np.cos(major_axis_angle_rad), np.sin(major_axis_angle_rad))
        # get main axis of ellipse
        B = np.array(major_axis_vector)


        dot = np.dot(A, B)
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)

        self.alignment_score = 1 - np.abs(dot / (magA * magB))
        return self.alignment_score



    def measure_orientation(self, impact_position: tuple[float,float]) -> float:
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
    def analyze_marked_image(marked_image, px_per_mm=1, return_thresh=False):
        # step 1: find markers and background
        red_pixels = marked_image[:,:,2] == 255

        # step 2: create binary mask for image
        thresh = cv2.threshold(marked_image[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = erodeImg(thresh, sz = 5)
        plotImage(thresh, "Thresholded Image")

        # step 3: create mask by copying red pixels from marked image
        #           only if the destination pixel is not black
        # create binary mask for non-black pixels in thresh
        mask1 = cv2.inRange(to_rgb(thresh), (1,1,1), (255,255,255))
        # create binary mask for red pixels
        mask2 = red_pixels.astype(np.uint8) * 255
        # combine the two binary masks using a bitwise AND operation
        red_mask = cv2.bitwise_and(mask1, mask2)
        plotImage(red_mask, "Red Mask")

        # step 4: create connectedcomponents and run watershed to identify sure foreground
        _, markers = cv2.connectedComponents(red_mask)
        markers = cv2.watershed(to_rgb(thresh),markers)

        # step 5: create binary stencil from the markers
        m_img = np.zeros_like(marked_image[:,:,0], dtype=np.uint8)
        m_img[markers == -1] = 255
        plotImage(m_img, "WS Contours 1")

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
        plotImage(cimg, "Detected contours")
        # draw contours on new image
        step1_img = np.zeros_like(marked_image[:,:,0])
        # cv2.drawContours(step1_img, contours, -1, 255, -1)
        # cv2.drawContours(step1_img, contours, -1, 0, 1)

        for c in filtered_contours:
            mask = np.zeros_like(marked_image[:,:,0])
            cv2.drawContours(mask, [c], -1, 255, -1)
            cv2.drawContours(mask, [c], -1, 0, 5)
            step1_img = cv2.add(step1_img, mask)

        plotImage(step1_img, "Step 1 Image")

        # create new set on markers, which are now the sure foreground
        _, markers = cv2.connectedComponents(step1_img)

        # perform second watershed to eliminate background
        markers = cv2.watershed(np.zeros_like(marked_image),markers)

        # create skeleton image from markers
        contour_image = np.zeros_like(marked_image[:,:,0], dtype=np.uint8)
        contour_image[markers == -1] = 255

        # FINAL STEP: Analyze the resulting contour image
        splinters = Splinter.analyze_contour_image(contour_image)

        if return_thresh:
            return splinters, to_rgb(thresh)

        return splinters

    @staticmethod
    def analyze_contour_image(contour_image, px_per_mm=1):
        """Analyze a contour image and return a list of splinters."""
        contour_image = to_gray(contour_image)

        contours = detect_fragments(contour_image, min_area=5, max_area=2000, filter=False)

        splinters = [Splinter(c, i, px_per_mm) for i, c in enumerate(contours)]

        return splinters

    @staticmethod
    def analyze_image(image, debug: bool = False, px_per_mm: float = 1.0):
        """Analyze an unprocessed image and return a list of splinters.

        Parameters:
            image: The input image to be analyzed. Must be in RGB format for watershed algorithm.
            debug: A boolean flag indicating whether to display intermediate images for debugging purposes. Default is False.
            px_per_mm: The number of pixels per millimeter in the image. Default is 1.0.

        Returns:
            (list[Splinter]): A list of Splinter objects representing the splinters detected in the input image.
        """
        assert is_rgb(image), "Image must be in RGB format for watershed algorithm."

        # thresh: black is crack, white is splinter
        gray = to_gray(image)

        # gray = cv2.GaussianBlur(gray, (5, 5), 1)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 35)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        if debug:
            plotImage(thresh, "Thresholded Image")

        # sure background: white is splinter, black is crack
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        if debug:
            plotImage(sure_bg, "Sure Background")

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3,)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        ret, sure_fg = cv2.threshold(dist_transform, 0, 255, 0)
        sure_fg = erodeImg(sure_fg, it=4)

        if debug:
            plotImages([("Distance Transform", dist_transform),("Sure Foreground", sure_fg)])


        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)


        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[sure_fg!=255] = 0

        if debug:
            plotImages([
                ("Sure Background", sure_bg),
                ("Sure Foreground", sure_fg),
                ("Back - Foreground", unknown),
                ("Markers", np.abs(markers).astype(np.uint8)),
            ])

        markers = cv2.watershed(np.zeros_like(image),markers)

        m_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        m_img[markers == -1] = 255
        # m_img = dilateImg(m_img)

        return Splinter.analyze_contour_image(m_img, px_per_mm=px_per_mm)

    @staticmethod
    def __analyze_image_legacy(image):
        pass