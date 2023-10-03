import numpy as np
import cv2

from fracsuite.core.image import to_gray
from fracsuite.core.imageplotting import plotImage, plotImages
from fracsuite.core.detection import detect_fragments
from fracsuite.splinters.processing import dilateImg, erodeImg


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
        dx = origin[0] - centroid[0]
        dy = origin[1] - centroid[1]

        angle_radians = np.deg2rad(self.angle)
        line_direction = np.array([dx, dy])
        angle_vector = np.array([np.cos(angle_radians), np.sin(angle_radians)])
        dot_product = np.dot(line_direction, angle_vector)
        magnitude_line = np.linalg.norm(line_direction)
        self.alignment_score = np.abs(dot_product) / magnitude_line
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
    def from_image(image, debug: bool = False, px_per_mm: float = 1.0):
        # thresh: black is crack, white is splinter
        gray = to_gray(image)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 14)
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
        m_img = dilateImg(m_img)
        ## find contours on watershed markered image
        contours = detect_fragments(m_img, min_area=5, max_area=2000, filter=False)

        splinters = [Splinter(c, i, px_per_mm) for i, c in enumerate(contours)]

        return splinters