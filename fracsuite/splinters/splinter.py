import numpy as np
import cv2


class Splinter: 

    angle: float
    "Orientation of the splinter in degrees."
    angle_vector: tuple[float, float]
    "Normalized orientation vector"
    alignment_score: float = np.nan
    "Score indicating how much the splinter points into the direction of the impact point."
    
    def __init__(self, contour, index, mm_px: float, config = None):
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
        self.roundness = 4 * np.pi * self.area / self.circumfence ** 2
        # roughness
        self.roughness = self.calculate_roughness()
        
        # centroid
        try:
            M = cv2.moments(self.contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            self.centroid_mm =  (cX * mm_px, cY * mm_px)
            self.centroid_px =  (cX, cY)
        except:
            self.centroid_mm = (np.nan, np.nan)
            self.centroid_px = (np.nan, np.nan)
            
        
        self.has_centroid = not any(np.isnan(self.centroid_mm)) and not any(np.isnan(self.centroid_px))
        

        self.angle = self.__calculate_orientation()
        # try:
        #     _,_, self.angle = cv2.fitEllipse(self.contour)
        # except:
        #     self.angle = np.nan
        
        if config is not None:
            self.measure_orientation(config)

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
        
        return perimeter / hullperimeter
    
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



    def measure_orientation(self, config) -> float:
        """Calculate, how much the splinters orientation points to the impactpoint of config.

        Args:
            config (AnalyzerConfig): Configuration object.

        Returns:
            float: Orientation in degrees.
        """
        if not self.has_centroid:
            return np.nan

        # calculate the angle between the centroid and the impact point
        
        return self.__calculate_orientation_score(config.impact_position)
