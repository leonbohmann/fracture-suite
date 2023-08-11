import numpy as np
import cv2


class Splinter:    
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
    
        