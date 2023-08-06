import numpy as np
import cv2


class Splinter:    
    def __init__(self, contour, index):
        self.ID = index
        self.contour = contour
        
        self.area = cv2.contourArea(self.contour)
        self.circumfence = cv2.arcLength(self.contour, True)
        
        # roundness
        self.roundness = 4 * np.pi * self.area / self.circumfence ** 2
        # roughness
        self.roughness = self.calculate_roughness()
        
        # centroid
        try:
            M = cv2.moments(self.contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            self.centroid =  (cX, cY)
        except:
            self.centroid = (np.nan, np.nan)
            
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
    
        