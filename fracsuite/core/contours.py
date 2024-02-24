import cv2
import numpy as np

def center(contour):
    # return the cv2 centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    else:
        return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])