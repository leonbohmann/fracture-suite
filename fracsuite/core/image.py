"""
Image conversion functions.
"""

import cv2

def is_gray(img):
    return len(img.shape) == 2

def is_rgb(img):
    return len(img.shape) == 3

def to_gray(img):
    if is_gray(img):
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_rgb(img):
    if is_rgb(img):
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
