from pyzbar.pyzbar import decode
from pylibdmtx.pylibdmtx import decode as decode_2

from matplotlib import pyplot as plt
import cv2
import numpy as np

from fracsuite.splinters.analyzer import isgray
def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    
    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([
            pts[np.argmin(summ)],
            pts[np.argmin(diff)],
            pts[np.argmax(summ)],
            pts[np.argmax(diff)],
        ])
                    
def perspective_transform(points,  image_size=(200,200)):
    points = fourCornersSort(points)  # Sort by y-coordinate and then by x-coordinate
    # Define the source points (coordinates of the corners of the original box)
    src_pts = np.array(points, dtype=np.float32)
    
    image_width, image_height = image_size
    
    # Define the destination points (coordinates of the corners of the desired output box)
    # You can specify the dimensions of the output box as per your requirement
    # Here, we'll assume a square box of size 200x200 pixels
    dst_pts = np.array([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]], dtype=np.float32)
    
    # Calculate the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return perspective_matrix

def enlarge_roi_with_padding(image, x, y, w, h, padding):
    """
    Enlarges the region of interest (ROI) defined by (x, y, w, h) by adding padding while ensuring it remains inside the image bounds.

    Args:
        image (numpy.ndarray): The input image.
        x (int): X-coordinate of the top-left corner of the ROI.
        y (int): Y-coordinate of the top-left corner of the ROI.
        w (int): Width of the ROI.
        h (int): Height of the ROI.
        padding (int): Padding value to be added to all sides of the ROI.

    Returns:
        numpy.ndarray: The enlarged ROI.
    """
    # Ensure padding is non-negative
    padding = max(padding, 0)

    # Enlarge the ROI by adding padding while ensuring it stays inside the image bounds
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)

    # Crop the enlarged ROI from the image
    enlarged_roi = image[y:y+h, x:x+w]

    return enlarged_roi

def fix_datamatrix_image(image):
    # Convert the image to grayscale
    gray_image = image
    if not isgray(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Optionally, apply erosion and dilation to further enhance the barcode
    kernel = np.ones((3, 3), np.uint8)
    thresholded_image = cv2.erode(gray_image, kernel, iterations=1)
    thresholded_image = cv2.dilate(thresholded_image, kernel, iterations=1)

    return thresholded_image
def read_barcode_from_roi(original_image, plot = False):
    well = cv2.cvtColor(original_image, cv2.COLOR_BGRA2GRAY)
    well = cv2.GaussianBlur(well, (3, 3), 0.1)
    well = cv2.threshold(well, 60, 255, cv2.THRESH_BINARY)[1]

    if plot:
        plt.subplot(151); plt.title('A')
        plt.imshow(well)
    harris = cv2.cornerHarris(well, 20, 7, 0.04)
    if plot:
        plt.subplot(152); plt.title('B')
        plt.imshow(harris)

    x, thr = cv2.threshold(harris, 0.1 * harris.max(), 255, cv2.THRESH_BINARY)
    thr = thr.astype('uint8')
    if plot:
        plt.subplot(153); plt.title('C')
        plt.imshow(thr)
    
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    areas = [cv2.contourArea(cv2.convexHull(x)) for x in contours]
    max_i = areas.index(max(areas))
    d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)
    if plot:
        plt.subplot(154); plt.title('D')
        plt.imshow(d)

    rect = cv2.minAreaRect(contours[max_i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    e = cv2.drawContours(well,[box],0,1,1)    
    if plot:
        plt.subplot(155); plt.title('E')
        plt.imshow(e)
        plt.show()
    
    # extract box from image
    x,y,w,h = cv2.boundingRect(contours[max_i])    
    x -= 5
    y -= 5
    w += 10
    h += 10
    
    # create box from xywh
    box = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    
    roi_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2GRAY)
    roi_image = cv2.threshold(roi_image, 95, 255, cv2.THRESH_BINARY)[1]

    for i in range(2):
        M = perspective_transform(box, (200,200))
        roi = cv2.warpPerspective(roi_image, M, (200,200))    
        roi = cv2.resize(roi, (100,100))
        if plot:
            plt.imshow(roi)
            plt.show()
        
        decoded_objects = decode_2(roi)
        
        if decoded_objects:
                    for obj in decoded_objects:
                        data = obj.data.decode('utf-8')
                        return data  # Return the barcode content if found
        else:
            dispImage(roi)
            
        original_image = fix_datamatrix_image(original_image)
        
        if not isgray(original_image):
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2GRAY)
        
        roi_image = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61, 30)

    
    # If no Data Matrix barcode was found, return None
    return None

def read_barcode(image):
    """Reads a barcode from an image.

    Args:
        image (numpy.ndarray): Image to read barcode from.

    Returns:
        str: Barcode value.
    """
    
    
    # use top right corner as roi
    roi = image[10:500, 3000:4000]
    return read_barcode_from_roi(roi)    

def dispImage(roi):
    plt.imshow(roi)
    plt.show()