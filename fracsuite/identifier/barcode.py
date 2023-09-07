import os
from pyzbar.pyzbar import decode
from pylibdmtx.pylibdmtx import decode as decode_datamatrix

from matplotlib import pyplot as plt
import cv2
import numpy as np
import zxing

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


def display_image(image_array):
    plt.ion()
    
    plt.imshow(image_array)
    plt.show()
    plt.pause(0.1)
    
    text = input("Enter ID: ")
    plt.ioff()
    
    return text

def get_label(image):
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (11, 11), 20)
    # dispImage(image)
    image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)[1]
    # dispImage(image)
    contours, hierarchy = cv2.findContours(255-image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    # out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # dispImage(cv2.drawContours(out, contours, -1, (255,0,0), 1)    )
    areas = [cv2.contourArea(x) for x in contours]
    max_i = areas.index(max(areas))
    
    x,y,w,h = cv2.boundingRect(contours[max_i])    
    # x -= 5
    # y -= 5
    # w += 10
    # h += 10
    
    # create box from xywh
    box = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    M = perspective_transform(box, (700,370))
    roi = cv2.warpPerspective(orig, M, (700,370))
    
    return roi
    
def find_code(label_original, plot = False):
    well = cv2.cvtColor(label_original, cv2.COLOR_BGRA2GRAY)
    well = cv2.GaussianBlur(well, (3, 3), 0.3)
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
    
    roi_image = cv2.cvtColor(label_original, cv2.COLOR_BGRA2GRAY)        
    roi_image = cv2.threshold(roi_image, 95, 255, cv2.THRESH_BINARY)[1]

    M = perspective_transform(box, (w,h))
    roi = cv2.warpPerspective(roi_image, M, (w,h))    
    roi = cv2.GaussianBlur(roi, (3, 3), 0.8)
    roi = cv2.resize(roi, (int(w*0.7),int(h*0.7)))
    # roi = cv2.threshold(roi, 95, 255, cv2.THRESH_BINARY)[1]
    if plot:
        plt.imshow(roi)
        plt.show()
    
    return roi

def read_barcode(image):
    """Reads a barcode from an image.

    Args:
        image (numpy.ndarray): Image to read barcode from.

    Returns:
        str: Barcode value.
    """
    
    w,h = image.shape[:2]
    top_right = image[0:int(h/3), int(w/2):]
    label = get_label(top_right)
    code = find_code(label)    
    
    # code = improve_datamatrix(code)
    # dispImage(code)
    
    reader = zxing.BarCodeReader()
    
    decoded_objects = decode_datamatrix(code)
    
    if decoded_objects:
        for obj in decoded_objects:
            data = obj.data.decode('utf-8')
            return data 
        
    
    cv2.imwrite('temp.png', code)
    code_file = "temp.png"
    
    decoded_objects = reader.decode(code_file, possible_formats=['DATA_MATRIX'])
    
    if os.path.exists(code_file):
        os.remove(code_file)
    
    if decoded_objects:
        if isinstance(decoded_objects, list):
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                return data 
        else:
            data = decoded_objects.parsed
            return data
    
    return display_image(label)



def dispImage(roi):
    plt.imshow(roi)
    plt.show()