from pylibdmtx.pylibdmtx import decode as decode_datamatrix

from matplotlib import pyplot as plt
import cv2
import numpy as np


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

def get_label(image, debug = None):
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (15, 15), 20)
    dispImage(image, "LABEL: gaussian blur", "label",debug)
    
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 1.5)
    dispImage(image, "LABEL: adaptive threshold", "label",debug)
    image = cv2.erode(image, np.ones((3,3), np.uint8), iterations=3)
    dispImage(image, "LABEL: eroded", "label",debug)
    # image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    # dispImage(image, "LABEL: thresholded", "label",debug)    
    image = cv2.dilate(image, np.ones((3,3), np.uint8), iterations=3)
    dispImage(image, "LABEL: dilated", "label",debug)
    contours, hierarchy = cv2.findContours(255-image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    areas = [cv2.contourArea(x) for x in contours]
    max_i = areas.index(max(areas))
    
    
    x,y,w,h = cv2.boundingRect(contours[max_i])    
    x += 15
    y += 15
    w -= 15
    h -= 15
    
    # create box from xywh
    box = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)    
    dispImage(cv2.drawContours(out, [box], -1, (255,0,0), 10), "LABEL: Label-Contour", "label", debug)
    
    M = perspective_transform(box, (700,370))
    roi = cv2.warpPerspective(orig, M, (700,370))
    
    return roi
    
def find_code(label_original, debug = None):
    well = cv2.cvtColor(label_original, cv2.COLOR_BGRA2GRAY)
    
    # use only the top half of the image
    h,w = well.shape[:2]
    well = well[0:int(h/2), :]
    
    well = cv2.GaussianBlur(well, (5, 5), 1.3)
    dispImage(well, "CODE: Gaussian Blur", "code", debug)
    
    well = cv2.erode(well, np.ones((3,3), np.uint8), iterations=2)
    thr = cv2.threshold(well, 100, 255, cv2.THRESH_BINARY)[1]

    dispImage(thr, "CODE: Thresholded","code", debug)

    contours, hierarchy = cv2.findContours(255-thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    areas = [cv2.contourArea(cv2.convexHull(x)) for x in contours]
    max_i = areas.index(max(areas))        

    rect = cv2.minAreaRect(contours[max_i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)      
  
    # extract box from image
    x,y,w,h = cv2.boundingRect(contours[max_i])    
    x -= 15
    y -= 15
    w += 30
    h += 30
    
    dispImage(cv2.drawContours(label_original.copy(), [box], -1, (255,0,0), 10), "CODE: Code-Contour", "code", debug)
    
    # create box from xywh
    box = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    
    roi_image = label_original.copy()

    M = perspective_transform(box, (w,h))
    roi = cv2.warpPerspective(roi_image, M, (w,h))    
    roi = cv2.resize(roi, (int(w*0.5),int(h*0.5)))
    roi = cv2.GaussianBlur(roi, (3, 3), 1.8)
   
    return roi

def read_barcode(image, debug = None):
    """Reads a barcode from an image.

    Args:
        image (numpy.ndarray): Image to read barcode from.

    Returns:
        str: Barcode value.
    """
    
    h,w = image.shape[:2]
    top_right = image[0:int(h/3), int(w/2):]
    dispImage(top_right, "BC: top-right", "barcode", debug)
    label = get_label(top_right, debug)
    dispImage(label, "BC: Label", "barcode", debug)
    code = find_code(label, debug)    
    dispImage(code, "BC: Code", "barcode", debug)
    
    # code = improve_datamatrix(code)
    # dispImage(code)
    
    decoded_objects = decode_datamatrix(code)
    if debug:
        print(decoded_objects)
        
    if decoded_objects:
        for obj in decoded_objects:
            data = obj.data.decode('utf-8')
            return data 
    elif debug:
        dispImage(code, "NOTFOUND: Code", "error", debug)
        
    return display_image(label)



def dispImage(roi, title = "", section = "", debug = None):
    if debug and section not in debug:
        return
    elif not debug:
        return
    
    plt.imshow(roi)
    plt.title(title)
    plt.show()