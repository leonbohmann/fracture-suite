from pyzbar.pyzbar import decode
from pylibdmtx.pylibdmtx import decode as decode_2

from matplotlib import pyplot as plt
import cv2
import numpy as np

def perspective_transform(points):
    # Define the source points (coordinates of the corners of the original box)
    src_pts = np.array(points, dtype=np.float32)
    
    # Define the destination points (coordinates of the corners of the desired output box)
    # You can specify the dimensions of the output box as per your requirement
    # Here, we'll assume a square box of size 200x200 pixels
    dst_pts = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
    
    # Calculate the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    return perspective_matrix

def find_and_read_data_matrix_barcode(original_image):
    well = cv2.cvtColor(original_image, cv2.COLOR_BGRA2GRAY)
    plt.subplot(151); plt.title('A')
    plt.imshow(well)
    harris = cv2.cornerHarris(well, 20, 7, 0.04)
    plt.subplot(152); plt.title('B')
    plt.imshow(harris)

    x, thr = cv2.threshold(harris, 0.1 * harris.max(), 255, cv2.THRESH_BINARY)
    thr = thr.astype('uint8')
    plt.subplot(153); plt.title('C')
    plt.imshow(thr)
    
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    areas = [cv2.contourArea(cv2.convexHull(x)) for x in contours]
    max_i = areas.index(max(areas))
    d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)
    plt.subplot(154); plt.title('D')
    plt.imshow(d)

    rect = cv2.minAreaRect(contours[max_i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    e = cv2.drawContours(well,[box],0,1,1)
    plt.subplot(155); plt.title('E')
    plt.imshow(e)
    plt.show()
    
    # extract box from image
    x,y,w,h = cv2.boundingRect(contours[max_i])
    roi = original_image[y:y+h, x:x+w]

    
    M = perspective_transform(box)
    roi = cv2.warpPerspective(original_image, M, (200,200))
    roi = cv2.resize(roi, (100,100))
    plt.imshow(roi)
    plt.show()
    
    decoded_objects = decode_2(roi, )
    print(decoded_objects)
    
    
    
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
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # roi = cv2.GaussianBlur(roi, (3, 3), 0.5)
    roi = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY)[1]
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    find_and_read_data_matrix_barcode(roi)
    
    decoded_objects = decode(roi, )

    # print(decoded_objects)
    # Iterate through the detected barcodes
    for obj in decoded_objects:
        print(obj)
        # Check if the barcode type is Data Matrix
        if obj.type == 'DM':
            # Extract the data (barcode content)
            data = obj.data.decode('utf-8')
            return data  # Return the barcode content

    # If no Data Matrix barcode was found, return None
    return None

def dispImage(roi):
    plt.imshow(roi)
    plt.show()