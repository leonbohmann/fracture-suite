import cv2
import numpy as np
import numpy.typing as nptyp

from fracsuite.core.image import to_gray, to_rgb
from fracsuite.core.plotting import plotImage
from fracsuite.splinters.analyzerConfig import AnalyzerConfig

def preprocess_spot_detect(img):
    img = to_gray(img)
    img = cv2.GaussianBlur(img, (5,5), 3)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return img

def preprocess_image(image, config: AnalyzerConfig) -> nptyp.ArrayLike:
        """Preprocess a raw image.

        Args:
            image (nd.array): The input image.
            config (AnalyzerConfig): The configuration to use.

        Returns:
            np.array: Preprocessed image.
        """

        rsz_fac = config.prep.resize_factor # x times smaller

        image = to_gray(image)
        # image = np.clip(image - np.mean(image), 0, 255).astype(np.uint8)

        # Apply Gaussian blur to reduce noise and enhance edge detection
        image = cv2.GaussianBlur(image, config.prep.gauss_size, config.prep.gauss_sigma)
        image = cv2.resize(image,
                           (int(image.shape[1]/rsz_fac), int(image.shape[0]/rsz_fac)))

        if config.debug:
            plotImage(image, 'PREP: GaussianBlur -> Resize', region=config.interest_region)

        # Use adaptive thresholding
        image = 255-cv2.adaptiveThreshold(image, 255, config.prep.thresh_adapt_mode, \
            cv2.THRESH_BINARY_INV, config.prep.thresh_block_size, config.prep.thresh_c)

        if config.debug:
            plotImage(image, 'PREP: ... -> Adaptive Thresh', region=config.interest_region)

        return image


def crop_perspective(img,
                     cropped_image_size: tuple[int,int],
                     debug: bool,
                     return_matrix: bool = False):
    """
    Crops a given image to its containing pane bounds. Finds smallest pane countour with
    4 corner points and aligns, rotates and scales the pane to fit a resulting image.

    Args:
        img (Image): Input image with a clearly visible glass pane.
        size (int): Size of the resulting image after perspective transformation.
        dbg (bool): Displays debug image outputs during processing.

    Returns:
        img: A cropped image which only contains the glass pane. Size: 1000x1000.
            If no 4 point contour is found, the whole image is returned.
    """

    def fourCornersSort(pts):
        """ Sort corners: top-left, bot-left, bot-right, top-right """
        # Difference and sum of x and y value
        # Inspired by http://www.pyimagesearch.com
        diff = np.diff(pts, axis=1)
        summ = pts.sum(axis=1)

        # Top-left point has smallest sum...
        # np.argmin() returns INDEX of min
        return np.array([pts[np.argmin(summ)],
                        pts[np.argmax(diff)],
                        pts[np.argmax(summ)],
                        pts[np.argmin(diff)]])



    img_original = img.copy()

    im = img.copy()
    im0 = img.copy()

    im = to_gray(im)
    im0 = to_rgb(im0)

    # # apply gaussian blur to image to we can get rid of some noise
    im = cv2.GaussianBlur(im, (5,5), 5)
    # # restore original image by thresholding
    _,im = cv2.threshold(im,127,255,0)

    if debug:
        plotImage(im, 'CROP: Image for rectangle detection')

    # fetch contour information
    contour_info = []
    contours, _ = cv2.findContours(255-im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop through contours and find their properties
    for cnt in contours:
        contour_info.append((
            cnt,
            cv2.isContourConvex(cnt),
            cv2.contourArea(cnt)
        ))

    # os.makedirs('.debug', exist_ok=True)
    # # count files in debug folder
    # file_count = len(os.listdir('.debug'))

    # cv2.imwrite(os.path.join('.debug', f'contours_{file_count}.png'),
    #             cv2.drawContours(im0, contours, -1, (0,0,255), 10))

    # sort contours after their area
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    # take the second largest contour (this has to be the outer bounds of pane)
    if len(contour_info) > 0:
        max_contour = contour_info[0][0]
    else:
        return img_original

    cv2.drawContours(im0, contours, -1, (0,0,255), 10)
    if debug:
        plotImage(im0, 'CROP: Detected contours')

    # Simplify contour
    perimeter = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.03 * perimeter, True)

    # Page has 4 corners and it is convex
    # Page area must be bigger than maxAreaFound
    if (len(approx) == 4 and
            cv2.isContourConvex(approx)):
        pageContour = fourCornersSort(approx[:, 0])

    else:
        rect = cv2.boundingRect(approx)
        x, y, w, h = rect

        # Compute the four corners of the rectangle using cv2.boxPoints
        corners = cv2.boxPoints(((x, y), (w, h), 0))

        # Convert the corners to integer values and print the result
        corners = corners.astype(int)
        #raise CropException("Pane boundary could not be found.")
        pageContour = corners

    if debug:
        cv2.drawContours(im0, [pageContour],-1, (0,0,255), thickness=im.shape[0]//50)
        plotImage(im0, 'CROP: Found rectangle contour')



    im_h, im_w =im.shape[0],im.shape[1]

    if im_w > im_h:
        f = im_w / cropped_image_size[0]
        im_w = cropped_image_size[0]
        im_h = int(im_w / f)
    else:
        f = im_h / cropped_image_size[1]
        im_h = cropped_image_size[1]
        im_w = int(im_h / f)

    # Create target points
    if cropped_image_size is not None:
        width, height=cropped_image_size
    else:
        height,width=im_w, im_h

    tPoints = np.array([[0, 0],
                    [0, height],
                    [width, height],
                    [width, 0]], np.float32)
    # source points are contour corners
    sPoints = pageContour

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)
    # Warping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints)

    img_original = crop_matrix(img_original, M, (int(width), int(height)))
    # print(img_original.shape)
    # plotImage(img_original, 'CROP: Cropped image')
    # and return the transformed image

    if return_matrix:
        return img_original, M

    return img_original

def crop_matrix(img, M, size):
    """Crop an image using a transformation matrix.

    Args:
        img (Image): The image to crop.
        M (np.array): The transformation matrix.
        size (tuple[int,int]): The size of the resulting image.

    Returns:
        Image: The cropped image.
    """
    return cv2.warpPerspective(img, M, size)


def filter_contours(contours, hierarchy, config: AnalyzerConfig) \
    -> list[nptyp.ArrayLike]:
    """
    This function filters a list of contours.

    kwargs:
        debug: bool
            If True, debug output is printed.
        min_area_px: int
            Minimum area of a contour in pixels.
        max_area_px: int
            Maximum area of a contour in pixels.
    """
    if config.debug:
        len_0 = len(contours)
    # Sort the contours by area (desc)
    contours, hierarchy = zip(*sorted(zip(contours, hierarchy[0]), \
        key=lambda x: cv2.contourArea(x[0]), reverse=True))

    contours = list(contours)
    contours_areas = [cv2.contourArea(x) for x in contours]
    # contour_area_avg = np.average(contours_areas)
    # contour_area_med = np.average(contours_areas)
    # contour_area_std = np.std(contours_areas)

    # Create a list to store the indices of the contours to be deleted
    to_delete: list[int] = []
    As = []
    ks = []
    # Iterate over all contours
    for i in range(len(contours)):
        contour_area = contours_areas[i]
        contour_perim = cv2.arcLength(contours[i], False)


        if contour_area > 0:
            As.append(contour_area)
            ks.append(contour_perim / contour_area)

        if contour_area == 0:
            contour_area = 0.00001

        # small size
        if contour_perim < config.fragment_min_area_px:
            to_delete.append(i)

    # def reject_outliers(data, m = 2.):
    #     return data[abs(data - np.mean(data)) < m * np.std(data)]

        # filter outliers
        elif contour_area > config.fragment_max_area_px:
            to_delete.append(i)

        # # more line shaped contour
        # elif contour_perim / contour_area > 1:
        #     contours[i] = cv2.convexHull(contours[i])

        # # Check if the contour has a parent
        # elif hierarchy[i][1] != -1:
        #     # If so, mark the contour for deletion
        #     to_delete.append(i)

        # # all other cases
        # else:
        #     contours[i] = connect_closest_hard_angles(contours[i], 140)
        #     contours[i] = cv2.approxPolyDP(contours[i], 0.1, True)

    # Delete the marked contours
    for index in sorted(to_delete, reverse=True):
        del contours[index]

    if config.debug:
        len_1 = len(contours)
        print(f'FILT: Contours before: {len_0}, vs after: {len_1}')

    return contours


def detect_fragments(binary_image, config: AnalyzerConfig) -> list[nptyp.ArrayLike]:
    """Detects fragments in a binary image.

    Args:
        binary_image (img): The source image to detect.

    kwargs:
        debug: bool
            If True, debug output is printed.
        min_area_px: int
            Minimum area of a contour in pixels.
        max_area_px: int
            Maximum area of a contour in pixels.

    Raises:
        ValueError: Something went wrong.

    Returns:
        list[nptyp.ArrayLike]: The found contours.
    """
    try:
        # Find contours of objects in the binary image
        contours, hierar = \
            cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours = filter_contours(contours, hierar, config)
        return contours
    except Exception as e:
        raise ValueError(f"Error in fragment detection: {e}")

def closeImg(image, sz = 3, it=1):
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_closing, iterations=it)

def openImg(image, sz = 3, it=1):
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_opening, iterations=it)

def erodeImg(image, sz=3, it=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
    return cv2.erode(image, kernel, iterations=it)

def dilateImg(image, sz=3, it=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
    return cv2.dilate(image, kernel, iterations=it)