import cv2
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as nptyp

from fracsuite.core.image import is_rgb, to_gray, to_rgb, to_rgba
from fracsuite.core.imageplotting import plotImage
from fracsuite.core.preps import PrepMode, PreprocessorConfig, defaultPrepConfig

from rich import print

from fracsuite.state import State

W_FAC = 4000

def simplify_contour(contour, epsilon=0.01):
    perimeter = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon * perimeter, True)


def modify_border(
    image,
    border_percent: int = 5,
    default_alpha:float = 1.0,
    fill_skipped_with_mean: bool = True,
) -> tuple[nptyp.NDArray, nptyp.NDArray]:
    """
    Makes the border of an image transparent.

    This method creates a mask with a transparent border around the image. The width of the border
    is specified in percent of the image width and height. The alpha value of the border is linearly
    interpolated from the default alpha value to 0 at the edge.

    Args:
        image (nd.array): The input image.
        border_percent (int, optional): The width of the border in percent. Defaults to 5.
        default_alpha (float, optional): The alpha value of the image. Defaults to 1.0.
        fill_skipped_with_mean (bool, optional): If true, the transparent pixels are set to the
            mean value of the image. Defaults to True.
    Returns:
        nd.array: The image with a transparent border.
    """
    assert border_percent >= 0 and border_percent <= 100, "Border percent must be between 0 and 100"
    assert default_alpha >= 0 and default_alpha <= 1, "Default alpha must be between 0 and 1"

    border_percent = border_percent / 100
    height, width = image.shape[:2]

    border_width = int(width *  border_percent)
    border_height = int(height * border_percent)

    mask = np.ones((height, width), dtype=np.float64) * default_alpha

    # set image to mean value with mask
    if fill_skipped_with_mean:
        mean_value = np.mean(image)

    for i in range(border_height):
        f = i / border_height
        alpha_value = f * default_alpha
        mask[i, :] = alpha_value
        mask[-(i + 1), :] = alpha_value

        if fill_skipped_with_mean:
            # modify image data to reach mean_value
            v0 = image[border_height-1,:]
            image[i,:] = mean_value + f * (v0 - mean_value)
            v0 = image[-(border_height-1),:]
            image[-(i + 1),:] = mean_value + f * (v0 - mean_value)

    for i in range(border_width):
        f = i / border_height
        alpha_value = f * default_alpha

        mask[:, i] = np.minimum(mask[:, i], alpha_value)
        mask[:, -(i + 1)] = np.minimum(mask[:, -(i + 1)], alpha_value)

        if fill_skipped_with_mean:
            # modify image data to reach mean_value
            v0 = image[:,border_width-1]
            image[:,i] = mean_value + f * (v0 - mean_value)
            v0 = image[:,-border_width-1]
            image[:,-(i + 1)] = mean_value + f * (v0 - mean_value)

        # mean_img = np.ones(image.shape, dtype=np.uint8) * mean_value
    # print(np.max(mask))
    # print(np.min(mask))
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    return mask, image

def lightcorrect(image, strength=5, size=8):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # Split the LAB image into different channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=strength, tileGridSize=(size,size))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    # Convert back to BGR
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def sigmoid(image):
    img_normalized = image / 255.0
    high_contrast = 1 / (1 + np.exp(-10 * (img_normalized - 0.5)))
    high_contrast = np.uint8(high_contrast * 255)

    return high_contrast

def brightnesscorrect(image, target):
    average_brightness = np.mean(to_gray(image))
    factor = target / average_brightness
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def preprocess_spot_detect(img) -> nptyp.ArrayLike:
    img = to_gray(img)
    img = cv2.GaussianBlur(img, (5,5), 3)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return img

def preprocess_image(
    image,
    prep: PreprocessorConfig = None,
    interest_region: tuple[int,int,int,int] = None,
) -> nptyp.ArrayLike:
    """Preprocess a raw image.

    Args:
        image (nd.array): The input image.
        config (AnalyzerConfig): The configuration to use.

    Returns:
        np.array: Preprocessed image that is (m,n) 0-255.
    """
    assert is_rgb(image), "Image must be in RGB format for preprocessing."

    w,h = image.shape[:2]

    if prep is None:
        prep = defaultPrepConfig
    rsz_fac = prep.resize_factor # x times smaller

    # image = sigmoid(image)
    if prep.correct_light:
        image = lightcorrect(image, prep.clahe_strength, prep.clahe_size)

    image = to_gray(image)
    if prep.lum is not None and prep.lum != 0:
        image = brightnesscorrect(image, prep.lum)
    # image = np.clip(image - np.mean(image), 0, 255).astype(np.uint8)

    # Apply Gaussian blur to reduce noise and enhance edge detection
    image = cv2.GaussianBlur(image, prep.gauss_size, prep.gauss_sigma)
    image = cv2.resize(image,
                        (int(image.shape[1]/rsz_fac), int(image.shape[0]/rsz_fac)))

    if interest_region is not None:
        plotImage(image, 'PREP: GaussianBlur -> Resize', region=interest_region)


    if prep.mode == PrepMode.ADAPTIVE:
        # adapt blocksize and c to current image size
        thresh_block_size = int(prep.athresh_block_size)
        thresh_block_size = thresh_block_size + (thresh_block_size + 1 ) % 2
        thresh_c = prep.athresh_c

        # Use adaptive thresholding
        image = cv2.adaptiveThreshold(image, 255, prep.athresh_adapt_mode, \
            cv2.THRESH_BINARY, thresh_block_size, thresh_c)
    elif prep.mode == PrepMode.NORMAL:
        # Use normal thresholding
        if prep.nthresh_lower == -1:
            image = cv2.threshold(image, 0, prep.nthresh_upper, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        else:
            image = cv2.threshold(image, prep.nthresh_lower, prep.nthresh_upper, \
            cv2.THRESH_BINARY)[1]

    if interest_region is not None or State.debug:
        plotImage(image, 'PREP: ... -> Adaptive Thresh', region=interest_region)

    # # remove noise
    # image = cv2.GaussianBlur(image, (5,5), 3)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    # image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]



    # plotImage(image, 'PREP: Preprocessed image', region=interest_region)

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