from multiprocessing import Pool
import cv2
import numpy.typing as nptyp
from fracsuite.core.image import to_rgb
import multiprocessing.shared_memory as sm
import numpy as np

from fracsuite.core.imageprocessing import preprocess_spot_detect

SM_IMAGE = "IMAGE_SM_SPLINT_DETECT"

def check_splinter(kv):
    """Check if a splinter is valid.

    kv: tuple(int, splinter)
    """
    i,contour,sz = kv

    shm = sm.SharedMemory(name=SM_IMAGE)
    img = np.ndarray(sz, dtype=np.uint8, buffer=shm.buf)
    x, y, w, h = cv2.boundingRect(contour)
    roi_orig = img[y:y+h, x:x+w]

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    roi = mask[y:y+h, x:x+w]
    # Apply the mask to the original image
    result = cv2.bitwise_and(roi_orig, roi_orig, mask=roi)

    # Check if all pixels in the contour area are black
    #TODO! this value is crucial to the quality of histograms!
    if np.mean(result) < 5:
        return i
    else:
        return -1

def filter_contours(contours, hierarchy, min_area = 1, max_area=25000) \
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
        if contour_perim < min_area:
            to_delete.append(i)

    # def reject_outliers(data, m = 2.):
    #     return data[abs(data - np.mean(data)) < m * np.std(data)]

        # filter outliers
        elif contour_area > max_area:
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

    return contours


def detect_fragments(
    binary_image,
    min_area:float = 1.0,
    max_area:float = 25000,
    filter = True,
) -> list[nptyp.ArrayLike]:
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
        contours = list(contours)[1:]
        if filter:
            contours = filter_contours(contours, hierar, min_area=min_area, max_area=max_area)
        return contours
    except Exception as e:
        raise ValueError(f"Error in fragment detection: {e}")

def get_contour_centroid(contour):
    """Calculate the centroid of a contour.

    Args:
        contour: A list of points representing the contour.

    Returns:
        A tuple (cx, cy) representing the centroid of the contour.
    """
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)

def remove_dark_spots(
    original_image,
    skeleton_image,
    contours,
    progress = None,
    task = None,
    silent: bool = False
) -> np.ndarray:
    """
    Filter contours, that contain only dark spots in the original image.
    Fills the dark spots by connecting adjacent contours to the dark spot centroid.

    Args:
        original_image: The original image.
        skeleton_image: The skeleton image.
        contours: A list of contours to check.
        progress: A progress bar.
        task: The task id of the progress bar.
        silent: If True, no progress bar is shown.

    Returns:
        A patched image where dark spots are filled.
    """
    skeleton_image = skeleton_image.copy()
    # create normal threshold of original image to get dark spots
    img = preprocess_spot_detect(original_image)

    i_del = []
    removed_contours: list = []

    def update_task(task, advance=1, total=None, descr=None):
        if progress is None or task is None:
            return

        if silent:
            return
        if total is not None:
            progress.update(task, total=total)
        if descr is not None:
            progress.update(task, description=descr)

        progress.update(task, advance=advance)

    print(img.shape)
    shm = sm.SharedMemory(create=True, size=img.nbytes, name=SM_IMAGE)
    shm_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
    shm_img[:] = img[:]
    i_del = []
    update_task(task, advance=1, descr='Finding dark spots...', total = len(contours))
    with Pool(processes=4) as pool:
        for i,result in enumerate(pool.imap_unordered(check_splinter, [(i,s,img.shape) for i,s in enumerate(contours)])):
            if result != -1:
                i_del.append(result)
            update_task(task, advance=1)


    update_task(task, advance=1, total=len(i_del), descr="Remove splinters...")
    # remove splinters starting from the back
    for i in sorted(i_del, reverse=True):
        update_task(task, advance=1)
        removed_contours.append(contours[i])
        del contours[i]

    skel_mask = skeleton_image.copy()

    update_task(task, advance=1, total=len(removed_contours), descr="Fill dark spots...")
    for s in removed_contours:

        c = get_contour_centroid(s)

        if c is None:
            continue

        # Remove the original contour from the mask
        cv2.drawContours(skel_mask, [s], -1, 0, 1)
        cv2.drawContours(skel_mask, [s], -1, 0, -1)

        # cv2.drawContours(skel_mask, [s], -1, 0, -1)
        connections = []

        # Search for adjacent lines that were previously attached
        #   to the removed contour
        for p in s:
            p = p[0]
            # Search the perimeter of the original pixel
            for i,j in [(-1,-1), (-1,0), (-1,1),\
                        (0,-1), (0,1),\
                        (1,-1), (1,0), (1,1) ]:
                x = p[0] + j
                y = p[1] + i
                if x >= skeleton_image.shape[1] or x < 0:
                    continue
                if y >= skeleton_image.shape[0] or y < 0:
                    continue

                # Check if the pixel in the skeleton image is white
                if skel_mask[y][x] != 0:
                    # Draw a line from the point to the centroid
                    connections.append((x,y))

        # 2 connections -> connect them
        if len(connections) == 2:
            cv2.drawContours(skeleton_image, [s], -1, (0), -1)
            x,y = connections[0]
            a,b = connections[1]

            cv2.line(skeleton_image, (int(x), int(y)), (int(a), int(b)), 255)

        # more than 2 -> connect each of them to centroid
        elif len(connections) > 2:
            cv2.drawContours(skeleton_image, [s], -1, (0), -1)
            # cv2.drawContours(skeleton_image, [s], -1, (0), 1)
            for x,y in connections:
                cv2.line(skeleton_image, (int(x), int(y)), (int(c[0]), int(c[1])), 255)

        update_task(task, advance=1)

    del skel_mask

    if not silent and progress is not None:
        progress.remove_task(task)

    return skeleton_image