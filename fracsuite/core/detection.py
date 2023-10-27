import multiprocessing.shared_memory as sm
from multiprocessing import Pool

import cv2
import numpy as np
import numpy.typing as nptyp
from rich import inspect
from rich.progress import track
from fracsuite.core.image import to_rgb
from fracsuite.core.imageplotting import plotImage

from fracsuite.core.imageprocessing import preprocess_spot_detect
from fracsuite.core.progress import get_progress

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
    assert np.mean(binary_image) < 120, "detect_fragments needs white cracks and black background!"

    try:
        # Find contours of objects in the binary image
        contours, hierar = \
            cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
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

def check_splinter_adj(
    data
):
    # i0: startindex
    # i1: endindex
    # splinters: list of contours and their centroids
    # im_shape: shape of the image
    i0, i1, splinters, im_shape = data


    connected_splinters: np.ndarray = [[]] * (i1-i0)

    for i in range(i0, i1):
        # first contour
        ctr1 = splinters[i][0]
        # first centroid
        cen1 = np.asarray(splinters[i][1])

        distances: list[int,float] = [(0,0)] * (len(splinters)-i)
        # calculate distance to other splinters
        for j in range(i,len(splinters)):
            # second centroid
            cen2 = np.asarray(splinters[j][1])

            distance = np.linalg.norm(cen1 - cen2)
            distances[j-i] = (j, distance)

        # find the closest splinters
        distances = sorted(distances, key=lambda x: x[1])
        # use 20 nearest splinters
        nearest_splinters = [(i, splinters[i]) for i,d in distances[:20]]

        # create empty image
        img = np.zeros(im_shape, dtype=np.uint8)
        # draw nearest splinters into it
        cv2.drawContours(img, [ctr[0] for i,ctr in nearest_splinters], -1, 255, 1)
        # draw current splinter a second time
        cv2.drawContours(img, [ctr1], -1, 0, 2)

        # detect contours in the image
        ctrs = detect_fragments(img, filter=False)
        # find contour that has current centroid inside
        current_contour = None
        for c in ctrs:
            if cv2.pointPolygonTest(c, splinters[i][1], False) == 1:
                current_contour = c
                break
        # if contour cant be found proceed
        if current_contour is None:
            continue

        # check all adjacent splinters if they are inside
        for j, splinter2 in nearest_splinters:
            if j == i:
                continue

            # check if centroid of second splinter is inside the contour of first splinter
            if cv2.pointPolygonTest(current_contour, splinters[j][1], False) == 1:
                connected_splinters[i-i0].append(j)




    return i0,i1,connected_splinters

def chunk_len(length, n):
    indices = np.linspace(0, length, n+1, dtype=int)
    return [(indices[i], indices[i+1]) for i in range(n)]

def get_adjacent_splinters_parallel(splinters, im_shape):
    """
    This function calculates adjacent splinters by using their contours and centroids.

    Args:
        splinters (List[Splinter]): Splinter list.
        simplify (float): Percentage to simplify the splinter contours.
    """
    print("Starting parallel splinter check")
    # create list of splinters and their centroids
    splinters = [(s.contour, s.centroid_px) for s in splinters]
    # create empty list of connected splinters
    connected_splinters: np.ndarray = [[]] * len(splinters)

    # split splinters into n chunks
    print("Chunking")
    chunks = chunk_len(len(splinters), 8)
    print("Creating tasks")
    tasks = []
    for chunk in chunks:
        tasks.append((chunk[0], chunk[1], splinters, im_shape))

    # create 4 processes
    with Pool() as pool:
        results = []
        with get_progress() as progress:
            task = progress.add_task("Check splinters", total=len(tasks))
            for result in pool.imap_unordered(check_splinter_adj, tasks):
                progress.advance(task)
                results.append(result)

    # merge results
    for i0,i1,cs in results:
        for i in range(i0,i1):
            connected_splinters[i] = cs[i-i0]

    return splinters

def get_adjacent_splinters(splinters, im_shape):
    """
    This function calculates adjacent splinters by using their contours and centroids.

    Args:
        splinters (List[Splinter]): Splinter list.
        simplify (float): Percentage to simplify the splinter contours.
    """
    for s1 in track(splinters):
        # calculate distances to all other splinters
        c1 = np.asarray(s1.centroid_px)
        distances = [(None,0)] * len(splinters)
        for i, s2 in enumerate(splinters):
            # calculate distances to all other centroids
            c2 = np.asarray(s2.centroid_px)
            distances[i] = (s2, np.linalg.norm(c1 - c2))

        # sort distances
        distances = sorted(distances, key=lambda x: x[1])

        # use 20 nearest splinters
        nearest_splinters = [x[0] for x in distances[:20]]

        # create an image like the original and draw nearest splinters into it
        img = np.zeros(im_shape, dtype=np.uint8)
        cv2.drawContours(img, [s.contour for s in nearest_splinters], -1, 255, 1)

        # draw the current splinter a second time
        cv2.drawContours(img, [s1.contour], -1, 0, 2)


        # find contours in the image
        ctrs = detect_fragments(img, filter=False)

        # print(len(ctrs))
        # im1 = to_rgb(img.copy())
        # cv2.drawContours(im1, ctrs, -1, (0,255,0), 1)
        # cv2.drawContours(im1, [s1.contour], -1, (255,255,0), 2)
        # plotImage(im1, "adjacent splinters", force=True)
        # find the contour that is the current splinter
        current_contour = None
        for c in ctrs:
            # print(c)
            if cv2.pointPolygonTest(c, s1.centroid_px, False) == 1:
                current_contour = c
                break

        # sometime, the contour is not enclosed by other contours so we have to skip it
        if current_contour is None:
            continue

        s1_adjacent_splinters = []
        # now, check all adjacent splinters if they are inside
        for s2 in nearest_splinters:
            if s2 == s1:
                continue

            # check if the centroid of s2 is inside the contour of s1
            if cv2.pointPolygonTest(current_contour, s2.centroid_px, False) == 1:
                s1_adjacent_splinters.append(s2)

        s1.touching_splinters = [1 for _ in s1_adjacent_splinters]
