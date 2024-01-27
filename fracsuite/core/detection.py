import multiprocessing
import multiprocessing.shared_memory as sm
from multiprocessing import Pool
from typing import Literal

import cv2
import numpy as np
import numpy.typing as nptyp
from rich import inspect, print
from rich.progress import track
from tqdm import tqdm
from fracsuite.core.image import to_rgb
from fracsuite.core.imageplotting import plotImage

from fracsuite.core.imageprocessing import preprocess_spot_detect
from fracsuite.core.progress import get_progress
from fracsuite.state import State

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fracsuite.core.splinter import Splinter

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
        if contour_area < min_area:
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
    min_area_px: int = 1,
    max_area_px: int = 25000,
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
            contours = filter_contours(contours, hierar, min_area=min_area_px, max_area=max_area_px)
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
    all_contours: list[tuple[list,tuple[int,int]]]
    i0: int
    i1: int
    im_shape: tuple[int,int]

    i0, i1, all_contours, im_shape = data


    print(f"{i0} - {i1}: Running...")
    all_ctrs = [c[0] for c in all_contours]

    connected_contours = [None] * (i1-i0)

    for i in range(i0, i1):
        i_connections = connected_contours[i-i0] = []

        # first contour
        ctr1 = all_contours[i][0]
        # first centroid
        cen1 = np.asarray(all_contours[i][1])

        distances: list[int,float] = []
        # calculate distance to other splinters
        for j in range(len(all_contours)):
            # second centroid
            cen2 = np.asarray(all_contours[j][1])

            distance = np.linalg.norm(cen1 - cen2)
            distances.append((j, distance))

        # find the closest splinters
        distances = sorted(distances, key=lambda x: x[1])
        # use nearest splinters
        nearest_contours = [(ii, all_contours[ii][0], all_contours[ii][1]) for ii,_ in distances[:30]]

        # create empty image
        img = np.zeros(im_shape, dtype=np.uint8)
        # draw nearest splinters into it
        img = cv2.drawContours(img, [ctr[1] for ctr in nearest_contours], -1, 255, 1)
        # draw current splinter a second time
        img = cv2.drawContours(img, [ctr1], 0, 0, 2)

        # detect contours in the image
        ctrs = detect_fragments(img, filter=False)
        # find contour that has current centroid inside
        current_contour = None
        for c in ctrs:
            if cv2.pointPolygonTest(c, all_contours[i][1], False) >= 0:
                current_contour = c
                break

        # if contour cant be found proceed
        if current_contour is None:
            if State.debug:
                cimg = to_rgb(img.copy())
                cv2.drawContours(cimg, all_ctrs, -1, (255,255,255), 1)
                cv2.ellipse(cimg, cen1, (5,5), 0, 0, 360, (0,255,0), 1)
                cv2.drawContours(cimg, [ctr1], 0, (255,0,0), 1)
                cv2.drawContours(cimg, ctrs, -1, (0,100,255), 1)
                cv2.drawContours(img, [ctr[1] for ctr in nearest_contours], -1, (120,0,120), 1)
                State.output(cimg, "faulty_splinters", f"splinter_adj_err_{i}", force=True)
            continue

        # check all adjacent splinters if they are inside
        for j, _, cen3 in nearest_contours:
            if i == j:
                continue
            # check if centroid of second splinter is inside the contour of first splinter
            if cv2.pointPolygonTest(current_contour, cen3, False) >= 0:
                i_connections.append(j)


    print(f"{i0} - {i1}: Finished.")
    return (i0,i1,connected_contours)

def chunk_len(length, n):
    indices = np.linspace(0, length, n+1, dtype=int)
    return [(int(indices[i]), int(indices[i+1])) for i in range(n)]

def get_adjacent_splinters_parallel(splinters, im_shape):
    """
    This function calculates adjacent splinters by using their contours and centroids.

    Args:
        splinters (List[Splinter]): Splinter list.
        simplify (float): Percentage to simplify the splinter contours.
    """
    print("Starting parallel splinter check")
    # create list of splinters and their centroids
    contour_centroids = [(s.contour, s.centroid_px) for s in splinters]
    # create empty list of connected splinters
    connected_splinters = [None] * len(contour_centroids)

    # split splinters into n chunks
    chunks = chunk_len(len(contour_centroids), multiprocessing.cpu_count()*2)
    print(f"Chunked into [cyan]{len(chunks)}[/cyan] chunks.")
    print(chunks)

    tasks = []
    for chunk in chunks:
        tasks.append((chunk[0], chunk[1], contour_centroids, im_shape))

    # create 4 processes
    with Pool() as pool:
        with get_progress(title="Checking adjacency...") as progress:
            for result in pool.imap_unordered(check_splinter_adj, tasks):
                progress.advance()
                i0 = result[0]
                i1 = result[1]
                connected_splinters[i0:i1] = result[2]



    return connected_splinters

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


def attach_connections(splinters, connections):
    for i in range(len(splinters)):
        for j in connections[i]:
            if j not in splinters[i].adjacent_splinter_ids:
                splinters[i].adjacent_splinter_ids.append(j)
            if i not in splinters[j].adjacent_splinter_ids:
                splinters[j].adjacent_splinter_ids.append(i)



def get_splinter_surface_area(args):
    """
    This function calculates the surface area of a splinter.

    Args:
        splinter (Splinter): Splinter object.
        image (img): Original image that has been preprocessed.
        t (float): Thickness of the splinter.

    Returns:
        None. This function modifies the splinter object.
    """
    splinter, image, t = args

    # contour is always closed!
    crack_area = 0

    for ip in range(len(splinter.contour)-1):
        if ip == len(splinter.contour)-1:
            # current point
            p0 = splinter.contour[ip+1][0]
            # next point is the first point of contour
            p1 = splinter.contour[0][0]
        else:
            # current point
            p0 = splinter.contour[ip][0]
            # next point
            p1 = splinter.contour[ip+1][0]

        # calculate vector between points
        dc = p1 - p0
        dc_norm = np.linalg.norm(dc)
        dc = dc / dc_norm
        # get perpendicular vector
        dp = np.array([dc[1], -dc[0]])
        dp_norm = np.linalg.norm(dp)
        dp = dp / dp_norm

        # t_star in each direction
        lstar1 = 0
        lstar2 = 0
        # check the perpendicular line for maximum 50 pixels in each direction!
        for i in range(1, 50):
            p1 = p0 + i * dp

            # clip to image size
            if p1[0] >= image.shape[1] or p1[0] < 0:
                break
            if p1[1] >= image.shape[0] or p1[1] < 0:
                break

            # check if p1 or p2 has black pixel in image
            if image[int(p1[1]), int(p1[0])] == 0:
                lstar1 = np.linalg.norm(p1 - p0)
            else:
                break

        for i in range(1, 50):
            p2 = p0 - i * dp

            # clip to image size
            if p2[0] >= image.shape[1] or p2[0] < 0:
                break
            if p2[1] >= image.shape[0] or p2[1] < 0:
                break

            if image[int(p2[1]), int(p2[0])] == 0:
                lstar2 = np.linalg.norm(p2 - p0)
            else:
                break
        #
        lstar = lstar1 + lstar2

        # maximum possible length
        Lm = t + lstar
        # minimum posible length
        Ld = np.sqrt(t**2 + lstar**2)

        crack_area += (Lm+Ld)/2 * dc_norm * 0.5 # 0.5 because the crack is divided to the adjacent splinter!

    return crack_area

# methods for crack surface calculation
def get_crack_surface(splinters: list, image, t):
    """
    Functions checks for every splinter contour the thickness of lines in the original image.

    Args:
        splinters (List[Splinter]): Splinter list.
        image (img): Original image.

    Returns:
        None. This function modifies the splinter objects.
    """
    total_crack_area = 0

    # create args
    args = [(s, image, t) for s in splinters]

    with Pool() as pool:
        for crack_area in tqdm(pool.imap_unordered(get_splinter_surface_area, args), total=len(splinters)):
            total_crack_area += crack_area


    return total_crack_area


def get_crack_surface_r(splinters, image, t, pxpmm):
    """
    Functions checks for every splinter contour the thickness of lines in the original image and calculates the total
    fracture surface of all splinters.

    Args:
        splinters (list[Splinter]): Splinter list.
        image (np.ndarray): Base image, grayscale, 0-255.
        t (float): Thickness of the glass.
        pxpmm (float): Pixel per millimeter.

    Returns:
        float: The total fracture surface.
    """
    from splintaz import calculate_fracture_surface

    fracture_surface = calculate_fracture_surface([s.contour[:,0] for s in splinters], image, t, pxpmm)

    return fracture_surface