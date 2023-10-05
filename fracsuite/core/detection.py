import cv2
import numpy.typing as nptyp



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