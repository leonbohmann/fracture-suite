from __future__ import annotations
from typing import Any, Callable, TypeVar
import numpy as np
from rich.progress import track
from torch import pdist

from fracsuite.core.image import split_image, SplitImage

def csintkern(events, region, h):
    n, d = events.shape

    # Get the ranges for x and y
    minx = np.min(region[:][0])
    maxx = np.max(region[:][0])
    miny = np.min(region[:][1])
    maxy = np.max(region[:][1])

    # Get 50 linearly spaced points
    xd = np.linspace(minx, maxx, 50)
    yd = np.linspace(miny, maxy, 50)
    X, Y = np.meshgrid(xd, yd)
    st = np.column_stack((X.ravel(), Y.ravel()))
    ns = len(st)
    xt = np.vstack(([0, 0], events))
    z = np.zeros(X.ravel().shape)

    for i in track(range(ns), leave=False):
        # for each point location, s, find the distances
        # that are less than h.
        xt[0] = st[i]
        # find the distances. First n points in dist
        # are the distances between the point s and the
        # n event locations.
        dist = pdist(xt)
        ind = np.where(dist[:n] <= h)[0]
        t = (1 - dist[ind]**2 / h**2)**2
        z[i] = np.sum(t)

    z = z * 3 / (np.pi * h)
    Z = z.reshape(X.shape)

    return X, Y, Z

T=TypeVar('T')
def csintkern_objects(region,
                    objects: list[T],
                    object_in_region: Callable[[T, tuple[int,int,int,int]], bool],
                    h=200,
                    z_value: Callable[[T], Any] = None,
                    no_track=False,
                    skip_edge=False
                ):
    """Calculate an intensity based on splinters in a region with size h and use z_action
    to perform calculations on the splinters in that region.

    Args:
        region (tuple): The region to scan.
        objects (list[Splinter]): The objects to check for every region.
        object_in_region (Callable[[Splinter, tuple[x1,y1,x2,y2]], bool]): Determines if the object is in a region.
        h (int, optional): Scan region size in px. Defaults to 200px.
        z_value (def(list[Specimen]), optional): Gets called for every region. Defaults to None, which
            will return the length of the specimen..
        no_track(bool): If true, will not use the progress bar.

    Returns:
        X,Y,Z: The meshgrid and the intensity values.
    """
    px_w, px_h = region

    # Get the ranges for x and y
    minx = 0
    maxx = np.max(region[0])
    miny = 0
    maxy = np.max(region[1])

    # Get 50 linearly spaced points
    xd = np.linspace(minx, maxx, int(np.round(px_w/h)))
    yd = np.linspace(miny, maxy, int(np.round(px_h/h)))
    X, Y = np.meshgrid(xd, yd)

    # perform the action on every area element
    if z_value is None:
        def z_value(x: list[T]):
            return len(x)

    result = np.empty(X.shape)
    # print(result.shape)
    # print(len(xd))
    # print(len(yd))
    skip_i = int(len(xd)*0.1) if skip_edge else 0
    if no_track:
        d = range(skip_i,len(xd) - skip_i)
    else:
        d = track(range(skip_i,len(xd) - skip_i), transient=True,
                   description="Calculating intensity...")
    # Iterate over all points and find splinters in the area of X, Y and intensity_h
    for i in d:
        for j in range(skip_i,len(yd) - skip_i):
            x1,y1=xd[i]-h//2,yd[j]-h//2
            x2,y2=xd[i]+h//2,yd[j]+h//2

            # Create a region (x1, y1, x2, y2)
            region_rect = (x1, y1, x2, y2)


            # Collect splinters in the current region
            splinters_in_region = [obj for obj in objects if object_in_region(obj, region_rect)]

            # Apply z_action to collected splinters
            result[j, i] = z_value(splinters_in_region) if len(splinters_in_region) > 0 else 0

    Z = result.reshape(X.shape)

    return X,Y,Z
def csintkern_objects_diagonal(region: tuple[int,int],
                    objects: list[T],
                    object_in_region: Callable[[T, tuple[int,int,int,int]], bool],
                    kernel_width=200,
                    n_points:int = 50,
                    z_value: Callable[[T], Any] = None,
                ):
    """Calculate the intensity on a diagonal kernel inside a region.

    Args:
        region (tuple): The region to scan.
        objects (list[Splinter]): The objects to check for every region.
        object_in_region (Callable[[Splinter, tuple[x1,y1,x2,y2]], bool]): Determines if the object is in a region.
        kernel_width (int, optional): Scan region size in px. Defaults to 200px.
        n_points (int, optional): Amount of points on the diagonal to be evaluated.
        z_value (def(list[Specimen]), optional): Gets called for every region. Defaults to None, which
            will return the length of the specimen..

    Returns:
        results (list[float]): The intensity values on the diagonal.
    """
    if z_value is None:
        def z_value(x: list[T]):
            return len(x)

    px_w, px_h = region

    assert kernel_width < px_w, "Kernel width must be smaller than the region width."
    assert kernel_width < px_h, "Kernel width must be smaller than the region height."
    assert kernel_width > 10, "Kernel width must be greater than 10."
    assert len(objects) > 0, "There must be at least one object in the list."
    assert isinstance(object_in_region(objects[0], (-100,-100, -100, -100)), bool), "object_in_region must return a bool."
    assert n_points > 0, "n_points must be greater than 0."
    # Get the ranges for x and y
    i_w = n_points
    i_h = n_points

    results = []
    for w in range(i_w):
        for h in range(i_h):
            # get diagonal only
            if w != h:
                continue

            x1 = (w/n_points) * px_w - kernel_width // 2
            y1 = (h/n_points) * px_h - kernel_width // 2
            x2 = x1 + kernel_width
            y2 = y1 + kernel_width

            objects_in_region = [obj for obj in objects if object_in_region(obj, (x1, y1, x2, y2))]

            result = z_value(objects_in_region)

            results.append(result)

    return results


def csintkern_image(image,
                    grid,
                    z_value: Callable[[Any], float],
                    skip_edge=False):
    """
    Performs an intensity calculation on an image by splitting it into a grid
    and calling z_value on each grid element.

    Args:
        image (np.array): The image to perform the calculation on.
        grid (int): The grid size.
        z_value (Callable[[Any], float]): The function to call on each grid element.

    Returns:
        X,Y,Z: The meshgrid and the intensity values.
    """
    px_w, px_h = (image.shape[1], image.shape[0])

    # Get the ranges for x and y
    minx = 0
    maxx = px_w
    miny = 0
    maxy = px_h

    split = split_image(image, grid)

    # Get 50 linearly spaced points
    xd = np.linspace(minx, maxx, split.rows)
    yd = np.linspace(miny, maxy, split.cols)
    X, Y = np.meshgrid(xd, yd)

    # perform the action on every area element
    if z_value is None:
        def z_value(x: Any):
            return 1

    result = np.empty(X.shape)
    # print(result.shape)
    # print(len(xd))
    # print(len(yd))
    skip_i = int(len(xd)*0.1) if skip_edge else 0
    # Iterate over all points and find splinters in the area of X, Y and intensity_h
    for i in track(range(skip_i,len(xd)-skip_i), transient=True,
                   description="Calculating intensity..."):
        for j in range(skip_i,len(yd)-skip_i):
            part = split.get_part(i,j)

            value = z_value(part)

            # Apply z_action to collected splinters
            result[j, i] = value

    Z = result.reshape(X.shape)

    return X,Y,Z


def csintkern_optimized_distances(events, region, r=500):
    n, d = events.shape

    # Get the ranges for x and y
    minx = np.min(region[:][0])
    maxx = np.max(region[:][0])
    miny = np.min(region[:][1])
    maxy = np.max(region[:][1])

    w = maxx - minx
    h = maxy - miny

    # Get 50 linearly spaced points
    xd = np.linspace(minx, maxx, w//r)
    yd = np.linspace(miny, maxy,    h//r)
    X, Y = np.meshgrid(xd, yd)
    st = np.column_stack((X.ravel(), Y.ravel()))
    ns = len(st)

    # Precompute distances
    dist_matrix = np.linalg.norm(events[:, np.newaxis, :] - st[np.newaxis, :, :], axis=-1)
    t = np.maximum(0, 1 - (dist_matrix / r) ** 2) ** 2

    # Calculate intensities
    z = np.sum(t, axis=0)
    # z *= 3 / (np.pi * h)

    Z = z.reshape(X.shape)

    return X, Y, Z