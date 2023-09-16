from __future__ import annotations
from typing import Any, Callable, TypeVar
import numpy as np
from rich.progress import track
from torch import pdist

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
def csintkern_splinters(region, splinters: list[T],  h=200, z_action: Callable[[T], Any] = None):
    """Calculate an intensity based on splinters in a region with size h and use z_action
    to perform calculations on the splinters in that region.

    Args:
        region (tuple): _description_
        splinters (list[Splinter]): _description_
        h (int, optional): Scan region size in px. Defaults to 200px.
        z_action (def(list[Specimen]), optional): Gets called for every region. Defaults to None, which
            will return the length of the specimen..

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
    xd = np.linspace(minx, maxx, int(np.ceil(px_w/h)))
    yd = np.linspace(miny, maxy, int(np.ceil(px_h/h)))
    X, Y = np.meshgrid(xd, yd)
    
    # perform the action on every area element
    if z_action is None:
        def z_action(x: list[T]):
            return len(x)
        
    result = np.empty(X.shape)
    # print(result.shape)
    # print(len(xd))
    # print(len(yd))
    
    # Iterate over all points and find splinters in the area of X, Y and intensity_h
    for i in track(range(1,len(xd) - 1), transient=True):
        for j in range(1,len(yd) - 1):
            x1,y1=xd[i]-h//2,yd[j]-h//2
            x2,y2=xd[i]+h//2,yd[j]+h//2

            # Create a region (x1, y1, x2, y2)
            region_rect = (x1, y1, x2, y2)


            # Collect splinters in the current region
            splinters_in_region = [splinter for splinter in splinters if splinter.in_region_px(region_rect)]

            # Apply z_action to collected splinters
            result[j, i] = z_action(splinters_in_region) if len(splinters_in_region) > 0 else 0
            
    Z = result.reshape(X.shape)

    return X,Y,Z
    # Z = np.zeros(X.shape)
    # for i in track(range(X.shape[0]), leave=False):
    #     for j in range(X.shape[1]):
    #         # find splinters in the area
    #         splinters_in_area = []
    #         for splinter in splinters:
    #             if splinter.in_region((X[i,j], Y[i,j])):
    #                 splinters_in_area.append(splinter)
    #         x = X[i,j]
    #         y = Y[i,j]
    #         Z[i,j] = z_action(x,y)
    
    pass

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