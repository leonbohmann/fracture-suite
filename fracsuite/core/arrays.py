import numpy as np

def sort_two_arrays(array1, array2, reversed=False, keyoverride=None) -> tuple[list, list]:
    # Combine x and y into pairs
    pairs = list(zip(array1, array2))
    # Sort the pairs based on the values in x
    sorted_pairs = sorted(pairs, key=keyoverride or (lambda pair: pair[0]), reverse=reversed)
    # Separate the sorted pairs back into separate arrays
    return zip(*sorted_pairs)

def sort_arrays(*arrays, reversed=False,keyoverride=None) -> tuple:
    # Combine x and y into pairs
    pairs = list(zip(*arrays))
    # Sort the pairs based on the values in x
    sorted_pairs = sorted(pairs, key=keyoverride or (lambda pair: pair[0]), reverse=reversed)
    # Separate the sorted pairs back into separate arrays
    return zip(*sorted_pairs)

def resample(X, Y, Z, nx=50, ny=30):
    """
    Resample lists of arrays to a new size using spline interpolation.

    The resulting array has shape (nx+1, ny+1) and the first row and column
    are the x and y values. The rest is the z values.

    Returns:
        np.ndarray: The resampled array, now 2D with shape (nx+1, ny+1).

    Example:
    ```python
        result = resample(X,Y,Z, 30, 15)
        rx = result[0,1:]
        ry = result[1:,0]
        rz = result[1:,1:]
    ```
    """
    assert len(X) == len(Y) == len(Z), "X, Y and Z must have the same length"

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    x = np.linspace(X.min(), X.max(), nx+1, endpoint=True)
    y = np.linspace(Y.min(), Y.max(), ny+1, endpoint=True)

    # create a new array with the new shape
    results = np.full((ny+1, nx+1), -1, dtype=np.float64)

    for ix in range(len(x)-1):
        curx = x[ix]
        nextx = x[ix+1]
        centx = (curx + nextx) / 2
        results[0,ix+1] = centx        # real current x value is the center of the cell

        for iy in range(len(y)-1):
            cury = y[iy]
            nexty = y[iy+1]
            centy = (cury + nexty) / 2
            results[iy+1,0] = centy         # real current y value is the center of the cell

            zvals = []
            for i in range(len(X)):
                if curx <= X[i] < nextx and cury <= Y[i] < nexty:
                    zvals.append(Z[i])

            # interpolate using simple mean value
            results[iy+1,ix+1] = np.nanmean(zvals) if len(zvals) > 0 else np.nan

    # get the indices that would sort the first row (excluding the first element) and first column (excluding the first element)
    row_sort_indices = np.argsort(results[0, 1:])
    col_sort_indices = np.argsort(results[1:, 0])

    # add 1 to the indices to account for the excluded first element
    row_sort_indices += 1
    col_sort_indices += 1

    # insert the index of the first element (0) at the beginning of the indices arrays
    row_sort_indices = np.insert(row_sort_indices, 0, 0)
    col_sort_indices = np.insert(col_sort_indices, 0, 0)

    # use these indices to sort the rows and columns of the result array
    results = results[:, row_sort_indices]
    results = results[col_sort_indices, :]

    results[0,0] = np.nan

    return results


def fill_nan(Z: np.ndarray):
    # horizontally interpolate missing values (nan)
    nans = np.isnan(Z)
    non_nans = ~nans
    interpolated_Z = np.interp(np.flatnonzero(nans), np.flatnonzero(non_nans), Z[non_nans])
    Z[nans] = interpolated_Z