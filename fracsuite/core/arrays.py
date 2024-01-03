import numpy as np

def resample(X, Y, Z, nx=50, ny=30):
    """
    Resample a 2D array to a new size using spline interpolation.

    The resulting array has shape (nx+1, ny+1) and the first row and column
    are the x and y values. The rest is the z values.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    x = np.linspace(X.min(), X.max(), nx)
    y = np.linspace(Y.min(), Y.max(), ny)

    # create a new array with the new shape
    results = np.zeros((ny+1, nx+1))

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
            results[iy+1,ix+1] = np.nanmean(zvals) if len(zvals) > 0 else 0

    return results