import numpy as np
from fracsuite.core.mechanics import Ud as calc_Ud, U as calc_U

nfifty_sigm_t = np.array([
    [5, 1.26, 2.10, 2.80, 4.20, 5.60, 7.00, 8.39, 10.49, 13.29],
    [10, 1.37, 2.29, 3.06, 4.58, 6.11, 7.64, 9.17, 11.46, 14.51],
    [15, 1.49, 2.49, 3.32, 4.98, 6.64, 8.30, 9.95, 12.44, 15.76],
    [20, 1.61, 2.69, 3.59, 5.38, 7.17, 8.97, 10.76, 13.45, 17.04],
    [25, 1.74, 2.90, 3.86, 5.79, 7.72, 9.65, 11.58, 14.48, 18.34],
    [30, 1.86, 3.11, 4.14, 6.21, 8.28, 10.35, 12.42, 15.53, 19.67],
    [35, 1.99, 3.32, 4.43, 6.64, 8.85, 11.07, 13.28, 16.60, 21.03],
    [40, 2.12, 3.54, 4.72, 7.08, 9.44, 11.80, 14.16, 17.70, 22.42],
    [45, 2.26, 3.76, 5.02, 7.53, 10.03, 12.54, 15.05, 18.81, 23.83],
    [50, 2.39, 3.99, 5.32, 7.98, 10.64, 13.30, 15.96, 19.95, 25.27],
    [55, 2.53, 4.22, 5.63, 8.44, 11.26, 14.07, 16.89, 21.11, 26.74],
    [60, 2.68, 4.46, 5.94, 8.92, 11.89, 14.86, 17.83, 22.29, 28.24],
    [65, 2.82, 4.70, 6.27, 9.40, 12.53, 15.66, 18.80, 23.50, 29.76],
    [70, 2.97, 4.94, 6.59, 9.89, 13.18, 16.48, 19.78, 24.72, 31.31],
    [75, 3.12, 5.19, 6.92, 10.39, 13.85, 17.31, 20.77, 25.97, 32.89],
    [80, 3.27, 5.45, 7.26, 10.89, 14.53, 18.16, 21.79, 27.23, 34.50],
    [85, 3.42, 5.70, 7.61, 11.41, 15.21, 19.02, 22.82, 28.52, 36.13],
    [90, 3.58, 5.97, 7.96, 11.93, 15.91, 19.89, 23.87, 29.84, 37.79],
    [95, 3.74, 6.23, 8.31, 12.47, 16.62, 20.78, 24.94, 31.17, 39.48],
    [100, 3.90, 6.50, 8.67, 13.01, 17.35, 21.68, 26.02, 32.52, 41.20],
    [105, 4.07, 6.78, 9.04, 13.56, 18.08, 22.60, 27.12, 33.90, 42.94],
    [110, 4.24, 7.06, 9.41, 14.12, 18.83, 23.53, 28.24, 35.30, 44.71],
    [115, 4.41, 7.34, 9.79, 14.69, 19.58, 24.48, 29.38, 36.72, 46.51],
    [120, 4.58, 7.63, 10.18, 15.27, 20.35, 25.44, 30.53, 38.16, 48.34],
    [125, 4.76, 7.93, 10.57, 15.85, 21.13, 26.42, 31.70, 39.63, 50.19],
    [130, 4.93, 8.22, 10.96, 16.44, 21.93, 27.41, 32.89, 41.11, 52.08],
    [135, 5.11, 8.52, 11.37, 17.05, 22.73, 28.41, 34.10, 42.62, 53.98],
    [140, 5.30, 8.83, 11.77, 17.66, 23.55, 29.43, 35.32, 44.15, 55.92],
    [145, 5.48, 9.14, 12.19, 18.28, 24.37, 30.47, 36.56, 45.70, 57.89],
    [150, 5.67, 9.45, 12.61, 18.91, 25.21, 31.51, 37.82, 47.27, 59.88],
    [155, 5.86, 9.77, 13.03, 19.55, 26.06, 32.58, 39.09, 48.87, 61.90],
    [160, 6.06, 10.10, 13.46, 20.19, 26.92, 33.65, 40.39, 50.48, 63.94],
    [165, 6.25, 10.42, 13.90, 20.85, 27.80, 34.75, 41.70, 52.12, 66.02],
    [170, 6.45, 10.76, 14.34, 21.51, 28.68, 35.85, 43.02, 53.78, 68.12],
    [175, 6.66, 11.09, 14.79, 22.18, 29.58, 36.97, 44.37, 55.46, 70.25],
    [180, 6.86, 11.43, 15.24, 22.87, 30.49, 38.11, 45.73, 57.16, 72.41],
    [185, 7.07, 11.78, 15.70, 23.56, 31.41, 39.26, 47.11, 58.89, 74.59],
    [190, 7.28, 12.13, 16.17, 24.25, 32.34, 40.42, 48.51, 60.63, 76.80],
    [195, 7.49, 12.48, 16.64, 24.96, 33.28, 41.60, 49.92, 62.40, 79.04],
    [200, 7.70, 12.84, 17.12, 25.68, 34.24, 42.79, 51.35, 64.19, 81.31],
    [205, 7.92, 13.20, 17.60, 26.40, 35.20, 44.00, 52.80, 66.00, 83.60],
    [210, 8.14, 13.57, 18.09, 27.13, 36.18, 45.22, 54.27, 67.84, 85.93],
    [215, 8.36, 13.94, 18.58, 27.88, 37.17, 46.46, 55.75, 69.69, 88.28],
    [220, 8.59, 14.31, 19.08, 28.63, 38.17, 47.71, 57.25, 71.57, 90.65],
    [225, 8.82, 14.69, 19.59, 29.39, 39.18, 48.98, 58.77, 73.47, 93.06],
    [230, 9.05, 15.08, 20.10, 30.15, 40.21, 50.26, 60.31, 75.39, 95.49],
    [235, 9.28, 15.47, 20.62, 30.93, 41.24, 51.55, 61.86, 77.33, 97.95],
    [240, 9.52, 15.86, 21.14, 31.72, 42.29, 52.86, 63.43, 79.29, 100.44],
    [245, 9.75, 16.26, 21.67, 32.51, 43.35, 54.19, 65.02, 81.28, 102.95],
    [250, 9.99, 16.66, 22.21, 33.31, 44.42, 55.52, 66.63, 83.29, 105.49],
    [255, 10.24, 17.06, 22.75, 34.13, 45.50, 56.88, 68.25, 85.31, 108.06],
    [260, 10.48, 17.47, 23.30, 34.95, 46.59, 58.24, 69.89, 87.36, 110.66],
    [265, 10.73, 17.89, 23.85, 35.77, 47.70, 59.62, 71.55, 89.44, 113.29],
    [270, 10.98, 18.31, 24.41, 36.61, 48.82, 61.02, 73.22, 91.53, 115.94],
    [275, 11.24, 18.73, 24.97, 37.46, 49.95, 62.43, 74.92, 93.65, 118.62],
    [280, 11.49, 19.16, 25.54, 38.31, 51.09, 63.86, 76.63, 95.78, 121.33],
    [285, 11.75, 19.59, 26.12, 39.18, 52.24, 65.30, 78.36, 97.94, 124.06],
    [290, 12.01, 20.02, 26.70, 40.05, 53.40, 66.75, 80.10, 100.12, 126.82],
    [295, 12.28, 20.47, 27.29, 40.93, 54.57, 68.22, 81.86, 102.33, 129.61],
    [300, 12.55, 20.91, 27.88, 41.82, 55.76, 69.70, 83.64, 104.55, 132.43],
    [305, 12.82, 21.36, 28.48, 42.72, 56.96, 71.20, 85.44, 106.80, 135.28],
    [310, 13.09, 21.81, 29.08, 43.63, 58.17, 72.71, 87.25, 109.07, 138.15],
    [315, 13.36, 22.27, 29.69, 44.54, 59.39, 74.24, 89.08, 111.36, 141.05],
    [320, 13.64, 22.73, 30.31, 45.47, 60.62, 75.78, 90.93, 113.67, 143.98],
    [325, 13.92, 23.20, 30.93, 46.40, 61.87, 77.33, 92.80, 116.00, 146.93],
    [330, 14.20, 23.67, 31.56, 47.34, 63.12, 78.90, 94.68, 118.36, 149.92],
    [335, 14.49, 24.15, 32.20, 48.29, 64.39, 80.49, 96.59, 120.73, 152.93],
    [340, 14.78, 24.63, 32.83, 49.25, 65.67, 82.09, 98.50, 123.13, 155.96],
    [345, 15.07, 25.11, 33.48, 50.22, 66.96, 83.70, 100.44, 125.55, 159.03],
    [350, 15.36, 25.60, 34.13, 51.20, 68.26, 85.33, 102.39, 127.99, 162.12],
    [355, 15.65, 26.09, 34.79, 52.18, 69.58, 86.97, 104.36, 130.46, 165.24],
    [360, 15.95, 26.59, 35.45, 53.18, 70.90, 88.63, 106.35, 132.94, 168.39],
    [365, 16.25, 27.09, 36.12, 54.18, 72.24, 90.30, 108.36, 135.45, 171.57],
    [370, 16.56, 27.60, 36.79, 55.19, 73.59, 91.98, 110.38, 137.98, 174.77],
    [375, 16.86, 28.11, 37.47, 56.21, 74.95, 93.68, 112.42, 140.53, 178.00],
    [380, 17.17, 28.62, 38.16, 57.24, 76.32, 95.40, 114.48, 143.10, 181.26],
    [385, 17.48, 29.14, 38.85, 58.28, 77.70, 97.13, 116.55, 145.69, 184.54],
    [390, 17.80, 29.66, 39.55, 59.32, 79.10, 98.87, 118.65, 148.31, 187.86],
    [395, 18.11, 30.19, 40.25, 60.38, 80.50, 100.63, 120.76, 150.95, 191.20],
    [400, 18.43, 30.72, 40.96, 61.44, 81.92, 102.40, 122.88, 153.61, 194.57]
])

# n50, u
nfifty_u_4mm = np.array([
    [2.988570242547909,	54.43219732397485],
    [3.9591401203877608,	48.318625631898925],
    [2.988570242547909,	48.7634778502579],
    [44.105516438678706,	74.67398000372832],
    [46.84521242514626,	68.13469310951773],
    [50.7646754467195,	69.07779295927216],

    # [2.990686739636244	,48.67970111902714],
    # [3.0211772898656832	,54.65749377301492],
    # [4.0135088880623355	,48.230721445360004],
    # [47.207261814768835	,68.26988807286239],
    # [51.197776639187175	,68.90541230201934],
    # [44.419850496213165	,74.2072768294387],
])
"(n50, u)"
# n50, u
nfifty_u_8mm = np.array([
    [3.9730034550451268	,55.42247692969174],
    [4.966326207445054	,55.93840463842206],
    [5.961154754531081	,58.86306511656379],
    [41.797025335806204	,90.14746879833663],
    [42.22315298185902	,93.11894435695726],
    [42.22315298185902	,95.30120811199536],
    [94.09537598368814	,143.9371346002303],
    [94.09537598368814	,147.31033426450102],
])
"(n50, u)"
# n50, u
nfifty_u_12mm = np.array([
    [9.700277108603037	,75.59529817644722],
    [10.955887828441798	,78.08709948681407],
    [11.64338599416454	,76.65332564600337],
    [20.135608460451905	,94.42223177617967],
    [19.93239437459763	,97.08378411945606],
    [19.93239437459763	,97.98753683983995],
    [29.306256905915465	,102.15967264931217],
    [29.90686739636164	,104.07053385456253],
    [30.21177289865603	,106.01713705517209],
    [63.99814718591155	,154.29574130434008],
    [63.3522603232277	,157.91169728548246],
])
"(n50, u)"

# nfify, ud
nfifty_ud = np.array([
    [2.98452e+0, 5.75806e+3],
    [3.00184e+0, 5.17661e+3],
    [3.98568e+0, 5.12891e+3],
    [4.00881e+0, 6.97740e+3],
    [4.99451e+0, 7.07496e+3],
    [5.97561e+0, 7.58358e+3],
    [9.77124e+0, 6.30191e+3],
    [1.09699e+1, 6.53963e+3],
    [1.16232e+1, 6.39003e+3],
    [1.99065e+1, 8.09124e+3],
    [2.00220e+1, 7.90615e+3],
    [2.00220e+1, 8.18541e+3],
    [2.93317e+1, 8.51385e+3],
    [3.03677e+1, 8.71317e+3],
    [4.19871e+1, 1.22721e+4],
    [4.24757e+1, 1.17171e+4],
    [4.44878e+1, 1.37775e+4],
    [4.65953e+1, 1.26762e+4],
    [5.05263e+1, 1.27941e+4],
    [6.36825e+1, 1.30936e+4],
    [6.36825e+1, 1.27941e+4],
    [9.43788e+1, 1.81875e+4],
    [9.54772e+1, 1.86997e+4],
    [1.13573e+2, 2.05133e+4],
    [1.36672e+2, 2.53804e+4],
    [1.39871e+2, 2.69543e+4],
    [1.59778e+2, 2.99126e+4],
    [1.60705e+2, 2.87587e+4],
    [1.87875e+2, 3.45276e+4],
])
"(n50, ud)"
t = [1.8, 3, 4, 6, 8, 10, 12, 15, 19]
n50 = nfifty_sigm_t[:, 0]
E = 70e3
nue  = 0.23

# sig_m[nfifty,sigm]

########################################
# transform u data to ud while preserving the thickness
#
# With this, we can plot all data from the N50(U) plot to the N50(Ud) plot.
# It is shown using fracsuite splinters nfifty, that the data has been filtered to better fit the N50(Ud) curve.
########################################
total_len = len(nfifty_u_4mm) + len(nfifty_u_8mm) + len(nfifty_u_12mm)

# t, n50, u, ud
total_data = np.full((total_len, 4), np.nan)

for i in range(len(nfifty_u_4mm)):
    i4 = i

    total_data[i4, 0] = 4
    total_data[i4, 1] = nfifty_u_4mm[i, 0]
    total_data[i4, 2] = nfifty_u_4mm[i, 1]
    total_data[i4, 3] = nfifty_u_4mm[i, 1] / 0.004
for i in range(len(nfifty_u_8mm)):
    i8 = i + len(nfifty_u_4mm)

    total_data[i8, 0] = 8
    total_data[i8, 1] = nfifty_u_8mm[i, 0]
    total_data[i8, 2] = nfifty_u_8mm[i, 1]
    total_data[i8, 3] = nfifty_u_8mm[i, 1] /  0.008

for i in range(len(nfifty_u_12mm)):
    i12 = i + len(nfifty_u_4mm) + len(nfifty_u_8mm)

    total_data[i12, 0] = 12
    total_data[i12, 1] = nfifty_u_12mm[i, 0]
    total_data[i12, 2] = nfifty_u_12mm[i, 1]
    total_data[i12, 3] = nfifty_u_12mm[i, 1] /  0.012

    # sig_m[:, i] = np.sqrt(sig_m[:, i])

    # Ud[:, i] = calc_Ud(sig_m[:, i] * 2)
    # U[:, i] = calc_U(sig_m[:, i] * 2, t[i])

def navid_nfifty_ud() -> np.ndarray:
    """
    Returns the nfifty table (n,3)

    Returns:
        np.ndarray: NDarray with (n50, Ud, t)
    """
    return total_data[:, [1, 3, 0]]

def navid_nfifty_sigm():
    """
    Return (n50, sigm) for all thicknesses.
    """
    n50_navid = navid_nfifty_ud()
    n50s_navid = n50_navid[:,0].flatten()
    sigm_navid = np.sqrt(n50_navid[:,1].flatten() * 5 / 4e6 * 70e3 / (1-0.25))
    return np.column_stack((n50s_navid, sigm_navid))

def navid_nfifty(thickness: int, as_ud: bool = False) -> np.ndarray:
    """
    Returns the nfifty table (n,2) for the given thickness.

    Args:
        thickness (float): The thickness. Has to be 4, 8 or 12.

    Returns:
        np.ndarray: NDarray with X,Y values per Row.
    """
    assert thickness in [4, 8, 12], f"Thickness {thickness} can only be 4, 8 or 12"

    if thickness == 4:
        data = nfifty_u_4mm.copy()
        data[:, 1] = data[:, 1] / (8e-3 if as_ud else 1)
    elif thickness == 8:
        data = nfifty_u_8mm.copy()
        data[:, 1] = data[:, 1] / (8e-3 if as_ud else 1)
    elif thickness == 12:
        data = nfifty_u_12mm.copy()
        data[:, 1] = data[:, 1] / (12e-3 if as_ud else 1)

    return data

def navid_nfifty_interpolated(thickness: float):
    assert thickness in t, f"Thickness {thickness} not in table"

    sig_m2_t = nfifty_sigm_t[:, 1:]
    Ud = np.zeros_like(sig_m2_t)
    U = np.zeros_like(sig_m2_t)

    for i in range(len(t)):
        U[:, i] = (1e6*sig_m2_t[:, i] / (5*E))*4*(1-nue)
        Ud[:, i] = U[:, i] / (t[i]*1e-3)

    i = t.index(thickness)

    return n50.ravel(), Ud[:, i].ravel(), U[:, i].ravel()