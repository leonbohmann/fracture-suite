import numpy as np
from scipy.interpolate import interp1d

# from Nielsen et. al. (2016), 10.1007/s40940-016-0036-z
urr_data = np.array([
    [0.05, 0.003, 0.002, 0.001, 0.001],
    [0.25, 0.052, 0.032, 0.028, 0.026],
    [0.45, 0.121, 0.083, 0.074, 0.069],
    [0.65, 0.186, 0.141, 0.129, 0.121],
    [0.85, 0.241, 0.198, 0.184, 0.173],
    [1.05, 0.287, 0.250, 0.235, 0.223],
    [1.25, 0.326, 0.295, 0.281, 0.269],
    [1.45, 0.358, 0.335, 0.323, 0.310],
    [1.65, 0.386, 0.370, 0.359, 0.346],
    [1.85, 0.411, 0.401, 0.391, 0.379],
    [2.05, 0.432, 0.427, 0.419, 0.407],
    [2.25, 0.452, 0.451, 0.444, 0.433],
    [2.45, 0.469, 0.472, 0.466, 0.456],
    [2.65, 0.485, 0.490, 0.486, 0.476],
    [2.85, 0.499, 0.507, 0.504, 0.495],
    [3.05, 0.512, 0.522, 0.520, 0.512],
    [3.25, 0.524, 0.536, 0.535, 0.527],
    [3.45, 0.535, 0.549, 0.548, 0.541],
    [3.65, 0.546, 0.560, 0.560, 0.554],
    [3.85, 0.556, 0.571, 0.572, 0.566],
    [4.05, 0.565, 0.581, 0.582, 0.577],
    [5.65, 0.621, 0.642, 0.645, 0.643],
    [7.25, 0.660, 0.681, 0.686, 0.686],
    [8.85, 0.688, 0.710, 0.715, 0.717],
    [10.45, 0.711, 0.732, 0.738, 0.740],
    [12.05, 0.729, 0.750, 0.755, 0.759],
    [13.65, 0.744, 0.764, 0.770, 0.774],
    [15.25, 0.756, 0.776, 0.782, 0.786],
    [16.85, 0.767, 0.787, 0.793, 0.797],
    [18.45, 0.777, 0.796, 0.802, 0.806]
])

ns = {
    3: 1,
    4: 2,
    5: 3,
    -1: 4,
}

def relative_remaining_stress(A: float, h: float, n: int = -1):
    """
    Logarithmically interpolates value for A/h² from URR-Data taken from Nielsen et. al. (2016), 10.1007/s40940-016-0036-z.

    Args:
        A (float): The mean fragment area.
        h (float): The thickness of the glass plate.
        n (int): The number of polygon edges to use. Defaults to -1, which is equivalent to Infinite.

    Returns:
        The relative remaining strain energy density in fragments.
    """
    d = A/h**2
    assert d > 0.05 and d < 18.45, f"A/h²={d:.2f} is out of range of the URR-Data."
    assert n in ns, "n is not a valid number of polygon edges."

    n = ns[n]

    # find lower and upper bounds
    i = 0
    while urr_data[i,0] < d:
        i += 1

    x1 = urr_data[i-1, 0]
    y1 = urr_data[i-1, n]
    x2 = urr_data[i, 0]
    y2 = urr_data[i, n]

    b = (np.log10(y2) - np.log10(y1))/(np.log10(x2) - np.log10(x1))
    # print(d)
    # print(b)
    # print(y1)
    # print(x1)

    y = y1 * (d/x1)**b

    return y