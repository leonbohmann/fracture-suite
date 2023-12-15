import numpy as np

def alignment_between(A, B) -> float:
    """
    Aligns a vector to a point.

    Returns:
        A value between 0 and 1, where 1 is perfectly aligned and 0 is perfectly
        perpendicular.
    """

    dot = np.dot(A, B)
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    return 1 - np.abs(dot / (magA * magB))