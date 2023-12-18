import numpy as np

def angle_between(A, B) -> float:
    """Returns the angle between two vectors."""
    dot = np.dot(A, B)
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    return np.arccos(dot / (magA * magB))

def alignment_between(A, B) -> float:
    """
    Aligns a vector to a point.

    Returns:
        A value between 0 and 1, where 1 is perfectly aligned and 0 is perfectly
        perpendicular.
    """
    return np.abs(alignment_cossim(A,B))

def alignment_cossim(A,B):
    """Calculate the cosine similarity between two vectors."""
    dot = np.dot(A, B)
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    return dot / (magA * magB)
