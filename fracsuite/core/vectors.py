import numpy as np

def angle_abs(v1):
    """Returns the absolute angle of a vector in radians."""
    return np.arctan2(v1[1], v1[0])

def angle_deg(v1):
    """Returns the absolute angle of a vector in degrees."""
    return np.degrees(np.arctan2(v1[1], v1[0]))

def angle_between(v1, v2):
    # Berechnung des Winkels in Radiant
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Umwandlung in Grad
    angle_deg = np.degrees(angle_rad)

    # Bestimmung der Orientierung des Winkels
    orientation = np.sign(np.cross(v1, v2))
    if orientation < 0:
        angle_deg = 360 - angle_deg

    return np.deg2rad(angle_deg)

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
