import numpy as np

def ellipse_radius(a, b, theta):
    """Returns the radius of an ellipse at a given angle."""
    return a*b/np.sqrt((b*np.cos(theta))**2 + (a*np.sin(theta))**2)