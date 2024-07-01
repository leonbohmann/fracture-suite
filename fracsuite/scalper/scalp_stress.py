from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt

descr = """
Calculate actual stress state at the surface of a scalp measured glass ply.

This takes three functions of [Depth, Stress]. They can be obtained from the scalp
project files either by converting them to txt or by using scalper.py to create
text files with the needed values automatically.

Specimen identification has to look as follows: THICKNESS.NOM_STRESS.BOUND.ID
From this format the needed information are read, such as the thickness.
"""

# read the functions of stress over depth
# fit a function into the points
# create a derivation of the functions
# apply magic


if __name__ == "__main__":
    # data: [[depth], [stress]]
    sig_x_data = []
    sig_y_data = []
    sig_xy_data = []
    
    
    

def calculate_simple(meas_x: list[float], meas_y: list[float], meas_xy: list[float], \
    beta: float, C: float, mohr: False) -> Tuple[float,float]:
    """
    Calculate principal stresses assuming planar and isotropic stress state.
    Uses the stress values obtained from scalp measurements directly.

    Args:
        meas_x (list[float]): Data in x-Direction.
        meas_y (list[float]): Data in y-Direction.
        meas_xy (list[float]): Data in xy-Direction.
        beta (float): Angle of the scalp Laser.
        C (float): Lightfracturing constant of glass ply.
    
    Returns:
        Tuple with (sig_1, sig_2).
    """
    # evaluate at surface
    sig_x_0 = meas_x[0]
    sig_y_0 = meas_y[0]
    sig_xy_0 = meas_xy[0]
    
    # simplified approach for principal stresses
    t_xy = sig_xy_0 - (sig_y_0 + sig_x_0) / 2
    phi1 = 0.5 * np.arctan(2*t_xy/(sig_x_0 - sig_y_0))
    phi2 = phi1 + np.pi / 2
    
    sig_1 = (sig_x_0 + sig_y_0) / 2 + ((sig_x_0 - sig_y_0) / 2) * np.cos(2*phi1) + t_xy * np.sin(2*phi1)
    sig_2 = (sig_x_0 + sig_y_0) / 2 + ((sig_x_0 - sig_y_0) / 2) * np.cos(2*phi2) + t_xy * np.sin(2*phi2)
    
    
    if np.abs(sig_x_0) > np.abs(sig_y_0):
        s1 = sig_y_0
        s2 = sig_x_0
    else:    
        s1 = sig_x_0
        s2 = sig_y_0
    
    # draw_mohrs_stress_circle(s1, s2, sig_x_0, t_xy)
    if mohr:
        draw_mohrs_stress_circle(sig_1, sig_2, s1, np.abs(t_xy), s2, -np.abs(t_xy))

    return sig_1, sig_2

def calculate_simple_ret(d_x: list[float], d_y: list[float], d_xy: list[float], \
    ret_x: list[float], ret_y: list[float], ret_xy: list[float], \
    beta: float, C: float) -> Tuple[float,float]:
    """
    Calculate principal stresses assuming planar and isotropic stress state.
    Uses retardation derivation to calculate stresses.

    Args:
        meas_x (Measurement): Measurement in x-Direction.
        meas_y (Measurement): Measurement in y-Direction.
        meas_xy (Measurement): Measurement in xy-Direction.
        beta (float): Angle of the scalp Laser.
        C (float): Lightfracturing constant of glass ply.
    
    Returns:
        Tuple with (sig_1, sig_2).
    """
    
    # parabolic polyfit on the functions
    ret_x = np.polyfit(d_x, ret_x, 2)
    ret_y = np.polyfit(d_y, ret_y, 2)
    ret_xy = np.polyfit(d_xy, ret_xy, 2)
    
    # derivation of the functions
    dret_x = np.polyder(ret_x)
    dret_y = np.polyder(ret_y)
    dret_xy = np.polyder(ret_xy)
    
    # evaluate at surface
    sig_x_0 = dret_x[0] / ( C * np.cos(beta)**2)
    sig_y_0 = dret_y[0] / ( C * np.cos(beta)**2)
    sig_xy_0 = dret_xy[0] / ( C * np.cos(beta)**2)
    
    # simplified approach for principal stresses
    t_xy = sig_xy_0 - (sig_y_0 - sig_x_0) / 2
    phi1 = 0.5 * np.arctan(2*t_xy/(sig_x_0 - sig_y_0))
    phi2 = phi1 + np.pi / 2
    
    sig_1 = (sig_x_0 + sig_y_0) / 2 + ((sig_x_0 - sig_y_0) / 2) * np.cos(2*phi1) + t_xy * np.sin(2*phi1)
    sig_2 = (sig_x_0 + sig_y_0) / 2 + ((sig_x_0 - sig_y_0) / 2) * np.cos(2*phi2) + t_xy * np.sin(2*phi2)
    
    
    return sig_1, sig_2

def calculate_nlgs(d_x: list[float], d_y: list[float], d_xy: list[float], \
    ret_x: list[float], ret_y: list[float], ret_xy: list[float]):
    """Calculate actual stress using three measurements.

    Args:
        meas_x (Measurement): Measurement in x-Direction.
        meas_y (Measurement): Measurement in y-Direction.
        meas_xy (Measurement): Measurement in xy-Direction.
    """
    
     # parabolic polyfit on the functions
    ret_x = np.polyfit(d_x, ret_x, 2)
    ret_y = np.polyfit(d_y, ret_y, 2)
    ret_xy = np.polyfit(d_xy, ret_xy, 2)
    
    # derivation of the functions
    dret_x = np.polyder(ret_x)
    dret_y = np.polyder(ret_y)
    dret_xy = np.polyder(ret_xy)
    
def draw_mohrs_stress_circle(sigma_1, sigma_2, sigma_n1, tau_n1, sigma_n2, tau_n2):
    # Calculate the center of the circle (average of principal stresses)
    center_stress = (sigma_1 + sigma_2) / 2.0

    # Calculate the radius of the circle
    radius = (sigma_1 - sigma_2) / 2.0

    # Plot the circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = center_stress + radius * np.cos(theta)
    y_circle = radius * np.sin(theta)

    plt.plot(x_circle, y_circle)

    # Plot the principal stresses
    plt.plot([sigma_1, sigma_2], [0, 0], 'ro', label='Principal Stresses')

    # Plot the center stress
    plt.plot(center_stress, 0, 'go', label='Center Stress')

    # Plot the normal stress
    plt.plot(sigma_n1, 0, 'bo', label='σ_a')
    plt.plot(sigma_n2, 0, 'bo', label='σ_b')

    # Connect the center stress with the shear stress on the circle
    plt.plot([center_stress, sigma_n1], [0, tau_n1], 'g--', label='τ_xy')
    plt.plot([center_stress, sigma_n2], [0, tau_n2], 'g--')

    # Connect the normal stress with the shear stress on the circle
    plt.plot([sigma_n1, sigma_n1], [0, tau_n1], 'b--')
    plt.plot([sigma_n2, sigma_n2], [0, tau_n2], 'b--')

    # Set axis limits and labels
    plt.axis('equal')
    plt.xlabel('Normal Stress')
    plt.ylabel('Shear Stress')
    plt.title("Mohr's Stress Circle")

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    # Show the plot
    plt.grid()
    plt.show()

    
if __name__ == "__main__":
    # parse arguments
    # if input is a file, use file
    # if input is a folder, use on every subfolder and find "scalp" subdirectories
    #   in folders that are named after an ID
    # create a new file in "scalp" folder, that contains the calculations
    
    pass