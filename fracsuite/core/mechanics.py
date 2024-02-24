import numpy as np


def U(sigma_s, t0):
    """
    Calculates the elastic strain energy for a given stress and thickness.

    Args:
        sigma_s (float): Surface stress in MPa.
        t0 (float): Thickness in m.
    """
    # print('Thickness: ', t0)
    return Ud(sigma_s) * t0 * 1e-3 # thickness in mm

def Ud(sigma_s):
    """
    Calculates the elastic strain energy density for a given stress.

    Args:
        sigma_s (float): Surface stress in MPa.
    """
    nue = 0.23
    E = 70e3
    # print('Sigma_h: ', self.scalp.sig_h)
    return 1e6/5 * (1-nue)/E * (sigma_s ** 2)


def Ud2sigm(ud):
    return np.sqrt(ud * 5 / 4e6 * 70e3 / (1-0.25))