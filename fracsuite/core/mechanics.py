import numpy as np

nue = 0.23
E = 70e3

def U(sigma_s, t0):
    """
    Calculates the elastic strain energy for a given stress and thickness.

    Args:
        sigma_s (float): Surface stress in MPa.
        t0 (float): Thickness in mm.
    """
    # print('Thickness: ', t0)
    return Ud(sigma_s) * t0 * 1e-3 # thickness in mm

def Ud(sigma_s):
    """
    Calculates the elastic strain energy density for a given stress.

    Args:
        sigma_s (float): Surface stress in MPa.
    """
    # print('Sigma_h: ', self.scalp.sig_h)
    return 1e6/5 * (1-nue)/E * (sigma_s ** 2)

# np.sqrt(ud / 1e6/5 * (1-nue)/E)

def Ub(sigma_s, t0, sz):
    E = 70e3
    return 1e6 * 6 * (sigma_s/2)**2 * sz[0]*1e-3 * sz[1]*1e-3 * t0 * 1e-3 / E

def Ud2sigms(ud):
    return np.sqrt((ud * 5 * E) / ((1-nue) * 1e6))

def U2sigs(u, t):
    return np.sqrt((u * 5 * E) / ((1-nue) * 1e6 * t))