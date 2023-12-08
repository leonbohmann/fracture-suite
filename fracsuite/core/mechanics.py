def U(sigma_s, t0):
        # print('Thickness: ', t0)
        return Ud(sigma_s) * t0 * 1e-3 # thickness in mm

def Ud(sigma_s):
    nue = 0.23
    E = 70e3
    # print('Sigma_h: ', self.scalp.sig_h)
    return 1e6/5 * (1-nue)/E * (sigma_s ** 2)