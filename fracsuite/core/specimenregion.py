from fracsuite.core.model_layers import arrange_regions


class SpecimenRegion:
    def __init__(self, d_r_mm: float, d_t_deg: float, impact_pos_mm: tuple, realsize: tuple):

        self.radii, self.theta = arrange_regions(
            d_r_mm=d_r_mm,
            d_t_deg=d_t_deg,
            break_pos=(impact_pos_mm[0], impact_pos_mm[1]),
            w_mm=realsize[0],
            h_mm=realsize[1]
        )