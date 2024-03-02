from fracsuite.core.logging import info
from fracsuite.core.mechanics import U
from fracsuite.core.specimen import Specimen
from fracsuite.scalper.scalpSpecimen import ScalpSpecimen


data = { #Specimen	 t [mm]	    σs	  σm	   M [g]	     V	  Abase [mm2]
"ScA-8-a": [3.80,	-91.3	,45.5,	0.457	,182.9	,48.1],
"ScA-8-b": [3.80,	-92.3	,45.3,	0.461	,184.3	,48.5],
"ScA-8-c": [3.80,	-93.1	,47.6,	0.344	,137.4	,36.2],
"ScB-6-a": [7.90,	-73.1	,36.6,	1.459	,583.6	,73.9],
"ScB-6-b": [7.70,	-72.9	,36.3,	1.619	,647.5	,84.1],
"ScB-6-c": [7.80,	-74.9	,37.2,	1.387	,554.8	,71.1],
"ScB-7-a": [7.90,	-90.4	,45.4,	0.613	,245.1	,31],
"ScB-7-b": [7.90,	-91.1	,45.5,	0.586	,234.6	,29.7],
"ScB-7-c": [7.90,	-93.1	,46.7,	0.549	,219.7	,27.8],
"ScB-8-a": [7.80,	-107	,53.5,	0.325	,130.2	,16.7],
"ScB-8-b": [7.90,	-106.3	,53.7,	0.307	,122.7	,15.5],
"ScB-8-c": [7.90,	-109	,55.2,	0.31	,124.1	,15.7],
"ScC-3-a": [12.0,	-55.2	,27.1,	9.488	,3795.2,	316.3],
"ScC-3-b": [12.0,	-54.3	,26.9,	10.798	,4319.1,	359.9],
"ScC-3-c": [12.0,	-54.2	,26.9,	11.122	,4448.8,	370.7],
"ScC-4-a": [12.0,	-59.4	,30	,   5.474	,2189.5,	182.5],
"ScC-4-b": [12.0,	-59.6	,29.2,	6.022	,2408.6,	200.7],
"ScC-4-c": [12.0,	-58.8	,30.1,	5.729	,2291.5,	191],
"ScC-5-a": [12.1,	-63.6	,31.5,	3.806	,1522.5,	125.8],
"ScC-5-b": [12.0,	-63.1	,31.5,	4.252	,1700.8,	141.7],
"ScC-5-c": [12.0,	-63.5	,31.7,	3.424	,1369.4,	114.1],
"ScC-6-a": [12.1,	-77.1	,37.6,	1.98	,792.1	,65.5],
"ScC-6-b": [12.0,	-77.7	,38.4,	1.899	,759.6	,63.3],
"ScC-6-c": [12.0,	-77.4	,38.9,	1.913	,765.4	,63.8],
}





class VirtualSpecimen():
    def __init__(self, t, sig_s, M, V, A, size, boundary):
        self.M = M
        self.V = V

        self.__measured_thickness = t
        self.__sigma_h = sig_s
        self.__mean_splinter_area = A

        self.__U = U(sig_s, t)
        self.__real_size = size
        self.boundary = boundary

    @property
    def splinter_area(self):
        return self.__mean_splinter_area
    @property
    def thickness(self):
        return int(round(self.__measured_thickness,0))

    @property
    def measured_thickness(self):
        return self.__measured_thickness

    @property
    def U(self) -> float:
        return self.__U

    def load_scalp(self, file=None):
        return
    def load_splinters(self, file=None):
        return
    def load(self, log_missing_data: bool = False):
        return
    def get_real_size(self):
        return self.__real_size
    def get_fall_height_m(self):
        return 0


def load_virtual_specimens(thickness):
    virtual_specimens = []
    for name in data:
        t, sig_s, sig_m, M, V, A = data[name]
        if round(t, 0) == thickness:
            virtual_specimens.append(VirtualSpecimen(t, sig_s, M, V, A, (1100,360), 'B'))
            info(f"Loaded virtual specimen {name} (t={t}mm, sigs={sig_s:.2f}MPa, M={M:.2f}g, V={V:.2f}mm3, A={A:.2f}mm2)")
    return virtual_specimens