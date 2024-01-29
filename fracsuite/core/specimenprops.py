from enum import Enum

import numpy as np


class SpecimenBreakPosition(str,Enum):
    """Break position of the specimen."""
    CENTER = "center"
    CORNER = "corner"
    NAVID = "navid"

    def position(self):
        if self == SpecimenBreakPosition.CENTER:
            return np.array((250,250))
        elif self == SpecimenBreakPosition.CORNER:
            return np.array((50,50))
        elif self == SpecimenBreakPosition.NAVID:
            return np.array((1050,10))
        else:
            raise Exception(f"Invalid break position {self}.")

class SpecimenBreakMode(str,Enum):
    PUNCH = "punch"
    DRILL = "drill"

class SpecimenBoundary(str, Enum):
    A = "A"
    B = "B"
    Z = "Z"
    Unknown = "unknown"