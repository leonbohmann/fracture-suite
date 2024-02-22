from enum import Enum

class SplinterProp(str,Enum):
    AREA = 'area'
    ORIENTATION = 'orientation'
    IMPACT_DEPENDENCY = 'impdep'
    ROUNDNESS = 'roundness'
    ROUGHNESS = 'roughness'
    ASP = 'asp'
    ASP0 = 'asp0'
    L1 = 'l1'
    L2 = 'l2'
    CIRCUMFENCE = 'circ'
    L1_WEIGHTED = 'l1w'
    ANGLE = 'angle'
    ANGLE0 = 'angle0'

    # global properties
    INTENSITY = 'intensity'
    RHC = 'rhc'
    ACCEPTANCE = 'acceptance'
    NFIFTY = 'nfifty'
    COUNT = 'count'