from enum import Enum
import cv2

class PrepMode(str, Enum):
    ADAPTIVE = 'ADAPTIVE'
    NORMAL = 'NORMAL'

class PreprocessorConfig:
    mode: PrepMode
    "Mode of the preprocessor. Either 'adaptive' or 'normal'."

    resize_factor: float
    "factor to resize input image in preprocess"

    gauss_size: tuple[int,int]
    "gaussian filter size before adaptive thold"
    gauss_sigma: float
    "gaussian sigma before adaptive thold"

    athresh_block_size: int
    "adaptive threshold block size"
    athresh_c: float
    "adaptive threshold c value"
    athresh_adapt_mode: int
    "adaptive threshold mode"

    nthresh_lower: int
    "lower threshold for normal thresholding"
    nthresh_upper: int
    "upper threshold for normal thresholding"


    lum: float
    "Lumincance correction"
    correct_light: bool
    "Instruct preprocessor to use CLERC to correct light"
    clahe_strength: float
    "CLAHE strength"
    clahe_size: int
    "CLAHE size"

    min_area: int
    max_area: int

    def __init__(
        self,
        name="default",
        mode=PrepMode.ADAPTIVE,
        block: int = 213,
        c: float = 0,
        gauss_size: tuple[int,int] = (3,3),
        gauss_sigma: float = 1,
        resize_factor: float = 1,
        adapt_mode: str = "mean",
        lum: float = None,
        correct_light: bool = False,
        clahe_strength: float = 5.0,
        clahe_size: int = 8,
        nthresh_lower: int = -1,
        nthresh_upper: int = 255
    ):
        self.mode = mode
        self.name = name
        self.athresh_block_size = block
        self.athresh_c = c
        self.gauss_size = gauss_size
        self.gauss_sigma = gauss_sigma
        self.resize_factor = resize_factor
        self.athresh_adapt_mode = 1
        self.athresh_adapt_mode : int  = cv2.ADAPTIVE_THRESH_GAUSSIAN_C \
            if adapt_mode == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

        self.nthresh_lower = nthresh_lower
        self.nthresh_upper = nthresh_upper

        self.lum = lum
        self.correct_light = correct_light
        self.clahe_strength = clahe_strength
        self.clahe_size = clahe_size

        self.min_area = 10
        self.max_area = 25000

    def print(self):
        print(self.__dict__)

    def __json__(self):
        dic = self.__dict__.copy()

        dic['thresh_adapt_mode'] = 'gaussian' if self.athresh_adapt_mode == cv2.ADAPTIVE_THRESH_GAUSSIAN_C else 'mean'
        return dic

    @classmethod
    def from_json(cls, json_obj):
        c = cls()

        for key in json_obj:
            if hasattr(c, key):
                setattr(c, key, json_obj[key])

        return c

    @classmethod
    def load(cls, filepath):
        import json
        with open(filepath, "r") as f:
            return cls.from_json(json.load(f))

# softerPrepConfig = PreprocessorConfig("soft", block=413)
# softerPrepConfig = PreprocessorConfig("softer", block=313)
# defaultPrepConfig = PreprocessorConfig("default")
# defaultPrepConfig = PreprocessorConfig("test1", block=1100, c=0, gauss_size=(3,3), gauss_sigma=0)
# defaultPrepConfig = PreprocessorConfig("test2", block=55, c=0, gauss_size=(5,5), gauss_sigma=1)
# defaultPrepConfig = PreprocessorConfig("test3", block=333, c=1.16, gauss_size=(11,11), gauss_sigma=8)
# defaultPrepConfig = PreprocessorConfig("default", block=1100, c=4, gauss_size=(3,3), gauss_sigma=28)
# harderPrepConfig = PreprocessorConfig("harder", block=113)
# hardPrepConfig = PreprocessorConfig("hard", block=53)

# defaultPrepConfig = PreprocessorConfig("test3-final", block=61, c=0, gauss_size=(3,3), gauss_sigma=1)
defaultPrepConfig = PreprocessorConfig("test2-final", block=23, c=0, gauss_size=(5,5), gauss_sigma=1)
# defaultPrepConfig = PreprocessorConfig("test1-final", block=41, c=0, gauss_size=(5,5), gauss_sigma=1)