import cv2

class PreprocessorConfig:
    resize_factor: float
    "factor to resize input image in preprocess"

    gauss_size: tuple[int,int]
    "gaussian filter size before adaptive thold"
    gauss_sigma: float
    "gaussian sigma before adaptive thold"

    thresh_block_size: int
    "adaptive threshold block size"
    thresh_c: float
    "adaptive threshold c value"
    thresh_adapt_mode: int
    "adaptive threshold mode"

    lum: float
    correct_light: bool
    "Instruct preprocessor to use CLERC to correct light"
    clahe_strength: float
    "CLAHE strength"
    clahe_size: int
    "CLAHE size"

    def __init__(self,
                 name="",
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
            ):
        self.name = name
        self.thresh_block_size = block
        self.thresh_c = c
        self.gauss_size = gauss_size
        self.gauss_sigma = gauss_sigma
        self.resize_factor = resize_factor
        self.thresh_adapt_mode = 1
        self.thresh_adapt_mode : int  = cv2.ADAPTIVE_THRESH_GAUSSIAN_C \
            if adapt_mode == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

        self.lum = lum
        self.correct_light = correct_light
        self.clahe_strength = clahe_strength
        self.clahe_size = clahe_size

    def print(self):
        print(self.__dict__)

    def __json__(self):
        return {
            'name': self.name,
            'thresh_block_size': self.thresh_block_size,
            'thresh_c': self.thresh_c,
            'gauss_size': self.gauss_size,
            'gauss_sigma': self.gauss_sigma,
            'resize_factor': self.resize_factor,
            'thresh_adapt_mode': 'gaussian' if self.thresh_adapt_mode == cv2.ADAPTIVE_THRESH_GAUSSIAN_C else 'mean'
        }

    @classmethod
    def from_json(cls, json_obj):
        return cls(
            name=json_obj['name'],
            block=json_obj['thresh_block_size'],
            c=json_obj['thresh_c'],
            gauss_size=tuple(json_obj['gauss_size']),
            gauss_sigma=json_obj['gauss_sigma'],
            resize_factor=json_obj['resize_factor'],
            adapt_mode=json_obj['thresh_adapt_mode']
        )

# softerPrepConfig = PreprocessorConfig("soft", block=413)
# softerPrepConfig = PreprocessorConfig("softer", block=313)
# defaultPrepConfig = PreprocessorConfig("default")
# defaultPrepConfig = PreprocessorConfig("test1", block=1100, c=0, gauss_size=(3,3), gauss_sigma=0)
# defaultPrepConfig = PreprocessorConfig("test2", block=55, c=0, gauss_size=(5,5), gauss_sigma=1)
# defaultPrepConfig = PreprocessorConfig("test3", block=333, c=1.16, gauss_size=(11,11), gauss_sigma=8)
# defaultPrepConfig = PreprocessorConfig("default", block=1100, c=4, gauss_size=(3,3), gauss_sigma=28)
# harderPrepConfig = PreprocessorConfig("harder", block=113)
# hardPrepConfig = PreprocessorConfig("hard", block=53)

defaultPrepConfig = PreprocessorConfig("test1_3-final", block=68, c=0, gauss_size=(3,3), gauss_sigma=0)

# defaultPrepConfig = PreprocessorConfig("test1-2", block=80, c=0, gauss_size=(3,3), gauss_sigma=9)
# defaultPrepConfig = PreprocessorConfig("test2-2", block=41, c=1, gauss_size=(5,5),
#                                        gauss_sigma=5,correct_light=True)
# defaultPrepConfig = PreprocessorConfig("test3-2", block=77, c=2, gauss_size=(5,5), gauss_sigma=1, correct_light=True)
# defaultPrepConfig = PreprocessorConfig("test3-3", block=159, c=0, gauss_size=(3,3), gauss_sigma=0, correct_light=False)