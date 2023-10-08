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

    def __init__(self,
                 name="",
                 block: int = 213,
                 c: float = 0,
                 gauss_size: tuple[int,int] = (3,3),
                 gauss_sigma: float = 1,
                 resize_factor: float = 1,
                 adapt_mode: str = "mean"):
        self.name = name
        self.thresh_block_size = block
        self.thresh_c = c
        self.gauss_size = gauss_size
        self.gauss_sigma = gauss_sigma
        self.resize_factor = resize_factor
        self.thresh_adapt_mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C \
            if adapt_mode == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

    def print(self):
        print(self.__dict__)

softerPrepConfig = PreprocessorConfig("soft", block=413)
softerPrepConfig = PreprocessorConfig("softer", block=313)
defaultPrepConfig = PreprocessorConfig("default")
harderPrepConfig = PreprocessorConfig("harder", block=113)
hardPrepConfig = PreprocessorConfig("hard", block=53)
