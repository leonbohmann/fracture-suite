import cv2

class PreprocessorConfig:
    resize_factor: float = 1.0
    "factor to resize input image in preprocess"

    gauss_size: tuple[int,int] = (5,5)
    "gaussian filter size before adaptive thold"
    gauss_sigma: float = 5.0
    "gaussian sigma before adaptive thold"

    thresh_block_size: int = 11
    "adaptive threshold block size"
    thresh_c: float = 5.0
    "adaptive threshold c value"
    thresh_adapt_mode: int = cv2.ADAPTIVE_THRESH_MEAN_C
    "adaptive threshold mode"

    def __init__(self,
                 name="",
                 block: int = 11,
                 c: float = 0.6,
                 gauss_size: tuple[int,int] = (5,5),
                 gauss_sigma: float = 8,
                 resize_factor: float = 1,
                 adapt_mode: str = "mean"):
        self.name = name
        self.threshold_block_size = block
        self.threshold_c = c
        self.gauss_size = gauss_size
        self.gauss_sigma = gauss_sigma
        self.resize_factor = resize_factor
        self.thresh_adapt_mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C \
            if adapt_mode == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

    def print(self):
        print(self.__dict__)

defaultPrepConfig = PreprocessorConfig("default")
softgaussPrepConfig = PreprocessorConfig("softgauss",adapt_mode="gaussian", block=5, c=1)
softmeanPrepConfig = PreprocessorConfig("softmean", adapt_mode="mean", block=5, c=1)
aggressivegaussPrepConfig = PreprocessorConfig("aggressivegauss", adapt_mode="gaussian", block=11, c=0.6)
aggressivemeanPrepConfig = PreprocessorConfig("aggressivemean", adapt_mode="mean", block=11, c=0.6)
ultrameanPrepConfig = PreprocessorConfig("ultramean", adapt_mode="mean",
                                         block=7, c=0.5,
                                         gauss_sigma=0.3, gauss_size=(3,3))