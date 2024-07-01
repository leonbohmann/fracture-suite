import numpy.typing as npt


class RectRegion:

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    def is_point_in(self, *p) -> bool:
        if len(p) == 1:
            x, y = p[0]
        else:
            x, y = p

        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def clipImg(self, img: npt.NDArray) -> npt.NDArray:
        return img[self.y1:self.y2, self.x1:self.x2]

    def short(self) -> str:
        return f"p1=({self.x1:.0f}, {self.y1:.0f}), p2=({self.x2:.0f}, {self.y2:.0f})"

    def long(self) -> str:
        return f"Region(x1={self.x1:.0f}, y1={self.y1:.0f}, x2={self.x2:.0f}, y2={self.y2:.0f})"

    def wh_center(self) -> str:
        return f"w={self.x2-self.x1:.0f}, h={self.y2-self.y1:.0f}, p=({(self.x1+self.x2)/2:.0f}, {(self.y1+self.y2)/2:.0f})"