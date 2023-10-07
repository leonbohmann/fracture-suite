from typing import Any, Callable, TypeVar
import numpy as np
from fracsuite.core.image import split_image
from fracsuite.tools.state import State
from rich.progress import track
from fracsuite.tools.general import GeneralSettings

class ImageKerneler():
    def __init__(
        self,
        image,
        grid,
        skip_edge=False
    ):
        self.image = image
        self.grid = grid
        self.skip_edge = skip_edge

    def run(
        self,
        z_value: Callable[[Any], float] = None,
    ):
        px_w, px_h = (self.image.shape[1], self.image.shape[0])

        # Get the ranges for x and y
        minx = 0
        maxx = px_w
        miny = 0
        maxy = px_h

        split = split_image(self.image, self.grid)

        # Get 50 linearly spaced points
        xd = np.linspace(minx, maxx, split.rows)
        yd = np.linspace(miny, maxy, split.cols)
        X, Y = np.meshgrid(xd, yd)

        # perform the action on every area element
        if z_value is None:
            def z_value(x: Any):
                return 1

        result = np.zeros_like(X, dtype=np.float64)
        # print(result.shape)
        # print(len(xd))
        # print(len(yd))
        skip_i = int(len(xd)*0.1) if self.skip_edge else 0
        # Iterate over all points and find splinters in the area of X, Y and intensity_h
        for i in track(range(skip_i,len(xd)-skip_i), transient=True,
                    description="Calculating intensity..."):
            for j in range(skip_i,len(yd)-skip_i):
                part = split.get_part(i,j)

                value = z_value(part)

                # Apply z_action to collected splinters
                result[j, i] = value

        Z = result.reshape(X.shape)

        return X,Y,Z


T = TypeVar("T")
class ObjectKerneler():
    """Can kernel any data in a region."""
    __modes = ["area", "diag"]
    general: GeneralSettings = GeneralSettings.get()

    def __init__(
        self,
        region: tuple[int, int],
        data_objects: list[T],
        collector: Callable[[T, tuple[int,int,int,int]], bool],
        kernel_width: float = 200,
        skip_edge: bool = False,
        skip_edge_factor: float = 0.02,
    ):
        self.region = region
        self.data_objects = data_objects
        self.collector = collector
        self.kernel_width = kernel_width
        self.skip_edge = skip_edge
        self.edgeskip_factor = skip_edge_factor

    def run(
        self,
        calculator: Callable[[list[T]], float],
        n_points: int = 50,
        mode: str = "area",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Run the kerneler."""

        assert mode in ObjectKerneler.__modes, \
            f"Mode must be one of '{ObjectKerneler.__modes}'."

        if mode == "area":
            return self.__run_area(calculator)
        elif mode == "diag":
            return self.__run_diag(calculator, n_points)

    def __run_diag(
        self,
        calculator: Callable[[list[T]], float],
        n_points: int,
    ) -> np.ndarray:
        assert self.kernel_width < self.region[0], \
            "Kernel width must be smaller than the region width."
        assert self.kernel_width < self.region[1], \
            "Kernel width must be smaller than the region height."
        assert self.kernel_width > 10, \
            "Kernel width must be greater than 10."
        assert len(self.data_objects) > 0, \
            "There must be at least one object in the list."
        assert n_points > 0, \
            "n_points must be greater than 0."


        # Get the ranges for x and y
        i_w = n_points
        i_h = n_points

        px_w, px_h = self.region

        Z = []
        for w in range(i_w):
            for h in range(i_h):
                # get diagonal only
                if w != h:
                    continue

                x1 = (w/n_points) * px_w - self.kernel_width // 2
                y1 = (h/n_points) * px_h - self.kernel_width // 2
                x2 = x1 + self.kernel_width
                y2 = y1 + self.kernel_width

                objects_in_region = \
                    [obj for obj in self.data_objects \
                        if self.collector(obj, (x1, y1, x2, y2))]

                result = calculator(objects_in_region)

                Z.append(result)

        return np.array(Z)

    def __run_area(
        self,
        calculator: Callable[[list[T]], float],
    ):
        # Get the ranges for x and y
        minx = 0.0
        maxx = np.max(self.region[0])
        miny = 0.0
        maxy = np.max(self.region[1])

        xd = np.linspace(minx, maxx, int(np.round(maxx/self.kernel_width)))
        yd = np.linspace(miny, maxy, int(np.round(maxy/self.kernel_width)))
        X, Y = np.meshgrid(xd, yd)

        Z = np.zeros_like(X, dtype=np.float64)

        skip_i = int(len(xd)*self.edgeskip_factor) if self.skip_edge else 0

        if State.has_progress():
            d = range(skip_i,len(xd) - skip_i)
        else:
            d = track(range(skip_i,len(xd) - skip_i), transient=True,
                    description="Running kernel over region...")

        for i in d:
            for j in range(skip_i,len(yd) - skip_i):
                x1,y1=xd[i]-self.kernel_width//2,yd[j]-self.kernel_width//2
                x2,y2=xd[i]+self.kernel_width//2,yd[j]+self.kernel_width//2

                # Create a region (x1, y1, x2, y2)
                region_rect = (x1, y1, x2, y2)

                # Collect objects in the current region
                objects_in_region = [obj for obj in self.data_objects \
                    if self.collector(obj, region_rect)]

                # Apply z_action to collected splinters
                Z[j, i] = calculator(objects_in_region) \
                    if len(objects_in_region) > 0 else 0

        return X,Y,Z
