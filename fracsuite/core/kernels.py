from typing import Any, Callable, TypeVar
import numpy as np
from fracsuite.core.image import split_image
from fracsuite.core.region import RectRegion
from fracsuite.state import State
from rich.progress import track
from fracsuite.general import GeneralSettings
from rich import print

SKIP_VALUE = np.nan

def convert_npoints(n_points, region, kw_px) -> tuple[int,int]:
    if isinstance(n_points, tuple):
        # Get the ranges for x and y
        i_w = n_points[0]
        i_h = n_points[1]
    elif n_points == -1:
        i_w = int(np.ceil(region[0]/kw_px))
        i_h = int(np.ceil(region[1]/kw_px))
    else:
        i_w = n_points
        i_h = n_points

    return i_w, i_h

class ImageKerneler():
    def __init__(
        self,
        image,
        skip_edge=False,
    ):
        self.image = image
        self.skip_edge = skip_edge

    def run(
        self,
        n_points: int,
        kw_px: int,
        z_value: Callable[[Any], float] = None,
        exclude_points: list[tuple[int,int]] = None,
        fill_skipped_with_mean: bool = True,
    ):
        print('[cyan]IM-KERNELER[/cyan] [green]END[/green]')

        px_w, px_h = (self.image.shape[1], self.image.shape[0])
        iw, ih = convert_npoints(n_points, (px_w, px_h), kw_px)
        print(f'[cyan]IM-KERNELER[/cyan] Kernel Width: {kw_px} px')
        print(f'[cyan]IM-KERNELER[/cyan] Points:       {n_points},{n_points} Points')
        print(f'[cyan]IM-KERNELER[/cyan] Region:       {px_w},{px_h} px')
        # Get the ranges for x and y
        # Get the ranges for x and y
        minx = kw_px // 2
        maxx = px_w - kw_px // 2
        miny = kw_px // 2
        maxy = px_h - kw_px // 2

        # Get 50 linearly spaced points
        xd = np.linspace(minx, maxx, iw,endpoint=True)
        yd = np.linspace(miny, maxy, ih,endpoint=True)
        X, Y = np.meshgrid(xd, yd)

        # perform the action on every area element
        if z_value is None:
            def z_value(x: Any):
                return 1

        result = np.zeros_like(X, dtype=np.float64)
        # print(result.shape)
        # print(len(xd))
        # print(len(yd))
        skip_i = 1 if self.skip_edge else 0
        # Iterate over all points and find splinters in the area of X, Y and intensity_h
        for i in track(range(len(xd)), transient=True,
                    description="Calculating intensity..."):
            for j in range(len(yd)):
                if (j < skip_i or j >= len(yd) - skip_i) or (i < skip_i or i >= len(xd) - skip_i):
                    result[j,i] = SKIP_VALUE
                    continue

                x1,x2=xd[i]-kw_px//2,xd[i]+kw_px//2
                y1,y2=yd[j]-kw_px//2,yd[j]+kw_px//2

                # Create a region (x1, y1, x2, y2)
                region = RectRegion(x1, y1, x2, y2)

                part = region.clipImg(self.image)

                is_excluded = False
                # Skip points that are in the exclude list
                if exclude_points is not None:
                    for p in exclude_points:
                        if region.is_point_in(p):
                            result[j, i] = SKIP_VALUE
                            is_excluded = True
                            break

                if is_excluded:
                    continue

                value = z_value(part)

                # Apply z_action to collected splinters
                result[j, i] = value

        Z = result.reshape(X.shape)

        invalid = np.bitwise_or(np.isnan(Z), Z == SKIP_VALUE)
        mean = np.mean(Z[~invalid])
        if fill_skipped_with_mean:
            Z[invalid] = mean
        else:
            Z[invalid] = 0
        print('[cyan]IM-KERNELER[/cyan] [green]END[/green]')

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
        collector: Callable[[T, RectRegion], bool],
        skip_edge: bool = False,
    ):
        self.region = region
        self.data_objects = data_objects
        self.collector = collector
        self.skip_edge = skip_edge

    def run(
        self,
        calculator: Callable[[list[T]], float],
        kw_px: int,
        n_points: int = -1,
        mode: str = "area",
        exclude_points: list[tuple[int,int]] = None,
        fill_skipped_with_mean: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """
        Run the kerneler.

        If n_points is -1, then the kerneler will use the kernel width to determine
        the number of points to use.

        For mode=='diag', the kerneler will only use the diagonal of the region and
        return a 1D array with results.



        Args:
            calculator (Callable[[list[T]], float]): The calculator function to use.
            kw_px (int): The kernel width in pixels. Defaults to 10.
            n_points (int, optional): The number of points to use. Defaults to -1.
            mode (str, optional): The mode to use. Defaults to "area".
            exclude_points (list[tuple[int,int]], optional): A list of points to exclude. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray: The result of the kerneler.
        """

        assert mode in ObjectKerneler.__modes, \
            f"Mode must be one of '{ObjectKerneler.__modes}'."

        if mode == "area":
            return self.__run_area(calculator, n_points, kw_px, exclude_points=exclude_points, fill_skipped_with_mean=fill_skipped_with_mean)
        elif mode == "diag":
            return self.__run_diag(calculator, n_points, kw_px, exclude_points=exclude_points, fill_skipped_with_mean=fill_skipped_with_mean)

    def __run_diag(
        self,
        calculator: Callable[[list[T]], float],
        n_points: int | tuple[int,int],
        kw_px: int,
        exclude_points: list[tuple[int,int]] = None,
        fill_skipped_with_mean: bool = True
    ) -> np.ndarray:
        assert kw_px < self.region[0], \
            "Kernel width must be smaller than the region width."
        assert kw_px < self.region[1], \
            "Kernel width must be smaller than the region height."
        assert kw_px > 10, \
            "Kernel width must be greater than 10."
        assert len(self.data_objects) > 0, \
            "There must be at least one object in the list."
        assert n_points > 0 or n_points == -1, \
            "n_points must be greater than 0."

        i_w, i_h = convert_npoints(n_points, self.region, kw_px)

        px_w, px_h = self.region

        Z = []
        for w in range(i_w):
            for h in range(i_h):
                # get diagonal only
                if w != h:
                    continue

                x1 = (w/n_points) * px_w - kw_px // 2
                y1 = (h/n_points) * px_h - kw_px // 2
                x2 = x1 + kw_px
                y2 = y1 + kw_px

                objects_in_region = \
                    [obj for obj in self.data_objects \
                        if self.collector(obj, (x1, y1, x2, y2))]

                result = calculator(objects_in_region)

                Z.append(result)

        return np.array(Z)

    def __run_area(
        self,
        calculator: Callable[[list[T]], float],
        n_points: int | tuple[int,int],
        kw: int,
        exclude_points: list[tuple[int,int]] = None,
        fill_skipped_with_mean: bool = True
    ):
        assert n_points > 0 or n_points == -1, \
            "n_points must be greater than 0 or -1."
        assert kw < self.region[0], \
            "Kernel width must be smaller than the region width."
        assert kw < self.region[1], \
            "Kernel width must be smaller than the region height."
        assert kw > 10, \
            "Kernel width must be greater than 10."
        assert len(self.data_objects) > 0, \
            "There must be at least one object in the list."

        print(f'[cyan]KERNELER[/cyan] [green]START[/green]')
        print(f'[cyan]KERNELER[/cyan] Kernel Width: {kw}')
        print(f'[cyan]KERNELER[/cyan] Points:       {n_points},{n_points} Points')
        print(f'[cyan]KERNELER[/cyan] Region:       {self.region}')

        # Get the ranges for x and y
        minx = kw // 2
        maxx = self.region[0] - kw // 2
        miny = kw // 2
        maxy = self.region[1] - kw // 2

        i_w, i_h = convert_npoints(n_points, self.region, kw)

        # xd = np.linspace(minx, maxx, int(np.round(maxx/self.kernel_width)))
        # yd = np.linspace(miny, maxy, int(np.round(maxy/self.kernel_width)))
        xd = np.linspace(minx, maxx, i_w, endpoint=True)
        yd = np.linspace(miny, maxy, i_h, endpoint=True)
        X, Y = np.meshgrid(xd, yd)


        Z = np.zeros_like(X, dtype=np.float64)
        if State.debug:
            print(f'[cyan]KERNELER[/cyan] "{len(xd)}x{len(xd)}" Points to process.')

        skip_i = 1 if self.skip_edge else 0

        if State.has_progress():
            d = range(len(xd))
        else:
            d = track(range(len(xd)), transient=True,
                    description="Running kernel over region...")

        for i in d:
            for j in range(len(yd)):
                if (j < skip_i or j >= len(yd) - skip_i) or (i < skip_i or i >= len(xd) - skip_i):
                    Z[j,i] = SKIP_VALUE
                    continue

                x1,x2=(xd[i]-kw//2,xd[i]+kw//2)
                y1,y2=(yd[j]-kw//2,yd[j]+kw//2)

                # Create a region (x1, y1, x2, y2)
                region = RectRegion(x1, y1, x2, y2)
                if State.debug:
                    print(f'[cyan]KERNELER[/cyan] Processing region ({region.wh_center()})...')

                is_excluded = False
                if exclude_points is not None:
                    for p in exclude_points:
                        if region.is_point_in(p):
                            Z[j, i] = SKIP_VALUE
                            is_excluded = True
                            if State.debug:
                                print(f'[cyan]KERNELER[/cyan] Region {region.wh_center()} is excluded.')
                            break

                if is_excluded:
                    continue


                # Collect objects in the current region
                objects_in_region = [obj for obj in self.data_objects \
                    if self.collector(obj, region)]
                # input('Continue?')

                if State.debug:
                    print(f'[cyan]KERNELER[/cyan] Found {len(objects_in_region)} objects in region.')
                # Apply z_action to collected splinters
                Z[j, i] = calculator(objects_in_region) \
                    if len(objects_in_region) > 0 else 0

                if State.debug:
                    print(f'[cyan]KERNELER[/cyan] Result: {Z[j,i]}')

        # input("continue?")
        # this is for testing
        # Z[0,0] = 1000
        # Z[-1,-1] = 1000
        invalid = np.bitwise_or(np.isnan(Z), Z == SKIP_VALUE)
        mean = np.mean(Z[~invalid])
        if fill_skipped_with_mean:
            Z[invalid] = mean
        else:
            Z[invalid] = 0
        print('[cyan]KERNELER[/cyan] [green]END[/green]')

        return X,Y,Z
