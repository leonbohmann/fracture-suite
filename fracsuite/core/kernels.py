import json
from fracsuite.core.arrays import fill_nan
from fracsuite.core.logging import debug
from multiprocessing import Pool
import tempfile
from typing import Any, Callable, TypeVar
from deprecated import deprecated
from matplotlib import pyplot as plt
import numpy as np
from fracsuite.core.progress import get_progress
from fracsuite.core.region import RectRegion
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stochastics import rhc_minimum, lhatc_xy
from fracsuite.core.vectors import angle_deg
from fracsuite.state import State
from tqdm import tqdm
from fracsuite.general import GeneralSettings
from rich import print
from dataclasses import dataclass

SKIP_VALUE = np.nan

def deepcopy(obj):
    return json.loads(json.dumps(obj))

def convert_npoints(n_points, region, kw_px) -> tuple[int,int]:
    if isinstance(n_points, tuple):
        # Get the ranges for x and y
        i_w = n_points[0]
        i_h = n_points[1]

        if i_w == -1:
            i_w = int(np.ceil(region[0]/kw_px))
        if i_h == -1:
            i_h = int(np.ceil(region[1]/kw_px))

    elif n_points == -1:
        i_w = int(np.ceil(region[0]/kw_px))
        i_h = int(np.ceil(region[1]/kw_px))
    else:
        i_w = n_points
        i_h = n_points

    return i_w, i_h


def splinterproperty_kernel(spl: list[Splinter], prop: SplinterProp, ip, pxpmm, **kwargs):
    """Calculates a property for a list of splinters."""
    if len(spl) == 0:
        return (np.nan,np.nan)

    values = np.asarray([s.get_splinter_data(prop=prop, ip_mm=ip,px_p_mm=pxpmm) for s in spl])

    return (np.nanmean(values), np.nanstd(values))

def nfifty_kernel(spl: list[Splinter], *args,**kwargs):
    """Count splinters and relate them to an area of 50x50."""
    # factor to convert the area to 50x50
    f = 2500/kwargs["window_size"]
    return int(len(spl)*f), 0

def count_kernel(spl: list[Splinter], *args, **kwargs):
    """Count splinters and relate them to an area of 50x50."""
    return int(len(spl)), 0

def intensity_kernel(spl: list[Splinter], *args,**kwargs):
    """Calculate the mean intensity parameter lambda for a given set of splinters."""
    # remark: Normally, the intensity is N/A, where A is the field of view.
    #   len(spl) / kwargs["window_size"], 0
    # in the context of fracture patterns, we need to use the mean
    # area of splinters

    area = np.sum([s.area for s in spl])
    # area = kwargs["window_size"]
    return len(spl) / area, 0 # 0 has to stay!

def rhc_kernel(spl: list[Splinter], *args, **kwargs):
    """Calculate the hard core radius for a given set of splinters."""
    all_centroids = np.array([s.centroid_mm for s in spl])
    total_area = np.sum([s.area for s in spl])
    w = np.sqrt(total_area)
    d_max = 7.5 # default(CALC_DMAX, estimate_dmax(spl))
    x2,y2 = lhatc_xy(all_centroids, w, w, d_max, use_weights=False)
    min_idx = rhc_minimum(y2,x2)
    r1 = x2[min_idx]
    return r1, 0 # 0 has to stay!

def acceptance_kernel(spl: list[Splinter], *args, **kwargs):
    """Calculate the hard core radius for a given set of splinters."""
    assert 'max_distance' in kwargs, "max_distance not passed to kernel function!"

    all_centroids = np.array([s.centroid_mm for s in spl])
    total_area = np.sum([s.area for s in spl])
    w = np.sqrt(total_area)
    d_max = 5 # default(CALC_DMAX, estimate_dmax(spl))
    x2,y2 = lhatc_xy(all_centroids, w, w, d_max, use_weights=False)
    min_idx = np.argmin(y2)
    acceptance = x2[min_idx] / kwargs['max_distance']

    # this is debug output
    if "debug" in kwargs and kwargs["debug"]:
        fig,axs = plt.subplots(1,1)
        axs.plot(x2,y2)
        file = tempfile.mktemp(".png", "acc")
        fig.savefig(file)
        plt.close(fig)

    return acceptance, 0 # 0 has to stay!

kernels = {
    'Any': splinterproperty_kernel,
    SplinterProp.INTENSITY: intensity_kernel,
    SplinterProp.RHC: rhc_kernel,
    SplinterProp.ACCEPTANCE: acceptance_kernel,
    SplinterProp.NFIFTY: nfifty_kernel,
    SplinterProp.COUNT: count_kernel,

}

@dataclass
class KernelerData:
    window_object_counts: np.ndarray


@dataclass
class WindowArguments:
    i: int
    j: int
    x1: float
    x2: float
    y1: float
    y2: float
    max_d: float
    objects_in_region: list
    calculator: Callable[[list], float]
    prop: str
    impact_position: tuple[float,float]
    pxpmm: float
    window_size: float
    kwargs: dict

@dataclass
class WindowResult:
    i: int
    "Index i"
    j: int
    "Index j"
    x: float
    "Center x coordinate"
    y: float
    "Center y coordinate"
    value: float
    "Mean value of the region"
    stddev: float
    "Standard deviation of the region"



def process_window(args: WindowArguments):
    """Wrapper function for the window kerneler."""
    if len(args.objects_in_region) < 50 and args.prop == SplinterProp.RHC:
        debug(f'> Window ({args.i},{args.j}) discard, less than 50 splinters: {len(args.objects_in_region)}')
        return (args.i, args.j, (args.x1+args.x2)/2, (args.y1+args.y2)/2, np.nan, np.nan)

    mean_value, stddev = args.calculator(args.objects_in_region, args.prop, args.impact_position, args.pxpmm, max_distance=args.max_d, window_size=args.window_size, **args.kwargs)
    return (args.i, args.j, (args.x1+args.x2)/2, (args.y1+args.y2)/2, mean_value, stddev)


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
        for i in tqdm(range(len(xd)), leave=False, desc="Calculating intensity..."):
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
    general: GeneralSettings = GeneralSettings.get()

    def __init__(
        self,
        region: tuple[int, int],
        data_objects: list[T],
        *args,
        **kwargs
    ):
        self.region = region
        self.data_objects = data_objects

        if len(args) > 0 or len(kwargs) > 0:
            debug('ObjectKerneler: Additional arguments passed. Consider removing them.')


    def __internalrun(
        self,
        prop,
        x_range,
        y_range,
        center_function,
        window_size_function,
        window_location_function,
        ip_mm,
        pxpmm,
        calculator: Callable[[list[Splinter]], float] = None,
        return_data = False,
        meshgrid_xy = False,
        **kwargs
    ):
        """
        Run a kernel over a polar domain. Uses non-overlapping windows that are defined by `r_range_mm` and `t_range_deg`.
        The impact position is the center of the domains, and the pixel per millimeter factor is used to convert the
        window size to pixels.

        The `prop`erty is used to get an according kernel function from the kernels dictionary. If the property is not
        found, the 'Any' kernel function is used, which passes the prop to the splinter's `get_splinter_data` method.

        Using `calculator` a custom kernel function can be passed. It's function header must include the following
        arguments: `spl: list[Splinter], prop: SplinterProp, ip: tuple[float,float], pxpmm: float, **kwargs`. Alternatively
        one can encapsulate the positional arguments in `*args`.

        Args:
            mode (SplinterProp): Property to calculate.
            r_range_mm (range): Created with arrange_regions.
            t_range_deg (range): Created with arrange_regions.
            ip_mm (tuple[float,float]): Impact position in mm.
            pxpmm (float): Pixel per millimeter factor.
            calculator (Callable[[list[Splinter]], float]): The calculator function to use.
            return_data (bool): If True, the function will return additional data.
            kwargs: Additional arguments for the calculator.

        Returns:
            R,T,Z,Zstd [, data]:
                Radii and thetas contain the centers of each region. They have to be used for interpolation.
                Z contains the calculated values.

                If R(1,n) and T(1,m) then Z(n,m) and Zstd(n,m) contain the calculated values and their standard deviations respectively.

                If return_data is True, the function will return additional data.
        """
        X = np.zeros(len(x_range)-1, dtype=np.float64)
        Y = np.zeros(len(y_range)-1, dtype=np.float64)
        Z = np.zeros((len(y_range)-1, len(x_range)-1), dtype=np.float64)
        Zstd = np.zeros((len(y_range)-1, len(x_range)-1), dtype=np.float64)

        window_object_counts = np.zeros((len(y_range)-1, len(x_range)-1), dtype=np.uint32)
        rData = KernelerData(window_object_counts)

        # the maximum possible distance between any two points
        max_d = np.sqrt(self.region[0]**2 + self.region[1]**2)

        # find appropriate kernel function
        if calculator is None:
            if prop in kernels:
                calculator = kernels[prop]
            else:
                calculator = kernels['Any']

        # create empty groups
        spl_groups = list()
        for i in range(len(y_range)-1):
            new_l = list()
            spl_groups.append(new_l)
            for j in range(len(x_range)-1):
                new_l.append(list())


        with get_progress(title="Sorting splinters...", total=len(self.data_objects)) as progress:
            # put all splinters in the correct group
            for s in self.data_objects:
                x,y = center_function(s)
                for i in range(len(x_range)-1):
                    for j in range(len(y_range)-1):
                        x0, x1, y0, y1 = window_location_function(x_range[i],x_range[i+1], y_range[j],y_range[j+1])

                        if not (x0 <= x < x1):
                            continue
                        if not (y0 <= y < y1):
                            continue

                        # add the splinter to each group it fits in
                        spl_groups[j][i].append(s)
                        progress.advance()

        # create args for every (r,t) region
        args: list[WindowArguments] = []
        with get_progress(title="Sorting splinters...", total=(len(x_range)-1)*(len(y_range)-1)) as progress:
            for i in range(len(x_range)-1):
                for j in range(len(y_range)-1):
                    # last r includes all remaining radii
                    x0, x1, y0, y1 = window_location_function(x_range[i], x_range[i+1], y_range[j], y_range[j+1])
                    window_size = window_size_function(x0,x1,y0,y1)

                    # get the objects in the region
                    spl = spl_groups[j][i]

                    if i % 30 == 0:
                        debug(f'Defining window ({i},{j}): x: [{x0:.1f}, {x1:.1f}], y: [{y0:.1f}, {y1:.1f}], sz: {window_size:.1f}, len: {len(spl)}.')
                    if State.debug:
                        kwargs['debug'] = True

                    # save additional data
                    window_object_counts[j,i] = len(spl)

                    args.append(
                        WindowArguments(
                            i,j,
                            x0,x1,
                            y0,y1,
                            max_d,
                            spl,
                            calculator,
                            prop,
                            ip_mm,
                            pxpmm,
                            window_size,
                            kwargs)
                        )
                    progress.advance()

        def put_result(result):
            i, j, r_c, t_c, mean_value, stddev = result
            Z[j,i] = mean_value
            Zstd[j,i] = stddev
            X[i] = r_c
            Y[j] = t_c

        # this might raise an error
        result = process_window(args[0])
        debug(f'Testing output: {result}')

        debug(f'Using {len(args)} windows. Kernel function: {calculator.__name__}.')
        with get_progress(title=f'Running {prop} calculation...', total=len(args)) as progress:
            if len(args) > 50 and not State.debug:
                debug('len(args) > 50, using multiprocessing pool.')
                # iterate to calculate the values
                with Pool() as pool:
                    for result in pool.imap_unordered(process_window, args):
                        # print(result)
                        put_result(result)

                        progress.advance()
            else:
                debug('len(args) < 50 or debug mode, using synchronous calculation.')
                for arg in args:
                    result = process_window(arg)
                    put_result(result)

                    progress.advance()

        if meshgrid_xy:
            debug('Returning meshgrid X,Y.')
            X,Y = np.meshgrid(X,Y)

        # fill NaN values
        fill_nan(Z)
        fill_nan(Zstd)

        if return_data:
            debug('Returning rData...')
            return X,Y,Z,Zstd, rData


        return X,Y,Z,Zstd


    def polar(
        self,
        prop,
        r_range_mm,
        t_range_deg,
        ip_mm,
        pxpmm,
        calculator: Callable[[list[Splinter]], float] = None,
        return_data = False,
        **kwargs
    ):
        def center_function(s):
            dr = s.centroid_mm - ip_mm
            r = np.linalg.norm(dr)
            a = angle_deg(dr)
            return r,a

        def window_size_function(x0,x1,y0,y1):
            # area factor for circle segment
            cf = np.abs(y1-y0) / 360
            c = np.pi*(x1**2-x0**2)
            return c * cf

        def window_location_function(x0,x1,y0,y1):
            return x0,x1,y0,y1

        return self.__internalrun(
            prop,
            r_range_mm,
            t_range_deg,
            center_function,
            window_size_function,
            window_location_function,
            ip_mm,
            pxpmm,
            calculator,
            return_data,
            **kwargs
        )

    def window(
        self,
        prop: SplinterProp,
        kw: float,
        n_points: int | tuple[int,int],
        impact_position: tuple[float,float],
        pxpmm: float,
        calculator: Callable[[list[T]], float] = None,
        return_data = False,
        **kwargs
    ):
        """
        Applies a window function to the given property within a specified region.

        If no calculator function is passed, the default calculator function for the property will be used.

        If `n_points = -1`, the number of points will be calculated from the region and the window size. If
        `n_points` is a tuple, the first value will be used for the x direction and the second value for the y direction.

        Args:
            prop: The property to be analyzed.
            kw (float): The size of the window in millimeters.
            n_points (int | tuple[int,int]): The number of points in the x and y directions or a tuple specifying the x and y dimensions of the region.
            impact_position (tuple[float,float]): The position of the impact in millimeters.
            pxpmm (float): The number of pixels per millimeter.
            calculator (Callable[[list[T]], float], optional): A function to calculate a value from the property values within each window. Defaults to None.
            return_data (bool, optional): Whether to return the analyzed data. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the calculator function.

        Returns:
            The result of the calculator function if return_data is True, otherwise None.
        """

        def center_function(s):
            return s.centroid_mm

        def window_size_function(x0,x1,y0,y1):
            return kw**2

        def window_location_function(x0,x1,y0,y1):
            return x0-kw//2,x0+kw//2,y0-kw//2,y0+kw//2

        i_w, i_h = convert_npoints(n_points, self.region, kw)

        debug(f'region: {self.region}')
        debug(f'kw: {kw}, n_points: {n_points}, i_w: {i_w}, i_h: {i_h}')
        x_centers = np.linspace(kw, self.region[0]-kw, i_w, endpoint=True)
        y_centers = np.linspace(kw, self.region[1]-kw, i_h, endpoint=True)
        debug(f'x_centers: {x_centers}')
        debug(f'y_centers: {y_centers}')

        return self.__internalrun(
            prop,
            x_centers,
            y_centers,
            center_function,
            window_size_function,
            window_location_function,
            impact_position,
            pxpmm,
            calculator,
            return_data,
            **kwargs
        )

    @deprecated(reason="Use the window/polar method instead.")
    def run(
        self,
        calculator: Callable[[list[T]], float],
        kw_px: int,
        n_points: int = -1,
        mode: str = "area",
        exclude_points: list[tuple[int,int]] = None,
        fill_skipped_with_mean: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        return None