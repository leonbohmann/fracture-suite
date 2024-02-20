import json
from multiprocessing import Pool
import tempfile
from typing import Any, Callable, TypeVar
from matplotlib import pyplot as plt
import numpy as np
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

def intensity_kernel(spl: list[Splinter], *args,**kwargs):
    """Calculate the mean intensity parameter lambda for a given set of splinters."""
    # remark: Normally, the intensity is N/A, where A is the field of view.
    #   len(spl) / kwargs["window_size"], 0
    # in the context of fracture patterns, we need to use the mean
    # area of splinters

    area = np.sum([s.area for s in spl])
    # area = kwargs["window_size"]
    return len(spl) / area, 0

def rhc_kernel(spl: list[Splinter], *args, **kwargs):
    """Calculate the hard core radius for a given set of splinters."""
    all_centroids = np.array([s.centroid_mm for s in spl])
    total_area = np.sum([s.area for s in spl])
    w = np.sqrt(total_area)
    d_max = 5 # default(CALC_DMAX, estimate_dmax(spl))
    x2,y2 = lhatc_xy(all_centroids, w, w, d_max, use_weights=False)
    min_idx = rhc_minimum(y2,x2)
    r1 = x2[min_idx]

    if "debug" in kwargs and kwargs["debug"]:
        fig,axs = plt.subplots(1,1)
        axs.plot(x2,y2)
        file = tempfile.mktemp(".png", "rhc")
        fig.savefig(file)
        plt.close(fig)

    return r1, 0

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

    return acceptance, 0

kernels = {
    'Any': splinterproperty_kernel,
    SplinterProp.INTENSITY: intensity_kernel,
    SplinterProp.RHC: rhc_kernel,
    SplinterProp.ACCEPTANCE: acceptance_kernel
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
    if State.debug:
        print(f"""Processing window ({args.i},{args.j}):
    x1: {args.x1:.1f}, x2: {args.x2:.1f}
    y1: {args.y1:.1f}, y2: {args.y2:.1f}
    window_size: {args.window_size:.1f}
    objects: {len(args.objects_in_region)}""")
    mean_value, stddev = args.calculator(args.objects_in_region, args.prop, args.impact_position, args.pxpmm, max_distance=args.max_d, window_size=args.window_size, **args.kwargs) \
        if len(args.objects_in_region) > 0 else (SKIP_VALUE, SKIP_VALUE)
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
        # create 2d array to store the results
        X = np.zeros(len(r_range_mm)-1, dtype=np.float64)
        Y = np.zeros(len(t_range_deg)-1, dtype=np.float64)
        Z = np.zeros((len(r_range_mm)-1, len(t_range_deg)-1), dtype=np.float64)
        Zstd = np.zeros((len(r_range_mm)-1, len(t_range_deg)-1), dtype=np.float64)


        window_object_counts = np.zeros((len(r_range_mm)-1, len(t_range_deg)-1), dtype=np.uint32)

        rData = KernelerData(window_object_counts)


        # the maximum possible distance between any two points
        max_d = np.sqrt(self.region[0]**2 + self.region[1]**2)

        if State.debug:
            print('[cyan]POLAR[/cyan] [green]START[/green]')
            print(f'[cyan]POLAR[/cyan] r_range:       {r_range_mm}')
            print(f'[cyan]POLAR[/cyan] t_range:       {t_range_deg}')
            print(f'[cyan]POLAR[/cyan] Region:       {self.region}')


        # find appropriate kernel function
        if calculator is None:
            if prop in kernels:
                calculator = kernels[prop]
            else:
                calculator = kernels['Any']

            if State.debug:
                print(f'[cyan]POLAR[/cyan] Using default kernel function "{calculator.__name__}".')

        spl_groups = []
        for i in range(len(r_range_mm)-1):
            new_l = []
            spl_groups.append(new_l)
            for j in range(len(t_range_deg)-1):
                new_l.append([])

        # put all splinters in the correct group
        for s in tqdm(self.data_objects, desc='Sorting splinters...', leave=False):
            dr = s.centroid_mm - ip_mm
            r = np.linalg.norm(dr)
            a = angle_deg(dr)
            for i in range(len(r_range_mm)-1):
                for j in range(len(t_range_deg)-1):
                    r0,r1 = (r_range_mm[i],r_range_mm[i+1])
                    t0,t1 = (t_range_deg[j],t_range_deg[j+1])
                    if not r0 <= r < r1:
                        continue
                    if not t0 <= a < t1:
                        continue

                    # add the splinter to a group and break
                    spl_groups[i][j].append(s)
                    break

        # create args for every (r,t) region
        args: list[WindowArguments] = []
        for i in range(len(r_range_mm)-1):
            for j in range(len(t_range_deg)-1):
                # last r includes all remaining radii
                r0,r1 = (r_range_mm[i],r_range_mm[i+1])
                t0,t1 = (t_range_deg[j],t_range_deg[j+1])
                spl = spl_groups[i][j]

                # area factor for circle segment
                cf = np.abs(t1-t0) / 360
                c = np.pi*(r1**2-r0**2)
                window_size = c * cf

                if State.debug:
                    print(f'Processing window ({i},{j}): r0: {r0:.1f}, r1: {r1:.1f}, t0: {t0:.1f}, t1: {t1:.1f}, window_size: {window_size:.1f}')
                    kwargs['debug'] = True

                # save additional data
                window_object_counts[i,j] = len(spl)

                args.append(
                    WindowArguments(
                        i,j,
                        r0,r1,
                        t0,t1,
                        max_d,
                        spl,
                        calculator,
                        prop,
                        ip_mm,
                        pxpmm,
                        window_size,
                        kwargs)
                    )

        def put_result(result):
            i, j, r_c, t_c, mean_value, stddev = result
            Z[i,j] = mean_value
            Zstd[i,j] = stddev
            X[i] = r_c
            Y[j] = t_c

        # this might raise an error
        process_window(args[0])

        if len(args) > 20 and not State.debug:
            # use multiprocessing pool
            with Pool() as pool:
                # create unordered imap and track progress
                for result in tqdm(pool.imap_unordered(process_window, args), desc='Asynchronous polar...', total=len(args), leave=False):
                    put_result(result)
        else:
            for arg in tqdm(args, desc='Synchronous calculation...', total=len(args), leave=False):
                result = process_window(arg)
                put_result(result)

        if return_data:
            return X,Y,Z,Zstd, rData

        return X,Y,Z,Zstd

    def window(
        self,
        prop,
        kw: float,
        n_points: int | tuple[int,int],
        impact_position: tuple[float,float],
        pxpmm: float,
        calculator: Callable[[list[T]], float] = None,
        return_data = False,
        **kwargs
    ):
        """
        Run a kernel over rectangular domain. Generates non-overlapping windows for `n_points == -1`. Otherwise divides the
        region into `n_points` x `n_points` windows. +The impact position is the center of the domains, and the pixel per millimeter factor is used to convert the
        window size to pixels.

        The `prop`erty is used to get an according kernel function from the kernels dictionary. If the property is not
        found, the 'Any' kernel function is used, which passes the prop to the splinter's `get_splinter_data` method.

        Using `calculator` a custom kernel function can be passed. It's function header must include the following
        arguments: `spl: list[Splinter], prop: SplinterProp, ip: tuple[float,float], pxpmm: float, **kwargs`. Alternatively
        one can encapsulate the positional arguments in `*args`.

        Args:
            prop (SplinterProp): Property to calculate.
            kw (float): Width of the kernel window.
            n_points (int | tuple[int,int]): Amount of points to calculate. If -1, the kerneler will produce non-overlapping windows.
            impact_position (tuple[float,float]): Impact position in mm. Is needed for some splinter properties.
            pxpmm (float): Scale factor.
            calculator (Callable[[list[T]], float]): Custom kernel function.
            **kwargs: Any arguments that are passed here are forwarded to the calculator function.

        Returns:
            X,Y,Z,Zstd
        """
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


        i_w, i_h = convert_npoints(n_points, self.region, kw)

        if State.debug:
            print('[cyan]WINDOW[/cyan] [green]START[/green]')
            print(f'[cyan]WINDOW[/cyan] Kernel Width: {kw}')
            print(f'[cyan]WINDOW[/cyan] Desired Points:       {n_points},{n_points} Points')
            print(f'[cyan]WINDOW[/cyan] Actual Points:       {i_w},{i_h} Points')
            print(f'[cyan]WINDOW[/cyan] Region:       {self.region}')

        # maximum possible distance between any two points
        max_d = np.sqrt(self.region[0]**2 + self.region[1]**2)

        # find appropriate kernel function
        if calculator is None:
            if prop in kernels:
                calculator = kernels[prop]
            else:
                calculator = kernels['Any']

            if State.debug:
                print(f'[cyan]WINDOW[/cyan] Using default kernel function "{calculator.__name__}".')


        # create X Y and Z and Zstd arrays
        X = np.zeros(i_w, dtype=np.float64)
        Y = np.zeros(i_h, dtype=np.float64)
        Z = np.zeros((i_h, i_w), dtype=np.float64)
        Zstd = np.zeros((i_h, i_w), dtype=np.float64)
        window_object_counts = np.zeros((i_h, i_w), dtype=np.uint32)


        rData = KernelerData(window_object_counts)


        def put_result(result):
            """Puts results from a kernel function to the arrays."""
            i,j = result[0], result[1]
            X[i] = result[2]
            Y[j] = result[3]
            Z[i, j] = result[4]
            Zstd[i, j] = result[5]


        # create groups of objects that lie in each window
        windows = []
        for w in range(i_w):
            new_l = []
            windows.append(new_l)
            for h in range(i_h):
                new_l.append([])

        # sort objects into windows
        for obj in tqdm(self.data_objects, desc='Sorting objects...', leave=False):
            c = obj.centroid_mm
            is_sorted = False
            for w in range(i_w):
                for h in range(i_h):
                    x1 = (w/i_w) * self.region[0]
                    y1 = (h/i_h) * self.region[1]
                    x2 = x1 + kw
                    y2 = y1 + kw

                    if x1 < c[0] < x2 and y1 < c[1] < y2:
                        windows[w][h].append(obj)
                        is_sorted = True
                        break

                if is_sorted:
                    break

        # create args for every (x,y) region
        args = []
        for w in range(i_w):
            for h in range(i_h):
                x1 = float(w/i_w) * self.region[0]
                y1 = float(h/i_h) * self.region[1]
                x2 = x1 + kw
                y2 = y1 + kw

                objects_in_region = windows[w][h]

                if State.debug:
                    print(f'Processing window ({w},{h}): x1: {x1:.1f}, x2: {x2:.1f}, y1: {y1:.1f}, y2: {y2:.1f}, window_size: {kw**2:.1f}')
                    kwargs['debug'] = True

                # save additional data
                window_object_counts[h,w] = len(objects_in_region)

                window_size = kw**2
                args.append(WindowArguments(w,h,x1,x2,y1,y2, max_d,objects_in_region,calculator,prop, impact_position, pxpmm, window_size, kwargs))

        # make a test run, it may raise an exception
        process_window(args[0])



        if len(args) > 120 and not State.debug:
            # iterate to calculate the values
            with Pool() as pool:
                for result in tqdm(pool.imap_unordered(process_window, args), desc='Calculating windows...', total=len(args), leave=False):
                    # print(result)
                    put_result(result)
        else:
            for arg in tqdm(args, desc='Synchronous calculation...', total=len(args), leave=False):
                result = process_window(arg)
                put_result(result)

        if return_data:
            return X,Y,Z,Zstd, rData

        return X,Y,Z,Zstd

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
        print(f'[cyan]KERNELER[/cyan] Excluded Ps:  {exclude_points}')
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

        d = range(len(xd))
        if State.has_progress():
            task = State.progress.add_task("Running kernel over region...", total=len(xd))
            def advance():
                State.progress.progress.advance(task)
            def stop():
                State.progress.remove_task(task)
        else:
            def advance():
                pass
            def stop():
                pass
            d = tqdm(range(len(xd)), leave=False, desc="Running kernel over region...")

        for i in d:
            advance()
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
        stop()
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
