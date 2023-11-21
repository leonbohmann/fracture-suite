import random
from typing import Annotated
import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from rich import inspect, print
from rich.progress import track
import typer
import numpy as np
from fracsuite.callbacks import main_callback
from fracsuite.core.contours import center
from fracsuite.core.image import to_rgb
from fracsuite.core.imageplotting import plotImage
from fracsuite.core.imageprocessing import dilateImg
from fracsuite.core.plotting import FigureSize, get_fig_width
from fracsuite.core.point_process import CSR_process, strauss_process, gibbs_strauss_process
from fracsuite.core.region import RectRegion
from fracsuite.core.specimen import Specimen
from fracsuite.core.splinter import Splinter
from fracsuite.core.stochastics import calculate_khat, calculate_fhat, calculate_ghat
from fracsuite.core.stress import relative_remaining_stress
from scipy.spatial import Voronoi, voronoi_plot_2d

from fracsuite.state import State

sim_app = typer.Typer(help=__doc__, callback=main_callback)


@sim_app.command()
def sim_break(
    specimen_name: Annotated[str, typer.Argument(help="Specimen name.")],
    nue: Annotated[float, typer.Option(help="Poissons ratio.")] = 0.23,
    E: Annotated[float, typer.Option(help="Youngs modulus [MPa].")] = 70e3,
):
    """
    Create a voronoi tesselation using the BREAK approach.

    The created voronoi plot should have similar statistical properties as the
    real specimen.
    """
    specimen = Specimen.get(specimen_name)

    thickness = specimen.measured_thickness
    sig_h = specimen.sig_h

    lam = specimen.break_lambda
    rhc = specimen.break_rhc

    mean_area = specimen.mean_splinter_area

    # estimate relative remaining stress
    urr = relative_remaining_stress(mean_area, thickness)
    # relaxation factor
    nue = 1 - urr

    # calculate initial strain energy U0 or U_D (in navid diss)
    U0 = (1-nue)/5/E * sig_h**2 * thickness
    # remaining strain energy
    U1 = urr * U0

    area = specimen.get_real_size()
    A = area[0] * area[1]
    n_points = lam * A
    points = gibbs_strauss_process(n_points, rhc, nue, area=area)
    fig,axs = plt.subplots()
    axs.scatter(*zip(*points))
    plt.show()


    # create voronoi of points
    vor = Voronoi(points)
    fig,axs = plt.subplots()
    voronoi_plot_2d(vor, ax=axs, show_points=False, show_vertices=False)
    plt.show()

@sim_app.command()
def est_break(
    specimen_name: Annotated[str, typer.Argument(help="Specimen name.")],
):
    specimen = Specimen.get(specimen_name)

    # find fracture intensity from first order statistics
    intensity = specimen.calculate_break_lambda(force_recalc=True)

    # find the hard core radius using second order statistics
    rhc = specimen.calculate_break_rhc(force_recalc=True)
    print(f'Fracture intensity: {intensity:.2f} 1/mm²')
    print(f'HC Radius: {rhc:.2f} mm')

    def Kpois(d):
        # see Baddeley et al. S.206 K_pois
        return np.pi*d**2

    x,y = specimen.kfun()
    fig, ax = plt.subplots(figsize=get_fig_width(FigureSize.ROW2))
    ax.plot(x,y, label='$\hat{K}$')
    ax.plot(x,Kpois(np.asarray(x)), label='$\hat{K}_{t}$')
    ax.legend()
    ax.set_ylabel('$\hat{K}(d)$')
    ax.set_xlabel('$d$ [mm]')
    State.output(fig,'kfunc', spec=specimen, figwidth=FigureSize.ROW2)


    x2,y2 = specimen.lfun()
    min_L = rhc

    ax: Axes
    fig, ax = plt.subplots(figsize=get_fig_width(FigureSize.ROW2))
    ax.plot(x2,y2, label='$\hat{L}$')
    ax.axvline(rhc, linestyle='--', color='r', label=f'$r_{{hc}}={min_L:.1f}mm$')
    ax.legend()
    ax.set_ylabel('$\hat{L}(d)$')
    ax.set_xlabel('$d$ [mm]')
    State.output(fig, 'lfunc', spec=specimen, figwidth=FigureSize.ROW2)






@sim_app.command()
def create_spatial():

    area = (500,500)
    intensity = 0.1
    n_points = 500
    image = np.zeros((area[0],area[1],3), dtype=np.uint8)

    points = gibbs_strauss_process(n_points, 20, intensity, area=area)
    # points = CSR_process(10, area[0])
    x,y = zip(*points)

    plt.figure()
    plt.scatter(x,y)
    plt.show()

    # perform watershed on the points
    markers = np.zeros(area, dtype=np.uint8)
    for point in points:
        markers[int(point[0]), int(point[1])] = 255

    markers = dilateImg(markers, 5)
    markers_rgb = to_rgb(markers)
    # from top left elongate markers
    p0 = np.asarray((area[0]*0.2,area[1]*0.2))
    max_elong = 20
    elong_d = 2 #px
    ds = np.linspace(0, np.sqrt(area[0]**2+area[1]**2), 100)

    dmax = np.max(ds)
    done = []
    print(ds)
    for d in track(ds):
        fd = 1-d/dmax
        print(fd)

        for i,p in enumerate(points):
            if i in done:
                continue

            p = np.asarray(p)

            # get vector from p0 to p
            v = p-p0
            # normalize
            dp = np.linalg.norm(v)

            if dp > d:
                continue

            done.append(i)

            v = v/dp
            for j in np.linspace(0, max_elong*fd, int(max_elong*fd/elong_d)*5):
                # calculate new point from p with v0 and elong_d distance
                new_point = p+v*j
                # and other side as well
                new_point2 = p-v*j
                # print(p)
                # print(v)
                # print(new_point)
                # print(new_point2)
                # input("")
                # add filled circle to both new points
                # cv2.circle(markers_rgb, (int(p[1]), int(p[0])), elong_d, (0,255,255), -1)
                # cv2.circle(markers_rgb, (int(new_point[1]), int(new_point[0])), elong_d, (255,0,0), -1)
                cv2.circle(markers, (int(new_point[1]), int(new_point[0])), elong_d, 255, -1)
                cv2.circle(markers, (int(new_point2[1]), int(new_point2[0])), elong_d, 255, -1)



    plotImage(markers, 'elongated markers', force=True)

    markers = cv2.connectedComponents(np.uint8(markers))[1]

    markers = cv2.watershed(np.zeros_like(image), markers)

    m_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    m_img[markers == -1] = 255

    splinters = Splinter.analyze_contour_image(m_img)

    out_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for s in splinters:
        clr = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.drawContours(out_img, [s.contour], -1, clr, -1)

    plt.imshow(out_img)
    plt.show()

@sim_app.command()
def compare_spatial(specimen_name):
    specimen = Specimen.get(specimen_name)


    size = 300
    x1 = 500
    x2 = x1 + size
    y1 = 500
    y2 = y1 + size

    # choose a region to replicate
    region = RectRegion(x1,y1,x2,y2)

    # get region from specimen fracture image
    splinters_in_region = specimen.get_splinters_in_region(region)
    frac_img = specimen.get_fracture_image()

    image = np.zeros((size,size,3), dtype=np.uint8)

    orig_contours = to_rgb(np.zeros_like(frac_img, dtype=np.uint8))

    # perform watershed on the points
    markers = np.zeros((size,size), dtype=np.uint8)
    for s in splinters_in_region:
        point = s.centroid_px
        ix = np.max([np.min([int(point[0])-x1, size-1]), 0])
        iy = np.max([np.min([int(point[1])-y1, size-1]), 0])
        markers[ix,iy] = 255

        cv2.drawContours(orig_contours, [s.contour], -1, (0,0,255), 2)

    orig_contours = orig_contours[x1:x2, y1:y2,:]

    markers = dilateImg(markers, 3, it=3)
    plt.imshow( markers)
    plt.show()

    markers = cv2.connectedComponents(np.uint8(markers))[1]

    markers = cv2.watershed(np.zeros_like(image), markers)

    m_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    m_img[markers == -1] = 255

    splinters = Splinter.analyze_contour_image(m_img)

    gen_contours = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for s in splinters:
        clr = (255,0,0)
        cv2.drawContours(gen_contours, [s.contour], -1, clr, 1)

    print(orig_contours.shape)
    print(gen_contours.shape)
    comparison = cv2.addWeighted(orig_contours, 0.5, gen_contours, 0.5, 0)


    plt.imshow(comparison)
    plt.show()


@sim_app.command()
def create_voronoi():

    area = (500,500)
    intensity = 0.3
    n_points = 400
    image = np.zeros((area[0],area[1],3), dtype=np.uint8)

    points = gibbs_strauss_process(n_points, 50, 0.1, area=area)
    # points = CSR_process(10, area[0])
    x0,y0 = zip(*points)

    plt.figure()
    plt.scatter(x0,y0)
    plt.show()

    # from top left elongate markers
    p0 = np.asarray((0,0))
    max_elong = 5
    elong_d = 120 #px
    ds = np.linspace(0, np.sqrt(area[0]**2+area[1]**2), 100)

    dmax = np.max(ds)
    done = []
    print(ds)
    points2 = []
    for d in track(ds):
        fd = d/dmax
        print(fd)

        for i,p in enumerate(points):

            if i in done:
                continue

            p = np.asarray(p)

            # get vector from p0 to p
            v = p-p0
            # normalize
            dp = np.linalg.norm(v)

            if dp > d:
                continue

            done.append(i)

            v = v/dp

            dv = v*fd*elong_d
            ldv = np.linalg.norm(dv)

            if ldv > 20:
                new_point = p-dv
                points2.append(new_point)
            points2.append(p)

    x,y = zip(*points2)
    plt.figure()
    plt.scatter(x0,y0)
    plt.scatter(x,y)
    plt.show()

    # create voronoi from points
    vor = Voronoi(points2)
    fig,axs = plt.subplots()
    voronoi_plot_2d(vor, ax=axs)

    plt.show()
