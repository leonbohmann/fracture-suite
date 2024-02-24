"""
Commands for simulating and analyzing fracture morphologies.
"""
from tqdm import tqdm
from fracsuite.core.logging import critical, debug, info
import random
from typing import Annotated
import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from rich import print
from rich.progress import track
import typer
import numpy as np
from fracsuite.callbacks import main_callback
from fracsuite.core.geometry import delta_hcp
from fracsuite.core.image import is_gray, to_gray, to_rgb
from fracsuite.core.imageplotting import plotImage
from fracsuite.core.imageprocessing import dilateImg
from fracsuite.core.mechanics import U
from fracsuite.core.model_layers import ModelLayer, arrange_regions, has_layer, interp_layer
from fracsuite.core.plotting import DataHistMode, FigureSize, annotate_corner, datahist_2d, datahist_plot, datahist_to_ax, get_fig_width, get_log_range, renew_ticks_ax, renew_ticks_cb, voronoi_to_image
from fracsuite.core.point_process import gibbs_strauss_process
from fracsuite.core.progress import get_progress, tracker
from fracsuite.core.region import RectRegion
from fracsuite.core.simulation import Simulation
from fracsuite.core.specimen import Specimen, SpecimenBoundary, SpecimenBreakPosition
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stress import relative_remaining_stress
from scipy.spatial import Voronoi, voronoi_plot_2d
from fracsuite.core.vectors import angle_between

from fracsuite.state import State

from spazial import csstraussproc2, bohmann_process
import matplotlib.ticker as ticker



sim_app = typer.Typer(help=__doc__, callback=main_callback)

def section(title: str):
    print(f'[yellow]{title}')

rng = np.random.default_rng()
def stdrand(mean, stddev):
    """Returns a random number between -0.5 and 0.5"""
    d = rng.standard_normal()
    return mean + d * stddev

def meanrand(mean, stddev):
    """Returns a random number between mean-stdev and mean+stdev"""
    d = rng.random() - 0.5
    return mean + d * stddev

@sim_app.command()
def nbreak(
    specimen_name: Annotated[str, typer.Argument(help="Specimen name.")],
    sz: Annotated[FigureSize, typer.Option(help="Figure size for spatial plots.")] = FigureSize.ROW2,
    plotsz: Annotated[FigureSize, typer.Option(help="Figure size for graph plots.")] = FigureSize.ROW2,
    with_poisson: Annotated[bool, typer.Option(help="Plot poisson's distribution functions.")] = False,
    legend_outside: Annotated[bool, typer.Option(help="Place legend outside of plot.")] = False,
    region: Annotated[tuple[float,float,float,float], typer.Option(help="Region to plot in mm. (x1,x2,y1,y2)")] = (-1,-1,-1,-1)
):
    specimen = Specimen.get(specimen_name)
    section('Estimating parameters...')
    intensity, rhc,acceptance = specimen.calculate_break_params(force_recalc=True)
    info(f'Fracture intensity: {intensity:.2f} 1/mm²')
    info(f'HC Radius: {rhc:.2f} mm')
    info(f'Acceptance: {acceptance:.2e}')
    info(f'Uniformity: {delta_hcp(intensity)/rhc:.2f}')

    def Kpois(d):
        # see Baddeley et al. S.206 K_pois
        return np.pi*d**2
    def Lpois(d):
        # see Baddeley et al. S.206 K_pois
        return d

    section('Plotting kfunc...')
    x,y = specimen.kfun()
    fig, ax = plt.subplots(figsize=get_fig_width(plotsz))
    ax.plot(x,y, label='$\hat{K}$') # , marker='x' # to display the points
    if with_poisson:
        ax.plot(x,Kpois(np.asarray(x)), label='$\hat{K}_{t}$')
    if legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend()
    ax.set_ylabel('$\hat{K}(d)$')
    ax.set_xlabel('$d$ (mm)')
    State.output(fig,'kfunc', spec=specimen, figwidth=plotsz, open=State.debug)


    section('Plotting lfunc...')
    x2,y2 = specimen.lfun()
    min_L = rhc

    ax: Axes
    fig, ax = plt.subplots(figsize=get_fig_width(plotsz))
    ax.plot(x2,y2, label='$\hat{L}$')
    ax.axvline(rhc, linestyle='--', color='r', label='$r_{{hc}}$')
    if with_poisson:
        ax.plot(x2,Lpois(np.asarray(x)), label='$\hat{L}_{t}$')
    if legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend()
    annotate_corner(ax, f'$r_{{hc}}={rhc:.2f}$')
    ax.set_ylabel('$\hat{L}(d)$')
    ax.set_xlabel('$d$ (mm)')
    State.output(fig, 'lfunc', spec=specimen, figwidth=plotsz, open=State.debug)

    section('Plotting centered lfunc...')
    x2,y2 = specimen.lcfun()
    min_L = rhc
    ax: Axes
    fig, ax = plt.subplots(figsize=get_fig_width(plotsz))
    ax.plot(x2,y2, label='$\hat{L}-d$')
    if with_poisson:
        ax.plot(x2,x2-x2, label='$\hat{L}_{t}-d$')
    ax.axvline(rhc, linestyle='--', color='r', label='$r_{{hc}}$')
    if legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend()
    annotate_corner(ax, f'$r_{{hc}}={rhc:.2f}$')
    ax.set_ylabel('$\hat{L}(d)-d$')
    ax.set_xlabel('$d$ (mm)')
    State.output(fig, 'lcfunc', spec=specimen, figwidth=plotsz, open=State.debug)

    section('Creating voronoi and plotting points...')
    # create actual voronoi plots
    size = specimen.get_real_size()
    area = size[0] * size[1]
    n_points = int(intensity * area)
    acceptance = acceptance
    points = csstraussproc2(size[0], size[1], rhc, n_points, acceptance, int(1e6))
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    axs.scatter(*zip(*points), s=1)
    if State.debug:
        plt.show()
    axs.set_xlabel('x (mm)')
    axs.set_ylabel('y (mm)')
    renew_ticks_ax(axs, (0, size[1]), (0, size[0]), 0)
    axs.grid(False)
    State.output(fig, 'points', spec=specimen, figwidth=sz, open=State.debug)


    # create voronoi of points
    voronoi = Voronoi(points)


    section('Transforming voronoi...')
    voronoi_img = np.full((int(size[1]), int(size[0])), 0, dtype=np.uint8)
    if not is_gray(voronoi_img):
        voronoi_img = cv2.cvtColor(voronoi_img, cv2.COLOR_BGR2GRAY)
    voronoi_to_image(voronoi_img, voronoi)
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    axs.imshow(255-voronoi_img, cmap='gray')
    axs.set_xlabel('x (mm)')
    axs.set_ylabel('y (mm)')
    axs.grid(False)


    State.output(fig, 'voronoi', spec=specimen, open=State.debug, figwidth=sz)

    section('Digitalize splinters...')
    splinters = Splinter.analyze_contour_image(voronoi_img, px_per_mm=1)
    info(f'Found {len(splinters)} splinters.')
    if State.debug:
        imgtest = np.zeros((int(size[1]), int(size[0]), 3), dtype=np.uint8)
        for s in splinters:
            clr = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            cv2.drawContours(imgtest, [s.contour], -1, clr, -1)

        plt.imshow(imgtest)
        plt.title('Detected Splinters in Voronoi')
        plt.show()

    ###
    # VORONOI

    section('Plotting real voronoi of specimen...')
    # create plot of voronoi that is generated from real splinters
    centroids = np.asarray([s.centroid_px for s in specimen.splinters])
    f = specimen.calculate_px_per_mm()
    voronoi = Voronoi(centroids)
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    base_img = specimen.get_fracture_image()
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    voronoi_to_image(base_img, voronoi, color=(255,0,0), thickness=int(f))
    for i, p in enumerate(voronoi.points):
        cv2.circle(base_img, (int(p[0]), int(p[1])), int(f/1.5), (0,0,255), -1)

    def millimeter_formatter(x, pos):
        return f'{x / f:.0f}'
    axs.xaxis.set_major_formatter(ticker.FuncFormatter(millimeter_formatter))
    axs.yaxis.set_major_formatter(ticker.FuncFormatter(millimeter_formatter))



    # base_img = cv2.resize(base_img, (int(size[0]), int(size[1])))
    axs.imshow(base_img)
    axs.grid(False)

    if not any([x == -1 for x in region]):
        region = np.asarray(region) * f
        axs.set_xlim(region[0], region[1])
        axs.set_ylim(region[2], region[3])
    axs.set_xlabel('x (mm)')
    axs.set_ylabel('y (mm)')
    State.output(fig, 'voronoi_overlay', spec=specimen, figwidth=sz)

    section('Print original image in a plot')
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    base_img = specimen.get_fracture_image()
    axs.imshow(base_img)
    axs.set_xlabel('x (mm)')
    axs.set_ylabel('y (mm)')
    axs.xaxis.set_major_formatter(ticker.FuncFormatter(millimeter_formatter))
    axs.yaxis.set_major_formatter(ticker.FuncFormatter(millimeter_formatter))
    axs.grid(False)
    State.output(fig, 'original', spec=specimen, figwidth=sz)

    section('Comparing break to splinters')
    # create pdf
    areas_original = [s.area for s in specimen.splinters]
    areas_voronoi =[s.area for s in splinters]

    fig,axs = datahist_plot(figwidth=plotsz)
    binrange = get_log_range(areas_original, 30)
    datahist_to_ax(axs, areas_original, binrange=binrange, label='Probekörper')
    datahist_to_ax(axs, areas_voronoi, binrange=binrange, label='Voronoi')
    axs[0].legend()
    State.output(fig, 'compare_pdf', spec=specimen, figwidth=plotsz)

    # create cdf
    fig,axs = datahist_plot(figwidth=plotsz)
    datahist_to_ax(axs, areas_original, binrange=binrange, label='Probekörper', data_mode=DataHistMode.CDF)
    datahist_to_ax(axs, areas_voronoi, binrange=binrange, label='Voronoi', data_mode=DataHistMode.CDF)
    axs[0].legend()
    State.output(fig, 'compare_cdf', spec=specimen, figwidth=plotsz)



    return splinters


@sim_app.command()
def lbreak(
    sigma_s: float,
    thickness: float,
    size: tuple[float,float] = (500,500),
    boundary: SpecimenBoundary = SpecimenBoundary.A,
    break_pos: SpecimenBreakPosition = SpecimenBreakPosition.CORNER,
    E: float = 70e3,
    nue: float = 0.23,
    impact_position: tuple[float,float] = (-1,-1)
):
    """
    Simulate a fracture morphology using the given parameters.

    Args:
        sigma_s (float): Surface compressive stress [MPa].
        thickness (float): Thickness of the specimen [mm].
        size (tuple[float,float], optional): Real dimensions of the specimen. Defaults to (500,500).
        break_pos (SpecimenBreakPosition, optional): The position, the specimen was broken in. Defaults to SpecimenBreakPosition.CORNER.

    Returns:
        None: This function creates data that is saved in its output folder.
    """
    assert has_layer(SplinterProp.INTENSITY, boundary, thickness, break_pos, False), f'Intensity layer not found for {boundary} {thickness} {break_pos}'
    assert has_layer(SplinterProp.RHC, boundary, thickness, break_pos, False), f'RHC layer not found for {boundary} {thickness} {break_pos}'
    assert has_layer(SplinterProp.ORIENTATION, boundary, thickness, break_pos, False), f'RHC layer not found for {boundary} {thickness} {break_pos}'

    if impact_position[0] == -1 or impact_position[1] == -1:
        impact_position = None

    # this will create the
    sim = Simulation.create(thickness, sigma_s, boundary, None)


    # calculate energy
    energy_u = U(sigma_s, thickness)
    # load layer for rhc and intensity
    intensity, intensity_std = interp_layer(SplinterProp.INTENSITY, boundary, thickness, break_pos,energy_u)
    rhc, rhc_std = interp_layer(SplinterProp.RHC, boundary, thickness, break_pos, energy_u)
    acc, acc_std = interp_layer(SplinterProp.ACCEPTANCE, boundary, thickness, break_pos, energy_u)
    # create radii
    r_range, t_range = arrange_regions(break_pos=break_pos, w_mm=size[0], h_mm=size[1])
    info('Radius range', r_range)

    fracture_intensity = np.mean(intensity(r_range))
    hc_radius = np.mean(rhc(r_range))
    fint_std = np.mean(intensity_std(r_range))
    rhc_stddev = np.mean(rhc_std(r_range))

    fracture_intensity = meanrand(fracture_intensity, fint_std)
    hc_radius = meanrand(hc_radius, rhc_stddev)

    # fetch fracture intensity and hc radius from energy
    mean_area = 1 / fracture_intensity
    if impact_position is None:
        impact_position = break_pos.default_position()
    area = size[0] * size[1]
    c = np.mean(acc(r_range))

    info(f'Energy:             {energy_u:<15.2f} J/m²')
    info(f'Fracture intensity: {fracture_intensity:<15.4f} 1/mm²')
    info(f'HC Radius:          {hc_radius:<15.2f} mm')
    info(f'Mean area:          {mean_area:<15.2f} mm²')
    info(f'Impact position:    {impact_position}')
    info(f'Area:               {area:<15.2f} mm²')
    info(f'Real size:          {size[0]:<15.2f} x {size[1]:<15.2f} mm²')
    # estimate acceptance probability
    urr = relative_remaining_stress(mean_area, thickness)
    nue = 1 - urr
    info(f'Urr: {urr:.2f}', f'nue: {nue:.2f}')

    # create spatial points
    # points = spazial_gibbs_strauss_process(fracture_intensity, hc_radius, 0.55, size)
    # points = csstraussproc(size, hc_radius, int(fracture_intensity*area), c, int(1e6))
    # points = csstraussproc2(size[0], size[1], hc_radius, int(fracture_intensity*area), c, int(1e6))

    section("Spatial point distribution...")
    # interpolate rhc values
    rhc_values = rhc(r_range)
    # create array with r_range in column 0 and rhc_values in 1
    rhc_array = np.column_stack((r_range, rhc_values))
    # # start point process with rhc values
    # points = csstraussproc_rhciter(size[0], size[1], rhc_array, impact_position, int(fracture_intensity*area), c, int(1e6))

    l_values = intensity(r_range)
    l_array = np.column_stack((r_range, l_values))
    points = bohmann_process(size[0], size[1], r_range, l_array, rhc_array, impact_position, c, int(1e6), False)

    # print region
    center_point = (0.5,0.5) # in percent
    region_size  = (200,200) # in mm

    section("Plotting points...")
    x,y = zip(*points)
    # plot points
    fig,axs = plt.subplots()
    axs.scatter(x,y)
    # find maximum region from points
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    w = x_max - x_min
    h = y_max - y_min

    x_min = w * center_point[0] - region_size[0]/2
    x_max = w * center_point[0] + region_size[0]/2
    y_min = h * center_point[1] - region_size[1]/2
    y_max = h * center_point[1] + region_size[1]/2



    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)

    if State.debug:
        plt.show()
    axs.set_xlabel('x (mm)')
    axs.set_ylabel('y (mm)')
    axs.set_aspect('equal', 'box')
    fig.savefig(sim.get_file('pointsfig.png'))


    # scaling factor (realsize > pixel size)
    size_f = 20
    info('Scaling factor: ', size_f)
    region_scaled = (x_min, x_max, y_min, y_max)
    # create output image store
    markers = np.zeros((int(size[1]*size_f),int(size[0]*size_f)), dtype=np.uint8)
    for point in points:
        markers[int(point[1]*size_f), int(point[0]*size_f)] = 255

    # clip region from markers
    markers_clipped = cropimg(region_scaled, size_f, markers)
    cv2.imwrite(sim.get_file('points.png'), markers_clipped)

    section("Loading layers...")
    # apply layers to points
    # modification 1: impact layer
    il_orientation, il_orientation_stddev = interp_layer(
        SplinterProp.ORIENTATION,
        boundary,
        thickness,
        SpecimenBreakPosition.CORNER,
        energy_u
    )

    il_l1, il_l1_stddev = interp_layer(
        SplinterProp.L1,
        boundary,
        thickness,
        SpecimenBreakPosition.CORNER,
        energy_u
    )

    il_l2, il_l2_stddev = interp_layer(
        SplinterProp.L2,
        boundary,
        thickness,
        SpecimenBreakPosition.CORNER,
        energy_u
    )
    il_l1l2, il_l1l2_stddev = interp_layer(
        SplinterProp.ASP0,
        boundary,
        thickness,
        SpecimenBreakPosition.CORNER,
        energy_u
    )

    layers = {
        'orientation': (il_orientation, il_orientation_stddev, Splinter.get_property_label(SplinterProp.ORIENTATION), 'orientation'),
        'l1': (il_l1, il_l1_stddev, Splinter.get_property_label(SplinterProp.L1), 'l1'),
        'l2': (il_l2, il_l2_stddev, Splinter.get_property_label(SplinterProp.L2), 'l2'),
        'l1l2': (il_l1l2, il_l1l2_stddev, Splinter.get_property_label(SplinterProp.ASP0), 'l1l2'),
        'intensity': (intensity, intensity_std, Splinter.get_property_label(SplinterProp.INTENSITY), 'intensity'),
        'rhc': (rhc, rhc_std, Splinter.get_property_label(SplinterProp.RHC), 'rhc'),
    }

    section("Plotting layers...")
    track
    for layer in tracker(layers):
        il, il_stddev, mode_labels, name = layers[layer]
        r_range = np.linspace(0, np.sqrt(450**2+450**2), 100)
        fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW3))
        r_il = il(r_range)
        r_il_stddev = il_stddev(r_range)
        axs.plot(r_range, r_il)
        # add error bars to plot
        axs.fill_between(r_range, r_il-r_il_stddev/2, r_il+r_il_stddev/2, alpha=0.3)
        axs.set_ylabel(mode_labels)
        if State.debug:
            plt.show()
        fig.savefig(sim.get_file(f'{name}.pdf'))

    section("Modify spatial points...")
    exceptions = []
    with get_progress() as progress:
        progress.set_description('Creating spatial points')
        progress.set_total(len(points))
        # iterate points
        for p in points:
            progress.advance()
            # calculate distance to impact position
            r = np.linalg.norm(p-impact_position)
            # calculate orientation
            orientation = il_orientation(r)
            orientation_stddev = il_orientation_stddev(r)
            orientation = meanrand(orientation,orientation_stddev)

            # calculate l1
            l1 = il_l1(r)
            l1_stddev = il_l1_stddev(r)
            l1 = meanrand(l1,l1_stddev)

            # calculate l1
            l2 = il_l2(r)
            l2_stddev = il_l2_stddev(r)
            l2 = meanrand(l2,l2_stddev)


            # calculate aspect ratio
            l1l2 = il_l1l2(r)
            l1l2_stddev = il_l1l2_stddev(r)
            l1l2 = meanrand(l1l2,l1l2_stddev)

            ## calculate major axis vector using orientation and its deviation
            # create major axis vector
            v0 = (p-impact_position)/r
            # calculate angle from current point to impact position
            angle0 = angle_between(v0, np.asarray((1,0)))
            angle = float(meanrand(angle0, (1-orientation) * np.pi/2))
            # print(angle)
            # angle = int(angle)

            # rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # v1 = np.dot(rotation_matrix, v0)

            ## calculate length of major axis using l1 and its deviation
            # calculate length of major axis (this is radius, so divide by 2)
            l_major = l1 / 2 # this will be filled from the algorithm
            l_minor = l2 / 2 # before: l_major / l1l2

            # make dimensions smaller so that the algorithm can fill it
            l_major = l_major
            l_minor = l_minor

            ## modify the point using the major axis
            markers = cv2.ellipse(
                    markers,
                    (int(p[1]*size_f), int(p[0]*size_f)), # location
                    # (int(2), int(2)), # axes lengths
                    # (int(2), int(l_major * size_f)), # axes lengths
                    (int(l_minor * size_f), int(l_major * size_f)), # axes lengths
                    np.rad2deg(angle), # angle
                    0, 360, # start and end angle
                    255, # color
                    -1, # thickness
                    cv2.LINE_8 # line type
                )

            markers = cv2.ellipse(
                    markers,
                    (int(p[1]*size_f), int(p[0]*size_f)), # location
                    # (int(2), int(2)), # axes lengths
                    # (int(2), int(l_major * size_f)), # axes lengths
                    (int(l_minor * size_f), int(l_major * size_f)), # axes lengths
                    np.rad2deg(angle), # angle
                    0, 360, # start and end angle
                    0, # color
                    size_f, # thickness
                    cv2.LINE_8 # line type
                )
            # try:
            #     markers = cv2.ellipse(
            #         markers,
            #         (int(p[1]*size_f), int(p[0]*size_f)), # location
            #         # (int(1), int(l_major * size_f / 2)), # axes lengths
            #         (int(l_minor * size_f / 2), int(l_major * size_f / 2)), # axes lengths
            #         np.rad2deg(angle)-180, # angle
            #         0, 360, # start and end angle
            #         255, # color
            #         -1 # thickness
            #     )
            # except Exception as e:
            #     print(f'Point: {p}, r: {r}, angle: {np.rad2deg(angle)}, l_major: {l_major}, l_minor: {l_minor}')
            #     exceptions.append(e)

    if len(exceptions) > 0:
        print(f'Exceptions: {[e for e in exceptions]}')

    # cv2.ellipse(markers, impact_position.astype(np.uint8)*size_f, (int(20), int(5)), 0, 0, 360, (0,255,0), 1)
    # cv2.ellipse(markers, (400*size_f,400*size_f), (int(5), int(20)), 0, 0, 360, (0,255,0), -1)
    # cv2.ellipse(markers, (200*size_f,400*size_f), (int(5), int(20)), 0, 0, 360, (255,120,0), -1)
    plotImage(markers, 'markers')
    markers_clipped = cropimg(region_scaled, size_f, markers)
    cv2.imwrite(sim.get_file('layered_points.png'), markers_clipped)


    # remove small regions
    detected_contours = cv2.findContours(to_gray(markers), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in detected_contours[0] if cv2.contourArea(c) < size_f]
    info(f'Removing {len(contours)} small regions, smaller than {size_f} px².')
    cv2.drawContours(markers, contours, -1, (0,0,0), -1)
    plotImage(markers, 'markers after removing small regions')

    section("Watershed")
    markers = cv2.connectedComponents(np.uint8(markers))[1]
    shape = (int(size[1]*size_f),int(size[0]*size_f),3)
    blank_image = np.zeros(shape, dtype=np.uint8)
    markers = cv2.watershed(blank_image, markers)

    m_img = np.zeros(shape, dtype=np.uint8)
    m_img[markers == -1] = 255
    m_img_clipped = cropimg(region_scaled, size_f, m_img)
    cv2.imwrite(sim.get_file('watershed_0.png'), m_img_clipped)


    section('Digitalizing image')
    black_white_img = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    splinters = Splinter.analyze_contour_image(m_img, px_per_mm=size_f)

    out_img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for s in splinters:
        clr = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.drawContours(out_img, [s.contour], -1, clr, -1)
        cv2.drawContours(black_white_img, [s.contour], -1, 255, 1)

    if State.debug:
        plt.imshow(out_img)
        plt.show()

        plt.imshow(black_white_img)
        plt.show()

    State.output(black_white_img, f'generated_{sigma_s}_{thickness}', spec=None, figwidth=FigureSize.ROW2, open=State.debug)
    cv2.imwrite(sim.get_file('filled.png'), out_img)
    cv2.imwrite(sim.get_file('contours.png'), 255-black_white_img)

    section('Saving...')
    sim.put_splinters(splinters)
    info(f'Simulation created: {sim.name}_{sim.nbr}')
    return sim

def cropimg(region, size_f, markers):
    x_min, x_max, y_min, y_max = region
    markers_clipped = markers[int(x_min*size_f):int(x_max*size_f), int(y_min*size_f):int(y_max*size_f)]
    return markers_clipped

@sim_app.command()
def lbreak_like(
    name: str,
    sigma_s: float = None,
    thickness: float = None,
    boundary: SpecimenBoundary = None,
    break_pos: SpecimenBreakPosition = None
):

    specimen = Specimen.get(name)

    if sigma_s is None:
        sigma_s = np.abs(specimen.sig_h)
    if thickness is None:
        thickness = specimen.thickness
    size = specimen.get_real_size()
    if boundary is None:
        boundary = specimen.boundary
    if break_pos is None:
        break_pos = specimen.break_pos
    E = 70e3
    nue = 0.23

    # create simulation
    simulation = lbreak(sigma_s, thickness, size, boundary, break_pos, E, nue, impact_position=specimen.get_impact_position())
    # compare simulation with input
    compare(simulation.fullname, specimen.name)

    return simulation

@sim_app.command()
def compare_all(name):
    specimen = Specimen.get(name)

    if not specimen.has_scalp:
        critical('[red]Specimen has not been scalped.')
        return

    info(f'Comparing specimen {specimen.name} with nbreak and lbreak...')

    sigma_s = np.abs(specimen.sig_h)
    thickness = specimen.thickness
    size = specimen.get_real_size()
    boundary = specimen.boundary
    break_pos = specimen.break_pos
    E = 70e3
    nue = 0.23

    # create simulation
    simulation = lbreak(sigma_s, thickness, size, boundary, break_pos, E, nue)

    voronoi = nbreak(specimen.name)
    # compare simulation with input
    compare(simulation.fullname, specimen.name, voronoi)

@sim_app.command()
def compare(
    simulation_name: Annotated[str, typer.Argument(help="Simulation name.")],
    specimen_name: Annotated[str, typer.Argument(help="Specimen name.")],
    voronoi = None
):
    """
    Compare a simulation with a real specimen.

    Args:
        simulation_name (str): Name of the simulation.
        specimen_name (str):
    """
    sim = Simulation.get(simulation_name)
    specimen = Specimen.get(specimen_name)

    max_area = np.min([np.max([s.area for s in sim.splinters]), np.max([s.area for s in specimen.splinters])])

    # create histogram of simulation
    sim_splinters = sim.splinters
    sim_areas = [s.area for s in sim_splinters]
    sim_hist, sim_bins = np.histogram(sim_areas, bins=30, density=True, range=(0, max_area))

    # create histogram of specimen
    spec_splinters = specimen.splinters
    spec_areas = [s.area for s in spec_splinters]

    # plot histograms
    fig,axs = datahist_plot(figwidth=FigureSize.ROW2)
    binrange = get_log_range(spec_areas, 30)

    if voronoi is not None:
        vor_splinters = voronoi
        vor_areas = [s.area for s in vor_splinters]
        datahist_to_ax(axs, vor_areas, binrange=binrange, label='Voronoi')
    datahist_to_ax(axs, spec_areas, binrange=binrange, label='Probekörper')
    datahist_to_ax(axs, sim_areas, binrange=binrange, label='Simulation')


    axs[0].legend()
    State.output(fig, f'{specimen.name}--{sim.name}_{sim.nbr}', figwidth=FigureSize.ROW2)

    info('Specimen: ', len(spec_splinters))
    info('Simulation: ', len(sim_splinters))


    # data_list = [
    #         (U(sim.nom_stress,sim.thickness), 'Simulation', sim_areas),
    #         (specimen.U, 'Probekörper', spec_areas),
    #     ]
    # if voronoi is not None:
    #     data_list.append((U(sim.nom_stress,sim.thickness), 'Voronoi', vor_areas))

    # fig2d = datahist_2d(
    #     data_list,
    #     30,
    #     False
    # )

    # State.output(fig2d, f'{specimen.name}--{sim.fullname}--voronoi', figwidth=FigureSize.ROW2)

# @sim_app.command()
# def create_spatial():
#     area = (500,500)
#     intensity = 0.1
#     n_points = 500
#     image = np.zeros((area[0],area[1],3), dtype=np.uint8)

#     points = gibbs_strauss_process(n_points, 20, intensity, area=area)
#     # points = CSR_process(10, area[0])
#     x,y = zip(*points)

#     plt.figure()
#     plt.scatter(x,y)
#     plt.show()

#     # perform watershed on the points
#     markers = np.zeros(area, dtype=np.uint8)
#     for point in points:
#         markers[int(point[0]), int(point[1])] = 255

#     markers = dilateImg(markers, 5)
#     markers_rgb = to_rgb(markers)
#     # from top left elongate markers
#     p0 = np.asarray((area[0]*0.2,area[1]*0.2))
#     max_elong = 20
#     elong_d = 2 #px
#     ds = np.linspace(0, np.sqrt(area[0]**2+area[1]**2), 100)

#     dmax = np.max(ds)
#     done = []
#     print(ds)
#     for d in track(ds):
#         fd = 1-d/dmax
#         print(fd)

#         for i,p in enumerate(points):
#             if i in done:
#                 continue

#             p = np.asarray(p)

#             # get vector from p0 to p
#             v = p-p0
#             # normalize
#             dp = np.linalg.norm(v)

#             if dp > d:
#                 continue

#             done.append(i)

#             v = v/dp
#             for j in np.linspace(0, max_elong*fd, int(max_elong*fd/elong_d)*5):
#                 # calculate new point from p with v0 and elong_d distance
#                 new_point = p+v*j
#                 # and other side as well
#                 new_point2 = p-v*j
#                 # print(p)
#                 # print(v)
#                 # print(new_point)
#                 # print(new_point2)
#                 # input("")
#                 # add filled circle to both new points
#                 # cv2.circle(markers_rgb, (int(p[1]), int(p[0])), elong_d, (0,255,255), -1)
#                 # cv2.circle(markers_rgb, (int(new_point[1]), int(new_point[0])), elong_d, (255,0,0), -1)
#                 cv2.circle(markers, (int(new_point[1]), int(new_point[0])), elong_d, 255, -1)
#                 cv2.circle(markers, (int(new_point2[1]), int(new_point2[0])), elong_d, 255, -1)



#     plotImage(markers, 'elongated markers', force=True)

#     markers = cv2.connectedComponents(np.uint8(markers))[1]

#     markers = cv2.watershed(np.zeros_like(image), markers)

#     m_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#     m_img[markers == -1] = 255

#     splinters = Splinter.analyze_contour_image(m_img)

#     out_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
#     for s in splinters:
#         clr = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
#         cv2.drawContours(out_img, [s.contour], -1, clr, -1)

#     plt.imshow(out_img)
#     plt.show()

# @sim_app.command()
# def compare_spatial(specimen_name):
#     specimen = Specimen.get(specimen_name)


#     size = 300
#     x1 = 500
#     x2 = x1 + size
#     y1 = 500
#     y2 = y1 + size

#     # choose a region to replicate
#     region = RectRegion(x1,y1,x2,y2)

#     # get region from specimen fracture image
#     splinters_in_region = specimen.get_splinters_in_region(region)
#     frac_img = specimen.get_fracture_image()

#     image = np.zeros((size,size,3), dtype=np.uint8)

#     orig_contours = to_rgb(np.zeros_like(frac_img, dtype=np.uint8))

#     # perform watershed on the points
#     markers = np.zeros((size,size), dtype=np.uint8)
#     for s in splinters_in_region:
#         point = s.centroid_px
#         ix = np.max([np.min([int(point[0])-x1, size-1]), 0])
#         iy = np.max([np.min([int(point[1])-y1, size-1]), 0])
#         markers[ix,iy] = 255

#         cv2.drawContours(orig_contours, [s.contour], -1, (0,0,255), 2)

#     orig_contours = orig_contours[x1:x2, y1:y2,:]

#     markers = dilateImg(markers, 3, it=3)
#     plt.imshow( markers)
#     plt.show()

#     markers = cv2.connectedComponents(np.uint8(markers))[1]

#     markers = cv2.watershed(np.zeros_like(image), markers)

#     m_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#     m_img[markers == -1] = 255

#     splinters = Splinter.analyze_contour_image(m_img)

#     gen_contours = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
#     for s in splinters:
#         clr = (255,0,0)
#         cv2.drawContours(gen_contours, [s.contour], -1, clr, 1)

#     print(orig_contours.shape)
#     print(gen_contours.shape)
#     comparison = cv2.addWeighted(orig_contours, 0.5, gen_contours, 0.5, 0)


#     plt.imshow(comparison)
#     plt.show()
# endregion

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
