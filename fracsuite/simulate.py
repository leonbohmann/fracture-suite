"""
Commands for simulating and analyzing fracture morphologies.
"""
import json
import shutil
from turtle import color
from fracsuite.core.logging import critical, info
import random
import os
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
from fracsuite.core.image import is_gray, to_gray
from fracsuite.core.imageplotting import plotImage
from fracsuite.core.mechanics import U
from fracsuite.core.model_layers import arrange_regions, has_layer, interp_layer, region_sizes
from fracsuite.core.outputtable import NumpyEncoder
from fracsuite.core.plotting import DataHistMode, DataHistPlotMode, FigureSize, annotate_corner, datahist_plot, datahist_to_ax, get_fig_width, get_log_range, legend_without_duplicate_labels, renew_ticks_ax, voronoi_to_image
from fracsuite.core.point_process import gibbs_strauss_process
from fracsuite.core.progress import get_progress, tracker
from fracsuite.core.simulation import Simulation
from fracsuite.core.specimen import Specimen, SpecimenBoundary, SpecimenBreakPosition
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stochastics import calculate_dmode, calculate_dmodei, data_mse
from fracsuite.core.stress import relative_remaining_stress
from scipy.spatial import Voronoi, voronoi_plot_2d
from fracsuite.core.vectors import angle_between
from fracsuite.general import GeneralSettings

from fracsuite.state import State

from spazial import csstraussproc2, bohmann_process
import matplotlib.ticker as ticker

general = GeneralSettings.get()

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
    d = 2.0*(rng.random() - 0.5)
    return mean + d * stddev

def image_to_fig(image, ret_ax=False, figwidth=FigureSize.ROW3, unit='mm'):
    fig,axs = plt.subplots(figsize=get_fig_width(figwidth))
    axs.imshow(image, cmap='gray')
    axs.set_xlabel(f'x ({unit})')
    axs.set_ylabel(f'y ({unit})')
    axs.grid(False)

    if ret_ax:
        return fig, axs

    return fig

@sim_app.command()
def nbreak(
    specimen_name: Annotated[str, typer.Argument(help="Specimen name.")],
    sz: Annotated[FigureSize, typer.Option(help="Figure size for spatial plots.")] = FigureSize.ROW2,
    plotsz: Annotated[FigureSize, typer.Option(help="Figure size for graph plots.")] = FigureSize.ROW2,
    with_poisson: Annotated[bool, typer.Option(help="Plot poisson's distribution functions.")] = False,
    legend_outside: Annotated[bool, typer.Option(help="Place legend outside of plot.")] = False,
    region: Annotated[tuple[float,float,float,float], typer.Option(help="Region to plot in mm. (x1,x2,y1,y2)")] = (-1,-1,-1,-1),
    force_recalc: Annotated[bool, typer.Option(help="Force recalculation of parameters.")] = True,
    no_plot_creation: Annotated[bool, typer.Option(help="Do not create plots.")] = False
):
    specimen = Specimen.get(specimen_name)
    section('Estimating parameters...')
    intensity, rhc,acceptance = specimen.calculate_break_params(force_recalc=force_recalc)
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

    if not no_plot_creation:
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
    if not no_plot_creation:
        fig,axs = plt.subplots(figsize=get_fig_width(sz))
        axs.scatter(*zip(*points), s=1)
        if State.debug:
            plt.show()
        axs.set_xlabel('x (mm)')
        axs.set_ylabel('y (mm)')
        renew_ticks_ax(axs, (0, size[1]), (0, size[0]), 0)
        axs.grid(False)
        State.output(fig, 'points', spec=specimen, figwidth=sz, open=State.debug)

    voronoi_scale = 5
    points = np.asarray(points) * voronoi_scale
    # create voronoi of points
    voronoi = Voronoi(points)


    section('Transforming voronoi...')
    voronoi_img = np.full((int(size[1]*voronoi_scale), int(size[0]*voronoi_scale)), 0, dtype=np.uint8)
    if not is_gray(voronoi_img):
        voronoi_img = cv2.cvtColor(voronoi_img, cv2.COLOR_BGR2GRAY)
    voronoi_to_image(voronoi_img, voronoi)
    if not no_plot_creation:
        fig,axs = plt.subplots(figsize=get_fig_width(sz))
        axs.imshow(255-voronoi_img, cmap='gray')
        axs.set_xlabel('x (mm)')
        axs.set_ylabel('y (mm)')
        axs.grid(False)
        State.output(fig, 'voronoi', spec=specimen, open=State.debug, figwidth=sz)

    section('Digitalize splinters...')
    splinters = Splinter.analyze_contour_image(voronoi_img, px_per_mm=voronoi_scale)
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
    if not no_plot_creation:
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
    if not no_plot_creation:
        fig,axs = plt.subplots(figsize=get_fig_width(sz))
        base_img = specimen.get_fracture_image()
        axs.imshow(base_img)
        axs.set_xlabel('x (mm)')
        axs.set_ylabel('y (mm)')
        axs.xaxis.set_major_formatter(ticker.FuncFormatter(millimeter_formatter))
        axs.yaxis.set_major_formatter(ticker.FuncFormatter(millimeter_formatter))
        axs.grid(False)
        State.output(fig, 'original', spec=specimen, figwidth=sz)

    if not no_plot_creation:
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
def alfa(
    sigma_s: float,
    thickness: float,
    size: tuple[float,float] = (500,500),
    boundary: SpecimenBoundary = SpecimenBoundary.A,
    break_pos: SpecimenBreakPosition = SpecimenBreakPosition.CORNER,
    E: float = 70e3,
    nue: float = 0.23,
    impact_position: tuple[float,float] = (-1,-1),
    no_region_crop: bool = False,
    reference: str = None
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

    # this will create the simulation
    sim = Simulation.create(thickness, sigma_s, boundary, None)

    pointsz = FigureSize.ROW3 if 'override_figwidth' not in State.kwargs else State.kwargs['override_figwidth']

    # calculate energy
    energy_u = U(sigma_s, thickness)
    # load layer for rhc and intensity
    intensity, intensity_std = interp_layer(SplinterProp.INTENSITY, boundary, thickness, break_pos,energy_u)
    rhc, rhc_std = interp_layer(SplinterProp.RHC, boundary, thickness, break_pos, energy_u)
    acc, acc_std = interp_layer(SplinterProp.ACCEPTANCE, boundary, thickness, break_pos, energy_u)
    # create radii
    r_range, t_range = arrange_regions(d_r_mm=40, break_pos=break_pos, w_mm=size[0], h_mm=size[1])
    info('Radius range', r_range)
    info('Theta range', t_range)

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
    info(f'Acceptance:         {c:<15.4f}')
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
    r_areas = region_sizes(r_range, impact_position, size)

    points = bohmann_process(size[0], size[1], r_range, r_areas, l_array, rhc_array, impact_position, c, int(1e6), False)

    # print region
    center_point = (0.3,0.3) # in percent
    region_size  = (50,50) # in mm

    if no_region_crop:
        center_point = (0.5,0.5)
        region_size = (size[0], size[1])

    section("Plotting points...")
    x,y = zip(*points)
    # plot points
    fig,axs = plt.subplots(figsize=get_fig_width(pointsz))
    axs.scatter(x,y, s=0.01 if no_region_crop else 1)
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

    # make 0,0 top left
    axs.invert_yaxis()
    axs.set_xlabel('x (mm)')
    axs.set_ylabel('y (mm)')
    axs.set_aspect('equal', 'box')
    fig.savefig(sim.get_file('points_original.pdf'))


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
    cv2.imwrite(sim.get_file('markers.png'), markers_clipped)

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
        'orientation': (il_orientation, il_orientation_stddev, Splinter.get_property_label(SplinterProp.ORIENTATION, row3=True), 'orientation'),
        'l1': (il_l1, il_l1_stddev, Splinter.get_property_label(SplinterProp.L1, row3=True), 'l1'),
        'l2': (il_l2, il_l2_stddev, Splinter.get_property_label(SplinterProp.L2, row3=True), 'l2'),
        'l1l2': (il_l1l2, il_l1l2_stddev, Splinter.get_property_label(SplinterProp.ASP0, row3=True), 'l1l2'),
        'intensity': (intensity, intensity_std, Splinter.get_property_label(SplinterProp.INTENSITY, row3=True), 'intensity'),
        'rhc': (rhc, rhc_std, Splinter.get_property_label(SplinterProp.RHC, row3=True), 'rhc'),
        'accprob': (acc, acc_std, Splinter.get_property_label(SplinterProp.ACCEPTANCE, row3=True), 'acc'),
    }

    section("Plotting layers...")
    for layer in tracker(layers):
        il, il_stddev, mode_labels, name = layers[layer]
        r_range = np.linspace(0, np.sqrt(450**2+450**2), 100)
        fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW3))
        r_il = il(r_range)
        r_il_stddev = il_stddev(r_range)
        axs.plot(r_range, r_il)
        # add error bars to plot
        axs.fill_between(r_range, r_il-r_il_stddev/2, r_il+r_il_stddev/2, alpha=0.3)
        axs.set_xlabel('R (mm)')
        axs.set_ylabel(mode_labels)
        if State.debug:
            plt.show()
        fig.savefig(sim.get_file(f'layer_{name}.pdf'))

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
            l_minor = l2 / 2

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
    fig = image_to_fig(markers_clipped, figwidth=pointsz, unit='px')
    cv2.imwrite(sim.get_file('points_modified.png'), markers_clipped)
    fig.savefig(sim.get_file('points_modified.pdf'))


    # remove small regions
    detected_contours = cv2.findContours(to_gray(markers), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in detected_contours[0] if cv2.contourArea(c) < size_f/2]
    info(f'Removing {len(contours)} small regions, smaller than {size_f/2:.2f} px².')
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
        cv2.drawContours(black_white_img, [s.contour], -1, 255, 3)

    if State.debug:
        plt.imshow(out_img)
        plt.show()

        plt.imshow(black_white_img)
        plt.show()

    State.output(black_white_img, f'generated_{sigma_s}_{thickness}', spec=None, figwidth=FigureSize.ROW2, open=State.debug)
    cv2.imwrite(sim.get_file('splinters_filled.png'), out_img)
    fig = image_to_fig(cropimg(region_scaled, size_f, out_img), figwidth=pointsz, unit='px')
    fig.savefig(sim.get_file('splinters_filled.pdf'))
    cv2.imwrite(sim.get_file('splinters_contours.png'), 255-black_white_img)
    fig = image_to_fig(cropimg(region_scaled, size_f, 255-black_white_img), figwidth=pointsz, unit='px')
    fig.savefig(sim.get_file('splinters_contours.pdf'))

    section('Saving...')
    sim.put_splinters(splinters)
    info(f'Simulation created: {sim.name}_{sim.nbr}')

    sim_areas = [s.area for s in splinters]
    spec_areas = [s.area for s in Specimen.get(reference).splinters]
    # sim_vs_spec = calculate_chi2(sim_areas, spec_areas)
    sim_vs_spec = data_mse(spec_areas, sim_areas)
    # sim_vs_spec_chi2 = data_chi2(sim_areas, spec_areas)
    info('MSE: ', sim_vs_spec)

    simulation_data = {
        'energy': energy_u,
        'fracture_intensity': fracture_intensity,
        'hc_radius': hc_radius,
        'acceptance': c,
        'impact_position': impact_position,
        'area': float(area),
        'size': tuple(size),
        'nue': nue,
        'sizef': size_f,
        'reference': reference if reference is not None else '',
        'reference_sigma': sigma_s,
        'reference_thickness': thickness,
        # 'chi2': sim_vs_spec_chi2,
        'mse': sim_vs_spec,
    }

    with open(sim.get_file('simulation.json'), 'w') as f:
        json.dump(simulation_data, f, indent=4 ,cls=NumpyEncoder)


    if 'output_to' in State.kwargs:
        shutil.copytree(sim.get_file(""), os.path.join(general.to_base_path, State.kwargs['output_to'], sim.name), dirs_exist_ok=True)


    return sim

def cropimg(region, size_f, markers):
    x_min, x_max, y_min, y_max = region
    markers_clipped = markers[int(x_min*size_f):int(x_max*size_f), int(y_min*size_f):int(y_max*size_f)]
    return markers_clipped

@sim_app.command()
def alfa_like(
    name: str,
    sigma_s: float = None,
    thickness: float = None,
    boundary: SpecimenBoundary = None,
    break_pos: SpecimenBreakPosition = None,
    no_region_crop: bool = False,
    validate: int = -1
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

    info(f'Creating simulation for {specimen.name}')
    info(f'> Sigma_s: {sigma_s}')
    info(f'> Thickness: {thickness} (real: {specimen.measured_thickness})')
    info(f'> Size: {size}')
    info(f'> Boundary: {boundary}')
    info(f'> Break position: {break_pos}')

    if validate == -1:
        # create simulation
        simulation = alfa(sigma_s, thickness, size, boundary, break_pos, E, nue, impact_position=specimen.get_impact_position(),
                            no_region_crop=no_region_crop, reference=specimen.name)
        # compare simulation with input
        compare(simulation.fullname, specimen.name)

        # put the original fracture image on a figure into the simulatio
        fig0,axs0 = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
        axs0.imshow(specimen.get_fracture_image())
        axs0.set_xlabel('x (px)')
        axs0.set_ylabel('y (px)')
        axs0.grid(False)
        fig0.savefig(simulation.get_file('original_fracture_image.pdf'))

        return simulation
    else:
        spec_areas = [s.area for s in specimen.splinters]
        binrange = get_log_range(spec_areas, 30)

        fig0,axs0 = None,None
        with plt.ion():
            fig0,axs0 = datahist_plot(figwidth=FigureSize.ROW1)
            datahist_to_ax(axs0, spec_areas, binrange=binrange, color='C0', data_mode=DataHistMode.CDF)
            fig0.canvas.draw()
            fig0.canvas.flush_events()

        for i in range(validate):
            sim = alfa(sigma_s, thickness, size, boundary, break_pos, E, nue, impact_position=specimen.get_impact_position(),
                            no_region_crop=no_region_crop, reference=specimen.name)
            areas = [s.area for s in sim.splinters]
            info(f'Validation {i+1}/{validate}: {len(sim.splinters)} splinters, mean area: {np.mean(areas):.2f} mm²')
            datahist_to_ax(axs0, areas, binrange=binrange, color='C1', data_mode=DataHistMode.CDF, linewidth=0.3)
            fig0.canvas.draw()
            fig0.canvas.flush_events()

        axs0[0].set_xlabel('Bruchstückflächeninhalt $A_\mathrm{S}$ (mm²)')
        axs0[0].set_ylabel('CDF')
        legend_without_duplicate_labels(axs0[0])
        State.output(fig0, f'validation_{specimen.name}', figwidth=FigureSize.ROW1)

@sim_app.command()
def compare_all(
    name,
    voronoi_count: int = 1,
    alfa_count: int = 1
):
    """
    Compares both the nbreak and alfa of a specimen.

    Args:
        name (str): Name of the specimen for which to create the comparison.
    """
    specimen = Specimen.get(name)

    if not specimen.has_scalp:
        critical('[red]Specimen has not been scalped.')
        return

    info(f'Comparing specimen {specimen.name} with nbreak and alfa...')

    sigma_s = np.abs(specimen.sig_h)
    thickness = specimen.thickness
    size = specimen.get_real_size()
    boundary = specimen.boundary
    break_pos = specimen.break_pos
    E = 70e3
    nue = 0.23

    specimen_areas = [s.area for s in specimen.splinters]

    ## ALFA
    # create alfa simulations
    simulations = []
    for i in range(alfa_count):
        State.console.rule(f'ALFA {i+1}/{alfa_count}', align='right')
        simulation = alfa(sigma_s, thickness, size, boundary, break_pos, E, nue, reference=name)
        simulations.append(simulation)

        plt.close('all')
    # transform areas to bins
    simulation_areas = []
    for sim in simulations:
        simulation_area = [s.area for s in sim.splinters]
        simulation_areas.append(simulation_area)

    ## VORONOI
    # create voronoi simulations
    voronois = []
    for i in range(voronoi_count):
        State.console.rule(f'Voronoi {i+1}/{voronoi_count}', align='right')
        voronoi = nbreak(specimen.name, force_recalc=False, no_plot_creation=True)
        voronois.append(voronoi)

        plt.close('all')
    # transform areas to bins
    voronoi_areas = []
    for vor in voronois:
        vor_areas = [s.area for s in vor]
        voronoi_areas.append(vor_areas)


    create_validation_plots(
        specimen_areas,
        simulation_areas,
        voronoi_areas,
        'lbreak--nbreak--real',
    )

def create_validation_plots(
    spec_areas: list[float],
    sim_areas: list[list[float]],
    vor_areas: list[list[float]],
    name: str,
    plot_all: bool = False,
):
    """
    Create validation plots for the given areas. The areas will be transformed into bins and then plotted as a CDF and PDF.

    Args:
        spec_areas (): Areas from the reference specimen.
        sim_areas (): Areas from the alfa simulations.
        vor_areas (): Areas from the voronoi simulations.
        name (str): Name of the output plots.
        plot_all (bool, optional): Not implemented yet. Defaults to False.
    """
    spec_areas = np.asarray(spec_areas)
    binrange = get_log_range(spec_areas, 30)
    spec_areas_bins = np.histogram(np.log10(spec_areas), bins=binrange, density=True)[0]

    sim_mpv = []
    vor_mpv = []
    sim_areas_bins = []
    vor_areas_bins = []
    # transform data into bins
    for i in range(len(sim_areas)):
        area_hist = np.histogram(np.log10(sim_areas[i]), bins=binrange, density=True)[0]
        sim_mpv.append(calculate_dmodei(sim_areas[i]))
        sim_areas_bins.append(area_hist)
    for i in range(len(vor_areas)):
        area_hist = np.histogram(np.log10(vor_areas[i]), bins=binrange, density=True)[0]
        vor_mpv.append(calculate_dmodei(vor_areas[i]))
        vor_areas_bins.append(area_hist)

    # calculate mean of all simulations
    mean_simulation_area_bins = np.mean(sim_areas_bins, axis=0)
    # calculate mean of all simulations
    mean_vor_area_bins = np.nanmean(vor_areas_bins, axis=0)

    mpvs = []
    mpvs.append(calculate_dmodei(spec_areas))
    mpvs.append(np.mean(sim_mpv))
    mpvs.append(np.mean(vor_mpv))

    print(mpvs)
    def plot_mpv(ax: Axes):
        ymax = ax.get_ylim()[1] * 0.1
        for i in range(len(mpvs)):
            mpv = mpvs[i]
            ax.arrow(np.log10(mpv), ymax, 0, -ymax, head_width=0.05, head_length=ymax*0.2, fc=f'C{i}', ec=f'C{i}')

    def annotate_mse(ax: Axes):
        mse_sim = np.mean(np.abs(spec_areas_bins - mean_simulation_area_bins))
        mse_vor = np.mean(np.abs(spec_areas_bins - mean_vor_area_bins))
        ax.annotate(f'$MAE_\mathrm{{ALFA}}$: {mse_sim:.2f}', xy=(0.98, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=6)
        ax.annotate(f'$MAE_\mathrm{{BREAK}}$: {mse_vor:.2f}', xy=(0.98, 0.90), xycoords='axes fraction', ha='right', va='top', fontsize=6)

    def mklegend(ax):
        ax.plot([], [], label=f'Referenz ({mpvs[0]:.1f}mm²)', color='C0')
        ax.plot([], [], label=f'ALFA ({mpvs[1]:.1f}mm²)', color='C1')
        ax.plot([], [], label=f'BREAK ({mpvs[2]:.1f}mm²)', color='C2')

    fig,axs = datahist_plot(figwidth=FigureSize.ROW2)
    axs[0].plot(binrange[:-1], np.cumsum(spec_areas_bins), label='Referenz', color='C0')
    axs[0].plot(binrange[:-1], np.cumsum(mean_simulation_area_bins), label='ALFA', color='C1')
    axs[0].plot(binrange[:-1], np.cumsum(mean_vor_area_bins), label='BREAK', color='C2')
    axs[0].legend()
    State.output(fig, f'{name}_cdf', figwidth=FigureSize.ROW2)


    fig,axs = datahist_plot(figwidth=FigureSize.ROW2, data_mode=DataHistMode.CDF)
    axs[0].stairs(spec_areas_bins, binrange, color='C0')
    axs[0].stairs(mean_simulation_area_bins, binrange, color='C1')
    axs[0].stairs(mean_vor_area_bins, binrange, color='C2')
    mklegend(axs[0])
    annotate_mse(axs[0])
    plot_mpv(axs[0])
    axs[0].legend(fontsize=7)
    State.output(fig, f'{name}_pdf', figwidth=FigureSize.ROW2)


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

    # create histogram of simulation
    sim_splinters = sim.splinters
    # create histogram of specimen
    spec_splinters = specimen.splinters

    plotmode = 'hist'
    sim_areas = [s.area for s in sim_splinters]
    spec_areas = [s.area for s in spec_splinters]
    binrange = get_log_range(spec_areas, 30)
    # spec_areas_h = np.histogram(spec_areas, bins=binrange)[0]
    # sim_areas_h = np.histogram(sim_areas, bins=binrange)[0]

    # create voronoi if available
    vor_areas = None
    # vor_compare = "Voronoi" if voronoi is not None else ""
    # lbr_compare = ""
    if voronoi is not None:
        vor_splinters = voronoi
        vor_areas = [s.area for s in vor_splinters]
        # vor_areas_h = np.histogram(vor_areas, bins=binrange)[0]
        plotmode = 'steps'
        # mse_vor = data_mse(spec_areas_h, vor_areas_h)
        # chi2_vor = data_chi2(vor_areas, spec_areas)
        # vor_compare = f'$MSE_{{break}}$: {mse_vor:.2f}'

    # lbr_compare = f'$MSE_{{sim}}$: {data_mse(spec_areas_h, sim_areas_h):.2f}'


    for mode in [DataHistMode.PDF, DataHistMode.CDF]:
        # plot histograms
        fig,axs = datahist_plot(figwidth=FigureSize.ROW2, data_mode=mode)
        if vor_areas is not None:
            datahist_to_ax(axs, vor_areas, binrange=binrange, label='BREAK', color="C2", data_mode=mode, plot_mode=plotmode)
        datahist_to_ax(axs, spec_areas, binrange=binrange, label='Probekörper', color="C0", data_mode=mode, plot_mode=plotmode)
        datahist_to_ax(axs, sim_areas, binrange=binrange, label='ALFA', color="C1", data_mode=mode, plot_mode=plotmode)

        # axs[0].annotate(lbr_compare, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=6)
        # axs[0].annotate(vor_compare, xy=(0.95, 0.90), xycoords='axes fraction', ha='right', va='top', fontsize=6)
        axs[0].legend()
        #save figure to simulation
        State.output(fig, f'lbreak--nbreak--real_{mode}', figwidth=FigureSize.ROW2)


        fig.savefig(sim.get_file(f'comparison_{mode}.pdf'))

    info('Specimen: ', len(spec_splinters), ' Splinters')
    info('Simulation: ', len(sim_splinters), ' Splinters')


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

@sim_app.command()
def compare_polar(
    prop: SplinterProp,
    simulation_name: str,
    specimen_name: str = None,
    sz: FigureSize = FigureSize.ROW3
):
    simulation = Simulation.get(simulation_name)


    if specimen_name is None and (specimen := simulation.reference) is not None:
        info(f'Using reference from simulation: {specimen.name}')
    elif specimen_name is not None:
        specimen = Specimen.get(specimen_name)
    else:
        assert False, 'Simulation has no reference. Please provide a specimen name.'


    r_range_mm, t_range_deg = arrange_regions(break_pos=SpecimenBreakPosition.CORNER)
    # get splinters of both simulation
    r,_,simZ,_ = simulation.calculate_2d_polar(prop, r_range_mm,t_range_deg)
    _,_,specZ,_ = specimen.calculate_2d_polar(prop, r_range_mm,t_range_deg)
    r = r.flatten()
    simZ = simZ.flatten()
    specZ = specZ.flatten()

    # plot in the same fig
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    axs.plot(r, specZ, label='Referenz')
    axs.plot(r, simZ, label='ALFA')
    axs.set_xlabel('R (mm)')
    axs.set_ylabel(Splinter.get_property_label(prop, row3=sz==FigureSize.ROW3))
    axs.legend()#
    # axs.annotate(f'NRMSE: {nrmse1:.1f}%', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    State.output(fig, f'lbreak--real--{prop}', figwidth=sz)



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


@sim_app.command()
def alfa_validate(
    name: str,
    sigma_s: float = None,
    thickness: float = None,
    boundary: SpecimenBoundary = None,
    break_pos: SpecimenBreakPosition = None,
    no_region_crop: bool = False,
    validation_count: int = 10
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

    info(f'Creating simulation for {specimen.name}')
    info(f'> Sigma_s: {sigma_s}')
    info(f'> Thickness: {thickness} (real: {specimen.measured_thickness})')
    info(f'> Size: {size}')
    info(f'> Boundary: {boundary}')
    info(f'> Break position: {break_pos}')

    spec_areas = [s.area for s in specimen.splinters]
    vor_splinters = nbreak(specimen.name, force_recalc=False)
    vor_areas = [s.area for s in vor_splinters]
    binrange = get_log_range(spec_areas, 30)

    sim_areas_all = []
    mpv_areas_all = []

    fig0,axs0 = None,None
    with plt.ion():
        fig0,axs0 = datahist_plot(figwidth=FigureSize.ROW1)
        datahist_to_ax(axs0, spec_areas, binrange=binrange, color='C0', data_mode=DataHistMode.PDF)
        fig0.canvas.draw()
        fig0.canvas.flush_events()

    for i in range(validation_count):
        sim = alfa(sigma_s, thickness, size, boundary, break_pos, E, nue, impact_position=specimen.get_impact_position(),
                        no_region_crop=no_region_crop, reference=specimen.name)
        areas = [s.area for s in sim.splinters]
        sim_areas_all.append(areas)

        most_prob_area = calculate_dmode(np.asarray(areas))
        mpv_areas_all.append(most_prob_area)

        info(f'Validation {i+1}/{validation_count}: {len(sim.splinters)} splinters, mean area: {np.mean(areas):.2f} mm²')
        datahist_to_ax(axs0, areas, binrange=binrange, color='C1', data_mode=DataHistMode.PDF, plot_mode=DataHistPlotMode.STEPS, linewidth=0.3)
        fig0.canvas.draw()
        fig0.canvas.flush_events()

    axs0[0].set_xlabel('Bruchstückflächeninhalt $A_\mathrm{S}$ (mm²)')
    axs0[0].set_ylabel('CDF')
    legend_without_duplicate_labels(axs0[0])
    State.output(fig0, f'validation_{specimen.name}', figwidth=FigureSize.ROW1)

    print(mpv_areas_all)

    sim_areas_all_hist = []
    # calculate mean areas
    for sim_area in sim_areas_all:
        sim_area_hist = np.histogram(np.log10(sim_area), bins=binrange, density=True)[0]
        sim_areas_all_hist.append(sim_area_hist)

    sim_areas_all_hist = np.array(sim_areas_all_hist)
    sim_areas_all_mean = np.mean(sim_areas_all_hist, axis=0)

    for mode in [DataHistMode.PDF, DataHistMode.CDF]:
        dmode = DataHistPlotMode.STEPS
        fig,axs = datahist_plot(figwidth=FigureSize.ROW2)
        datahist_to_ax(axs, spec_areas, binrange=binrange, label="Probekörper", color='C0', data_mode=mode, plot_mode=dmode)
        datahist_to_ax(axs, vor_areas, binrange=binrange, label="Voronoi", color='C2', data_mode=mode, plot_mode=dmode)
        datahist_to_ax(axs, 10**sim_areas_all_mean, binrange=binrange, label="Simulation", color='C1', data_mode=mode, plot_mode=dmode)
        State.output(fig, f'validation_{specimen.name}_mean_{dmode}', figwidth=FigureSize.ROW2)