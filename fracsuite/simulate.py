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
from fracsuite.core.image import to_rgb
from fracsuite.core.imageplotting import plotImage
from fracsuite.core.imageprocessing import dilateImg
from fracsuite.core.mechanics import U
from fracsuite.core.model_layers import ModelLayer, arrange_regions, has_layer, interp_layer
from fracsuite.core.plotting import FigureSize, datahist_plot, datahist_to_ax, get_fig_width
from fracsuite.core.point_process import gibbs_strauss_process
from fracsuite.core.progress import get_progress
from fracsuite.core.region import RectRegion
from fracsuite.core.simulation import Simulation
from fracsuite.core.specimen import Specimen, SpecimenBoundary, SpecimenBreakPosition
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stress import relative_remaining_stress
from scipy.spatial import Voronoi, voronoi_plot_2d
from fracsuite.core.vectors import angle_between

from fracsuite.state import State

from spazial import csstraussproc2, csstraussproc_rhciter, bohmann_process

sim_app = typer.Typer(help=__doc__, callback=main_callback)

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

    print('Estimating intensity...')
    # find fracture intensity from first order statistics
    intensity = specimen.calculate_break_lambda(force_recalc=True)

    print('Estimating rhc...')
    # find the hard core radius using second order statistics
    rhc,acceptance = specimen.calculate_break_rhc(force_recalc=True)
    print(f'Fracture intensity: {intensity:.2f} 1/mm²')
    print(f'HC Radius: {rhc:.2f} mm')
    print(f'Acceptance: {acceptance:.2f}')

    def Kpois(d):
        # see Baddeley et al. S.206 K_pois
        return np.pi*d**2
    def Lpois(d):
        # see Baddeley et al. S.206 K_pois
        return d

    print('Plotting kfunc...')
    x,y = specimen.kfun()
    fig, ax = plt.subplots(figsize=get_fig_width(FigureSize.ROW2))
    ax.plot(x,y, label='$\hat{K}$')
    ax.plot(x,Kpois(np.asarray(x)), label='$\hat{K}_{t}$')
    ax.legend()
    ax.set_ylabel('$\hat{K}(d)$')
    ax.set_xlabel('$d$ (mm)')
    State.output(fig,'kfunc', spec=specimen, figwidth=FigureSize.ROW2)


    print('Plotting lfunc...')
    x2,y2 = specimen.lfun()
    min_L = rhc

    ax: Axes
    fig, ax = plt.subplots(figsize=get_fig_width(FigureSize.ROW2))
    ax.plot(x2,y2, label='$\hat{L}$')
    ax.axvline(rhc, linestyle='--', color='r', label=f'$r_{{hc}}={min_L:.1f}mm$')
    ax.plot(x2,Lpois(np.asarray(x)), label='$\hat{K}_{t}$')
    ax.legend()
    ax.set_ylabel('$\hat{L}(d)$')
    ax.set_xlabel('$d$ (mm)')
    State.output(fig, 'lfunc', spec=specimen, figwidth=FigureSize.ROW2)

    print('Plotting centered lfunc...')
    x2,y2 = specimen.lcfun()
    min_L = rhc

    ax: Axes
    fig, ax = plt.subplots(figsize=get_fig_width(FigureSize.ROW2))
    ax.plot(x2,y2, label='$\hat{L}-d$')
    ax.axvline(rhc, linestyle='--', color='r', label=f'$r_{{hc}}={min_L:.1f}mm$')
    ax.legend()

    ax.set_ylabel('$\hat{L}(d)-d$')
    ax.set_xlabel('$d$ (mm)')
    State.output(fig, 'lcfunc', spec=specimen, figwidth=FigureSize.ROW2)

    # create actual voronoi plots
    size = specimen.get_real_size()
    area = size[0] * size[1]
    n_points = int(intensity * area)
    points = gibbs_strauss_process(
        n_points,
        rhc,
        acceptance_possibility=acceptance,
        area=area)
    fig,axs = plt.subplots()
    axs.scatter(*zip(*points))
    plt.show()


    # create voronoi of points
    vor = Voronoi(points)
    fig,axs = plt.subplots()
    voronoi_plot_2d(vor, ax=axs, show_points=False, show_vertices=False)
    plt.show()



@sim_app.command()
def lbreak(
    sigma_s: float,
    thickness: float,
    size: tuple[float,float] = (500,500),
    boundary: SpecimenBoundary = SpecimenBoundary.A,
    break_pos: SpecimenBreakPosition = SpecimenBreakPosition.CORNER,
    E: float = 70e3,
    nue: float = 0.23,
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

    # this will create the
    sim = Simulation.create(thickness, sigma_s, boundary, None)


    # calculate energy
    energy_u = U(sigma_s, thickness)
    # load layer for rhc and intensity
    intensity, intensity_std = interp_layer(SplinterProp.INTENSITY, boundary, thickness, break_pos,energy_u)
    rhc, rhc_std = interp_layer(SplinterProp.RHC, boundary, thickness, break_pos, energy_u)
    # create radii
    r_range, t_range = arrange_regions(break_pos=break_pos, w_mm=size[0], h_mm=size[1])
    print(r_range)
    print(intensity(r_range))

    fracture_intensity = np.mean(intensity(r_range))
    hc_radius = np.mean(rhc(r_range))
    fint_std = np.mean(intensity_std(r_range))
    rhc_stddev = np.mean(rhc_std(r_range))

    fracture_intensity = meanrand(fracture_intensity, fint_std)
    hc_radius = meanrand(hc_radius, rhc_stddev)

    # fetch fracture intensity and hc radius from energy
    mean_area = 1 / fracture_intensity
    impact_position = break_pos.default_position()
    area = size[0] * size[1]
    c = 4e-5

    print(f'Energy:             {energy_u:>15.2f} J/m²')
    print(f'Fracture intensity: {fracture_intensity:>15.4f} 1/mm²')
    print(f'HC Radius:          {hc_radius:>15.2f} mm')
    print(f'Mean area:          {mean_area:>15.2f} mm²')
    print(f'Impact position:    {impact_position}')
    # estimate acceptance probability
    urr = relative_remaining_stress(mean_area, thickness)
    nue = 1 - urr
    print(f'Urr: {urr:.2f}', f'nue: {nue:.2f}')

    # create spatial points
    # points = spazial_gibbs_strauss_process(fracture_intensity, hc_radius, 0.55, size)
    # points = csstraussproc(size, hc_radius, int(fracture_intensity*area), c, int(1e6))
    # points = csstraussproc2(size[0], size[1], hc_radius, int(fracture_intensity*area), c, int(1e6))

    # interpolate rhc values
    rhc_values = rhc(r_range)
    # create array with r_range in column 0 and rhc_values in 1
    rhc_array = np.column_stack((r_range, rhc_values))
    # # start point process with rhc values
    # points = csstraussproc_rhciter(size[0], size[1], rhc_array, impact_position, int(fracture_intensity*area), c, int(1e6))

    l_values = intensity(r_range)
    l_array = np.column_stack((r_range, l_values))
    points = bohmann_process(size[0], size[1], r_range, l_array, rhc_array, impact_position, c, int(1e6))

    # print region
    region = (0,0,0.2,0.2)

    # plot points
    fig,axs = plt.subplots()
    axs.scatter(*zip(*points))
    # find maximum region from points
    x,y = zip(*points)
    x_min = np.min(x) * region[0]
    x_max = np.max(x) * region[2]
    y_min = np.min(y) * region[1]
    y_max = np.max(y) * region[3]
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)
    plt.show()

    # scaling factor (realsize > pixel size)
    size_f = 20
    region_scaled = (x_min, x_max, y_min, y_max)
    # create output image store
    markers = np.zeros((int(size[0]*size_f),int(size[1]*size_f)), dtype=np.uint8)
    for point in points:
        markers[int(point[0]*size_f), int(point[1]*size_f)] = 0

    # clip region from markers
    markers_clipped = cropimg(region_scaled, size_f, markers)
    cv2.imwrite(sim.get_file('points.png'), markers_clipped)

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
        'orientation': (il_orientation, il_orientation_stddev, Splinter.get_mode_labels(SplinterProp.ORIENTATION), 'orientation'),
        'l1': (il_l1, il_l1_stddev, Splinter.get_mode_labels(SplinterProp.L1), 'l1'),
        'l2': (il_l2, il_l2_stddev, Splinter.get_mode_labels(SplinterProp.L2), 'l2'),
        'l1l2': (il_l1l2, il_l1l2_stddev, Splinter.get_mode_labels(SplinterProp.ASP0), 'l1l2'),
        'intensity': (intensity, intensity_std, Splinter.get_mode_labels(SplinterProp.INTENSITY), 'intensity'),
        'rhc': (rhc, rhc_std, Splinter.get_mode_labels(SplinterProp.RHC), 'rhc'),
    }

    for layer in layers:
        print(f'Plotting {layer}...')
        il, il_stddev, mode_labels, name = layers[layer]
        r_range = np.linspace(0, np.sqrt(450**2+450**2), 100)
        fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW3))
        r_il = il(r_range)
        r_il_stddev = il_stddev(r_range)
        axs.plot(r_range, r_il)
        # add error bars to plot
        axs.fill_between(r_range, r_il-r_il_stddev/2, r_il+r_il_stddev/2, alpha=0.3)
        axs.set_ylabel(mode_labels)
        plt.show()
        fig.savefig(sim.get_file(f'{name}.pdf'))

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
            l_major = l_major * 0.5
            l_minor = l_minor * 0.5

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
                    cv2.LINE_AA # line type
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

    print('[yellow] Simulating...')
    # cv2.ellipse(markers, impact_position.astype(np.uint8)*size_f, (int(20), int(5)), 0, 0, 360, (0,255,0), 1)
    # cv2.ellipse(markers, (400*size_f,400*size_f), (int(5), int(20)), 0, 0, 360, (0,255,0), -1)
    # cv2.ellipse(markers, (200*size_f,400*size_f), (int(5), int(20)), 0, 0, 360, (255,120,0), -1)
    plotImage(markers, 'markers', force=True)
    markers_clipped = cropimg(region_scaled, size_f, markers)
    cv2.imwrite(sim.get_file('layered_points.png'), markers_clipped)


    markers = cv2.connectedComponents(np.uint8(markers))[1]
    shape = (int(size[0]*size_f),int(size[1]*size_f),3)
    blank_image = np.zeros(shape, dtype=np.uint8)
    markers = cv2.watershed(blank_image, markers)

    m_img = np.zeros(shape, dtype=np.uint8)
    m_img[markers == -1] = 255
    m_img_clipped = cropimg(region_scaled, size_f, m_img)
    cv2.imwrite(sim.get_file('watershed_0.png'), m_img_clipped)

    black_white_img = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    splinters = Splinter.analyze_contour_image(m_img, px_per_mm=size_f)

    out_img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for s in splinters:
        clr = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.drawContours(out_img, [s.contour], -1, clr, -1)
        cv2.drawContours(black_white_img, [s.contour], -1, 255, 1)

    plt.imshow(out_img)
    plt.show()

    plt.imshow(black_white_img)
    plt.show()

    State.output(black_white_img, f'generated_{sigma_s}_{thickness}', spec=None, figwidth=FigureSize.ROW2)
    cv2.imwrite(sim.get_file('filled.png'), out_img)

    print('[yellow] Saving...')
    sim.put_splinters(splinters)
    print(f'[green] Simulation created: {sim.name}_{sim.nbr}[/green]')
    return sim

def cropimg(region, size_f, markers):
    x_min, x_max, y_min, y_max = region
    markers_clipped = markers[int(x_min*size_f):int(x_max*size_f), int(y_min*size_f):int(y_max*size_f)]
    return markers_clipped

@sim_app.command()
def lbreak_like(name):

    specimen = Specimen.get(name)

    sigma_s = np.abs(specimen.sig_h)
    thickness = specimen.thickness
    size = specimen.get_real_size()
    boundary = specimen.boundary
    break_pos = specimen.break_pos
    E = 70e3
    nue = 0.23

    # create simulation
    simulation = lbreak(sigma_s, thickness, size, boundary, break_pos, E, nue)
    # compare simulation with input
    compare(simulation.fullname, specimen.name)

@sim_app.command()
def compare(
    simulation_name: Annotated[str, typer.Argument(help="Simulation name.")],
    specimen_name: Annotated[str, typer.Argument(help="Specimen name.")],
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
    spec_hist, spec_bins = np.histogram(spec_areas, bins=30, density=True, range=(0, max_area))

    # plot histograms
    fig,axs = datahist_plot(figwidth=FigureSize.ROW2)
    datahist_to_ax(axs, spec_areas, 30, label='Specimen')
    datahist_to_ax(axs, sim_areas, 30, label='Simulation')
    axs[0].legend()
    State.output(fig, f'{specimen.name}--{sim.name}_{sim.nbr}', figwidth=FigureSize.ROW2)

    print('Specimen: ', len(spec_splinters))
    print('Simulation: ', len(sim_splinters))


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
