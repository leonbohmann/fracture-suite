from inspect import Arguments
import re
from textwrap import indent
from typing import Annotated
import cv2
from matplotlib.axes import Axes

import numpy as np
from tqdm import tqdm
import typer
from matplotlib import colors as pltc
from matplotlib import pyplot as plt

from rich.progress import track
from rich import print

from fracsuite.callbacks import main_callback
from fracsuite.core.coloring import get_color, norm_color
from fracsuite.core.logging import debug, info
from fracsuite.core.mechanics import U
from fracsuite.core.model_layers import ModelLayer, arrange_regions, arrange_regions_px, has_layer, interp_layer, load_layer, load_layer_file, plt_layer, save_layer
from fracsuite.core.plotting import FigureSize, annotate_image, fill_polar_cell, get_fig_width, get_legend, renew_ticks_cb
from fracsuite.core.progress import get_progress, tracker
from fracsuite.core.specimen import Specimen
from fracsuite.core.specimenprops import SpecimenBreakMode, SpecimenBreakPosition, SpecimenBoundary
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stochastics import calculate_dmode, moving_average
from fracsuite.core.vectors import angle_deg
from fracsuite.general import GeneralSettings
from fracsuite.splinters import create_filter_function
from fracsuite.state import State, StateOutput

layer_app = typer.Typer(callback=main_callback, help="Model related commands.")
general = GeneralSettings.get()

bid = {
        'A': 1,
        'B': 2,
        'Z': 3,
    }

# @layer_app.command()
# def create_base_layer(
#     mode: Annotated[SplinterProp, typer.Argument(help='Mode for the aspect ratio.')],
#     break_pos: Annotated[SpecimenBreakPosition, typer.Option(help='Break position.')] = SpecimenBreakPosition.CORNER,
#     break_mode: Annotated[SpecimenBreakMode, typer.Option(help='Break mode.')] = SpecimenBreakMode.PUNCH,
# ):
#     bid = {
#         'A': 1,
#         'B': 2,
#         'Z': 3,
#     }
#     boundaries = {
#         1: '--',
#         2: '-',
#         3: ':',
#     }
#     # inverse bid
#     bid_r = {v: k for k, v in bid.items()}

#     def add_filter(specimen: Specimen):
#         if break_pos is not None and specimen.break_pos != break_pos:
#             return False

#         if break_mode is not None and specimen.break_mode != break_mode:
#             return False

#         if specimen.boundary == SpecimenBoundary.Unknown:
#             return False

#         if not specimen.has_splinters:
#             return False

#         if specimen.U_d is None or not np.isfinite(specimen.U_d):
#             return False

#         return True

#     specimens: list[Specimen] = Specimen.get_all_by(add_filter, load=False)

#     # [U, boundary, lambda, rhc]
#     values = np.zeros((len(specimens), 5))
#     with get_progress(title='Calculating base layer', total=len(specimens)) as progress:
#         progress.set_total(len(specimens))
#         for id, spec in enumerate(specimens):
#             progress.set_description(f"Specimen {spec.name}")
#             progress.advance()
#             values[id,0] = spec.U
#             values[id,1] = bid[spec.boundary]
#             values[id,2] = spec.measured_thickness
#             values[id,3] = spec.calculate_break_lambda()
#             values[id,4] = spec.calculate_break_rhc()[0]

#     # sort value after first column
#     values = values[values[:,0].argsort()]

#     sz = get_fig_width(FigureSize.ROW1)
#     fig,lam_axs = plt.subplots(figsize=sz)
#     fig2,rhc_axs = plt.subplots(figsize=sz)
#     # save values
#     for b in boundaries:
#         mask = values[:,1] == b

#         # values for the current boundary
#         b_values = values[mask,:]

#         # save layer
#         save_base_layer(b_values, bid_r[b], break_pos)

#         # plot fracture intensity parameter
#         lam_axs.plot(b_values[:,0], b_values[:,3], label=bid_r[b], linestyle=boundaries[b])

#         # plot hard core radius
#         rhc_axs.plot(b_values[:,0], b_values[:,4], label=bid_r[b], linestyle=boundaries[b])

#     lam_axs.set_xlabel("Strain Energy U [J/m²]")
#     lam_axs.set_ylabel("Fracture Intensity Parameter $\lambda$ [-]")
#     lam_axs.legend()
#     State.output(StateOutput(fig, FigureSize.ROW1), f"base-layer-lambda_{break_pos}", to_additional=True)

#     rhc_axs.set_xlabel("Strain Energy U [J/m²]")
#     rhc_axs.set_ylabel("Hard Core Radius $r_{hc}$ [mm]")
#     rhc_axs.legend()
#     State.output(StateOutput(fig2, FigureSize.ROW1), f"base-layer-rhc_{break_pos}", to_additional=True)

# @layer_app.command()
# def create_impact_layer_intensity(
#     mode: Annotated[SplinterProp, typer.Argument(help='Mode for the aspect ratio.')],
#     break_pos: Annotated[SpecimenBreakPosition, typer.Option(help='Break position.')] = SpecimenBreakPosition.CORNER,
#     break_mode: Annotated[SpecimenBreakMode, typer.Option(help='Break mode.')] = SpecimenBreakMode.PUNCH,
#     specimen_name: Annotated[str, typer.Option(help='Specimen name.')] = None,
# ):
#     """
#     Plots the fracture intensity parameter for every specimen via distance and angle to impact.

#     In specimen_name, supply the full specimen name!
#     """
#     bid = {
#         'A': 1,
#         'B': 2,
#         'Z': 3,
#     }
#     boundaries = {
#         1: '--',
#         2: '-',
#         3: ':',
#     }
#     # inverse bid
#     bid_r = {v: k for k, v in bid.items()}

#     def add_filter(specimen: Specimen):
#         if specimen_name is not None and not specimen.name.startswith(specimen_name):
#             return False

#         if break_pos is not None and specimen.break_pos != break_pos:
#             return False

#         if break_mode is not None and specimen.break_mode != break_mode:
#             return False

#         if specimen.boundary == SpecimenBoundary.Unknown:
#             return False

#         if not specimen.has_splinters:
#             return False

#         if specimen.U_d is None or not np.isfinite(specimen.U_d):
#             return False

#         return True

#     specimens: list[Specimen] = Specimen.get_all_by(add_filter, load=True, max_n=1)
#     sz = FigureSize.ROW1

#     if mode == 'intensity':
#         def value_calculator(x):
#             return len(x)

#         clrlabel = 'Fracture Intensity $N_{50}$ [-]'
#     else:

#         clrlabel = Splinter.get_mode_labels(mode, row3=sz == FigureSize.ROW3)

#     for spec in specimens:
#         if mode != 'intensity':
#             def value_calculator(x: list[Splinter]):
#                 return np.mean([s.get_splinter_data(prop=mode, ip_mm=spec.get_impact_position()) for s in x])


#         result = spec.calculate_2d_polar(value_calculator=value_calculator)

#         X = result[0,1:]
#         Y = result[1:,0]
#         Z = result[1:,1:]

#         # print(X)
#         # print(Y)
#         # print(Z)

#         # plot results as 2d contour plot
#         fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
#         # axs.imshow(Z, cmap='turbo')
#         X, Y = np.meshgrid(X, Y, indexing='xy')
#         mesh = axs.pcolormesh(X, Y, Z, shading='auto', cmap='turbo')
#         cbar = fig.colorbar(mesh, label=clrlabel)
#         renew_ticks_cb(cbar)


#         axs.set_xlabel("Distance to Impact [mm]")
#         axs.set_ylabel("Angle to Impact [°]")
#         axs.autoscale()
#         State.output(StateOutput(fig, sz), f"{mode}-2d_{spec.name}", to_additional=True)


@layer_app.command()
def create(
    prop: Annotated[str, typer.Argument(help='Mode for the aspect ratio.')],
    break_pos: Annotated[SpecimenBreakPosition, typer.Option(help='Break position.')] = SpecimenBreakPosition.CORNER,
    break_mode: Annotated[SpecimenBreakMode, typer.Option(help='Break mode.')] = SpecimenBreakMode.PUNCH,
    ignore_nan_u: Annotated[bool, typer.Option(help='Filter Ud values that are NaN from plot.')] = False,
    thickness: Annotated[float, typer.Option(help='Specimen thickness.')] = None,
    exclude_names: Annotated[str, typer.Option(help='Exclude specimens with these names. Seperated by comma.')] = "",
    exclude_name_filter: Annotated[str, typer.Option(help='Exclude specimens matching this filter.')] = None,
    normalize: Annotated[bool, typer.Option(help='Normalize specimen value ranges.')] = False,
    name_filter: Annotated[str, typer.Option(help='Filter specimen names.')] = "*",
    sz: FigureSize = typer.Option(FigureSize.ROW3, help='Figure size.'),
    no_save: Annotated[bool, typer.Option(help='Do not save the created layers.')] = False,
    with_std: Annotated[bool, typer.Option(help='Save standard deviation layers.')] = False,
):
    """
    Create layers based on the given parameters. Also generated useful plots for the layers.

    This function will create layers and save them to the output folder.
    If `no_save` is set to `True`, the layers will not be saved and only plots will be created.


    Args:
        prop (str): Mode for the aspect ratio.
        break_pos (SpecimenBreakPosition, optional): Break position. Defaults to SpecimenBreakPosition.CORNER.
        break_mode (SpecimenBreakMode, optional): Break mode. Defaults to SpecimenBreakMode.PUNCH.
        ignore_nan_u (bool, optional): Filter Ud values that are NaN from plot. Defaults to False.
        thickness (float, optional): Specimen thickness. Defaults to None, which uses all.
        exclude_names (str, optional): Exclude specimens with these names. Separated by comma. Defaults to "".
        exclude_name_filter (str, optional): Exclude specimens matching this filter. Defaults to None.
        normalize (bool, optional): Normalize specimen value ranges. Defaults to False.
        name_filter (str, optional): Filter specimen names. Defaults to "*", which uses all specimens.
        sz (FigureSize, optional): Figure size. Defaults to FigureSize.ROW3.
        no_save (bool, optional): Do not save the created layers. Defaults to False.
        with_std (bool, optional): Plot standard deviation layers. Defaults to False.
    """

    if "," in prop or prop == "all":
        info("Detected pattern in property...")
        if prop == "all":
            prop = "intensity,rhc,acceptance,orientation,l1,l2,asp,asp0"
            info(f"All properties will be calculated: {prop}")
            pass

        import gc
        props = prop.split(',')
        for p in (t := tracker(props, title='Calculating properties')):
            t.progress.set_description(f"Calculating '{p}'...")
            info(f"Calculating property '{p}'...")
            p = SplinterProp(p)
            create(p, break_pos, break_mode, ignore_nan_u, thickness, exclude_names, exclude_name_filter, normalize, name_filter, sz, no_save, with_std)
            gc.collect()

        return

    prop = SplinterProp(prop)

    bid = {
        'A': 1,
        'B': 2,
        'Z': 3,
    }
    boundaries = {
        1: '--',
        2: '-',
        3: ':',
    }
    # inverse bid
    bid_r = {v: k for k, v in bid.items()}

    if exclude_names != "":
        exclude_names = exclude_names.split(',')

    base_filter = create_filter_function(
        name_filter,
    )

    def add_filter(specimen: Specimen):

        ## this is not necessary, it was shown that the impactor energy is not big enough to cause
        ##  discrepancies in layer creation
        # if not specimen.broken_immediately:
        #     return False

        if exclude_name_filter is not None:
            if re.match(exclude_name_filter, specimen.name) is not None:
                return False

        if break_pos is not None and specimen.break_pos != break_pos:
            return False

        if break_mode is not None and specimen.break_mode != break_mode:
            return False

        if specimen.boundary == SpecimenBoundary.Unknown:
            return False

        if not specimen.has_splinters:
            return False

        if specimen.U_d is None or not np.isfinite(specimen.U_d):
            return False

        if thickness is not None and specimen.thickness != thickness:
            return False

        if specimen.name in exclude_names:
            return False



        return base_filter(specimen)

    specimens: list[Specimen] = Specimen.get_all_by(add_filter, load=True)

    # for spec in specimens:
    #     area = np.sum([s.area for s in spec.splinters])
    #     print(f'{spec.name}: {area:.2f} mm²')

    if thickness is None:
        thickness = 'all'
    else:
        thickness = f'{thickness:.0f}'

    xlabel = "Abstand zum Anschlagpunkt (mm)" if sz != FigureSize.ROW3 else "R (mm)"
    ylabel = Splinter.get_property_label(prop, row3=sz==FigureSize.ROW3)


    r_range, t_range = arrange_regions(d_r_mm=25,d_t_deg=360,break_pos=break_pos,w_mm=500,h_mm=500)

    debug(f"Using radius range {r_range}.")
    debug(f"Using theta range {t_range}.")

    ########################
    # Calculate value for every specimen and save results to arrays
    ########################
    results = np.ones((len(specimens), len(r_range)-1+4)) * np.nan
    stddevs = np.ones((len(specimens), len(r_range)-1+4)) * np.nan


    # iterate specimens and calculate polar values
    for si, specimen in tracker(enumerate(specimens), 'Calculating specimens', total=len(specimens)):
        specimen: Specimen

        _,_,Z,Zstd = specimen.calculate_2d_polar(
            prop=prop,
            r_range_mm=r_range,
            t_range_deg=t_range,
        )

        results[si,0] = specimen.U
        results[si,1] = specimen.thickness
        results[si,2] = bid[specimen.boundary]
        results[si,3] = si
        results[si,4:] = (Z / (np.max(Z) if normalize else 1)).flatten() # normalization

        stddevs[si,0] = results[si,0]
        stddevs[si,1] = results[si,1]
        stddevs[si,2] = results[si,2]
        stddevs[si,3] = results[si,3]
        stddevs[si,4:] = Zstd.flatten()


    # sort results after first column
    results = results[results[:,0].argsort()]
    stddevs = stddevs[stddevs[:,0].argsort()]

    if not no_save:
        ########################
        # Save results as layer
        ########################
        for b in bid:
            # mask all results for the current boundary
            b_mask = results[:,2] == bid[b]

            # mask thicknesses
            for t in [4,8,12]:
                t_mask = results[:,1] == t

                # mask both
                bt_mask = b_mask & t_mask

                # skip empty boundaries
                if np.sum(bt_mask) == 0:
                    continue

                # boundary results
                b_results = results[bt_mask,:]
                b_stddevs = stddevs[bt_mask,4:]

                b_energies = b_results[:,0]
                b_values = b_results[:,4:]
                b_r_range = r_range[:-1] # -1 because the last range is not closed (last r_range is calculated as lower bound)

                # save layer
                save_layer(
                    prop,
                    b,
                    t,
                    break_pos,
                    False,
                    b_r_range,      # 1d radius array
                    b_energies,     # 1d energy array
                    b_values        # 2d array
                )
                # save stddev
                save_layer(
                    prop,
                    b,
                    t,
                    break_pos,
                    True,
                    b_r_range,
                    b_energies,
                    b_stddevs
                )

    # define an energy range to plot mean values in
    n_ud = 7 # J/m²
    ud_min = np.min(results[:,0])
    ud_max = np.max(results[:,0])
    ud_range = np.linspace(ud_min, ud_max, n_ud)
    ud_colors = [norm_color(get_color(ud, ud_min, ud_max)) for ud in ud_range]

    # shift ranges to centers (alternatively use R and T from polar calculation)
    x = r_range[:-1] + (r_range[1]-r_range[0])/2

    # plot a figure for every boundary using the colorbar for energy
    for b in bid:
        b_mask = results[:,2] == bid[b]

        for t in [4,8,12]:
            t_mask = results[:,1] == t

            # mask both
            bt_mask = b_mask & t_mask

            # skip empty boundaries
            if np.sum(bt_mask) == 0:
                continue

            # boundary results
            b_results = results[bt_mask,:]
            b_stddevs = stddevs[bt_mask,:]


            max_u = np.nanmax(b_results[:,0])
            min_u = np.nanmin(b_results[:,0])

            fig,axs = plt.subplots(figsize=get_fig_width(sz))
            # axs.set_autoscaley_on(False)

            colors = []
            print('Current Boundary: ', b)

            # plot all individual results
            for i in range(len(b_results)):
                bt_energies = b_results[i,4:]
                # print specimen name
                print(f'{specimens[int(b_results[i,3])].name}: {np.nanmean(bt_energies):.2f} (mpv: {calculate_dmode(bt_energies)[0]}) +/- {np.nanmax(b_stddevs[i,4:]):.2f} (max: {np.nanmax(bt_energies):.2f}, min: {np.nanmin(bt_energies):.2f})')

                # get color for current energy
                c = norm_color(get_color(b_results[i,0], min_u, max_u))
                colors.append(c)

                # plot individual lines
                axs.plot(x, bt_energies, color=c, linewidth=1, markersize=1.5) #

                # fill stddev
                if with_std:
                    axs.fill_between(x, b_results[i,4:] - b_stddevs[i,4:], b_results[i,4:] + b_stddevs[i,4:],
                                 color=c, alpha=0.1)

            # plot mean values as thick lines
            for i in range(len(ud_range)-1):
                cud = (ud_range[i] + ud_range[i+1]) / 2
                if i < len(ud_range)-1:
                    ud_mask = (b_results[:,0] >= ud_range[i]) & (b_results[:,0] < ud_range[i+1])
                else:
                    ud_mask = (b_results[:,0] >= ud_range[i])

                if np.sum(ud_mask) == 0:
                    continue

                ud_results = b_results[ud_mask,4:]
                if np.nansum(ud_results) == 0:
                    continue

                mean = np.nanmean(ud_results, axis=0)
                c = norm_color(get_color(cud, min_u, max_u))

                # axs.plot(x, mean, color=c, linewidth=2)

            # put plot data onto axs
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            cmap = pltc.ListedColormap(colors)
            norm = pltc.Normalize(min_u, max_u)
            clabel = "Formänderungsenergie $U$ (J/m²)" if sz != FigureSize.ROW3 else "U (J/m²)"
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label=clabel)
            # renew_ticks_cb(cbar)
            # renew_ticks_ax(axs)
            State.output(StateOutput(fig, sz), f"impact-layer_{b}_{t:.0f}_{prop}_{break_pos}", to_additional=True)

@layer_app.command()
def compare_boundaries(
    mode: Annotated[SplinterProp, typer.Argument(help='Mode for the aspect ratio.')],
    u: Annotated[str, typer.Argument(help='Energy level.')],
    break_pos: Annotated[SpecimenBreakPosition, typer.Option(help='Break position.')] = SpecimenBreakPosition.CORNER,
    break_mode: Annotated[SpecimenBreakMode, typer.Option(help='Break mode.')] = SpecimenBreakMode.PUNCH,
    ignore_nan_u: Annotated[bool, typer.Option(help='Filter Ud values that are NaN from plot.')] = False,
    energy_range: Annotated[float, typer.Option(help='Energy range around ud.')] = 0.05,
):
    bid = {
        'A': 1,
        'B': 2,
        'Z': 3,
    }
    boundaries = {
        1: '--',
        2: '-',
        3: ':',
    }
    # inverse bid
    bid_r = {v: k for k, v in bid.items()}

    if ',' in u:
        s,t = [float(x) for x in u.split(',')]
        u = U(s,t)
    else:
        u = float(u)


    min_energy = u * (1-energy_range)
    max_energy = u * (1+energy_range)

    def add_filter(specimen: Specimen):
        if break_pos is not None and specimen.break_pos != break_pos:
            return False

        if break_mode is not None and specimen.break_mode != break_mode:
            return False

        if specimen.boundary == SpecimenBoundary.Unknown:
            return False

        if not specimen.has_splinters:
            return False

        if specimen.U_d is None or not np.isfinite(specimen.U_d):
            return False

        if not max_energy > specimen.U > min_energy:
            return False

        return True

    specimens: list[Specimen] = Specimen.get_all_by(add_filter, load=True)


    sz = FigureSize.ROW1

    xlabel = "Abstand R zum Anschlagpunkt (mm)"
    ylabel = Splinter.get_property_label(mode, row3=sz == FigureSize.ROW3)
    ylabel_short = Splinter.get_property_label(mode, row3=True)


    # find aspect over R on every specimen

    r_range, t_range = arrange_regions(break_pos=break_pos,w_mm=500,h_mm=500)

    ########################
    # Calculate value for every specimen and save results to arrays
    ########################
    results = np.ones((len(specimens), len(r_range)+2)) * np.nan
    stddevs = np.ones((len(specimens), len(r_range)+2)) * np.nan

    with get_progress(title='Calculating', total=len(specimens)) as progress:
        for si, specimen in enumerate(specimens):
            print('Specimen: ', specimen.name)

            X,Y,Z,Zstd = specimen.calculate_2d_polar(mode, r_range_mm=r_range, t_range_deg=t_range)

            results[si,0] = specimen.U
            results[si,1] = bid[specimen.boundary]
            results[si,2] = si
            results[si,3:] = Z.flatten()

            stddevs[si,0] = results[si,0]
            stddevs[si,1] = results[si,1]
            stddevs[si,2] = results[si,2]
            stddevs[si,3:] = Zstd.flatten()

            progress.advance()

            # except:
            #     results[si+1,0] = specimen.U
            #     results[si+1,1] = bid[specimen.boundary]
            #     results[si+1,2] = si
            #     results[si+1,3:] = aspects[:,1]


    # sort results after U_d
    results = results[results[:,0].argsort()]

    ########################
    # Save results as layer
    ########################

    # first: results, second: stddevs
    results_per_boundary_unclustered = {
        1: ([],[]),
        2: ([],[]),
        3: ([],[])
    }

    # find all results for every boundary
    mask_A = results[:,1] == 1
    mask_B = results[:,1] == 2
    mask_Z = results[:,1] == 3

    # put results in dictionary
    results_per_boundary_unclustered[1] = (results[mask_A,:], stddevs[mask_A,:])
    results_per_boundary_unclustered[2] = (results[mask_B,:], stddevs[mask_B,:])
    results_per_boundary_unclustered[3] = (results[mask_Z,:], stddevs[mask_Z,:])
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    # sort the individual results after U_d
    for ib, b in enumerate(results_per_boundary_unclustered):
        b_mask = results_per_boundary_unclustered[b][0][:,0].argsort()
        results_b = results_per_boundary_unclustered[b][0][b_mask]
        stddev_b = results_per_boundary_unclustered[b][1][b_mask]

        # plot the current boundary
        x = r_range + (r_range[1]-r_range[0])/2
        x = x[:-1]
        y = results_b[:,0].ravel()
        z = results_b[:,3:]

        # scatter each line in results with x
        for i in range(len(y)):
            axs.scatter(x, z[i,:], color='rkb'[ib], marker='x', linewidth=0.5, s=1.5)


        # plot the stdmean of the current boundary
        mean = np.nanmean(z, axis=0)
        axs.plot(x, mean,  color='rkb'[ib], linestyle='-', alpha=1, label=bid_r[b])

    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.legend()
    State.output(StateOutput(fig,sz), f'compare_{min_energy:.0f}-{max_energy:.0f}_{mode}_{break_pos}', to_additional=True)

@layer_app.command()
def graph(
    specimen_name: Annotated[str, typer.Argument(help='Specimen name.')],
    mode: Annotated[SplinterProp, typer.Argument(help='Mode for the aspect ratio.')],
    colored_angle: Annotated[bool, typer.Option(help='Color the scatterplot by angle to impact.')] = False,
    plot_stddev: Annotated[bool, typer.Option(help='Plot standard deviation behind lineplot.')] = False,
):
    """
    Create a plot that contains all individual values for every splinter and the mean value as a line for every radius.

    Args:
        specimen_name (str): Specimen name.
        mode (SplinterProp): The splinter property to plot.
        colored_angle (bool, optional): Color the angle instead of shaded gray. Defaults to False.
        plot_stddev (bool, optional): Plot the standard deviation as a filled area behind the mean. Defaults to False.
    """
    specimen = Specimen.get(specimen_name, load=True)

    # find aspect over R on every specimen
    n_r = 30
    r_min = 0
    r_max = np.sqrt(450**2 + 450**2)
    r_range = np.linspace(r_min, r_max, n_r, endpoint=False)

    clr = "rkb"[bid[specimen.boundary]]

    ########################
    # Calculate value for every specimen and save results to arrays
    ########################
    results = np.full((len(r_range)+1),np.nan)
    stddevs = np.full((len(r_range)+1),np.nan)
    amounts = np.full((len(r_range)),np.nan)
    # now, find aspect ratio of all splinters
    aspects = np.zeros((len(specimen.splinters), 3)) # 0: radius, 1: aspect ratio
    ip = specimen.get_impact_position()
    px_p_mm = specimen.calculate_px_per_mm()
    for i, s in enumerate(specimen.splinters):
        # calculate distance to impact point
        p = np.asarray(s.centroid_mm)
        dp = p - ip

        # distance
        r = np.linalg.norm(dp)

        # get data from splinter
        data = s.get_splinter_data(prop=mode, px_p_mm=px_p_mm, ip_mm=ip)

        # calculate angle to impact
        # print(dv)
        angle = angle_deg(dp)
        # # angle should be between 30 and 60 degrees
        if np.deg2rad(-20) > angle > np.deg2rad(90):
            continue

        aspects[i,:] = (r, data, angle) # (r, a)

    # sort after the radius
    aspects = aspects[aspects[:,0].argsort()]

    # take moving average
    r1,l1,stddev1 = moving_average(aspects[:,0], aspects[:,1], r_range)

    results[0] = specimen.U_d
    results[1:] = l1

    stddevs[0] = results[0]
    stddevs[1:] = stddev1



    # plot the results in a graph
    sz = FigureSize.ROW1HL
    fig,axs = plt.subplots(figsize=get_fig_width(sz))


    # scatterplot aspects
    x_values = aspects[:,0]
    y_values = aspects[:,1]
    c_values = aspects[:,2]

    if not colored_angle:
        c_values = None
        axs.scatter(x_values, y_values, linewidth=0.5, marker='o', color=clr, s=1.5, alpha=0.1)
    else:
        # plot x and y with c as colors
        scatter = axs.scatter(x_values, y_values, c=c_values, cmap='turbo', s=1.5, alpha=0.1)
        # add colorbar
        cbar = fig.colorbar(scatter, label="Winkel zum Anschlagpunkt (°)", ax=axs)
        renew_ticks_cb(cbar)

    # lineplot
    axs.plot(r_range, results[1:], color='r', linestyle='-')

    # count amount of splinters in each bin
    for i in range(len(r_range)):
        if i == len(r_range)-1:
            mask = (aspects[:,0] >= r_range[i])
        else:
            mask = (aspects[:,0] >= r_range[i]) & (aspects[:,0] < r_range[i+1])
        amounts[i] = np.sum(mask)

    if plot_stddev:

        # fill stddev behind lineplot
        axs.fill_between(r_range, results[1:]-stddevs[1:], results[1:]+stddevs[1:], color='r', alpha=0.2)


    # plot amounts to secondary axis
    axs2 = axs.twinx()
    axs2.set_ylabel("Anzahl Bruchstücke")
    axs2.grid(False)
    axs2.plot(r_range, amounts, color='g', linestyle='--')

    axs.set_xlabel("Abstand vom Anschlagpunkt (mm)")
    axs.set_ylabel(Splinter.get_property_label(mode, row3=sz == FigureSize.ROW3))


    State.output(StateOutput(fig,sz), f'graph-{specimen_name}_{mode}', to_additional=True)


@layer_app.command()
def plot_regions(
    d_r: int = 20,
    d_t: int = 360,
    break_pos: SpecimenBreakPosition = SpecimenBreakPosition.CORNER,
    w_mm: int = 500,
    h_mm: int = 500,
    base_specimen_name: str = None
):
    """
    Plots a 2D representation of polar layer regions on a specimen fracture image or white background.
    """
    pxpmm = 5
    r_range, t_range = arrange_regions_px(pxpmm, d_r, d_t, break_pos, w_mm, h_mm)
    print(r_range)
    print(t_range)
    if base_specimen_name is not None:
            base_specimen = Specimen.get(base_specimen_name, load=True)

    img_w = int(w_mm * pxpmm)
    img_h = int(h_mm * pxpmm)
    ip_x,ip_y = break_pos.default_position()

    if base_specimen_name is not None:
        ip_x,ip_y = base_specimen.get_impact_position()

    ip_x = int(ip_x * pxpmm)
    ip_y = int(ip_y * pxpmm)



    stroke = 1 * pxpmm
    wht = (255,255,255)
    blk = (0,0,0)
    red = (0,0,255)
    img = np.full((img_w,img_h,3), blk, dtype=np.uint8)

    r_range = np.asarray(r_range)
    r_max = np.max(r_range) * 1.3
    t_range = np.deg2rad(np.asarray(t_range))
    # add circles
    for r in r_range:
        # draw circle
        cv2.circle(img, (int(ip_x),int(ip_y)), int(r), red, stroke)
        # annotate circle wth radius
        cv2.putText(img, f"{r/pxpmm:.0f}", (int(ip_x+r*np.sqrt(2)/2+5),int(ip_y+r*np.sqrt(2)/2+5)), cv2.FONT_HERSHEY_DUPLEX, 1, red, stroke // 2, bottomLeftOrigin=False)

    for t in t_range:
        cv2.line(img, (int(ip_x),int(ip_y)), (int(ip_x+r_max*np.cos(t)),int(ip_y+r_max*np.sin(t))), red, stroke)
        # annotate at a distance from ip
        cv2.putText(img, f"{np.degrees(t):.0f} deg", (int(ip_x+r_max/2*np.cos(t)),int(ip_y+r_max/2*np.sin(t))), cv2.FONT_HERSHEY_DUPLEX, 1, blk, stroke // 2)

    # mark impact red
    cv2.circle(img, (int(ip_x),int(ip_y)), stroke*2, red, -1)

    if base_specimen_name is not None:
        img_base = base_specimen.get_fracture_image()
        img_base = cv2.resize(img_base, (img_w,img_h))

        # overwrite img with img_base where img is black
        black_pixels = np.all(img == [0, 0, 0], axis=-1)
        img[black_pixels] = img_base[black_pixels]

    State.output(StateOutput(img, FigureSize.ROW1, img_ext = 'jpeg'), f'layer-regions_{d_r}_{d_t}_{break_pos}_{w_mm}x{h_mm}', to_additional=True)


@layer_app.command()
def plot_polar(
    specimen_name: str,
    prop: SplinterProp,
    d_r: float = 25,
    d_t: float = 360,
    plot_std: bool = False,
    plot_counts: bool = False,
):
    """
    Create an overlay of a specific splinter property on a fracture image using radial bands and also plot the mean values
    of each radius over the property.
    """
    specimen = Specimen.get(specimen_name, load=True)
    pxpmm = specimen.calculate_px_per_mm()
    ip_px = specimen.get_impact_position(True)
    realsize = specimen.get_real_size()

    # fetch radii and angles

    r_range,t_range = arrange_regions(d_r,d_t,specimen.get_impact_position(),realsize[0],realsize[1])
    _,_,Z,Zstd,kData = specimen.calculate_2d_polar(prop=prop, r_range_mm=r_range, t_range_deg=t_range, return_data=True)

    # plot the results in a colored plot
    sz = FigureSize.ROW1HL

    img = specimen.get_fracture_image()
    img_overlay = np.zeros_like(img)

    for r in track(range(len(r_range)-1),total=len(r_range)-1,description='Plotting regions...'):
        for t in range(len(t_range)-1):
            r0,r1 = r_range[r],r_range[r+1]
            t0,t1 = t_range[t],t_range[t+1]

            r0,r1,t0,t1 = int(r0*pxpmm),int(r1*pxpmm),int(t0),int(t1)

            # find value for this region
            z = Z[t,r]

            if np.isnan(z):
                continue

            # interpolate color value
            clr = get_color(z, np.nanmin(Z), np.nanmax(Z))
            # print(clr)
            # fill cell with color
            img_overlay = fill_polar_cell(img_overlay, ip_px, t0,t1,r0,r1, clr)

    result_img = cv2.addWeighted(img, 0.5, img_overlay, 0.5, 0.5)

    clr_label = Splinter.get_property_label(prop, row3=sz == FigureSize.ROW3)

    clr_format = ".0f" if np.nanmax(Z) > 10 else ".2f"

    output = annotate_image(
        result_img,
        cbar_range = (np.nanmin(Z),np.nanmax(Z)),
        cbar_title=clr_label,
        clr_format=clr_format,
        figwidth=sz
    )


    print('Results: ', Z)

    State.output(output, f'polar-{specimen_name}_{prop}_{d_r}mm_{d_t}deg', to_additional=True)

    ylabel = Splinter.get_property_label(prop, row3=sz == FigureSize.ROW3)
    fig,axs = plt.subplots(figsize=get_fig_width(sz))
    x_values = r_range[:-1]
    y_values = Z.flatten()
    axs.plot(x_values, y_values, label=ylabel)

    Zstd = Zstd.flatten()
    if np.any(Zstd != 0) and plot_std:
        axs.fill_between(x_values, y_values - Zstd, y_values + Zstd, color='C0', alpha=0.2, edgecolor='none')

    axs2: Axes = None
    if plot_counts:
        axs2 = axs.twinx()
        axs2.plot(x_values, kData.window_object_counts.flatten(), linestyle='--', label='Anzahl Bruchstücke', color='C1')
        axs2.set_ylabel("Anzahl Bruchstücke")
        axs2.grid(False)

    axs.set_xlabel("Abstand zum Anschlagpunkt (mm)")
    axs.set_ylabel(ylabel)

    axs.legend(*get_legend(axs, axs2))
    State.output(StateOutput(fig,sz), f'polargraph-{specimen_name}_{prop}_{d_r}mm_{d_t}deg', to_additional=True)

@layer_app.command()
def plot(
    mode: Annotated[SplinterProp, typer.Argument(help="The mode to display")],
    boundary: Annotated[SpecimenBoundary, typer.Argument(help="Boundary condition")],
    thickness: Annotated[int, typer.Argument(help="The thickness of the specimen")],
    layer: Annotated[ModelLayer, typer.Option(help="The layer to display")] = ModelLayer.IMPACT,
    break_pos: Annotated[SpecimenBreakPosition, typer.Option(help="Break position")] = SpecimenBreakPosition.CORNER,
    stddev: Annotated[bool, typer.Option(help="Plot standard deviation")] = False,
    ignore_nan_plot: Annotated[bool, typer.Option(help="Filter NaN values")] = True,
    file: Annotated[str, typer.Option(help="File that contains the model")] = None,
    figwidth: Annotated[FigureSize, typer.Option(help="Figure width")] = FigureSize.ROW2,
):
    """Plot a layer from database-file."""
    if file is not None:
        R,U,V = load_layer_file(file)
    else:
        model_name = ModelLayer.get_name(layer, mode, boundary, thickness, break_pos, stddev)
        R,U,V = load_layer(model_name)


    print('R:', R)
    print('U:', U)

    xlabel = 'Abstand $R$ zum Anschlagpunkt (mm)'
    ylabel = 'Formänderungsenergie $U$ (J/m²)'

    if figwidth == FigureSize.ROW3:
        xlabel = 'Abstand $R$ (mm)'
        ylabel = '$U$ (J/m²)'

    clabel = Splinter.get_property_label(mode)

    fig = plt_layer(R,U,V,ignore_nan=ignore_nan_plot, xlabel=xlabel, ylabel=ylabel, clabel=clabel,
                    interpolate=True,figwidth=figwidth)
    State.output(StateOutput(fig, figwidth), f"{layer}-2d_{mode}_{boundary}_{thickness}_{break_pos}")
    plt.close(fig)

@layer_app.command()
def test_interpolation(
    mode: Annotated[SplinterProp, typer.Argument(help="The mode to display")],
    boundary: Annotated[SpecimenBoundary, typer.Argument(help="Boundary condition")],
    thickness: Annotated[int, typer.Argument(help="The thickness of the specimen")],
    sigma: Annotated[float, typer.Argument(help="Prestress")],
    layer: Annotated[ModelLayer, typer.Option(help="The layer to display")] = ModelLayer.IMPACT,
    break_pos: Annotated[SpecimenBreakPosition, typer.Option(help="Break position")] = SpecimenBreakPosition.CORNER,
    stddev: Annotated[bool, typer.Option(help="Plot standard deviation")] = False,
    ignore_nan_plot: Annotated[bool, typer.Option(help="Filter NaN values")] = True,
    file: Annotated[str, typer.Option(help="File that contains the model")] = None,
    figwidth: Annotated[FigureSize, typer.Option(help="Figure width")] = FigureSize.ROW2,
):
    """Plot a layer from database-file."""
    energy = U(sigma, thickness)
    print('Energy :', energy)
    layer_f, layer_std = interp_layer(mode, boundary, thickness, break_pos, energy)

    r_range, _ = arrange_regions(d_r_mm=25,d_t_deg=360,break_pos=break_pos,w_mm=500,h_mm=500)

    # extract values using range
    values = layer_f(r_range)
    stddevs = layer_std(r_range)

    fig,axs = plt.subplots(figsize=get_fig_width(figwidth))

    # plot the results in a colored plot
    axs.plot(r_range, values, color='r', linestyle='-')
    # fill stddev
    axs.fill_between(r_range, values - stddevs, values + stddevs, \
                        color='r', alpha=0.2)


    xlabel = 'Abstand $R$ zum Anschlagpunkt (mm)'
    ylabel = Splinter.get_property_label(mode, row3=figwidth == FigureSize.ROW3)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)


    State.output(StateOutput(fig, figwidth), f"{layer}-interpolated_{mode}_{boundary}_{thickness}_{break_pos}")
    plt.close(fig)

# @layer_app.command()
# def plot_layer(
#     specimen_name: Annotated[str, typer.Argument(help="The specimen to display")],
#     mode: Annotated[SplinterProp, typer.Argument(help="The mode to display")],
# ):
#     """Overlay values on fracture image, using 2d kerneler and transforming it into polar coordinates."""
#     specimen = Specimen.get(specimen_name)

#     # calculate mode for every splinter
#     splinter_values = np.zeros((len(specimen.splinters), 2)) # 0: radius, 1: aspect ratio
#     ip = specimen.get_impact_position()
#     s_sz = specimen.get_image_size()
#     px_p_mm = specimen.calculate_px_per_mm()
#     for i, s in enumerate(specimen.splinters):
#         # calculate distance to impact point
#         r = np.linalg.norm(np.asarray(s.centroid_mm) - ip)

#         # get data from splinter
#         a = s.get_splinter_data(prop=mode, px_p_mm=px_p_mm, ip_mm=ip)

#         splinter_values[i,:] = (r, a)

#     # sort after the radius
#     splinter_values = splinter_values[splinter_values[:,0].argsort()]

#     fig,axs=plt.subplots(figsize=get_fig_width(FigureSize.ROW2))

#     # display the splinter fracture image
#     axs.imshow(specimen.get_fracture_image(), cmap='gray')

#     # now, overlay circles corresponding to the radii and the color the value
#     R,V,D = moving_average(splinter_values[:,0], splinter_values[:,1], np.linspace(0, np.max(splinter_values[:,0])))
#     R = R*px_p_mm

#     # Erstellen Sie ein Gitter für den Contour-Plot
#     ip = ip * px_p_mm
#     x = np.linspace(0, s_sz[0], 100) - ip[0]
#     y = np.linspace(0, s_sz[1], 100) - ip[1]
#     x = np.absolute(x)
#     y = np.absolute(y)

#     X, Y = np.meshgrid(x, y, indexing='xy')
#     Z = np.zeros_like(X)
#     Z.fill(np.nan)

#     # Berechnen Sie die Werte für den Contour-Plot
#     for i in range(len(R)-1):
#         if i == len(R):
#             mask = (X**2 + Y**2 >= R[i]**2)
#         else:
#             mask = (X**2 + Y**2 >= R[i]**2) & (X**2 + Y**2 <= R[i+1]**2)

#         Z[mask] = V[i]

#     # replace nan with mean
#     Z_mean = np.nanmean(Z)
#     Z[np.isnan(Z)] = Z_mean

#     output = plot_kernel_results(
#         original_image=specimen.get_fracture_image(),
#         clr_label=Splinter.get_mode_labels(mode),
#         no_ticks=True,
#         plot_vertices=False,
#         mode=KernelContourMode.FILLED,
#         X=X,
#         Y=Y,
#         results=Z,
#         kw_px=None,
#         figwidth=FigureSize.ROW2,
#         clr_format=".2f",
#         crange=None,
#         plot_kernel=False,
#         fill_skipped_with_mean=False,
#         make_border_transparent=False
#     )

#     # for i in range(len(R)):
#     #     c = norm_color(get_color(V[i], np.min(V), np.max(V)))

#     #     # scale radii and impact point to pixels
#     #     r = R[i] * px_p_mm

#     #     cp = axs.contourf(X, Y, Z, levels=np.linspace(np.min(V), np.max(V), 100), cmap='turbo', alpha=0.5)


#     #     # fill a circle with the color without overlapping smaller circles
#     #     # axs.add_patch(plt.Circle(ip, r, color=c, fill=True, linewidth=0, alpha=0.5))

#     #     # axs.add_patch(plt.Circle(ip, r, color=c, fill=True, linewidth=0, alpha=0.5))

#     #     # axs.add_patch(plt.Circle(ip, R[i], color=c, fill=False, linewidth=1))

#     # display the figure
#     State.output(output, f"impact-layer_{mode}", spec=specimen)
