from typing import Annotated

import numpy as np
import typer
from matplotlib import colors as pltc
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.progress import track

from fracsuite.callbacks import main_callback
from fracsuite.core.coloring import get_color, norm_color
from fracsuite.core.lbreak import ModelBoundary, load_layer, plt_layer
from fracsuite.core.plotting import FigureSize, get_fig_width, renew_ticks_cb
from fracsuite.core.specimen import Specimen
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stochastics import moving_average
from fracsuite.general import GeneralSettings
from fracsuite.state import State, StateOutput

layer_app = typer.Typer(callback=main_callback, help="Model related commands.")
general = GeneralSettings.get()

@layer_app.command()
def plot(
    layer: Annotated[str, typer.Argument(help="The layer to display")],
    mode: Annotated[SplinterProp, typer.Argument(help="The mode to display")],
    boundary: Annotated[ModelBoundary, typer.Argument(help="Boundary condition")],
    ignore_nan_plot: Annotated[bool, typer.Option(help="Filter NaN values")] = True,
):
    model_name = f'{layer}-layer_{mode}_{boundary}_corner.npy'
    R,U,V = load_layer(model_name)

    xlabel = 'Distance $R$ from Impact [mm]'
    ylabel = 'Elastic Strain Energy U [J/m²]'
    clabel = Splinter.get_mode_labels(mode)

    fig = plt_layer(R,U,V,ignore_nan=ignore_nan_plot, xlabel=xlabel, ylabel=ylabel, clabel=clabel,
                    interpolate=False)
    State.output(StateOutput(fig, FigureSize.ROW2), f"impact-layer-2d_{mode}_{boundary}")
    plt.close(fig)


@layer_app.command()
def create_impact_layer(
    break_pos: Annotated[str, typer.Option(help='Break position.')] = 'corner',
    mode: Annotated[SplinterProp, typer.Option(help='Mode for the aspect ratio.')] = 'asp',
    ignore_nan_u: Annotated[bool, typer.Option(help='Filter Ud values that are NaN from plot.')] = False,
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

    def add_filter(specimen: Specimen):
        if break_pos is not None and specimen.break_pos != break_pos:
            return False

        if specimen.boundary == "":
            return False

        if not specimen.has_splinters:
            return False

        if specimen.U_d is None or not np.isfinite(specimen.U_d):
            return False

        return True

    specimens: list[Specimen] = Specimen.get_all_by(add_filter, lazyload=False)


    sz = FigureSize.ROW1

    xlabel = "Distance from impact R [mm]"
    ylabel = Splinter.get_mode_labels(mode, row3=sz == FigureSize.ROW3)
    clabel = "Strain Energy U [J/m²]"


    # find aspect over R on every specimen
    n_r = 30
    r_min = 0
    r_max = np.sqrt(450**2 + 450**2)
    r_range = np.linspace(r_min, r_max, n_r)

    results = np.ones((len(specimens), len(r_range)+3)) * np.nan

    # all other rows: u_d, boundary, aspect ratios

    d_rand = 15 #mm

    for si, specimen in track(list(enumerate(specimens))):
        print('Specimen: ', specimen.name)

        # now, find aspect ratio of all splinters
        aspects = np.zeros((len(specimen.splinters), 2)) # 0: radius, 1: aspect ratio
        ip = specimen.get_impact_position()
        s_sz = specimen.get_real_size()
        px_p_mm = specimen.calculate_px_per_mm()
        for i, s in enumerate(specimen.splinters):
            # skip splinters that are too close to the edge
            if s.centroid_mm[0] > s_sz[0]-d_rand or s.centroid_mm[0] < d_rand or s.centroid_mm[1] > s_sz[1]-d_rand or s.centroid_mm[1] < d_rand:
                aspects[i,:] = (np.nan, np.nan)
                continue

            # calculate distance to impact point
            r = np.linalg.norm(np.asarray(s.centroid_mm) - ip)

            # get data from splinter
            a = s.get_splinter_data(mode=mode, px_p_mm=px_p_mm, ip=ip)

            aspects[i,:] = (r, a) # (r, a)

        # sort after the radius
        aspects = aspects[aspects[:,0].argsort()]

        # take moving average
        r1,l1 = moving_average(aspects[:,0], aspects[:,1], r_range)

        results[si,0] = specimen.U
        results[si,1] = bid[specimen.boundary]
        results[si,2] = si
        results[si,3:] = l1
        # except:
        #     results[si+1,0] = specimen.U
        #     results[si+1,1] = bid[specimen.boundary]
        #     results[si+1,2] = si
        #     results[si+1,3:] = aspects[:,1]


    # sort results after U_d
    results = results[results[:,0].argsort()]



    results_per_boundary_unclustered = {
        1: [],
        2: [],
        3: []
    }

    # find all results for every boundary
    mask_A = results[:,1] == 1
    mask_B = results[:,1] == 2
    mask_Z = results[:,1] == 3

    # put results in dictionary
    results_per_boundary_unclustered[1] = results[mask_A,:]
    results_per_boundary_unclustered[2] = results[mask_B,:]
    results_per_boundary_unclustered[3] = results[mask_Z,:]

    # sort the individual results after U_d
    for b in results_per_boundary_unclustered:
        results_per_boundary_unclustered[b] = \
            results_per_boundary_unclustered[b][results_per_boundary_unclustered[b][:,0].argsort()]

        # plot the current boundary
        x = r_range
        y = results_per_boundary_unclustered[b][:,0].ravel()
        z = results_per_boundary_unclustered[b][:,3:]

        # display the figure
        fig = plt_layer(x,y,z, xlabel=xlabel, ylabel=clabel, clabel=ylabel, ignore_nan=False,interpolate=False)

        # save output

        # save model
        print(f'> Saving impact layer {mode}_{bid_r[b]}_{break_pos}')
        Z_path = State.get_general_outputfile(f'layer/impact-layer_{mode}_{bid_r[b]}_{break_pos}.npy')
        data = np.zeros((len(y)+1, len(x)+1))
        data[1:,1:] = z
        data[1:,0] = y
        data[0,1:] = x
        np.save(Z_path, data)

        plot("impact", mode=mode, boundary=bid_r[b], ignore_nan_plot=ignore_nan_u)


    # results for interpolation with different boundaries
    results_per_boundary = {
        1: [],
        2: [],
        3: []
    }

    n_ud = 30
    ud_min = np.min(results[:,0])
    ud_max = np.max(results[:,0])
    ud_range = np.linspace(ud_min, ud_max, n_ud)


    fig,axs = plt.subplots(figsize=get_fig_width(sz))

    bfigs: list[tuple[Figure,Axes]] = []
    row3size = get_fig_width(FigureSize.ROW3)
    bfigs.append(plt.subplots(figsize=row3size))
    bfigs.append(plt.subplots(figsize=row3size))
    bfigs.append(plt.subplots(figsize=row3size))
    # cluster results for a range of energies
    for i in range(n_ud):
        for ib, b in enumerate(boundaries):
            # gruppieren von Ergebnissen nach erstem Eintrag (U,Ud,...)
            if i == n_ud-1:
                mask = (results[:,0] >= ud_range[i]) & (results[:,1] == b)
            else:
                mask = (results[:,0] >= ud_range[i]) & (results[:,0] < ud_range[i+1]) & (results[:,1] == b)

            # get mean of all results (specimens) in this range
            mean = np.nanmean(results[mask,3:], axis=0)

            clr = norm_color(get_color(ud_range[i], ud_min, ud_max))

            # plot all masked results as scatter plots
            for j in range(len(results)):
                if mask[j]:
                    axs.scatter(r_range, results[j,3:], color=clr, marker='x', linewidth=0.5, s=1.5)
                    bfigs[ib][1].scatter(r_range, results[j,3:], color=clr, marker='x', linewidth=0.5, s=1.5)

                    id = int(results[j,2])
                    s = specimens[id]
                    print(f"{s.name} ({np.nanmean(results[j,3:]):.1f})")

            ls = boundaries[b]
            alpha = 0.7
            axs.plot(r_range, mean,  color=clr, linestyle=ls, alpha=alpha)
            bfigs[ib][1].plot(r_range, mean,  color=clr, linestyle='-', alpha=1)
            # label=f"{uds[i]:.2f} - {uds[i+1]:.2f} J/m²",
            results_per_boundary[b].append(mean)



    axs.set_xlabel(xlabel)
    # plt.ylabel("Splinter Orientation Strength $\Delta$ [-]")
    axs.set_ylabel(ylabel)
    # create legend for boundaries
    axs.plot([], [], label="A", linestyle=boundaries[1], color='k')
    axs.plot([], [], label="B", linestyle=boundaries[2], color='k')
    axs.plot([], [], label="Z", linestyle=boundaries[3], color='k')
    colors = [norm_color(get_color(x, ud_min, ud_max)) for x in ud_range]
    cmap = pltc.ListedColormap(colors)
    norm = pltc.Normalize(np.min(ud_range), np.max(ud_range))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label=clabel, ax=axs)
    renew_ticks_cb(cbar)
    axs.legend(loc='upper right')

    if mode == 'asp':
        # plt.ylim((0.9, 2.6))
        pass
    elif mode == 'area':
        axs.set_ylim((0, 30))

    State.output(StateOutput(fig,sz), f'U_{mode}_all_{break_pos}_nr{n_r}_nud{n_ud}', to_additional=True)

    xlabel = "Distance R [mm]"
    ylabel = Splinter.get_mode_labels(mode, row3=True)
    clabel = "U [J/m²]"
    for i_f, (f,a) in enumerate(bfigs):
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        cbar = f.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label=clabel, ax=a)
        renew_ticks_cb(cbar)
        State.output(StateOutput(f,FigureSize.ROW3), f'U_{mode}_{bid_r[i_f+1]}_{break_pos}_nr{n_r}_nud{n_ud}', to_additional=True)

    # ##
    # ## PLOT interpolated data as 2d plot for every boundary
    # print('> Plotting interpolated data...')
    # # num_results[ud,R]
    # for br_key in results_per_boundary: # different boundaries
    #     print('Boundary: ', br_key)
    #     nr_r = np.asarray(results_per_boundary[br_key])

    #     # format data storage for saving and interpolation
    #     x = r_range
    #     y = ud_range
    #     data = np.zeros((len(y)+1, len(x)+1))
    #     print(nr_r.shape)
    #     data[1:,1:] = nr_r
    #     data[1:,0] = ud_range
    #     data[0,1:] = r_range

    #     # this should save the original data that has not been clustered
    #     Z_path = State.get_general_outputfile(f'model/interpolate_{mode}_{bid_r[br_key]}_{break_pos}.npy')
    #     np.save(Z_path, data)

    #     fig = plt_model(x,y,nr_r, xlabel=xlabel, ylabel=clabel, clabel=ylabel, exclude_nan=exclude_nan_u)
    #     State.output(StateOutput(fig, FigureSize.ROW1), f'2d-{mode}_{bid_r[br_key]}_{break_pos}_n{n}_nud{n_ud}', to_additional=True)
