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
from fracsuite.core.model_layers import ModelLayer, load_layer, load_layer_file, plt_layer
from fracsuite.core.plotting import FigureSize, KernelContourMode, get_fig_width, plot_kernel_results, renew_ticks_cb
from fracsuite.core.specimen import Specimen, SpecimenBoundary, SpecimenBreakPosition, SpecimenBreakMode
from fracsuite.core.splinter import Splinter
from fracsuite.core.splinter_props import SplinterProp
from fracsuite.core.stochastics import moving_average
from fracsuite.general import GeneralSettings
from fracsuite.state import State, StateOutput

layer_app = typer.Typer(callback=main_callback, help="Model related commands.")
general = GeneralSettings.get()

@layer_app.command()
def plot(
    layer: Annotated[ModelLayer, typer.Argument(help="The layer to display")],
    mode: Annotated[SplinterProp, typer.Argument(help="The mode to display")],
    boundary: Annotated[SpecimenBoundary, typer.Argument(help="Boundary condition")],
    ignore_nan_plot: Annotated[bool, typer.Option(help="Filter NaN values")] = True,
    file: Annotated[str, typer.Option(help="File that contains the model")] = None,
):
    if file is not None:
        R,U,V = load_layer_file(file)
    else:
        model_name = f'{layer}_{mode}_{boundary}_corner.npy'
        R,U,V = load_layer(model_name)

    xlabel = 'Distance $R$ from Impact [mm]'
    ylabel = 'Elastic Strain Energy U [J/m²]'
    clabel = Splinter.get_mode_labels(mode)

    fig = plt_layer(R,U,V,ignore_nan=ignore_nan_plot, xlabel=xlabel, ylabel=ylabel, clabel=clabel,
                    interpolate=False)
    State.output(StateOutput(fig, FigureSize.ROW2), f"{layer}-2d_{mode}_{boundary}")
    plt.close(fig)

@layer_app.command()
def plot_impact_layer(
    specimen_name: Annotated[str, typer.Argument(help="The specimen to display")],
    mode: Annotated[SplinterProp, typer.Argument(help="The mode to display")],
):
    specimen = Specimen.get(specimen_name)

    # calculate mode for every splinter
    splinter_values = np.zeros((len(specimen.splinters), 2)) # 0: radius, 1: aspect ratio
    ip = specimen.get_impact_position()
    s_sz = specimen.get_image_size()
    px_p_mm = specimen.calculate_px_per_mm()
    for i, s in enumerate(specimen.splinters):
        # calculate distance to impact point
        r = np.linalg.norm(np.asarray(s.centroid_mm) - ip)

        # get data from splinter
        a = s.get_splinter_data(prop=mode, px_p_mm=px_p_mm, ip=ip)

        splinter_values[i,:] = (r, a)

    # sort after the radius
    splinter_values = splinter_values[splinter_values[:,0].argsort()]

    fig,axs=plt.subplots(figsize=get_fig_width(FigureSize.ROW2))

    # display the splinter fracture image
    axs.imshow(specimen.get_fracture_image(), cmap='gray')

    # now, overlay circles corresponding to the radii and the color the value
    R,V,D = moving_average(splinter_values[:,0], splinter_values[:,1], np.linspace(0, np.max(splinter_values[:,0])))
    R = R*px_p_mm

    # Erstellen Sie ein Gitter für den Contour-Plot
    ip = ip * px_p_mm
    x = np.linspace(0, s_sz[0], 100) - ip[0]
    y = np.linspace(0, s_sz[1], 100) - ip[1]
    x = np.absolute(x)
    y = np.absolute(y)

    X, Y = np.meshgrid(x, y, indexing='xy')
    Z = np.zeros_like(X)
    Z.fill(np.nan)

    # Berechnen Sie die Werte für den Contour-Plot
    for i in range(len(R)-1):
        if i == len(R):
            mask = (X**2 + Y**2 >= R[i]**2)
        else:
            mask = (X**2 + Y**2 >= R[i]**2) & (X**2 + Y**2 <= R[i+1]**2)

        Z[mask] = V[i]

    # replace nan with mean
    Z_mean = np.nanmean(Z)
    Z[np.isnan(Z)] = Z_mean

    output = plot_kernel_results(
        original_image=specimen.get_fracture_image(),
        clr_label=Splinter.get_mode_labels(mode),
        no_ticks=True,
        plot_vertices=False,
        mode=KernelContourMode.FILLED,
        X=X,
        Y=Y,
        results=Z,
        kw_px=None,
        figwidth=FigureSize.ROW2,
        clr_format=".2f",
        crange=None,
        plot_kernel=False,
        fill_skipped_with_mean=False,
        make_border_transparent=False
    )

    # for i in range(len(R)):
    #     c = norm_color(get_color(V[i], np.min(V), np.max(V)))

    #     # scale radii and impact point to pixels
    #     r = R[i] * px_p_mm

    #     cp = axs.contourf(X, Y, Z, levels=np.linspace(np.min(V), np.max(V), 100), cmap='turbo', alpha=0.5)


    #     # fill a circle with the color without overlapping smaller circles
    #     # axs.add_patch(plt.Circle(ip, r, color=c, fill=True, linewidth=0, alpha=0.5))

    #     # axs.add_patch(plt.Circle(ip, r, color=c, fill=True, linewidth=0, alpha=0.5))

    #     # axs.add_patch(plt.Circle(ip, R[i], color=c, fill=False, linewidth=1))

    # display the figure
    State.output(output, f"impact-layer_{mode}", spec=specimen)

@layer_app.command()
def create_impact_layer(
    mode: Annotated[SplinterProp, typer.Argument(help='Mode for the aspect ratio.')],
    break_pos: Annotated[SpecimenBreakPosition, typer.Option(help='Break position.')] = SpecimenBreakPosition.CORNER,
    break_mode: Annotated[SpecimenBreakMode, typer.Option(help='Break mode.')] = SpecimenBreakMode.PUNCH,
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

        if break_mode is not None and specimen.break_mode != break_mode:
            return False

        if specimen.boundary == SpecimenBoundary.Unknown:
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
    ylabel_short = Splinter.get_mode_labels(mode, row3=True)
    clabel = "Strain Energy U [J/m²]"


    # find aspect over R on every specimen
    n_r = 30
    r_min = 0
    r_max = np.sqrt(450**2 + 450**2)
    r_range = np.linspace(r_min, r_max, n_r)



    ########################
    # Calculate value for every specimen and save results to arrays
    ########################
    results = np.ones((len(specimens), len(r_range)+3)) * np.nan
    stddevs = np.ones((len(specimens), len(r_range)+3)) * np.nan

    for si, specimen in track(list(enumerate(specimens))):
        print('Specimen: ', specimen.name)

        # now, find aspect ratio of all splinters
        aspects = np.zeros((len(specimen.splinters), 2)) # 0: radius, 1: aspect ratio
        ip = specimen.get_impact_position()
        px_p_mm = specimen.calculate_px_per_mm()
        for i, s in enumerate(specimen.splinters):
            # calculate distance to impact point
            r = np.linalg.norm(np.asarray(s.centroid_mm) - ip)

            # get data from splinter
            a = s.get_splinter_data(prop=mode, px_p_mm=px_p_mm, ip=ip)

            aspects[i,:] = (r, a) # (r, a)

        # sort after the radius
        aspects = aspects[aspects[:,0].argsort()]

        # take moving average
        r1,l1,stddev1 = moving_average(aspects[:,0], aspects[:,1], r_range)

        results[si,0] = specimen.U
        results[si,1] = bid[specimen.boundary]
        results[si,2] = si
        results[si,3:] = l1

        stddevs[si,0] = results[si,0]
        stddevs[si,1] = results[si,1]
        stddevs[si,2] = results[si,2]
        stddevs[si,3:] = stddev1

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
    for b in results_per_boundary_unclustered:
        b_mask = results_per_boundary_unclustered[b][0][:,0].argsort()
        results_b = results_per_boundary_unclustered[b][0][b_mask]
        stddev_b = results_per_boundary_unclustered[b][1][b_mask]

        # plot the current boundary
        x = r_range
        y = results_b[:,0].ravel()
        z = results_b[:,3:]

        # save model
        Z_path = State.get_output_file(f'{ModelLayer.IMPACT}_{mode}_{bid_r[b]}_{break_pos}.npy')
        print(f'> Saving impact layer {mode}_{bid_r[b]}_{break_pos} to {Z_path}')
        data = np.zeros((len(y)+1, len(x)+1))
        data[1:,1:] = z
        data[1:,0] = y
        data[0,1:] = x
        np.save(Z_path, data)
        stddev_path = State.get_output_file(f'{ModelLayer.IMPACT}-stddev_{mode}_{bid_r[b]}_{break_pos}.npy')
        data_stddev = np.zeros((len(y)+1, len(x)+1))
        data_stddev[1:,1:] = stddev_b[:,3:]
        data_stddev[1:,0] = stddev_b[:,0].ravel()
        data_stddev[0,1:] = r_range
        np.save(stddev_path, data_stddev)

        plot(ModelLayer.IMPACT, mode=mode, boundary=bid_r[b], ignore_nan_plot=ignore_nan_u,
             file=Z_path)


        # this plots the standard deviation between different energy levels
        c_stddev = np.nanstd(z, axis=0)
        # this however plots the maximum difference between lowest and highest property value
        max_diff = np.abs((np.nanmax(z,axis=0) - np.nanmin(z,axis=0)))
        # plot max_dif over r
        axs.plot(x, max_diff, label=bid_r[b], linestyle=boundaries[b])

    axs.set_xlabel(xlabel)
    axs.set_ylabel(f"Influence of energy on {ylabel_short}")
    axs.legend()
    State.output(StateOutput(fig,sz), f'max-diff_{mode}_{break_pos}', to_additional=True)

    ########################
    # cluster results for a range of energies and save plots
    ########################

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
