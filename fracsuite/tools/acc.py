"""
Acceleration tools.
"""

import os
from typing import Annotated
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
import numpy as np
import typer
from apread import APReader
from rich import print
from rich.progress import track

from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import find_file
from fracsuite.tools.specimen import Specimen

app = typer.Typer(help=__doc__)
general = GeneralSettings.get()

ns = 1e-9
us = 1e-6
ms = 1e-3

def mod_unit(unit):
    """
    Calculate the factor to convert the given unit to seconds.
    Also modifies the global ns, us, ms variables.
    """
    global ns, us, ms

    time_f = 1
    if unit == "ns":
        time_f = 1000000000
        # ns = 1
        # us = 1e3
        # ms = 1e6
    elif unit == "us":
        time_f = 1000000
        # ns = 1e-3
        # us = 1
        # ms = 1e3
    elif unit == "ms":
        time_f = 1000
        # ms = 1
        # us = 1e-3
        # ns = 1e-6
    elif unit == "s":
        time_f = 1
    else:
        raise Exception(f"Unknown time unit '{unit}'.")

    return time_f

def fft(data, time, plot=False, title=""):
    """Calculates the fft of the given data."""
    # calculate the fft
    fft = np.fft.fft(data)
    fft = np.abs(fft)
    fft = fft[:int(len(fft)/2)]

    # calculate the frequencies
    freq = np.fft.fftfreq(len(fft), time[1]-time[0])
    freq = np.abs(freq)


    # plot the fft
    if plot:
        plt.plot(freq, fft)
        # plt.hist(fft, bins=np.linspace(freq[0], freq[-1], 100), density=True)
        plt.title(title)
        plt.show()

    return freq, fft

def find_impact_time(data, time):
    mean_drop_g = np.max(np.abs(data[:18000]))*1.001
    xx1 = np.abs(data/mean_drop_g)**10
    impact_time_i: int = np.argwhere(xx1 >= 1)[0]-2
    impact_time = time.data[int(impact_time_i)]

    return impact_time_i, impact_time

def reader_to_csv(reader: APReader, out_dir, dot: str = "."):
    """Writes the reader data to a csv file."""
    # create csv file
    csv_file = os.path.join(out_dir, f"{reader.fileName}.csv")
    with open(csv_file, 'w') as f:
        # write header
        f.write(";")
        for chan in reader.Channels:
            f.write(f"{chan.Name} [{chan.unit}];")
        f.write("\n")

        # write header
        f.write("Maximum;")
        for chan in reader.Channels:
            f.write(f"{np.max(chan.data)};")
        f.write("\n")

        f.write("Minimum;")
        for chan in reader.Channels:
            f.write(f"{np.min(chan.data)};")
        f.write("\n")

        f.write("Time of Maximum;")
        for chan in reader.Channels:
            max_i = np.argmax(chan.data)
            if not chan.isTime:
                time = chan.Time.data[max_i]
                f.write(f"{time};")
            else:
                f.write(";")
        f.write("\n")

        # write data
        max_len = np.max([len(x.data) for x in reader.Channels])
        for i in track(range(0, max_len)):
            f.write(";")
            for g in reader.Groups:
                if i < len(g.ChannelX.data):
                    f.write(f"{g.ChannelX.data[i]};")
                for chan in g.ChannelsY:
                    if i < len(chan.data):
                        f.write(f"{chan.data[i]};")
                    else:
                        f.write(";")
            f.write("\n")

    with open(csv_file, 'r') as f:
        content = f.read()
        content = content.replace(".", dot)
    with open(csv_file, 'w') as f:
        f.write(content)

def convert_time(t, unit):
    if unit == "ns":
        t *= 1000000000
    elif unit == "us":
        t *= 1000000
    elif unit == "ms":
        t *= 1000

    return t

@app.command()
def wave_runtime(velocity_mps: float,
                 distance_mm: float,
                 unit: str = "s"
                ):
    """Calculates the runtime of a wave."""
    t = distance_mm/velocity_mps/1e3

    t = convert_time(t, unit)

    print(f"Runtime: {t:.2f} {unit}")

@app.command()
def freq_calc(
    frequency: float,
    unit: str = "s"
):
    """Calculates the period time of a frequency."""
    t = 1/frequency

    t = convert_time(t, unit)

    print(f"Runtime: {t:.2f} {unit}")

@app.command()
def integrate_fall(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
    time_unit: Annotated[str, typer.Option(help="The unit to show on the x-axis.", )] = "s",
    normalize_time: Annotated[bool, typer.Option('--normalize-time', help="Move 0-time to impact.", )] = False,
    file: Annotated[str, typer.Option(help="The file to plot.", )] = None,
    fig_title: Annotated[str, typer.Option(help="Title of the figure.", )] = None,
    apply_filter: Annotated[bool, typer.Option(help="Apply filter function.", )] = False,
):
    """Integrate the weight acceleration twice to get velocity and displacement."""
    if file is None:
        specimen = Specimen.get(specimen_name)

        reader = APReader(specimen.acc_file)
    else:
        reader = APReader(file)

    reader.printSummary()


    g_channels = reader.collectChannelsLike('Acc')
    drop_channels = reader.collectChannelsLike('Fall_g')

    imp_time, imp_time_i = find_impact_time(drop_channels[0].data, drop_channels[0].Time.data)

    g = 9.81
    drop_acc = drop_channels[0]
    drop_avg = np.average(drop_acc.data[-60000:])
    drop_data = drop_acc.data - drop_avg
    drop_data = drop_data * g

    a = drop_data # savgol_filter(drop_data, 51, 5)
    v = cumulative_trapezoid(a, drop_acc.Time.data, initial=0)
    s = cumulative_trapezoid(v, drop_acc.Time.data, initial=0)

    fig, axs = plt.subplots(figsize=general.figure_size)
    axa = axs
    axv = axs.twinx()
    axss = axs.twinx()

    axs.set_xlabel(f"Time [{time_unit}]")
    axa.set_ylabel("Acceleration [g]")
    axv.set_ylabel("Velocity [m/s]")
    axss.set_ylabel("Distance [m]")

    axa.plot(drop_channels[0].Time.data, a, 'r-',  label='Acceleration')
    axv.plot(drop_channels[0].Time.data, v, 'g-', label='Velocity')
    axss.plot(drop_channels[0].Time.data, s, 'k-', label='Distance')
    plt.legend()
    plt.show()

def tti(t, time):
    return np.argmin(np.abs(time - t))

@app.command()
def plot_impact(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
    time_unit: Annotated[str, typer.Option(help="The unit to show on the x-axis.", )] = "s",
    normalize_time: Annotated[bool, typer.Option('--normalize-time', help="Move 0-time to impact.", )] = False,
    file: Annotated[str, typer.Option(help="The file to plot.", )] = None,
    fig_title: Annotated[str, typer.Option(help="Title of the figure.", )] = None,
    apply_filter: Annotated[bool, typer.Option(help="Apply filter function.", )] = False,
    zoom_in: Annotated[bool, typer.Option(help="Zoom to impact.", )] = True,
    no_legend: Annotated[bool, typer.Option(help="Hide the plot legend.", )] = False,
    show_title: Annotated[bool, typer.Option(help="Show a title in the plot.", )] = False,
    mpss: Annotated[bool, typer.Option(help="Show velocity in m/s.", )] = False,
    plot_filtered_sep: Annotated[bool, typer.Option('--plot-sep', help="Draw the filtered function seperately.", )] = False
):
    """Plots the impact of the given specimen."""
    time_f = mod_unit(time_unit)

    g = 9.81
    prim_sensors = ['2', '6']
    sec_sensors = ['1', '3', '4', '5']

    if file is None:
        specimen = Specimen.get(specimen_name)

        # wave and crackfront velocities
        v_p = 5500 * 1e3
        v_s = 3500 * 1e3
        v_c = 1500 * 1e3

        # distances
        d_p = 450   # primary wave first registered on 2nd sensor
        d_s = 400   # secondary wave first registered on 1st sensor
        d_c = np.sqrt(450**2 + 450**2) # diagonal distance when crack is finished

        if specimen.break_pos == "center":
            d_p = 250
            d_s = 200
            d_c = np.sqrt(250**2 + 250**2)



        prim_runtime = (d_p / v_p)
        sec_runtime = (d_s / v_s)
        crackfront_runtime = (d_c / v_c)

        reader = APReader(specimen.acc_file)
    else:
        prim_runtime = 0
        sec_runtime = 0
        crackfront_runtime = 0
        reader = APReader(file)

    reader.printSummary()

    # get the channels
    g_channels = reader.collectChannelsLike('Acc')
    drop_channels = reader.collectChannelsLike('Fall_g')
    # g_channels = reader.collectChannelsLike('shock')
    # drop_channels = reader.collectChannelsLike('Force')

    mean_drop_g = np.max(np.abs(drop_channels[0].data[:10000]))*1.5
    xx1 = np.abs(drop_channels[0].data/mean_drop_g)**10
    impact_time_i: int = np.argwhere(xx1 >= 1)[0]-2
    impact_time = drop_channels[0].Time.data[impact_time_i]
    time0 = drop_channels[0].Time.data
    dtime = impact_time if normalize_time else 0

    before = impact_time - 30 * us
    after = impact_time + crackfront_runtime + 100 * us

    g_data = []
    # collect channel data and their times
    for chan in g_channels:
        data = chan.data * (g if mpss else 1.0)
        fdata = None
        # apply a filter to chan.data
        fdata = savgol_filter(data, 9, 3)
        # freq, dfft = fft(data, chan.Time.data)
        imp_data = data[int(impact_time_i+1000):] # int(impact_time_i+3000)
        freq, dfft = fft(imp_data, chan.Time.data, plot=False,title=chan.Name)
        g_data.append((chan, chan.Time.data, data, fdata, freq))

    drop_data = []
    # collect channel data and their times
    for chan in drop_channels:
        drop_data.append((chan, chan.Time.data, chan.data))

    # plot the data

    figsize = general.figure_size
    if not no_legend:
        figsize = (figsize[0] * 1.2, figsize[1])

    fig, ax = plt.subplots(figsize=figsize)
    if show_title:
        fig.suptitle(fig_title or f"Impact of specimen '{specimen_name}'")
    ax.set_xlabel(f"Time [{time_unit}]")
    ax.set_ylabel(f"Acceleration [{'g' if not mpss else 'm/s²'}]")
    ax.grid()


    # plot the g channels
    for chan, time, data, fdata, freq in g_data:
        time = time - dtime
        linestyle = '-'

        if any([x in chan.Name for x in prim_sensors]):
            linestyle = (0, (5, 1))

        line = ax.plot(time, data, linestyle=linestyle, label=chan.Name)
        if plot_filtered_sep and not apply_filter:
            ax.plot(time, fdata, "--", color = line[0].get_color(), label=chan.Name + " (filtered)")

    impact_time = impact_time - dtime
    # plot the impact time
    ax.axvline(impact_time, color="red")
    ax.text(impact_time, 0.01, 'Impact',rotation=90, va='bottom', transform=ax.get_xaxis_transform())
    ax.axvline(impact_time + prim_runtime, color='k', linestyle='--')
    ax.text(impact_time + prim_runtime, 0.01, 'P-Wave',rotation=90, va='bottom', transform=ax.get_xaxis_transform())
    ax.axvline(impact_time + sec_runtime, color='k', linestyle='--')
    ax.text(impact_time + sec_runtime, 0.01, 'S-Wave',rotation=90, va='bottom', transform=ax.get_xaxis_transform())
    ax.axvline(impact_time + crackfront_runtime, color='k', linestyle='--')
    ax.text(impact_time + crackfront_runtime, 0.01, 'Glass broken',rotation=90, va='bottom', transform=ax.get_xaxis_transform())

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*time_f:.2f}"))

    # plot the drop channels
    for chan, time, data in drop_data:
        time = time - dtime
        ax.plot(time, data, linestyle=(0, (3, 1, 1, 1)), label=chan.Name)

    # plot the 0.5s before and 3 seconds after the impact

    if zoom_in:
        ax.set_xlim(before - dtime, after - dtime)
        ib = tti(before, time0)
        ia = tti(after, time0)

        y_max= [np.max(x[ib:ia]) for _, _, x, _, _ in g_data]
        y_max = np.max(y_max) * 1.3
        ax.set_ylim((-y_max, y_max))

    if not no_legend:
        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))

    fig.tight_layout()
    plt.show()

    if specimen is not None:
        fig.savefig(specimen.get_acc_outfile(f'impact_w_waves.{general.image_extension}'), bbox_inches="tight")

    return
    drop1 = reader.collectChannels(['Fall_g1'])
    drop2 = reader.collectChannels(['Fall_g2'])
    time1 = drop1.Time
    time2 = drop2.Time

    # find peak in drop1
    peak1_i = np.argmax(drop1[0].data)
    time_peak = time1.data[peak1_i]


    fig = plt.figure()



@app.command()
def to_csv(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
    number_dot: Annotated[str, typer.Option(help="Number format dot.")] = ".",
    plot: Annotated[bool, typer.Option(help="Plot the reader before saving.")] = False):
    """Converts the given specimen to a csv file."""
    specimen = Specimen.get(specimen_name)

    acc_path = os.path.join(specimen.path, "fracture", "acceleration")
    acc_file = find_file(acc_path, "*.BIN")

    if acc_file is None:
        print(f"Could not find acceleration file for specimen '{specimen_name}'.")
        return

    reader = APReader(acc_file)

    if plot:
        reader.plot()

    reader_to_csv(reader, acc_path, number_dot)
