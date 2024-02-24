"""
Acceleration tools.
"""

import os
import re
from typing import Annotated
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
import numpy as np
import typer
from apread import APReader, Channel
from rich import print
from rich.progress import track
from fracsuite.core.accelerationdata import AccelerationData
from fracsuite.core.plotting import FigureSize, get_fig_width
from fracsuite.core.signal import lowpass
from fracsuite.state import State

from fracsuite.general import GeneralSettings
from fracsuite.helpers import find_file
from fracsuite.callbacks import main_callback
from fracsuite.core.specimen import Specimen
from fracsuite.splinters import create_filter_function

app = typer.Typer(help=__doc__, callback=main_callback)
general = GeneralSettings.get()

ns = 1e-9       # nanoseconds factor
us = 1e-6       # microseconds factor
ms = 1e-3       # milliseconds factor

prim_sensors = ['2', '6']           # sensor nbr that measure the primary wave
sec_sensors = ['1', '3', '4', '5']  # sensor nbr that measure the secondary wave

drop_ls = (0, (3, 1, 1, 1))     # linestyle for drop sensors
prim_ls = (0, (5, 1))           # linestyle for primary wave
sec_ls = '-'                    # linestyle for secondary wave


def set_prim_sec_sensors(specimen: Specimen):
    """Sets the primary and secondary sensors for the given specimen."""
    global prim_sensors, sec_sensors
    if specimen.break_pos == "center":
        prim_sensors, sec_sensors = ['2', '4'], ['1', '3', '6', '5']
    else:
        prim_sensors, sec_sensors = ['2', '6'], ['1', '3', '4', '5']

def annotate_runtimes(specimen: Specimen, ax: Axes, with_text = True) -> tuple[float, float, float]:
    # wave and crackfront velocities
    v_p = 5500 * 1e3 # mm/s
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

    ax.axvline(0, color="red", linestyle=drop_ls)
    ax.axvline(prim_runtime, linestyle=prim_ls)
    ax.axvline(sec_runtime, linestyle =sec_ls)
    ax.axvline(crackfront_runtime, color='k', linestyle='-')
    if with_text:
        ax.text(0, 0.01, 'Impact',rotation=90, va='bottom', transform=ax.get_xaxis_transform())
        ax.text(prim_runtime, 0.01, 'P-Wave',rotation=90, va='bottom', transform=ax.get_xaxis_transform())
        ax.text(sec_runtime, 0.01, 'S-Wave',rotation=90, va='bottom', transform=ax.get_xaxis_transform())
        ax.text(crackfront_runtime, 0.01, 'Glass broken',rotation=90, va='bottom', transform=ax.get_xaxis_transform())

    return prim_runtime, sec_runtime, crackfront_runtime


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
        unit = "µs"
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

    return time_f, unit

def fft_calc(data, time, plot=False, title=""):
    """Calculates the fft of the given data."""
    # calculate the fft
    fft = np.fft.fft(data)
    fft = np.abs(fft)

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
def wave_compare(
    name_filter: Annotated[str, typer.Argument(help="The name filter to use.")],
    sigmas: Annotated[str, typer.Option(help="The sigmas to plot.", )] = "all",
    time_unit: Annotated[str, typer.Option(help="The unit to show on the x-axis.", )] = "s",
    sensor_nr: Annotated[str, typer.Option(help="The sensors to plot.", )] = "2",
    sensor_filter: Annotated[str, typer.Option(help="The sensor filter. When set, sensor-nr is ignored!", )] = None,
    out: Annotated[str, typer.Option(help='Output file.')] = None,
    annotate: Annotated[bool, typer.Option(help='Annotate wave runtime into plots.')] = True,
    annotate_text: Annotated[bool, typer.Option(help='Annotate text to wave-runtimes.')] = True,
):
    tf, tunit = mod_unit(time_unit)

    filter_func = create_filter_function(name_filter, sigmas)
    specimens: list[Specimen] = Specimen.get_all_by(filter_func, load=False)

    fig, axs = plt.subplots(len(specimens), 1, sharey='all', sharex='all')

    # specimens = sorted(specimens, key=lambda x: x.sig_h)

    for i,specimen in enumerate(specimens):
        ax: Axes = axs[i]

        set_prim_sec_sensors(specimen)

        reader = APReader(specimen.acc_file)

        drop = reader.collectChannels('Fall_g1')[0]

        if sensor_filter is None:
            channels = reader.collectChannelsLike(f'Acc*{sensor_nr}')
        else:
            channels = reader.collectChannelsLike(sensor_filter.replace("!", "|"))


        if annotate:
            annotate_runtimes(specimen, ax, annotate_text)


        imp_i, imp_time = get_impact_time(drop)

        time = drop.Time.data - imp_time
        for chan in channels:
            ls = "-"
            if 'fall' in chan.Name.lower():
                ls = drop_ls
            if any([x in chan.Name for x in prim_sensors]):
                ls = prim_ls

            ax.plot(time, chan.data, label=chan.Name, linestyle=ls)

        ax.grid()
        ax.set_xlim((-1*ms, +3*ms))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*tf:.2f}"))

        # print sig_h to the right
        ax.text(0.99, 0.95, f"{specimen.sig_h:.0f} MPa", ha='right', va='top', transform=ax.transAxes)
        ax.text(0.01, 0.95, f"'{specimen.name}'", ha='left', va='top', transform=ax.transAxes)

    for ax in axs:
        ax.axhline(0, color='k', linestyle='-')

    axs[-1].set_xlabel(f"Time [{tunit}]")
    axs[len(specimens)//2].set_ylabel(f"Acceleration [g]")

    # lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    axs[0].legend(loc="upper left", bbox_to_anchor=(1.04, 1))
    fig.tight_layout()
    plt.show()

    State.output(fig)

@app.command()
def plot_mean(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
    time_unit: Annotated[str, typer.Option(help="The unit to show on the x-axis.", )] = "s",
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
    time_f, readable_unit = mod_unit(time_unit)

    g = 9.81


    if file is None:
        specimen = Specimen.get(specimen_name)

        set_prim_sec_sensors(specimen)
        reader = APReader(specimen.acc_file)
    else:
        reader = APReader(file)

    reader.printSummary()

    # get the channels
    drop_channels = reader.collectChannelsLike('Fall_g')
    # g_channels = reader.collectChannelsLike('shock')
    # drop_channels = reader.collectChannelsLike('Force')

    fall_time_i, fall_time, mean_data = get_drop_time(drop_channels[0], returnMean=True)

    time0 = drop_channels[0].Time.data

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
    ax.set_xlabel(f"Time [{readable_unit}]")
    ax.set_ylabel(f"Acceleration [{'g' if not mpss else 'm/s²'}]")
    ax.grid()

    ax.axvline(fall_time, color="red", linestyle=drop_ls)
    ax.text(fall_time, 0.01, 'Fall',rotation=90, va='bottom', transform=ax.get_xaxis_transform())

    # time for freefall of 7cm
    t = np.sqrt(2*0.20/g)
    ax.axvline(fall_time + t, color="red", linestyle=drop_ls)
    ax.text(fall_time + t, 0.01, 'Impact',rotation=90, va='bottom', transform=ax.get_xaxis_transform())



    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*time_f:.2f}"))

    ax.plot(time0, mean_data, linestyle=drop_ls, label="Mean")

    # plot the drop channels
    for chan, time, data in drop_data:
        ax.plot(time, data, linestyle=drop_ls, label=chan.Name)

    if not no_legend:
        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))

    fig.tight_layout()

    State.output(fig, specimen)

@app.command()
def plot_impact(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
    time_unit: Annotated[str, typer.Option(help="The unit to show on the x-axis.", )] = "ms",
    file: Annotated[str, typer.Option(help="The file to plot.", )] = None,
    fig_title: Annotated[str, typer.Option(help="Title of the figure.", )] = None,
    apply_filter: Annotated[bool, typer.Option(help="Apply filter function.", )] = False,
    zoom_in: Annotated[bool, typer.Option(help="Zoom to impact.", )] = True,
    zoom_drop: Annotated[bool, typer.Option(help="Zoom to drop.", )] = False,
    no_legend: Annotated[bool, typer.Option(help="Hide the plot legend.", )] = False,
    show_title: Annotated[bool, typer.Option(help="Show a title in the plot.", )] = False,
    mpss: Annotated[bool, typer.Option(help="Show velocity in m/s.", )] = False,
    plot_filtered_sep: Annotated[bool, typer.Option('--plot-sep', help="Draw the filtered function seperately.", )] = False
):
    """Plots the impact of the given specimen."""
    time_f, readable_unit = mod_unit(time_unit)

    g = 9.81

    assert not (zoom_in and zoom_drop), "Cannot zoom in and drop at the same time."

    if file is None:
        specimen = Specimen.get(specimen_name)

        set_prim_sec_sensors(specimen)
        reader = APReader(specimen.acc_file)
    else:
        reader = APReader(file)

    reader.printSummary()

    # get the channels
    g_channels = reader.collectChannelsLike('Acc')
    drop_channels = reader.collectChannelsLike('Fall_g')
    # g_channels = reader.collectChannelsLike('shock')
    # drop_channels = reader.collectChannelsLike('Force')


    # reader.plotGroup(0)
    reader.plot(sameAxis=True)

    impact_time_i, impact_time = get_impact_time(drop_channels[0])
    drop_time_i, drop_time = get_drop_time(drop_channels[0])

    time0 = drop_channels[0].Time.data

    g_data = []
    # collect channel data and their times
    for chan in g_channels:
        data = chan.data * (g if mpss else 1.0)
        fdata = None
        # apply a filter to chan.data
        fdata = savgol_filter(data, 9, 3)
        # freq, dfft = fft(data, chan.Time.data)
        imp_data = data[int(impact_time_i+1000):] # int(impact_time_i+3000)
        freq, dfft = fft_calc(imp_data, chan.Time.data, plot=False,title=chan.Name)
        g_data.append((chan, chan.Time.data, data, fdata, freq))

    drop_data = []
    # collect channel data and their times
    for chan in drop_channels:
        drop_data.append((chan, chan.Time.data, chan.data))

    # plot the data

    figsize = get_fig_width(FigureSize.ROW2)
    if not no_legend:
        figsize = (figsize[0] * 1.2, figsize[1])

    fig, ax = plt.subplots(figsize=figsize)
    if show_title:
        fig.suptitle(fig_title or f"Impact of specimen '{specimen_name}'")
    ax.set_xlabel(f"Time [{readable_unit}]")
    ax.set_ylabel(f"Acceleration [{'g' if not mpss else 'm/s²'}]")
    ax.grid()


    # plot the g channels
    for chan, time, data, fdata, freq in g_data:
        time = time - impact_time
        linestyle = '-'

        if any([x in chan.Name for x in prim_sensors]):
            linestyle = prim_ls

        line = ax.plot(time, data, linestyle=linestyle, label=chan.Name)
        if plot_filtered_sep and not apply_filter:
            ax.plot(time, fdata, "--", color = line[0].get_color(), label=chan.Name + " (filtered)")

    # plot the impact time
    if file is None:
        _,_,crack_runtime = annotate_runtimes(specimen, ax)
        # ax.axvline(drop_time - impact_time, color="red", linestyle=drop_ls)
        # ax.text(drop_time  - impact_time, 0.01, 'Drop',rotation=90, va='bottom', transform=ax.get_xaxis_transform())

        drop_height = specimen.fall_height_m
        drop_dur = np.sqrt(2*drop_height/g)

        # ax.axvline(drop_time - impact_time + drop_dur, color="red", linestyle=drop_ls)
        # ax.text(drop_time  - impact_time + drop_dur, 0.01, 'Impact',rotation=90, va='bottom', transform=ax.get_xaxis_transform())

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*time_f:.2f}"))

    # plot the drop channels
    for chan, time, data in drop_data:
        time = time - impact_time
        ax.plot(time, data, linestyle=drop_ls, label=chan.Name)

    if zoom_in and file is None:
        before = -30*us
        after = crack_runtime + 200*us
        ax.set_xlim(before, after)
        ib = tti(before, time0 - impact_time)
        ia = tti(after, time0 - impact_time)

        y_max= [np.max(x[ib:ia]) for _, _, x, _, _ in g_data]
        y_max = np.max(y_max) * 1.3
        ax.set_ylim((-y_max, y_max))

    if zoom_drop and file is None:
        before = -30*us
        after = 100*us
        ax.set_xlim(before, after)
        ib = tti(before, time0 - impact_time)
        ia = tti(after, time0 - impact_time)

        y_max= [np.max(x[ib:ia]) for _, _, x, _, _ in g_data]
        y_max = np.max(y_max) * 1.3
        ax.set_ylim((-y_max, y_max))

    if not no_legend:
        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))

    fig.tight_layout()
    plt.show()

    State.output(fig, "acceleration-impact", spec=specimen, figwidth=FigureSize.ROW1HL)


def get_impact_time(channel: Channel):
    mean_drop_g = np.max(np.abs(channel.data[:10000]))*1.5
    xx1 = np.abs(channel.data/mean_drop_g)**10
    impact_time_i: int = np.argwhere(xx1 >= 1)[0]-2
    impact_time = channel.Time.data[impact_time_i]
    return impact_time_i,impact_time

def around(x, y, eps=1e-3):
    return np.abs(x-y) < eps

def mean_data(data, h):
    # meaned = np.zeros(len(data))
    # for i in range(len(data)):
    #     l = np.max([0, i-w])
    #     t = np.min([len(data), i+w])

    #     meaned[i] = np.mean(data[l:t])

    # return meaned
    # ndata = signal.sosfilt(butter(5, 0.1, 'hp', output='sos'),data)
    ndata = savgol_filter(data, h, 5)

    # meaned = np.zeros(len(ndata))
    # for i in range(len(ndata)):
    #     l = np.max([0, i- h // 2])
    #     t = np.min([len(ndata), i+h // 2])

    #     meaned[i] = np.sum(ndata[l:t]) / h

    return ndata

def get_drop_time(channel: Channel, returnMean = False, h = 203):
    data = channel.data
    time = channel.Time.data
    # first, take running average of data
    data = mean_data(data, h)
    data = np.abs(data)
    # find the time when data is around -1
    fall_time_i = np.argwhere(np.abs(data - 1) < 0.01)[0] # + h // 2

    fall_time = time[fall_time_i]

    if not returnMean:
        return fall_time_i, fall_time

    return fall_time_i, fall_time, data




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

@app.command()
def fft(file, chan: str = "Fall_g"):
    """Calculates the fft of the given file."""

    if (spec := Specimen.get(file, panic=False)) is not None:
        file = spec.acc_file

    reader = APReader(file)
    reader.printSummary()

    chan = reader.collectChannelsLike(chan)[0]
    freq, ffts = fft_calc(chan.data, chan.Time.data, plot=False, title=chan.Name)

    fig, ax = plt.subplots()
    ax.plot(freq, ffts)
    plt.show()

@app.command()
def test_data(file):
    """Calculates the fft of the given file."""

    if (spec := Specimen.get(file, panic=False)) is not None:
        file = spec.acc_file


    accdata = AccelerationData(file)

@app.command()
def transform(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
):
    specimen = Specimen.get(specimen_name)

    accdata = specimen.accdata

    reader = accdata.reader

    groups = reader.Groups

    # perform lowpass filter on Fall_g1 sensor data
    for group in groups:
        for chan in group.ChannelsY:
            if re.match("[Ff]all(_?)g1", chan.Name):
                chan.data = lowpass(chan.Time.data, chan.data, 4500, 1/(chan.Time.data[1]-chan.Time.data[0]))
                break

    # create csv file
    csv_file = specimen.get_acc_outfile("data.csv")

    # write csv
    with open(csv_file, 'w') as f:
        # header
        for group in reader.Groups:
            f.write(f"{group.ChannelX.Name} [{group.ChannelX.unit}];")

            for chan in group.ChannelsY:
                f.write(f"{chan.Name} [{chan.unit}];")

        f.write("\n")

        # data
        max_len = np.max([np.max([len(x.data) for x in group.ChannelsY]) for group in reader.Groups])
        print(max_len)
        for i in range(max_len):
            for group in reader.Groups:
                if i < len(group.ChannelX.data):
                    f.write(f"{group.ChannelX.data[i]};")
                else:
                    f.write(";")
                for chan in group.ChannelsY:
                    if i < len(chan.data):
                        f.write(f"{chan.data[i]};")
                    else:
                        f.write(";")
            f.write("\n")
