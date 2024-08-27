"""
Acceleration tools.
"""

from ctypes.wintypes import RECT
import os
from pstats import StatsProfile
import re
import shutil
import time
from tracemalloc import start
from turtle import color
from typing import Annotated
from unittest import skip
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
import numpy as np
import typer
from apread import APReader, Channel # type: ignore
from fracsuite.core.coloring import get_color, norm_color
from fracsuite.core.logging import error
from fracsuite.core.series import betweenSeconds, untilSeconds, afterSeconds
from rich import inspect, print
from rich.progress import track
from fracsuite.core.accelerationdata import DEBUG, AccelerationData
from fracsuite.core.plotting import FigureSize, get_fig_width, legend_without_duplicate_labels, plot_series
from fracsuite.core.signal import bandstop, lowpass, bandpass
from fracsuite.core.specimenprops import SpecimenBreakPosition
from fracsuite.state import State, StateOutput

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
def to_csv(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
):
    """
    Export acceleration data of a specimen to a csv file.
    """
    specimen = Specimen.get(specimen_name, panic=False)
    
    if specimen is None:
        reader = APReader(specimen_name)
        
        csv_file = os.path.join(os.path.dirname(specimen_name), f"{os.path.basename(specimen_name)}.csv")
    else:
        accdata = specimen.accdata

        reader = accdata.reader
        
        
        # create csv file
        csv_file = specimen.get_acc_outfile("data.csv")

    groups = reader.Groups

    # perform lowpass filter on Fall_g1 sensor data
    for group in groups:
        for chan in group.ChannelsY:
            if re.match("[Ff]all(_?)g1", chan.Name):
                chan.data = lowpass(chan.Time.data, chan.data, 4500, 1/(chan.Time.data[1]-chan.Time.data[0]))
                break


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

@app.command()
def integrate_fall(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
    time_unit: Annotated[str, typer.Option(help="The unit to show on the x-axis.", )] = "s",
    normalize_time: Annotated[bool, typer.Option('--normalize-time', help="Move 0-time to impact.", )] = False,
    file: Annotated[str, typer.Option(help="The file to plot.", )] = None,
    fig_title: Annotated[str, typer.Option(help="Title of the figure.", )] = None,
    apply_filter: Annotated[bool, typer.Option(help="Apply filter function.", )] = False,
):
    """Integrate the fallweight acceleration twice to get velocity and displacement."""
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
    sigmas: Annotated[str, typer.Option(help="The sigmas to plot. For example '100-130'.", )] = "all",
    time_unit: Annotated[str, typer.Option(help="The unit to show on the x-axis.", )] = "s",
    sensor_nr: Annotated[str, typer.Option(help="The sensor number to plot. Works only if sensor names are 'Acc[_][Nr]", )] = "2",
    sensor_filter: Annotated[str, typer.Option(help="The sensor filter. When set, sensor-nr is ignored!", )] = None,
    out: Annotated[str, typer.Option(help='Output file.')] = None,
    annotate: Annotated[bool, typer.Option(help='Annotate wave runtime into plots using vertical lines.')] = True,
    annotate_text: Annotated[bool, typer.Option(help='Annotate text to wave-runtimes.')] = True,
):
    """
    Compares the wave runtimes of the given specimens.
    """
    
    tf, tunit = mod_unit(time_unit)

    filter_func = create_filter_function(name_filter, sigmas)
    specimens: list[Specimen] = Specimen.get_all_by(filter_func, load=True)

    fig, axs = plt.subplots(len(specimens), 1, sharey='all', sharex='all')

    # specimens = sorted(specimens, key=lambda x: x.sig_h)

    for i,specimen in enumerate(specimens):
        ax: Axes = axs[i]

        set_prim_sec_sensors(specimen)

        reader = APReader(specimen.acc_file)

        drop = reader.collectChannelsLike('Fall(_)?g1')[0]

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

    State.output(fig, figwidth=FigureSize.ROW1)

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
    plot_filtered_sep: Annotated[bool, typer.Option('--plot-sep', help="Draw the filtered function seperately.", )] = False,
    sensors: Annotated[str, typer.Option(help="The sensors to plot.", )] = "all",
):
    """Plots the impact of the given specimen."""
    time_f, readable_unit = mod_unit(time_unit)

    g = 9.81

    assert not (zoom_in and zoom_drop), "Cannot zoom in and drop at the same time."

    if file is None:
        specimen = Specimen.get(specimen_name)
        if not specimen.accdata.is_ok:
            error(f"Specimen '{specimen_name}' has faulty acceleration data.")
            return
        set_prim_sec_sensors(specimen)
        reader = APReader(specimen.acc_file)
        accdata = specimen.accdata
    else:
        accdata = AccelerationData(file)
        reader = accdata.reader

    reader.printSummary()
    
    print("filtering data")
    # accdata.filter_fallgewicht_lowpass(f0=low)
    # accdata.filter_fallgewicht_highpass(f0=high)

    # accdata.filter_fallgewicht_eigenfrequencies()
    accdata.filter_fallgewicht_wiener()
        
    
    # get the channels
    g_channels = accdata.get_channels_like('Acc')
    drop_channels = accdata.get_channels_like('Fall_g')
    # g_channels = reader.collectChannelsLike('shock')
    # drop_channels = reader.collectChannelsLike('Force')

    if any((x.sampling_frequency < 20000 for x in reader.Channels if 'Fall_g' in x.Name)):
        error("The sampling rate is too low for this analysis. A minimum of 20kHz is required.")
        return


    # reader.plotGroup(0)
    reader.plot(sameAxis=True)
    
    print("Channels:")
    for chan in g_channels:
        print(f"  {chan.Name}")
        
    print("Drop channels:")
    for chan in drop_channels:
        print(f"  {chan.Name}")

    print("Plotting impact...")
    impact_time_i, impact_time = get_impact_time(drop_channels[0])
    drop_time_i, drop_time = get_drop_time(drop_channels[0])

    time0 = drop_channels[0].Time.data

    g_data = []
    sensors = sensors.split(",")
    # collect channel data and their times
    for chan in g_channels:
        if not all or not any([re.match(x, chan.Name) for x in sensors]):
            continue
        
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

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*time_f:.5f}"))

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
def ffts(file, channel_names: list[str], seconds: float = None):
    if (spec := Specimen.get(file, panic=False)) is not None:
        file = spec.acc_file
        reader = spec.accdata.reader
    else:
        reader = APReader(file)


    channels: list[Channel] = []
    for name in channel_names:
        channels.extend(reader.collectChannelsLike(name))

    series = [(*untilSeconds(chan, seconds), chan.Name) for chan in channels]
    perform_plot_fft(series)

@app.command()
def fft(file, chan: str = "Fall_g", seconds: float = None, time: tuple[float,float] = (None, None), outdir: str = None):
    """
    Calculates the fft of the given file.
    
    Args:
        file (str): The file to calculate the fft of.
        chan (str, optional): The channel to calculate the fft of. Defaults to "Fall_g".
        seconds (float, optional): The time up until which to calculate the fft of. Defaults to None.
    """

    if (spec := Specimen.get(file, panic=False)) is not None:
        file = spec.acc_file
        reader = spec.accdata.reader
    else:
        reader = APReader(file)
    
    reader.printSummary()


    chan = reader.collectChannelsLike(chan)[0]
    
    # only fft until time
    if time[0] is not None and time[1] is not None:
        time, data = betweenSeconds(chan, time[0], time[1])
    else:
        time, data = untilSeconds(chan, seconds)

    # plot the base series
    plot_series(time, data, chan.Name)    
    # perform fft
    perform_plot_fft((time, data, chan.Name), output_dir=outdir)

def perform_plot_fft(series: list[tuple], time_bounds: tuple[float,float] = (None, None), output_dir = None):
    """
    Performs the fft on all series and plot them in a single figure.
    
    Args:
        series (list[tuple]): The series to perform the fft on. [(time, data, name)]
    """
    series = series if isinstance(series, list) else [series]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)


    fig, axs = plt.subplots(2,len(series),figsize=get_fig_width(FigureSize.ROW1))    
    for i, serie in enumerate(series):
        time, data, name = serie
        # plot the original series on the left in the current row
        ax = axs[0] if len(series) == 1 else axs[0, i]        
        time,data = betweenSeconds((time, data), time_bounds[0], time_bounds[1])
        plot_series(time, data, name, axs=ax)

        # fft
        ax = axs[1] if len(series) == 1 else axs[1, i]
        freq, ffts = fft_calc(data, time, plot=False)
        ax.plot(freq, ffts, label=name)

        # find eigenfrequencies
        # peaks, _ = find_peaks(ffts, height=0.1)
        if output_dir is not None:
            file = os.path.join(output_dir, f"{name}.csv")
            with open(file, 'w') as f:
                f.write("Freq;Ampl\n")
                for t, d in zip(freq, ffts):
                    f.write(f"{t};{d}\n")

    ax.legend()
    plt.show()

    State.output(StateOutput(fig, figwidth=FigureSize.ROW1))

@app.command()
def test_filter(
    file: str,
    low: float = None,
    high: float = None,
    bands: bool = False,    
    chan_name:str = None,
    fft_range: tuple[float,float] = (None, None),
    wiener: bool = False
):
    """
    Tests the filtering algorithm and displays both the unfiltered and filtered signal.
    
    Args:
        file (str): The file to test the filtering on. Can also be the name of a specimen.
    """
    accdata: AccelerationData = None
    if (spec := Specimen.get(file, panic=False)) is not None:
        assert spec.acc_file is not None, f"Specimen '{file}' has no acceleration data."
        accdata = spec.accdata
    elif file in general.aliases:
        file = general.aliases[file]
    
    if not os.path.exists(file) and accdata is None:
        raise FileNotFoundError(f"File '{file}' not found.")

    if accdata is None:
        accdata = AccelerationData(file)
        
    if chan_name is None:
        drop_channel = accdata.drop_channel
    else:
        drop_channel = accdata.get_channel_like(chan_name)

    original_data = drop_channel.data.copy()


    # filter out high frequencies
    if low is not None:
        accdata.filter_fallgewicht_lowpass(f0=low)
    # filter out low frequencies
    if high is not None:
        accdata.filter_fallgewicht_highpass(f0=high)

    if bands:
        accdata.filter_fallgewicht_eigenfrequencies()
        accdata.filter_fallgewicht_wiener()
        
    time = drop_channel.Time.data


    if wiener:
        from scipy.signal import wiener as w2
        drop_channel.data = w2(drop_channel.data, 11)


    # plot filtered data
    fig, axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    axs.set_title(f"Filtering of '{drop_channel.Name}'")
    axs.plot(time, original_data, label='Original' )
    axs.plot(time, drop_channel.data, label='Filtered')
    axs.legend()
    plt.show()
    State.output(fig, figwidth=FigureSize.ROW1)

    perform_plot_fft((time, drop_channel.data, drop_channel.Name + " filtered"), fft_range)

@app.command()
def wiener_test(
    file: str,
    chan_name: str = None
):
    """
    Compare the wiener filter with the original data.

    Args:
        name (str): File name, specimen name or alias.
    """
    accdata: AccelerationData = None
    if (spec := Specimen.get(file, panic=False)) is not None:
        assert spec.acc_file is not None, f"Specimen '{file}' has no acceleration data."
        accdata = spec.accdata
    elif file in general.aliases:
        file = general.aliases[file]
    
    if not os.path.exists(file) and accdata is None:
        raise FileNotFoundError(f"File '{file}' not found.")

    if accdata is None:
        accdata = AccelerationData(file)
        
    if chan_name is None:
        drop_channel = accdata.drop_channel
    else:
        drop_channel = accdata.get_channel_like(chan_name)

    original_data = drop_channel.data.copy()
    
    # filter using wiener
    from scipy.signal import wiener as w2
    data_wienered = {}

    wiener_sizes = [1, 3, 5, 7, 9, 11, 13, 15, 17]

    for i in wiener_sizes:
        data_wienered[i] = w2(original_data, i)    
    
    time = drop_channel.Time.data    
    # fig, axs = plt.subplots(len(wiener_sizes) // 3, 3, figsize=get_fig_width(FigureSize.ROW1), sharex=True, sharey=True)
    
    # for i, (size, data) in enumerate(data_wienered.items()):
    #     row = i // 3
    #     col = i % 3

    #     axs[row,col].set_title(f"Wiener filter size {size}")
    #     axs[row,col].plot(time, original_data, label='Original' )
    #     axs[row,col].plot(time, data, label='Wienered')
    #     axs[row,col].legend()
        
    # plt.show()

    # fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    # for size, data in data_wienered.items():
    #     freq, ffts = fft_calc(data, time, plot=False)
    #     axs.plot(freq, ffts, label=f"Wiener size {size}")
        
    # axs.legend()
    # plt.show()
    
    # plot all wieners in one plot
    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    axs.plot(time, original_data, label="Original")
    for size, data in data_wienered.items():
        axs.plot(time, data, label=f"{size}")
        
    axs.legend()
    plt.show()
        

@app.command()
def calculate_load_time(
    specimen_name: str,    
    skip_filters: bool = False,
):
    if ',' in specimen_name:
        specimen_names = specimen_name.split(',')
        for name in specimen_names:
            calculate_load_time(name, skip_filters)
        return
    
    
    # load data
    specimen = Specimen.get(specimen_name)
    accdata = specimen.accdata

    # filter drop channel
    # if not skip_filters:
        # accdata.filter_fallgewicht_highpass(f0=10)
        # accdata.filter_fallgewicht_eigenfrequencies()
        
    accdata.filter_fallgewicht_lowpass(f0=15000)
    accdata.filter_fallgewicht_wiener()


    # get drop channel
    drop_channel = accdata.drop_channel
    data = drop_channel.data
    time = drop_channel.Time.data
    
    # get the exact time of fall initiation
    ti = 0.5 # this is the trigger time from catmanAP
    
    sfall = specimen.fall_height_m
    tfall = np.sqrt(2*sfall/9.81)
    t0 = ti - tfall
    # fetch time and data of fall weight from fall to impact + 1 second
    time, data = betweenSeconds((time, data), t0, t0 + 0.5)

    # normalize time
    time = time - time[0]

    m = 2.41

    # acceleration
    a = data * 9.81 # g to m/s²
    # velocity
    v = cumulative_trapezoid(a, time, initial=0)
    # distance
    s = cumulative_trapezoid(v, time, initial=0) + sfall

    kin = 0.5 * 9.81 * v ** 2
    F = m * a

    # plot the data into subplots over each other
    fig, axs = plt.subplots(3, 1, figsize=general.figure_size, sharex=True, sharey=False)
    axs[0].plot(time, a, label="Acceleration")
    axs[1].plot(time, v, label="Velocity")
    axs[2].plot(time, s, label="Distance")

    axs[0].set_ylabel("Acceleration [m/s²]")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[2].set_ylabel("Distance [m]")

    for ax in axs:        
        # ax.autoscale()
        ax.legend()
        ax.grid()

    plt.show()

    State.output(StateOutput(fig, figwidth=FigureSize.ROW1), f'{specimen_name}-load-time', spec=specimen)


    # create csv file
    csv_file = State.get_output_file(f"{specimen_name}-load-time.csv")
    with open(csv_file, 'w') as f:
        f.write("Time [s];Acceleration [m/s²];Velocity[m/s];Distance[m];E_kin[J];F[N]\n")
        for i in range(len(time)):
            f.write(f"{time[i]};{a[i]};{v[i]};{s[i]};{kin[i]};{F[i]}\n")
            
    print(f"Data written to '{csv_file}'.")
    
    # copy contents to specimen folder
    shutil.copy(csv_file, specimen.get_acc_outfile(f"{specimen_name}-load-time.csv"))
    
    
@app.command()
def compare(
    sensor: str = "[Aa]cc(_?)6",
    sensor2: str = "",
            zero_impact: bool = False,
            zero_crackfront: bool = False,
            clr_bounds: bool = False,
            clr_sigma: bool = False,
            clr_specimen: bool = False,
            use_lims: bool = False,
            sigma_range:str = "40-160",
            no_zero: bool = False,
            show: bool = False):
    """Compare the acceleration of the specimens.

    Args:
        sensor (str, optional): Filter for the sensor to be compared. Defaults to "[Aa]cc(_?)6".
        zero_time (bool, optional): Use the time to zero instead of greatest peak. Defaults to True.
        plt_bounds (bool, optional): Plot the boundary condition instead of the stress. Defaults to False.
    """
    
    assert np.sum([zero_impact, zero_crackfront]) == 1, "One of zero_time or zero_impact must be set."
    assert np.sum([clr_sigma, clr_bounds, clr_specimen]) == 1, "One of the color options must be set."
    
    assert "-" in sigma_range, "Sigma range must be in format '[lower]-[upper]'"
    
    basefilt = create_filter_function("4.*.*.*", needs_scalp=True)
    
    # load all specimens
    def filt(s: Specimen):
        if not s.accdata.is_ok:
            return False
        if s.break_pos != SpecimenBreakPosition.CORNER:
            return False
        
        return basefilt(s)
    
    specimens = Specimen.get_all_by(filt, load=True)

    clrs = {
        'A': 'C0',
        'B': 'C1',
        'Z': 'C2',
    }
    
    def get_acc_sensor(s: Specimen) -> np.ndarray:
        if s.accdata is None:
            return None
        
        acc6 = s.accdata.get_channel_like(sensor, panic=False)
        
        if acc6 is None:
            return None
                
        return acc6.get_filtered()

    sig0 = int(sigma_range.split('-')[0])
    sig1 = int(sigma_range.split('-')[1])

    max_peak = {}
    max_peak_time = {}

    impact_crackinit_delta = {}

    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))    
    for ss, s in enumerate(specimens):
        # get acc_6 sensor data
        accdata = s.accdata
        acc1 = accdata.get_channel_like(sensor, panic=False)
        acc2 = accdata.get_channel_like(sensor2, panic=False) if sensor2 != "" else None
        
        accs: list[Channel] = [acc1, acc2]
        
        for i,acc in enumerate(accs):
            if acc is None:
                continue
            
            time = acc.Time.data
            data = acc.get_filtered()
            
            impact_time_corner = accdata.get_impacttime_from_corner()
            
            delta_ms = np.abs(impact_time_corner - 0.5) * 1e3                        
            impact_crackinit_delta[s] = delta_ms
            if delta_ms > 0.5:
                continue            
            
            
            
            # align the time, where the highest peak is found
            peak = np.argmax(data)
            if zero_impact:
                time = time - accdata.get_impacttime_from_fall()        
            elif zero_crackfront:
                time = time - impact_time_corner
            elif not no_zero:
                time = time - time[peak]
            
            max_peak[s] = data[peak]
            max_peak_time[s] = time[peak]
            
            if clr_bounds:
                clr = clrs[s.boundary.value]
            elif clr_specimen:
                clr = f"C{ss}"
            else:
                clr = norm_color(get_color(np.abs(s.sig_h), 40, 160))
                
            # * 1e3 to convert to ms
            axs.plot(time * 1e3, data, ls=['-', '--'][i], alpha=0.3, c=clr, label=f"{s.boundary.value}")
        
    if use_lims:
        axs.set_xlim(-0.00075, 0.00075)
        axs.set_ylim(-1400, 5200)
    legend_without_duplicate_labels(axs)
    axs.set_xlabel("Time [ms]")
    axs.set_ylabel("Acceleration [m/s²]")
    axs.set_title(f"Comparison of acceleration {accs[0].Name} and {accs[1].Name if accs[1] is not None else ''}")

    if not clr_bounds:
        fig.colorbar(ScalarMappable(norm=Normalize(40, 160), cmap='turbo'), ax=axs, label="Sigma_h [MPa]")
    
    if show:
        plt.show()
    State.output(StateOutput(fig, figwidth=FigureSize.ROW1), "compare-acc6")
    
    
    print('Plotting delta between impact and crack initiation')
    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    x = [s.sig_h for s in specimens]
    y = [impact_crackinit_delta[s] for s in specimens]
    axs.scatter(x, y)
    axs.set_xlabel("Sigma_h [MPa]")
    axs.set_ylabel("Delta between impact and crack initiation [ms]")
    plt.show()
    State.output(StateOutput(fig, figwidth=FigureSize.ROW1), "compare-impact-crackinit-delta")
    
    print('Calculating max-peak plots')
    # Auswertung: Maximaler Peak vs Vorspanngrad
    fig,axs = plt.subplots(1, figsize=get_fig_width(FigureSize.ROW1))
    t_clr = {4: 'C0', 8: 'C1', 12: 'C2'}
    for s in specimens:
        if s.fall_height_m != 0.07:
            continue
        
        clr = t_clr[s.thickness]
        data = get_acc_sensor(s)
        if data is None:
            continue   
        peak = data[np.argmax(data)]
        valley = data[np.argmin(data)]
        maxd = peak
        x = s.U * 0.25 # np.abs(s.sig_h)
        axs.scatter(x, maxd, c=clr, label=f"{s.thickness}mm")
        
    
    
    # x = [np.abs(s.sig_h) for s, _ in max_peak.items()]
    # y = [p for s, p in max_peak.items()]
    # axs.scatter(x, y)
    # axs.set_ylim(0,10)
    axs.set_xlabel("Strain Energy [J]")
    axs.set_ylabel("Max Acceleration [m/s²]")
    legend_without_duplicate_labels(axs)
    State.output(StateOutput(fig, figwidth=FigureSize.ROW1), "compare-acc6-maxpeak")
    
    
    
    # group specimens according to their sigma-level (0-10, 10-20, 20-30, ...)
    groups = {}
    for s in specimens:
        sig = s.sig_h
        key = (sig // 10) * 10
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    
    
    
    
    # create a single plot
    fig, axs = plt.subplots(1, figsize=get_fig_width(FigureSize.ROW1), sharex=True, sharey=True)
    
    
@app.command()
def dbg_compare_sensors(
    file: str = Annotated[str, typer.Argument(help="The file to load.")],
    sensor1: Annotated[str, typer.Option(help="The first sensor to compare.")] = "Acc(_?)5",
    sensor2: Annotated[str, typer.Option(help="The second sensor to compare.")] = "Acc(_?)6"
):
    # check if vispy is installed
    try:        
        import vispy.plot as vp
    except ImportError:
        print("Vispy is not installed. Please install it using 'pip install vispy'.")
        return
    
    
    accdata = APReader(file)
    
    # load both channels Acc_5 and Acc_6
    acc5 = accdata.collectChannelsLike(sensor1)[0]
    acc6 = accdata.collectChannelsLike(sensor2)[0]
    
    # get the filtered data
    data5 = acc5.get_filtered() * -1
    data6 = acc6.get_filtered()
    
    # get the time
    time5_ms = acc5.Time.data * 1e3
    time6_ms = acc6.Time.data * 1e3
    
    d5 = [(x,y) for x,y in zip(time5_ms, data5)]    
    d6 = [(x,y) for x,y in zip(time6_ms, data6)]
    
    # plot
    fig = vp.Fig()
    fig[0,0].plot(d5, marker_size=0, color='blue')
    fig[0,0].plot(d6, marker_size=0, color='red')    
    
    # find min and max
    max_g1 = np.argmax(data5)
    max_g2 = np.argmax(data6)
    
    max_time0 = time5_ms[max_g1]
    max_time1 = time6_ms[max_g2]
    
    # axs.axvline(min_time, color='red', linestyle='--', label=f"Min {min_time:.3f}")
    # axs.axvline(max_time, color='green', linestyle='--', label=f"Max {max_time:.3f}")
    
    print(f"Min at {max_time0} and max at {max_time1}")
    print(f"Delta: {(max_time1 - max_time0)} ms")
    
    
    # add axes
    fig[0,0].xlabel = "Time [ms]"
    fig[0,0].ylabel = "Acceleration [g]"
    fig.app.run()
    # axs.set_xlim(min_time - 0.1, max_time + 0.1)
    
    
    # find bounds of the fig
    bounds: RECT = fig[0,0].view.camera.rect
    
    # remove all data outside of bounds
    data5 = [y for x,y in d5 if bounds.left < x < bounds.right]
    data6 = [y for x,y in d6 if bounds.left < x < bounds.right]
    time5_ms = [x for x,y in d5 if bounds.left < x < bounds.right]
    time6_ms = [x for x,y in d6 if bounds.left < x < bounds.right]    
    
    fig,axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
    axs.plot(time5_ms, data5, color='blue')
    axs.plot(time6_ms, data6, color='red')
    axs.axvline(max_time0, color='red', linestyle='--', label=f"Min {max_time0:.3f}")
    axs.axvline(max_time1, color='green', linestyle='--', label=f"Max {max_time1:.3f}")
    axs.set_xlim(bounds.left, bounds.right)
    axs.set_ylim(bounds.bottom, bounds.top)
    
    axs.set_xlabel("Time [ms]")
    axs.set_ylabel("Acceleration [g]")
    plt.show()
    
    # plt.show()
    State.output(fig, "fwaves-in-fallweight", figwidth=FigureSize.ROW2)
    
    