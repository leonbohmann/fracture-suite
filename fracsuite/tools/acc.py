import os
from typing import Annotated
from matplotlib import pyplot as plt

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

app = typer.Typer()
general = GeneralSettings.get()

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

    fig, axs = plt.subplots()
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

@app.command()
def plot_impact(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
    time_unit: Annotated[str, typer.Option(help="The unit to show on the x-axis.", )] = "s",
    normalize_time: Annotated[bool, typer.Option('--normalize-time', help="Move 0-time to impact.", )] = False,
    file: Annotated[str, typer.Option(help="The file to plot.", )] = None,
    fig_title: Annotated[str, typer.Option(help="Title of the figure.", )] = None,
    apply_filter: Annotated[bool, typer.Option(help="Apply filter function.", )] = False,
    plot_filtered_sep: Annotated[bool, typer.Option('--plot-sep', help="Draw the filtered function seperately.", )] = False
):
    """Plots the impact of the given specimen."""
    if file is None:
        specimen = Specimen.get(specimen_name)

        reader = APReader(specimen.acc_file)
    else:
        reader = APReader(file)

    reader.printSummary()

    # get the channels
    g_channels = reader.collectChannelsLike('Acc')
    drop_channels = reader.collectChannelsLike('Fall_g')
    # g_channels = reader.collectChannelsLike('shock')
    # drop_channels = reader.collectChannelsLike('Force')

    mean_drop_g = np.max(np.abs(drop_channels[0].data[:18000]))*1.001
    xx1 = np.abs(drop_channels[0].data/mean_drop_g)**10
    impact_time_i: int = np.argwhere(xx1 >= 1)[0]-2
    impact_time = drop_channels[0].Time.data[impact_time_i]

    dtime = impact_time if normalize_time else 0

    before = impact_time - 0.003
    after = impact_time + 0.003

    g_data = []
    # collect channel data and their times
    for chan in g_channels:
        data = chan.data
        fdata = None
        # apply a filter to chan.data
        fdata = savgol_filter(data, 9, 3)
        # freq, dfft = fft(data, chan.Time.data)
        imp_data = data[int(impact_time_i+1000):] # int(impact_time_i+3000)
        freq, dfft = fft(imp_data, chan.Time.data, plot=True,title=chan.Name)
        g_data.append((chan, chan.Time.data, data, fdata, freq))

    drop_data = []
    # collect channel data and their times
    for chan in drop_channels:

        drop_data.append((chan, chan.Time.data, chan.data))

    # plot the data
    fig = plt.figure()
    fig.suptitle(fig_title or f"Impact of specimen '{specimen_name}'")
    ax = fig.add_subplot(111)
    ax.set_xlabel(f"Time [{time_unit}]")
    ax.set_ylabel("Acceleration [g]")
    ax.grid()

    time_f = 1
    if time_unit == "ns":
        time_f = 1000000000
    elif time_unit == "us":
        time_f = 1000000
    elif time_unit == "ms":
        time_f = 1000
    elif time_unit == "s":
        time_f = 1
    else:
        print(f"Unknown time unit '{time_unit}'.")
        return

    # plot the g channels
    for chan, time, data, fdata, freq in g_data:
        time = time * time_f - dtime * time_f
        line = ax.plot(time, data, label=chan.Name)

        if plot_filtered_sep and not apply_filter:
            ax.plot(time, fdata, "--", color = line[0].get_color(), label=chan.Name + " (filtered)")


    # plot the impact time
    ax.axvline(impact_time * time_f- dtime * time_f, color="red", label="Impact Time")

    # plot the 0.5s before and 3 seconds after the impact
    ax.set_xlim(before * time_f- dtime * time_f, after * time_f- dtime * time_f)

    # plot the drop channels
    for chan, time, data in drop_data:
        time = time * time_f - dtime * time_f
        ax.plot(time, data, "--", label=chan.Name)

    plt.legend(loc="upper right")
    plt.show()


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
