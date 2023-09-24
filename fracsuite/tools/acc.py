import os
from typing import Annotated
from matplotlib import pyplot as plt

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


@app.command()
def plot_impact(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
):
    """Plots the impact of the given specimen."""

    specimen = Specimen.get(specimen_name)

    reader = APReader(specimen.acc_file)

    reader.printSummary()

    time_peak = -np.inf
    peaks = []
    for group in reader.Groups:
        print(f"Group '{group.Name}'")

        time = group.ChannelX

        drops = [x for x in group.ChannelsY if "fall" in x.Name.lower()]

        if len(drops) >= 0:
            for drop in drops:
                max_i = np.argmax(drop.data)
                time_peak = time.data[max_i]
                peaks.append(time_peak)

    print(peaks)
    impact_time = np.mean(peaks)
    print(impact_time)
    # get 0.5s before and 3 seconds after the impact from all channels

    # get the channels
    g_channels = reader.collectChannels(['Acc1', 'Acc2', 'Acc3', 'Acc4', 'Acc5', 'Acc6'])
    drop_channels = reader.collectChannels(['Fall_g1', 'Fall_g2'])

    xx1 = np.abs(drop_channels[0].data/5)**10
    impact_time_i = np.argwhere(xx1 >= 1)[0]
    impact_time = drop_channels[0].Time.data[impact_time_i]

    before = impact_time - 0.003
    after = impact_time + 00.003

    g_data = []
    # collect channel data and their times
    for chan in g_channels:
        g_data.append((chan, chan.Time.data, chan.data))

    drop_data = []
    # collect channel data and their times
    for chan in drop_channels:
        drop_data.append((chan, chan.Time.data, chan.data))

    # plot the data
    fig = plt.figure()
    fig.suptitle(f"Impact of specimen '{specimen_name}'")
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [g]")
    ax.grid()

    # plot the g channels
    for chan, time, data in g_data:
        ax.plot(time, data, label=chan.Name)


    # plot the impact time
    ax.axvline(impact_time, color="red", label="Impact Time")

    # plot the 0.5s before and 3 seconds after the impact
    ax.set_xlim(before, after)

    ax1 = ax.twinx()
    # plot the drop channels
    for chan, time, data in drop_data:
        ax1.plot(time, data, "--", label=chan.Name)

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
