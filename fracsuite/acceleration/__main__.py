import argparse
from argparse import RawDescriptionHelpFormatter
import os
from tqdm import tqdm
from apread import APReader, plot_multiple_datasets, Channel
import numpy as np
from scipy import integrate, signal

from fracsuite.general import GeneralSettings

descr=\
"""
 █████╗  ██████╗ ██████╗███████╗██╗     ███████╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔════╝██╔════╝██╔════╝██║     ██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
███████║██║     ██║     █████╗  ██║     █████╗  ██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║
██╔══██║██║     ██║     ██╔══╝  ██║     ██╔══╝  ██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
██║  ██║╚██████╗╚██████╗███████╗███████╗███████╗██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝  ╚═╝ ╚═════╝ ╚═════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

Leon Bohmann            TUD - ISMD - GCC              www.tu-darmstadt.de/glass-cc

Description:
-------------------------
This module is used to analyze acceleration data, measured during the fracturing of the
glass plys. The sensor data is read using the apreader from https://github.com/leonbohmann/apreader.


Usage:
-------------------------
Command line usage is shown below. For further information visit:
https://github.com/leonbohmann/fracture-suite
"""

general = GeneralSettings()

def reader_to_csv(reader: APReader):
    """Writes the reader data to a csv file."""
    # create csv file
    csv_file = os.path.join(out_dir, f"{reader.fileName}.csv")
    with open(csv_file, 'w') as f:
        # write header
        f.write("Time [s];")
        for chan in reader.Channels:
            f.write(f"{chan.Name} [{chan.unit}];")
        f.write("\n")
        # write data
        for i in tqdm(range(0, len(reader.Channels[0].data))):
            f.write(f"{reader.Groups[0].ChannelX.data[i]};")
            for chan in reader.Channels:
                f.write(f"{chan.data[i]};")
            f.write("\n")


def time_to_index(time_array, t):
    """Returns the index of the time array that is closest to the given time t."""
    return np.argmin(np.abs(time_array - t))

def plotChannels(chans: list[Channel], lbs: list[str], title: str):
    styles = [None] * len(chans)
    for x in range(0,len(chans)):
        if len(lbs) > x:
            styles[x] = lbs[x]

    return plot_multiple_datasets([(x.Time.data, x.data, l, f'{x.Name} [{x.unit}]', x.Name) for x,l in zip(chans, styles)], title)


def test_1(reader: APReader):
    time = reader.Groups[0].ChannelX.data
    # impact_time_id = np.argmax([])
    # time = time - time[impact_time_id-5]
    print([x.Name for x in reader.Channels])

    ds = reader.collectChannels(['Force', 'Acc1', 'Fall_g2'])

    fig,axs = plot_multiple_datasets([ \
        (time, ds[0].data, "g-", "Force [N]", "Fall 2"),
        (time, ds[1].data, "r-", "PCB Acc [g]", "PCB Acc"),
        (time, ds[2].data, "b-", "Endevco Acc[g]", "Endevco Acc")],
        'Acceleration delay in fall weight')

    fig.savefig(os.path.join(out_dir, f"{reader.fileName}_detail.png"))

def test_4(reader: APReader):
    time = reader.Groups[0].ChannelX.data
    # impact_time_id = np.argmax([])
    # time = time - time[impact_time_id-5]
    print([x.Name for x in reader.Channels])

    ds = reader.collectChannels(['Force', 'Fall_g2'])

    fig,axs = plot_multiple_datasets([ \
        (time, ds[0].data, "g-", "Force [N]", "Force"),
        (time, ds[1].data, "r-", "Fall_g2 [g]", "Fall_g2")],
        'Acceleration delay in fall weight')

    fig.savefig(os.path.join(out_dir, f"{reader.fileName}_detail.png"))

def test_hammer_sensor_fallgewicht(reader: APReader):
    time = reader.Groups[0].ChannelX.data
    # impact_time_id = np.argmax([])
    # time = time - time[impact_time_id-5]

    ds = reader.collectChannels(['Force', 'Acc1', 'Fall_g2'])

    fig,axs = plotChannels(ds, ['r-', 'g-', 'b-'],
        'Acceleration delay in fall weight')

    fig.savefig(os.path.join(out_dir, f"{reader.fileName}_detail.png"))


def test_impact_delay(reader:APReader):
    ds = reader.collectChannels(['Fall_g1', 'Fall_g2', 'Acc2', 'Acc6', 'Acc3'])

    fig,axs = plotChannels(ds, ['b--', 'g--'],
        'Acceleration delay in fall weight')

    fig.savefig(os.path.join(out_dir, f"{reader.fileName}_impact_delay.png"))

def test_3(reader: APReader):
    time = reader.Groups[0].ChannelX.data
    # impact_time_id = np.argmax([])
    # time = time - time[impact_time_id-5]
    print([x.Name for x in reader.Channels])

    ds = reader.collectChannels(['Force', 'Acc1', 'Acc2', 'Acc3', 'Acc4', 'Acc7'])

    fig,axs = plot_multiple_datasets([ \
        (time, ds[0].data, "g-", "Force [N]", "Fall 2"),
        (time, ds[1].data, "r-", "Acc 1 [g]", "PCB Acc"),
        (time, ds[2].data, "k-", "Acc 2 (S) [g]", "PCB Acc"),
        (time, ds[3].data, "y-", "Acc 3 [g]", "PCB Acc"),
        (time, ds[4].data, "m-", "Acc 4 (S) [g]", "PCB Acc"),
        (time, ds[5].data, "b-", "Acc 7 [g]", "Acc 7")],
        'Acceleration delay in fall weight')

    fig.savefig(os.path.join(out_dir, f"{reader.fileName}_detail.png"))


def test_integrate(reader: APReader):
    channel_max = [(x.Name, np.max(x.data), np.min(x.data)) for x in reader.Channels]

    for max in channel_max:
        print(f'{max[0]}: {max[1]} / {max[2]}')

    # channels: 0: Piezoelectric, 1: Piezoresistive
    channels = [x for x in reader.Channels if x.Name.startswith("Fall")]

    g = 9.81 # m/s²
    drop_acc = channels[1]
    drop_acc1 = channels[0].data
    time = drop_acc.Time.data

    # this shifts the acc signal to zero before the impact
    drop_avg = np.average(drop_acc.data[-120000:])
    drop_data = drop_acc.data - drop_avg
    drop_data = drop_data * g

    # estimate savgol filter parameters based on the data time interval
    time_interval = time[1] - time[0]
    window_length = int(0.1 / time_interval)
    if window_length % 2 == 0:
        window_length += 1
    polyorder = 3


    # savgol
    drop_data_smooth = signal.savgol_filter(drop_data, 140, 5)

    # firwin
    cutoff_frequency = 0.1  # Adjust as needed
    num_taps = 101  # Filter length, adjust as needed
    fir_filter = signal.firwin(num_taps, cutoff_frequency)
    # drop_data_smooth = signal.lfilter(fir_filter, 1.0, drop_data)
    # np.savetxt("output.txt", np.column_stack([time, drop_data]), fmt = ['%.8f', '%.8f'])
    # drop_data = drop_data_smooth

    a = drop_data.copy()

    impact_w = 10
    impact_time_id = np.argmax(np.abs(a))
    impact_end_index = impact_time_id + impact_w//2
    impact_start_index = impact_time_id - impact_w//2

    # smooth g around the impact
    a[:impact_start_index] = drop_data_smooth[:impact_start_index]
    a[impact_end_index:] = drop_data_smooth[impact_end_index:]


    # shift a to zero around the impact
    impact_time = time[impact_time_id]
    impact_start_index = time_to_index(time, impact_time - 0.5)
    impact_end_index = time_to_index(time, impact_time + 1.5)


    # take aaverage around impact_start and end index

    # shift g into zero
    g_shift = np.average(a[:impact_start_index])
    g_shift2 = np.average(a[impact_end_index:])
    g_shift =( g_shift + g_shift2) / 2
    a = a - g_shift


    # initial is needed, so that the length of the result is the same as the input
    v = integrate.cumulative_trapezoid(a, time, initial = 0)
    s = integrate.cumulative_trapezoid(v, time, initial = 0)

    # shift the drop_data to zero after the impact
    # find the impact time index
    # impact_time_id = np.argmax(np.abs(v))
    # impact_time = time[impact_time_id]
    # impact_end_index = time_to_index(time, impact_time + 1.5)

    # g_shift = np.average(drop_data_smooth[impact_end_index:])
    # drop_data_s = drop_data_smooth.copy()
    # # shift data from impact to end
    # drop_data_s[impact_time_id+1:] = drop_data_s[impact_time_id+1:] - g_shift

    # # calculate g shift after impact
    # v_shift = np.average(v[impact_end_index:])
    # v_shifted = v.copy()
    # # shift data from impact to end
    # v_shifted[impact_time_id+1:] = v_shifted[impact_time_id+1:] - v_shift

    # use the corrected g now
    # v = integrate.cumulative_trapezoid(drop_data_s, time, initial = 0)

    # s_shifted = integrate.cumulative_trapezoid(v_shifted, time, initial = 0)
    # s = integrate.cumulative_trapezoid(v, time, initial = 0)

    # fig0 = plt.plot()

    # plt.plot(time, drop_data, "gray", label="Acceleration")
    # plt.plot(time, drop_data_smooth, "b", label="Smoothed")
    # plt.plot(time, drop_data_s, "g", label="Adjusted")

    # plt.legend()
    # plt.show()

    fig,axs = plot_multiple_datasets([ \
        (time, drop_acc1, "k--", "Acc 1 [g]", "Ac1"), \
        (time, a, "m-", "Acc [m/s²]", "Acceleration"),
        (time, drop_data, "r--", "Acc [m/s²]", "Acceleration"),
        # (time, v_shifted, "b--", "Speed [m/s]", "Acceleration"),
        (time, v, "b", "Speed [m/s]", "Speed"),
        # (time, s_shifted, "g--", "Distance [m]", "Distance"),
        (time, s, "g", "Distance [m]", "Distance")],
        'Time integrals of Acceleration')
    fig.savefig(os.path.join(out_dir, f"{reader.fileName}_fall.png"))

    time = time - time[impact_time_id-5]
    print([x.Name for x in reader.Channels])
    # Auswertung auf Impact beziehen und Zeit ab dort messen
    acc1 = reader.Channels[1]
    acc2 = reader.Channels[2]
    acc3 = reader.Channels[3]
    acc4 = reader.Channels[4]
    acc5 = reader.Channels[5]
    acc6 = reader.Channels[6]
    acc_g1 = reader.Channels[7]
    acc_g2 = reader.Channels[8]


    fig,axs = plot_multiple_datasets([ \
        # (time, drop_acc1, "k-", "Fall 1 [g]", "Fall 1"), \
        (time, acc_g2.data, "g-", "Fall 2 [g]", "Fall 2"), \
        (time, acc3.data, "r-", "Acc 3 [g]", "Acc 2"), \
        (time, acc2.data, "b-", "Acc 2 [g]", "Acc 2"), \
        (time, acc6.data, "y-", "Acc 6 [g]", "Acc 6")],
        'Comparison of different impact times')
    fig.savefig(os.path.join(out_dir, f"{reader.fileName}_fall2.png"))

# get all test-functions
test_funcs = [x for x in globals().values() if callable(x) and x.__name__.startswith('test')]
# strip leading underscore and test from names
test_names = sorted([x.__name__[5:] for x in test_funcs])

parser = argparse.ArgumentParser(description=descr, formatter_class=RawDescriptionHelpFormatter)

parser.add_argument('measurement', nargs="?", \
    help="""The measurement to be processed. This can either be a .bin file or a project folder that
    has subfolders 'fracture/acc', where the bin file is located.""")
parser.add_argument('-test', nargs=1, default=None, choices=test_names,
    help="""Run the test environment""")
parser.add_argument('--sameaxis', action="store_true", help='Plot all datasets in the same axis.')

args = parser.parse_args()

if args.measurement.lower().endswith('.bin'):
    file = args.measurement
else:
    if args.measurement.count('.') == 3:
        args.measurement = os.path.join(general.base_path, args.measurement)

    search_path = os.path.join(args.measurement, 'fracture', 'acceleration')
    for pfile in os.listdir(search_path):
        if pfile.lower().endswith('bin'):
            file = os.path.join(search_path, pfile)
            break

out_dir = os.path.dirname(file)


# file = r"d:\Forschung\Glasbruch\Versuche.Reihe\Proben\4.70.30.B\fracture\acc\4.70.30.A_fracture.bin"
# pfile = r"d:\Forschung\Glasbruch\Versuche.Reihe\Proben\8.140.Z.2\frac\8_140_Z_2.bin"
reader = APReader(file, verbose=True)

# reader_to_csv(reader)

reader.printSummary()
reader.plot(sameAxis=args.sameaxis)

if args.test is not None:
    fname = f'test_{args.test[0]}'
    print(f'Calling {fname}')
    func = globals()[fname]
    func(reader)

# slow_group = reader.Groups[0]
# fall_group = reader.collectDatasets(["Fall_g1", "Fall_g2"])

# plot_multiple_datasets(fall_group, 'Fallgewicht')

# slow_group.plot()
