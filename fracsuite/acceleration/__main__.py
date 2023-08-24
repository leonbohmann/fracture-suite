import argparse
from argparse import RawDescriptionHelpFormatter
import os
from matplotlib import pyplot as plt
from apread import APReader, plot_multiple_datasets
import numpy as np
from scipy import integrate, signal

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

def time_to_index(time_array, t):
    """Returns the index of the time array that is closest to the given time t."""
    return np.argmin(np.abs(time_array - t))


parser = argparse.ArgumentParser(description=descr, formatter_class=RawDescriptionHelpFormatter)    

parser.add_argument('measurement', nargs="?", \
    help="""The measurement to be processed. This can either be a .bin file or a project folder that
    has subfolders 'fracture/acc', where the bin file is located.""")

args = parser.parse_args()

if args.measurement.lower().endswith('.bin'):
    file = args.measurement
else:
    search_path = os.path.join(args.measurement, 'fracture', 'acc')
    for pfile in os.listdir(search_path):
        if pfile.lower().endswith('bin'):
            file = os.path.join(search_path, pfile)
            break

out_dir = os.path.dirname(file)


# file = r"d:\Forschung\Glasbruch\Versuche.Reihe\Proben\4.70.30.B\fracture\acc\4.70.30.A_fracture.bin"
# pfile = r"d:\Forschung\Glasbruch\Versuche.Reihe\Proben\8.140.Z.2\frac\8_140_Z_2.bin"
reader = APReader(file, verbose=True)

reader.printSummary()

# slow_group = reader.Groups[0]
# fall_group = reader.collectDatasets(["Fall_g1", "Fall_g2"])

# plot_multiple_datasets(fall_group, 'Fallgewicht')

# slow_group.plot()

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
fig.savefig(os.path.join(out_dir, "fall.png"))
