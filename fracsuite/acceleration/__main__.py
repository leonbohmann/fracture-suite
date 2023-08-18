import argparse
from argparse import RawDescriptionHelpFormatter
import os

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
time = drop_acc.Time.data
drop_avg = np.average(drop_acc.data[:1000])
drop_data = drop_acc.data - drop_avg
print(f'Average at 0: {drop_avg}')


# savgol
drop_data_smooth = signal.savgol_filter(drop_data, 70, 5)

# firwin
cutoff_frequency = 0.1  # Adjust as needed
num_taps = 101  # Filter length, adjust as needed
fir_filter = signal.firwin(num_taps, cutoff_frequency)
drop_data_smooth = signal.lfilter(fir_filter, 1.0, drop_data)
# np.savetxt("output.txt", np.column_stack([time, drop_data]), fmt = ['%.8f', '%.8f'])
# drop_data = drop_data_smooth

# initial is needed, so that the length of the result is the same as the input
v = integrate.cumulative_trapezoid(drop_data, time, initial = 0)
s = integrate.cumulative_trapezoid(v, time, initial = 0)

fig = plot_multiple_datasets([\
    (time, drop_data_smooth, "m--", "Smoothed [g]", "Smoothed"), \
    (time, drop_data, "gray", "Acc [g]", "Acceleration"), \
    (time, v, "b", "Speed [m/s]", "Speed"),\
    (time, s, "g", "Distance [m]", "Distance")], 'Time integrals of Acceleration')

fig.savefig(os.path.join(out_dir, "fall.png"))
    