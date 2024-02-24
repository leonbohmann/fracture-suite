import re
import os

import numpy as np

from rich import print
from apread import APReader
from apread.entries import Channel
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, spectrogram

from fracsuite.core.signal import lowpass, remove_freq

# enable this to display debug info and plots when loading any acceleration data
DEBUG = False


ns_f = 1e-9       # nanoseconds factor
us_f = 1e-6       # microseconds factor
ms_f = 1e-3       # milliseconds factor

def ms(n: float) -> float:
    return n / ms_f
def us(n: float) -> float:
    return n / us_f
def ns(n: float) -> float:
    return n / ns_f

def get_impact_time(channel: Channel, height=20, distance=10):



    mean_drop_g = np.max(np.abs(channel.data[:10000]))*1.5
    xx1 = np.abs(channel.data/mean_drop_g)**10
    impact_time_i: int = np.argwhere(xx1 >= 1)[0]-2
    impact_time = channel.Time.data[impact_time_i]
    return impact_time_i,impact_time

def runtimes(break_pos: str = "corner"):
    """
    Calculate the runtimes of the primary wave, secondary wave and crackfront.
    Measured between impact and drop weight (Fall_g1) and Acc_2 or Acc_1.

    Args:
        break_pos (str, optional): Break Position.. Defaults to "center".

    Returns:
        tuple[float,float,float]: The runtimes; primary, shear, crackfront.
    """

    # wave and crackfront velocities
    v_p = 5500 * 1e3 # mm/s
    v_s = 3500 * 1e3
    v_c = 1500 * 1e3

    # distances
    d_p = 450   # primary wave first registered on 2nd sensor
    d_s = 400   # secondary wave first registered on 1st sensor
    d_c = np.sqrt(450**2 + 450**2) # diagonal distance when crack is finished

    if break_pos == "center":
        d_p = 250
        d_s = 200
        d_c = np.sqrt(250**2 + 250**2)

    prim_runtime = (d_p / v_p)
    sec_runtime = (d_s / v_s)
    crackfront_runtime = (d_c / v_c)    # default: 0.4ms, eher weniger weil die Kante eigentlich näher liegt

    return prim_runtime, sec_runtime, crackfront_runtime

class AccelerationData:
    """
    This class is used to load and analyze acceleration data from a given file. It is used to determine the time of impact
    and the time of fracture of a glass specimen.

    Created from specimens using their acceleration file.
    """

    def __init__(self, file):
        """
        Create a new AccelerationData object from a given file.

        Args:
            file (str): The file to load the data from.
        """
        self.file = file
        self.broken_immediately = False


        self.channels: list[Channel] = None

        self.load()

    def load(self):
        # create APReader to read the file
        reader = APReader(self.file)
        self.reader = reader

        if not any([re.match("Fall_g1", channel.Name) for channel in reader.Channels]):
            print(f"\t> [red]File {os.path.basename(self.file)} does not contain the necessary channels, Fall_g1 not found.")
            return

        # save channels
        self.channels = reader.Channels

        # preliminary runtime calculations
        _, _, t_c = runtimes()

        # get channel data
        chan_g1 = self.get_channel("Fall_g1")           # sensor on the drop weight
        acc2 = self.get_channel_like("[Aa]cc(_?)[26]")  # sensor on the edge (either 2 or 6)
        time = chan_g1.Time.data
        time2 = acc2.Time.data

        # drop_data = remove_freq(time, chan_g1.data, 5000, 15000, 1/(time[1]-time[0]))

        # remove all frequencies that may originate from the drop weight
        drop_data = lowpass(time, chan_g1.data, 3500, 1/(time[1]-time[0]))

        # normalize time, backrecorded for 0.5 seconds
        time = time - time[0] - 0.5
        time2 = time2 - time2[0] - 0.5

        # find drop peak
        peak_ind,_ = find_peaks(drop_data, distance=10, height=20)
        # find crack acceleration peak
        peak_ind2,_ = find_peaks(acc2.data, distance=100, height=500)

        # find times
        impact_time = time[peak_ind[0]] if len(peak_ind) > 0 else time[0]
        frac_time = time2[peak_ind2[0]] if len(peak_ind2) > 0 else time2[0]

        # now, check if the impact made the glass break immediately
        if frac_time - impact_time < 1.5*t_c:
            self.broken_immediately = True

        if DEBUG:
            fig,axs = plt.subplots()
            axs.plot(time, drop_data, label='Filtered')
            axs.plot(time, chan_g1.data, label='Original')
            axs.legend()
            plt.show()
            # now, check if the impact made the glass break immediately
            print(f'Crackfront runtime: {ms(t_c):.3f}ms')
            print(f'Time of impact: {ms(impact_time):.3f}ms')
            print(f'Time of fracture: {ms(frac_time):.3f}ms')
            if frac_time - impact_time > t_c:
                print("> Glass didn't break immediately")
            else:
                print("> Glass broke immediately")
                self.broken_immediately = True

            # plot all peaks
            fig,axs = plt.subplots()
            axs.plot(ms(time2), acc2.data, label='Reaktion')
            axs.plot(ms(time), drop_data, label='Impaktor')
            axs.plot(ms(time[peak_ind]),drop_data[peak_ind],'x')
            axs.plot(ms(time2[peak_ind2]),acc2.data[peak_ind2],'o')
            axs.legend()
            plt.show()

            print('Create spectrogram')
            # compute spectrograms of the channels
            f, t, Sxx = spectrogram(drop_data, fs=1/(time[1]-time[0]), nperseg=1024, noverlap=512)
            fig,axs = plt.subplots()
            axs.pcolormesh(t, f, Sxx, shading='gouraud')
            axs.plot(time[peak_ind],drop_data[peak_ind],'x')
            axs.plot(time2[peak_ind2],acc2.data[peak_ind2],'o')
            plt.show()


    def get_acc_chans(self) -> list[Channel]:
        return [channel for channel in self.channels if re.match("[Aa]cc(_?)", channel.Name) is not None]

    def get_channel_like(self, filter: str) -> Channel:
        for channel in self.channels:
            if re.match(filter, channel.Name) is not None:
                return channel

        raise ValueError(f"Channel with filter {filter} not found in {self.file}")

    def get_channel(self, channel_name: str) -> Channel:
        for channel in self.channels:
            if channel.Name == channel_name:
                return channel

        raise ValueError(f"Channel {channel_name} not found in {self.file}")
