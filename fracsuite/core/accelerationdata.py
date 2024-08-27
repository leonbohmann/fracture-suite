import re
import os
from typing import Tuple

import numpy as np

from rich import print
from apread import APReader
from apread.entries import Channel
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, spectrogram
from scipy.signal import wiener as w2

from fracsuite.core.logging import error
from fracsuite.core.series import betweenSeconds
from fracsuite.core.signal import bandstop, highpass, lowpass

# enable this to display debug info and plots when loading any acceleration data
DEBUG = False


ns_f = 1e-9  # nanoseconds factor
us_f = 1e-6  # microseconds factor
ms_f = 1e-3  # milliseconds factor


def ms(n: float) -> float:
    return n / ms_f


def us(n: float) -> float:
    return n / us_f


def ns(n: float) -> float:
    return n / ns_f


def get_impact_time(channel: Channel, height=20, distance=10):
    mean_drop_g = np.max(np.abs(channel.data[:10000])) * 1.5
    xx1 = np.abs(channel.data / mean_drop_g) ** 10
    impact_time_i: int = np.argwhere(xx1 >= 1)[0] - 2
    impact_time = channel.Time.data[impact_time_i]
    return impact_time_i, impact_time


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
    v_p = 5500 * 1e3  # mm/s
    v_s = 3500 * 1e3
    v_c = 1500 * 1e3

    # distances
    d_p = 450  # primary wave first registered on 2nd sensor
    d_s = 400  # secondary wave first registered on 1st sensor
    d_c = np.sqrt(450**2 + 450**2)  # diagonal distance when crack is finished

    if break_pos == "center":
        d_p = 250
        d_s = 200
        d_c = np.sqrt(250**2 + 250**2)

    prim_runtime = d_p / v_p
    sec_runtime = d_s / v_s
    crackfront_runtime = (
        d_c / v_c
    )  # default: 0.4ms, eher weniger weil die Kante eigentlich näher liegt

    return prim_runtime, sec_runtime, crackfront_runtime


class AccelerationData:
    """
    This class is used to load and analyze acceleration data from a given file. It is used to determine the time of impact
    and the time of fracture of a glass specimen.

    Created from specimens using their acceleration file.
    """
    __is_ok: bool

    @property
    def is_ok(self) -> bool:
        return self.__is_ok

    channels: list[Channel]
    drop_channel: Channel

    def __init__(self, file, zero_channels: bool = True):
        """
        Create a new AccelerationData object from a given file.

        Args:
            file (str): The file to load the data from.
        """
        self.file = file
        self.broken_immediately = False

        self.channels: list[Channel] = None

        if file is not None:
            self.load(zero_channels=zero_channels)
        else:
            self.__is_ok = False

    def filter_fallgewicht_lowpass(self, f0: float = 5000):
        original_chan_data = None

        print(
            f"Applying lowpass filter for drop-channel {self.drop_channel.Name} at {f0}..."
        )
        original_chan_data = self.drop_channel.data
        # transform channel data
        time = self.drop_channel.Time

        self.drop_channel.data = lowpass(time, self.drop_channel.data, f0)

        return original_chan_data, self.drop_channel

    def filter_fallgewicht_wiener(self):
        """Filter out some more noise from the signal."""
        # self.drop_channel.data = bandstop(self.drop_channel.Time, self.drop_channel.data, 150,2500)
        self.drop_channel.data = w2(self.drop_channel.data, 17)

    def filter_fallgewicht_highpass(self, f0: float = 1000):
        """Filter out a highpass filter from the signal."""
        original_chan_data = None

        print(
            f"Applying highpass filter for drop-channel {self.drop_channel.Name} at {f0}..."
        )
        original_chan_data = self.drop_channel.data
        # transform channel data
        time = self.drop_channel.Time
        self.drop_channel.data = highpass(time, self.drop_channel.data, f0)
        return original_chan_data, self.drop_channel

    def filter_fallgewicht_eigenfrequencies(
        self,
    ) -> Tuple[list[np.ndarray], list[Channel]]:
        """
        Filter the drop channels from the acceleration data. This is done by applying a lowpass filter to the data.

        Args:
            lowpassfreq (int, optional): The frequency to filter. Defaults to 7500.
        Returns:
            list[np.ndarray]: The original channel data before filtering.
        """
        original_chan_data = None

        print("Filter eigenfrequencies of drop channel...")
        original_chan_data = self.drop_channel.data
        # transform channel data
        time = self.drop_channel.Time
        # self.drop_channel.data = lowpass(time, self.drop_channel.data, lowpassfreq, 1/(time[1]-time[0]))


        # # EIGENFREQUENCIES FROM ANYS
        # eigen_f = [
        #     1009,
        #     3107,
        #     7601,
        #     7605,
        #     7639,
        #     19724
        # ]
        # # space around the eigenfrequencies
        # df = 100
        # for f in eigen_f:
        #     f0, f1 = f-df, f+df
        #     print(f" > Filtering {f0}-{f1}...")
        #     self.drop_channel.data = bandstop(
        #         time, self.drop_channel.data, f0, f1, 1 / (time[1] - time[0])
        #     )

        ## EIGENFREQUENCIES FROM FFT (Experiment)
        # (100,130), (2050, 2100),
        for f in [ (2000,2200), (7100, 7300), (7700, 7900), (9200, 9800)]:
            f0, f1 = f
            print(f" > Filtering {f0}-{f1}...")
            self.drop_channel.data = bandstop(
                time, self.drop_channel.data, f0, f1, 1 / (time[1] - time[0])
            )

        return original_chan_data, self.drop_channel

    def load(self, zero_channels: bool = True):
        """
        Loads the acceleration data from the file and calculates the time of impact and fracture.
        By default, the channels are zeroed using the first 0.35 seconds of the data.

        Args:
            zero_channels (bool, optional): Whether to zero the channels. Defaults to True.
        """
        # create APReader to read the file
        reader = APReader(self.file)
        self.reader = reader
        self.__is_ok = True
        
        if not any([re.match("Fall_g1", channel.Name) for channel in reader.Channels]):
            print(
                f"\t> [red]File {os.path.basename(self.file)} does not contain the necessary channels, Fall_g1 not found."
            )   
            self.__is_ok = False
            return

        
        if any((x.sampling_frequency < 20000 for x in reader.Channels if 'Fall_g' in x.Name)):
            error("The sampling rate is too low. The acceleration data can not be considered for calculatations.")
            self.__is_ok = False
            return

        # save channels
        self.channels: list[Channel] = reader.Channels

        # when zeroing channels, calculate zeroing
        if zero_channels:
            # print("Zeroing channels...")
            for channel in self.channels:
                # time channel gets set to the start time
                if channel.isTime:
                    channel.data = channel.data - channel.data[0]
                # all other channels get zeroed in the first 0.2 seconds
                else:
                    channel.zero(seconds=0.2)

        # preliminary runtime calculations
        _, _, t_c = runtimes()

        # get channel data
        chan_g1 = self.get_channel("Fall_g1")  # sensor on the drop weight
        acc2 = self.get_channel_like(
            "[Aa]cc(_?)[26]"
        )  # sensor on the edge (either 2 or 6)
        time = chan_g1.Time.data
        time2 = acc2.Time.data

        # save main drop channel for easier access
        self.drop_channel = chan_g1

        # remove all frequencies that may originate from the drop weight
        drop_data = lowpass(time, chan_g1.data, 3500)

        # normalize time, backrecorded for 0.5 seconds
        time = time - time[0] - 0.5
        time2 = time2 - time2[0] - 0.5

        # find drop peak
        peak_ind, _ = find_peaks(drop_data, distance=10, height=20)
        # find crack acceleration peak
        peak_ind2, _ = find_peaks(acc2.data, distance=100, height=500)

        # find times
        impact_time = time[peak_ind[0]] if len(peak_ind) > 0 else time[0]
        frac_time = time2[peak_ind2[0]] if len(peak_ind2) > 0 else time2[0]

        # now, check if the impact made the glass break immediately
        if frac_time - impact_time < 1.5 * t_c:
            self.broken_immediately = True

        if DEBUG:
            fig, axs = plt.subplots()
            axs.plot(time, drop_data, label="Filtered")
            axs.plot(time, chan_g1.data, label="Original")
            axs.legend()
            plt.show()
            # now, check if the impact made the glass break immediately
            print(f"Crackfront runtime: {ms(t_c):.3f}ms")
            print(f"Time of impact: {ms(impact_time):.3f}ms")
            print(f"Time of fracture: {ms(frac_time):.3f}ms")
            if frac_time - impact_time > t_c:
                print("> Glass didn't break immediately")
            else:
                print("> Glass broke immediately")
                self.broken_immediately = True

            # plot all peaks
            fig, axs = plt.subplots()
            axs.plot(ms(time2), acc2.data, label="Reaktion")
            axs.plot(ms(time), drop_data, label="Impaktor")
            axs.plot(ms(time[peak_ind]), drop_data[peak_ind], "x")
            axs.plot(ms(time2[peak_ind2]), acc2.data[peak_ind2], "o")
            axs.legend()
            plt.show()

            print("Create spectrogram")
            # compute spectrograms of the channels
            f, t, Sxx = spectrogram(
                drop_data, fs=1 / (time[1] - time[0]), nperseg=1024, noverlap=512
            )
            fig, axs = plt.subplots()
            axs.pcolormesh(t, f, Sxx, shading="gouraud")
            axs.plot(time[peak_ind], drop_data[peak_ind], "x")
            axs.plot(time2[peak_ind2], acc2.data[peak_ind2], "o")
            plt.show()

    def get_acc_chans(self) -> list[Channel]:
        return [
            channel
            for channel in self.channels
            if re.match("[Aa]cc(_?)", channel.Name) is not None
        ]

    def get_channels_like(self, filter: str) -> list[Channel]:
        return [
            channel
            for channel in self.channels
            if re.match(filter, channel.Name) is not None
        ]

    def get_channel_like(self, filter: str, panic=True) -> Channel:
        for channel in self.channels:
            if re.match(filter, channel.Name) is not None:
                return channel

        if panic:
            raise ValueError(f"Channel with filter {filter} not found in {self.file}")
        else:
            return None

    def get_channel(self, channel_name: str) -> Channel:
        for channel in self.channels:
            if channel.Name == channel_name:
                return channel

        raise ValueError(f"Channel {channel_name} not found in {self.file}")
