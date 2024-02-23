from scipy import signal
from scipy.signal import butter, sosfiltfilt
import numpy as np

def smooth_moving_average(x, window_len=11):
    """smooth the data using a moving average window with requested size."""
    assert x.ndim == 1, "smooth only accepts 1 dimension arrays."
    assert x.size > window_len, "Input vector needs to be bigger than window size."

    s = np.pad(x, (window_len//2, window_len//2), mode='edge')
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

def smooth_hanning(x,window_len=11):
    """smooth the data using a hanning window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """

    assert x.ndim == 1, "smooth only accepts 1 dimension arrays."
    assert x.size > window_len, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    win = signal.windows.hann(window_len)
    filtered = signal.convolve(x, win, mode='same') / sum(win)
    return filtered

def remove_freq(t, data, f0, f1, fs):
    """
    Filtert Frequenzen zwischen f0 und f1 aus einem Signal heraus.

    :param t: Zeitvektor des Signals.
    :param data: Das Eingangssignal.
    :param f0: Die untere Grenzfrequenz des zu entfernenden Bereichs.
    :param f1: Die obere Grenzfrequenz des zu entfernenden Bereichs.
    :param fs: Die Abtastrate des Signals.
    :return: Das gefilterte Signal.
    """
    # Erstellt den Bandpassfilter
    sos = butter(4, [f0, f1], btype='bandstop', analog=False, output='sos', fs=fs)
    # Wendet den Filter an
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


def highpass(t, data, f0, fs):
    """
    Filtert Frequenzen unterhalb von f0 aus einem Signal heraus.

    :param t: Zeitvektor des Signals.
    :param data: Das Eingangssignal.
    :param f0: Die untere Grenzfrequenz des zu entfernenden Bereichs.
    :param fs: Die Abtastrate des Signals.
    :return: Das gefilterte Signal.
    """
    # Erstellt den Hochpassfilter
    sos = butter(4, f0, btype='highpass', analog=False, output='sos', fs=fs)
    # Wendet den Filter an
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def lowpass(t, data, f0, fs):
    """
    Filtert Frequenzen oberhalb von f0 aus einem Signal heraus.

    :param t: Zeitvektor des Signals.
    :param data: Das Eingangssignal.
    :param f0: Die obere Grenzfrequenz des zu entfernenden Bereichs.
    :param fs: Die Abtastrate des Signals.
    :return: Das gefilterte Signal.
    """
    # Erstellt den Tiefpassfilter
    sos = butter(4, f0, btype='lowpass', analog=False, output='sos', fs=fs)
    # Wendet den Filter an
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


def bands(t, data, fs, width, *f):
    """
    Filter all frequencies in f from the signal data. The width parameter is the width around each frequency.
    """
    assert len(f) > 0, "At least one frequency must be given."

    for freq in f:
        # Erstellt den Tiefpassfilter
        sos = butter(4, [freq-width//2, freq+width//2], btype='bandstop', analog=False, output='sos', fs=fs)
        # Wendet den Filter an
        filtered_data = sosfiltfilt(sos, data)

    return filtered_data
