from scipy.signal import butter, sosfiltfilt

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
