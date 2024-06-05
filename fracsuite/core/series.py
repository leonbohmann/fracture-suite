from typing import Any
import warnings
import functools

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

def untilSeconds(chan, seconds):
    """
    Get data from a channel from the start for a certain amount of seconds.
    """
    if chan.isTime and seconds is not None:
        d = chan.data[chan.data < chan.data[0] + seconds]
        return d,d

    if seconds is not None:
        time = chan.Time.data
        data = chan.data
        time = time[time < time[0] + seconds]
        if len(time) == 0:
            print(f"untilSeconds: No data until {seconds} in channel {chan.name}.")
        data = data[:len(time)]
        return time, data
    return chan.Time.data, chan.data

def afterSeconds(chan: Any | tuple, seconds):
    """
    Get data from a channel from the end for a certain amount of seconds.
    """
    if isinstance(chan, tuple):
        time = chan[0]
        data = chan[1]
        name = ""
    else:
        time = chan.Time.data
        data = chan.data
        name = chan.name
        if chan.isTime and seconds is not None:
            d = chan.data[chan.data > seconds]
            return d,d
        
    if seconds is not None:
        time = time[time > seconds]
        if len(time) == 0:
            print(f"afterSeconds: No data after {seconds} in channel {name}.")
        data = data[-len(time):]
        return time, data

def betweenSeconds(chan, start, end):
    if isinstance(chan, tuple):
        time = chan[0]
        data = chan[1]
    else:
        time = chan.Time.data
        data = chan.data

        if chan.isTime and start is not None and end is not None:
            d = chan.data[(chan.data > chan.data[0] + start) & (chan.data < chan.data[0] + end)]
            return d, d
        elif start is None or end is None:
            return chan.Time.data, chan.data

    
    if start is not None and end is not None:
        mask = (time > start) & (time <  end)

        time = time[mask]

        if len(time) == 0:
            print(f"betweenSeconds: No data between {start} and {end} in channel {chan.name}.")
        data = data[mask]
    
    return time, data
    


