from typing import Iterator, List, TypeVar
from rich.progress import Progress, SpinnerColumn, \
    TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from fracsuite.core.ProgWrapper import ProgWrapper

from fracsuite.state import State

def default_progress(start = False):
    global state
    if state is not None:
        if start:
            state['progress'].start()
        return state['progress']

    prog = get_progress()
    prog.start()
    return prog

def on_progress_exit():
    State.progress = None

def get_progress(expand = True, bw = 80, title="Progress...", total=None):
    if State.progress is None:
        prog = Progress(
                    TextColumn("[progress.description]{task.description:<50}"),
                    BarColumn(bar_width=bw),
                    TaskProgressColumn(justify="right"),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                transient=True, refresh_per_second=3, expand=expand)
        State.progress = ProgWrapper(prog, title=title, total=total)
        State.progress.set_exit_handler(on_progress_exit)


    State.progress.nset_description(title)
    State.progress.nset_total(total)

    return State.progress



def get_spinner(description: str = "Loading specimens...", with_bar: bool = False) -> ProgWrapper:
    return get_progress(False, title=description)

T = TypeVar("T")
class tracker:
    def __init__(self, data: List[T], title=None,total=None):
        self.current = 0
        self.high = len(data) if isinstance(data, (list, tuple)) else total
        if self.high is None:
            raise ValueError("Total length of data must be provided if data is not a list or tuple.")

        self.baseiterator = iter(data)
        self.progress = get_progress(total=self.high, title=title)  # Add this line to create a progress with total set to the length of data
        self.progress.start()

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T: # Python 2: def next(self)
        self.current += 1
        if self.current <= self.high:
            self.progress.advance()  # Add this line to advance the progress
            return next(self.baseiterator)

        self.progress.stop()
        raise StopIteration
