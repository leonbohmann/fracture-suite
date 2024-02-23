from typing import Any, Iterator, List, Sequence, TypeVar, Union
from rich.progress import Progress, SpinnerColumn, \
    TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from fracsuite.core.ProgWrapper import ProgWrapper

from fracsuite.state import State
from typing import Iterator, TypeVar
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

T = TypeVar('T')
def tracker(iterator: Union[Sequence[T] | Iterator[T] | dict[T,Any]], title=None, total=None) -> Iterator[T]:

    if total is None and hasattr(iterator, '__len__'):
        total = len(iterator)

    progress = get_progress(total=total, title=title)
    progress.start()

    try:
        for obj in iterator:
            yield obj
            progress.advance()
    finally:
        progress.stop()
