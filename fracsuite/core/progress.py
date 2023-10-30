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

    return State.progress



def get_spinner(description: str = "Loading specimens...", with_bar: bool = False) -> ProgWrapper:
    prog = Progress(
                TaskProgressColumn(),
                BarColumn(bar_width=10) if with_bar else SpinnerColumn(),
                TextColumn("[progress.description]{task.description:<50}"),
                TimeElapsedColumn(),
            transient=True)
    return ProgWrapper(prog, description)
