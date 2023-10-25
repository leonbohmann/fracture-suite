from rich.progress import Progress, SpinnerColumn, \
    TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn

def default_progress(start = False):
    global state
    if state is not None:
        if start:
            state['progress'].start()
        return state['progress']

    prog = get_progress()
    prog.start()
    return prog


def get_progress(expand = True, bw = 80):
    return Progress(
                TextColumn("[progress.description]{task.description:<50}"),
                BarColumn(bar_width=bw),
                TaskProgressColumn(justify="right"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            transient=True, refresh_per_second=3, expand=expand)

class ProgSpinner():

    def __init__(self, spinnerProgress: Progress, title: str = "Loading ..."):
        self.progress = spinnerProgress
        self.task = self.progress.add_task(description=title)

    def set_description(self, description: str):
        self.progress.update(self.task, description=description)
    def set_total(self, total: int):
        self.progress.update(self.task, total=total)
    def set_completed(self, completed: int):
        self.progress.update(self.task, completed=completed)
    def advance(self):
        self.progress.advance(self.task)
    def add_task(self, description: str, total: int = None):
        return self.progress.add_task(description=description, total=total)
    def remove_task(self, task_id):
        self.progress.remove_task(task_id)
    def __enter__(self):
        self.progress.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.progress.__exit__(exc_type, exc_val, exc_tb)


def get_spinner(description: str = "Loading specimens...", with_bar: bool = False) -> ProgSpinner:
    prog = Progress(
                TaskProgressColumn(),
                BarColumn(bar_width=10) if with_bar else SpinnerColumn(),
                TextColumn("[progress.description]{task.description:<50}"),
                TimeElapsedColumn(),
            transient=True)
    return ProgSpinner(prog, description)
