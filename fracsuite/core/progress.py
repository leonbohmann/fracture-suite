from rich.progress import Progress, SpinnerColumn, \
    TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn


def get_progress():
    return Progress(
                TextColumn("[progress.description]{task.description:<50}"),
                BarColumn(bar_width=80),
                TaskProgressColumn(justify="right"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            transient=True)

class ProgSpinner():

    def __init__(self, spinnerProgress: Progress):
        self.progress = spinnerProgress

    def __enter__(self):
        return self.progress.__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.progress.__exit__(exc_type, exc_val, exc_tb)


def get_spinner_prog(description: str = "Loading ..."):
    prog = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description:<50}"),
                TimeElapsedColumn(),
            transient=True)
    prog.add_task(description=description, total=1.0)
    return prog
