from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn


def get_progress():
    return Progress(
                TextColumn("[progress.description]{task.description:<50}"),
                BarColumn(bar_width=80),
                TaskProgressColumn(justify="right"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            transient=True)