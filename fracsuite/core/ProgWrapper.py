from rich.progress import Progress


class ProgWrapper():
    entered: bool
    enter_level: int
    total: int
    descr: str
    def __init__(self, spinnerProgress: Progress, title: str = None, total = None):
        self.progress = spinnerProgress
        self.task = self.progress.add_task(title, total=total)
        self.entered = False
        self.enter_level = 0
        self.set_description(title)
        self.set_total(total)

        self.total = total
        self.descr = title

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

    def pause(self):
        self.progress.stop()

    def resume(self):
        self.progress.start()

    def __enter__(self):
        if not self.entered:
            self.entered = True
            self.progress.__enter__()

            self.set_total(self.total)
            self.set_description(self.descr)
        else:
            self.enter_level += 1

        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enter_level == 0:
            return self.progress.__exit__(exc_type, exc_val, exc_tb)
        else:
            self.enter_level -= 1
            return False