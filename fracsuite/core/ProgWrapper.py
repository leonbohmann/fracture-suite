from rich.progress import Progress


class ProgWrapper():
    entered: bool
    enter_level: int
    total: int
    descr: str
    tasks: list
    def __init__(self, spinnerProgress: Progress, title: str = None, total = None):
        self.progress = spinnerProgress
        self.task = self.progress.add_task(title, total=total)
        self.tasks = [self.task]
        self.entered = False
        self.enter_level = 0

        self.total = total
        self.descr = title

        self.ntotal = None
        self.ndescr = None

        self.nset_description(title)
        self.nset_total(total)

        self.exithandler = None

    def nset_description(self, description: str):
        self.ndescr = description
    def nset_total(self, total: int):
        self.ntotal = total

    def set_description(self, description: str):
        self.progress.update(self.tasks[self.enter_level], description=description, refresh=True)
    def set_total(self, total: int):
        self.progress.update(self.tasks[self.enter_level], total=total, refresh=True)
    def set_completed(self, completed: int):
        self.progress.update(self.tasks[self.enter_level], completed=completed, refresh=True)
    def advance(self):
        self.progress.advance(self.tasks[self.enter_level])
    def advance_task(self, taskid):
        self.progress.advance(taskid)
    def add_task(self, description: str, total: int = None):
        return self.progress.add_task(description=description, total=total)
    def remove_task(self, task_id):
        self.progress.remove_task(task_id)

    def start(self):
        self.__enter__()
    def stop(self):
        self.__exit__(None, None, None)
    def pause(self):
        self.progress.stop()

    def resume(self):
        self.progress.start()

    def __enter__(self):
        if not self.entered:
            self.entered = True
            self.progress.__enter__()

            self.progress.update(self.tasks[self.enter_level], description=self.ndescr, total=self.ntotal, refresh=True)
        else:
            newtask = self.add_task('', None)
            self.tasks.insert(self.enter_level+1, newtask)
            self.enter_level += 1

            self.progress.update(self.tasks[self.enter_level], description=self.ndescr, total=self.ntotal, refresh=True)
            self.ndescr = None
            self.ntotal = None


        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enter_level == 0:
            self.entered = False

            if self.exithandler is not None:
                self.exithandler()

            return self.progress.__exit__(exc_type, exc_val, exc_tb)
        else:
            self.remove_task(self.tasks[self.enter_level])
            self.enter_level -= 1
            return False

    def set_exit_handler(self, handler):
        self.exithandler = handler