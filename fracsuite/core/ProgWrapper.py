from rich.progress import Progress
from fracsuite.core.logging import info



class ProgWrapper():

    entered: bool
    enter_level: int
    total: int
    descr: str
    tasks: list
    completed: int
    lastf: float
    def __init__(self, spinnerProgress: Progress, title: str = None, total = None):
        from fracsuite.state import State

        self.progress = spinnerProgress
        self.task = self.progress.add_task(title, total=total)
        self.tasks = [self.task]
        self.entered = False
        self.enter_level = 0

        self.total = total
        self.descr = title
        self.completed = 0

        self.ntotal = None
        self.ndescr = None

        self.nset_description(title)
        self.nset_total(total)

        self.exithandler = None
        self.lastf = -1
        if State.debug:
            self.progress.update(self.task, visible=False, refresh=True)

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
        import fracsuite.state as st

        if st.State.debug:
            f = (self.completed/self.progress.tasks[self.enter_level].total)*100
            if f % 10 == 0 and f > self.lastf:
                info(f"Step {self.completed}/{self.progress.tasks[self.enter_level].total}: {self.progress.tasks[self.enter_level].description}")
                self.lastf = f

        else:
            self.progress.advance(self.tasks[self.enter_level])

        self.completed += 1
    def advance_task(self, taskid):
        self.progress.advance(taskid)
    def add_task(self, description: str, total: int = None):
        return self.progress.add_task(description=description, total=total)
    def remove_task(self, task_id):
        self.progress.remove_task(task_id)

    def enter(self):
        self.__enter__()
    def exit(self):
        self.__exit__(None, None, None)
    def pause(self):
        self.progress.stop()

    def resume(self):
        self.progress.start()

    def __enter__(self):
        import fracsuite.state as st

        if not self.entered:
            self.entered = True
            self.completed = 0
            self.progress.__enter__()

            self.progress.update(self.tasks[self.enter_level], description=self.ndescr, total=self.ntotal, refresh=True)
        else:
            self.completed = 0
            newtask = self.add_task('', None)
            self.tasks.insert(self.enter_level+1, newtask)
            self.enter_level += 1

            self.progress.update(self.tasks[self.enter_level], description=self.ndescr, total=self.ntotal, refresh=True)
            self.ndescr = None
            self.ntotal = None

        if st.State.debug:
            self.progress.update(self.tasks[self.enter_level], visible=False, refresh=True)

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