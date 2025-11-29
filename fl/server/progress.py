import ray
import time
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.table import Table
from rich.console import Console

@ray.remote
class ProgressActor:
    def __init__(self):
        self.tasks = {}  # {task_id: {"total": total, "completed": completed, "description": desc, "metrics": metrics}}
        self.should_stop = False

    def add_task(self, task_id, total, description=""):
        self.tasks[task_id] = {
            "total": total,
            "completed": 0,
            "description": description,
            "metrics": {}
        }

    def update(self, task_id, advance=0, metrics=None):
        if task_id in self.tasks:
            self.tasks[task_id]["completed"] += advance
            if metrics:
                self.tasks[task_id]["metrics"] = metrics

    def get_state(self):
        return self.tasks

    def stop(self):
        self.should_stop = True

import threading

class ProgressMonitor:
    def __init__(self, progress_actor):
        self.progress_actor = progress_actor
        self.console = Console()
        self.stop_event = threading.Event()
        self.thread = None
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[metrics]}"),
            console=self.console
        )
        self.rich_tasks = {}

    def __enter__(self):
        self.progress.start()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.progress.stop()

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            try:
                state = ray.get(self.progress_actor.get_state.remote())
                for task_id, task_data in list(state.items()):
                    if task_id not in self.rich_tasks:
                        self.rich_tasks[task_id] = self.progress.add_task(
                            task_data["description"], 
                            total=task_data["total"],
                            metrics=""
                        )
                    
                    metrics_str = " ".join([f"{k}:{v}" for k, v in task_data["metrics"].items()])
                    rich_task_id = self.rich_tasks[task_id]
                    self.progress.update(
                        rich_task_id, 
                        completed=task_data["completed"],
                        metrics=metrics_str
                    )

                    # 如果任务已完成，从界面和缓存中移除，避免进度条堆积
                    if task_data["total"] > 0 and task_data["completed"] >= task_data["total"]:
                        self.progress.remove_task(rich_task_id)
                        del self.rich_tasks[task_id]
            except Exception:
                pass
            time.sleep(0.1)
