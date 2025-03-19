import datetime
import linecache
import tracemalloc
from contextlib import ContextDecorator
from pathlib import Path


class Timer(ContextDecorator):
    """Timer context manager and decorator.

    Usage:
        with Timer("My task"):
            do_something()

    or:
        @Timer("My task")
        def my_function():
            do_something()
    """

    started: datetime.datetime
    delta: datetime.timedelta

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        # We use relative time and not need to use timezone.
        self.started = datetime.datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        ended = datetime.datetime.now()
        self.delta = ended - self.started

        print(f"{self.name} completed in {self.delta}")
        
        
class TraceMem(ContextDecorator):
    """Context manager to trace memory usage."""

    def __enter__(self):
        tracemalloc.start()

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        snapshot = tracemalloc.take_snapshot()
        self.display_top(snapshot)

    @staticmethod
    def display_top(snapshot: tracemalloc.Snapshot, key_type: str = "lineno", limit: int = 3):
        snapshot = snapshot.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),  # noqa: FBT003
                tracemalloc.Filter(False, "<unknown>"),  # noqa: FBT003
            )
        )
        top_stats = snapshot.statistics(key_type)

        print(f"Top {limit} lines")
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            module_name, file_name = Path(frame.filename).parts[-2:]
            filename = Path(module_name) / Path(file_name)
            print(f"#{index}: {filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print(f"    {line}")

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print(f"{len(other)} other: {size / 1024:.1f} KiB")
        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f KiB" % (total / 1024))
        