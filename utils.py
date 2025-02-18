import datetime
import linecache
import os
import tracemalloc
from contextlib import ContextDecorator


BACKGROUND_COLOR_HEX = "#1e1e1e"


class Timer(ContextDecorator):
    started: datetime.datetime
    delta: datetime.timedelta

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.started = datetime.datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ended = datetime.datetime.now()
        self.delta = ended - self.started

        print(f"{self.name} completed in {self.delta}")


class TraceMem(ContextDecorator):
    def __enter__(self):
        tracemalloc.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        snapshot = tracemalloc.take_snapshot()
        self.display_top(snapshot)

    @staticmethod
    def display_top(snapshot, key_type='lineno', limit=3):
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = snapshot.statistics(key_type)

        print(f"Top {limit} lines")
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print(f"#{index}: {filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print(f'    {line}')

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print(f"{len(other)} other: {size / 1024:.1f} KiB")
        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f KiB" % (total / 1024))
