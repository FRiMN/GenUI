import datetime
from contextlib import ContextDecorator


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
