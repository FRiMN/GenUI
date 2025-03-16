from pathlib import Path
import collections
import time
from typing import Any

from .settings import settings


BACKGROUND_COLOR_HEX = "#1e1e1e"
TOOLBAR_MARGIN = (3, 0, 3, 0)


class FIFODict(collections.OrderedDict):
    def __init__(self, maxsize: int = 128):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key: Any, value: Any):
        if key in self:
            self.move_to_end(key)
        elif len(self) + 1 > self.maxsize:
            self.popitem(last=False)
        super().__setitem__(key, value)


def generate_image_filepath() -> Path:
    """Generate a unique image filepath."""
    t = int(time.time())
    return settings.autosave_image.path / Path(f"{t}.jpg")
