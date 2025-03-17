from pathlib import Path
import collections
import time
from typing import Any

from .settings import settings


BACKGROUND_COLOR_HEX = "#1e1e1e"
TOOLBAR_MARGIN = (3, 0, 3, 0)


class FIFODict(collections.OrderedDict):
    """FIFO dictionary that automatically removes the oldest item when the maximum size is reached."""
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
    
    
def get_aspect_ratios(labels: list[str]) -> list[tuple[str, float]]:
    labels_ = [(label, label.split(" ")) for label in labels]   # ["1:1"], ["P", "4:5"]
    labels_ = [(label, x[1]) if len(x) > 1 else (label, x[0]) for label, x in labels_]  # "1:1", "4:5"
    labels_ = [(label, x.split(":")) for label, x in labels_]   # ["1", "1"], ["4", "5"]
    labels_ = [(label, int(x[0]) / int(x[1])) for label, x in labels_]  # 1.0, 1.2555555
    return [(label, round(x, 2)) for label, x in labels_]   # 1.00, 1.26
