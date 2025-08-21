from pathlib import Path
import collections
import time
from typing import Any
import gc
import os.path

from PyQt6 import QtCore, QtGui

from .settings import settings


BACKGROUND_COLOR_HEX = "#1e1e1e"
TOOLBAR_MARGIN = (3, 0, 3, 0)


class FIFODict(collections.OrderedDict):
    """FIFO dictionary that automatically removes the oldest item when the maximum size is reached."""
    def __init__(self, maxsize: int = 128, remove_callback=None):
        super().__init__()
        self.maxsize = maxsize
        self.remove_callback = remove_callback

    def __setitem__(self, key: Any, value: Any):
        if key in self:
            self.move_to_end(key)
        elif len(self) + 1 > self.maxsize:
            item = self.popitem(last=False)
            del item
            gc.collect()
            if self.remove_callback:
                self.remove_callback(self)
        super().__setitem__(key, value)


def generate_image_filepath(suffix: str = "") -> Path:
    """Generate a unique image filepath."""
    t = int(time.time())
    file_path = settings.autosave_image.path / Path(f"{t}{suffix}.jpg")
    
    if os.path.exists(file_path):
        new_suffix = f"{suffix}0"
        file_path = generate_image_filepath(new_suffix)
    
    return file_path


def get_aspect_ratios(labels: list[str]) -> list[tuple[str, float]]:
    labels_ = [(label, label.split(" ")) for label in labels]   # ["1:1"], ["P", "4:5"]
    labels_ = [(label, x[1]) if len(x) > 1 else (label, x[0]) for label, x in labels_]  # "1:1", "4:5"
    labels_ = [(label, x.split(":")) for label, x in labels_]   # ["1", "1"], ["4", "5"]
    labels_ = [(label, int(x[1]) / int(x[0])) for label, x in labels_]  # 1.0, 1.2555555
    return [(label, round(x, 2)) for label, x in labels_]   # 1.00, 1.26
    
    
def pixmap_to_bytes(pixmap: QtGui.QPixmap) -> bytes:
    """Convert a QPixmap to bytes."""
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    buffer.close()
    return buffer.data().data()
