from __future__ import annotations

import datetime
import time
from multiprocessing import Pipe
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal
from torch import OutOfMemoryError

from .common.trace import Timer

if TYPE_CHECKING:
    from .generator.sdxl import GenerationPrompt
    from multiprocessing.connection import Connection
    from PIL import Image


class Worker(QObject):
    """Worker runs a generation task in a separate thread."""

    finished = pyqtSignal()  # Worker is finished and starts to close (close the main application).
    done = pyqtSignal()  # Worker is done with the generation task.
    error = pyqtSignal(str)  # Worker encountered an error.
    progress_preview = pyqtSignal(bytes, int, int, int, int, datetime.timedelta)

    poll_timeout = 0.3  # Poll timeout for checking data availability

    parent_conn: Connection
    child_conn: Connection

    def __init__(self):
        super().__init__()
        self._started = False
        self.step = 0  # Current step of the current generation process.
        self.steps = 0  # Total steps of the current generation process.
        # We really need to use multiprocessing.Pipe() instead of simple list?
        self.parent_conn, self.child_conn = Pipe()

    def callback_preview(self, image: Image.Image, step: int, gen_time: datetime.timedelta | None = None):
        self.step = step
        gen_time = gen_time or datetime.timedelta()
        image_data = image.tobytes()
        self.progress_preview.emit(image_data, step, self.steps, image.width, image.height, gen_time)

    def run(self):
        """Run in thread.

        NOTE: What about [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html)?
        """
        from .generator.sdxl import generate, load_pipeline, PIPELINE_CACHE

        self._started = True
        print("loop start")

        while self._started:
            is_data_exist = self.child_conn.poll(timeout=self.poll_timeout)
            if is_data_exist:
                prompt: GenerationPrompt = self.child_conn.recv()

                self.steps = prompt.inference_steps

                prompt.callback = self.callback_preview
                
                try:
                    with Timer("Image generation") as t:
                        image: Image.Image = generate(prompt)
                    
                except OutOfMemoryError as e:
                    self.error.emit(str(e))
                    # Clear pipeline cache, because it can store corrupted pipeline.
                    PIPELINE_CACHE.clear()
                    
                else:
                    pipe = load_pipeline(prompt.model_path)
                    if not pipe._interrupt:
                        # Set result image. We use `self.steps`, because in this case step -- it is last step.
                        self.callback_preview(image, self.steps, t.delta)
    
                    self.done.emit()

    def stop(self):
        print("stopping")
        self._started = False
        # This pause needed, because we wait data in pipe (self.child_conn.poll).
        time.sleep(self.poll_timeout)
        self.finished.emit()
