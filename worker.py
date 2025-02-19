from multiprocessing import Pipe
from multiprocessing.connection import Connection

import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    from generator.sdxl import GenerationPrompt


class Worker(QObject):
    finished = pyqtSignal()
    done = pyqtSignal()
    progress_preview = pyqtSignal(bytes, int, int, int, int)

    poll_timeout = 0.3

    parent_conn: Connection
    child_conn: Connection

    def __init__(self):
        super().__init__()
        self._started = False
        self.step = 0
        self.parent_conn, self.child_conn = Pipe()

    def callback_preview(self, image: Image.Image, step: int):
        self.step = step
        image_data = image.tobytes()
        self.progress_preview.emit(image_data, step, 20, image.width, image.height)

    def run(self):
        """ Run in thread

        NOTE: What about [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html)?
        """
        from generator.sdxl import generate

        self._started = True
        print("loop start")

        while self._started:
            is_data_exist = self.child_conn.poll(timeout=self.poll_timeout)
            if is_data_exist:
                prompt: GenerationPrompt = self.child_conn.recv()

                prompt.callback = self.callback_preview
                image: Image.Image = generate(prompt)

                self.callback_preview(image, self.step)
                self.save_image(image)
                self.done.emit()

    def stop(self):
        print("stopping")
        self._started = False
        # This pause needed, because we wait data in pipe (self.child_conn.poll)
        time.sleep(self.poll_timeout)
        self.finished.emit()

    def generate_filepath(self) -> Path:
        t = time.time()
        return Path(f"/media/frimn/archive31/ai/stable_diffusion/ComfyUI/output/genui/{t}.jpg")

    def save_image(self, image: Image.Image):
        p = self.generate_filepath()
        image.save(p)
