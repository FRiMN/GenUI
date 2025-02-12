from multiprocessing import Pipe
from multiprocessing.connection import Connection

import time
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal

from generator.sdxl import generate


class Worker(QObject):
    finished = pyqtSignal()
    progress_preview = pyqtSignal(bytes, int, int, int, int)

    def __init__(self):
        super().__init__()
        self._started = False
        self.step = 0

        self.parent_conn: Connection
        self.child_conn: Connection
        self.parent_conn, self.child_conn = Pipe()

    def callback_preview(self, image: Image.Image, step: int):
        self.step = step
        image_data = image.tobytes()
        self.progress_preview.emit(image_data, step, 20, image.width, image.height)

    def run(self):
        """ Run in thread """
        self._started = True
        print("loop start")

        while self._started:
            is_data_exist = self.child_conn.poll(timeout=1)
            if is_data_exist:
                data = self.child_conn.recv()

                image: Image.Image = generate(
                    **data,
                    callback=self.callback_preview,
                )

                self.callback_preview(image, 20)
                self.save_image(image)

    def stop(self):
        print("stopping")
        self._started = False
        # This pause needed, because we wait data in pipe (self.child_conn.poll)
        time.sleep(1)
        self.finished.emit()

    def generate_filepath(self) -> Path:
        t = time.time()
        return Path(f"/media/frimn/archive31/ai/stable_diffusion/ComfyUI/output/genui/{t}.jpg")

    def save_image(self, image: Image.Image):
        p = self.generate_filepath()
        image.save(p)
