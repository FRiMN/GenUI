import time
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal

from generator.sdxl import Generator


class Worker(QObject):
    finished = pyqtSignal()
    progress_preview = pyqtSignal(bytes, int, int, int, int)

    def __init__(
            self,
            generator: Generator,
            prompt: str,
            neg_prompt: str,
            seed: int,
            size: tuple[int, int],
            clip_skip: int,
            scheduler: str,
            model_path: str,
    ):
        super().__init__()
        self.generator = generator
        self.step = 0

        self.data = dict(
            model_path=model_path,
            scheduler_name=scheduler,
            prompt=prompt,
            neg_prompt=neg_prompt,
            seed=seed,
            size=size,
            clip_skip=clip_skip,
            # callback=self.callback_preview,
        )

    def callback_preview(self, image: Image.Image, step: int):
        # h = hpy()
        self.step = step
        image_data = image.tobytes()
        self.progress_preview.emit(image_data, step, 20, image.width, image.height)
        # print(h.heap())

    def run(self):
        # image: Image.Image = generate(
        #     self.prompt,
        #     self.neg_prompt,
        #     seed=self.seed,
        #     size=self.size,
        #     clip_skip=self.clip_skip,
        #     callback=self.callback_preview,
        # )

        self.generator.send(("generate", self.data))

        command, image_bytes, width, height = self.generator.parent_conn.recv()
        print("res", command, width, height)

        image = Image.frombytes(
            "RGB",
            (width, height),
            image_bytes,
        )

        self.callback_preview(image, self.step)
        # self.save_image(image)

        self.stop()

    def stop(self):
        print("stopping")
        self.finished.emit()

    def generate_filepath(self) -> Path:
        t = time.time()
        return Path(f"/media/frimn/archive31/ai/stable_diffusion/ComfyUI/output/genui/{t}.jpg")

    def save_image(self, image: Image.Image):
        p = self.generate_filepath()
        image.save(p)
