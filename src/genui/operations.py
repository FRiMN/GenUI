from queue import Queue
from multiprocessing.connection import Connection
from time import sleep
import datetime

from PyQt6.QtCore import QObject, pyqtSignal
from PIL import Image

from .process_manager import ProcessManager
from .generator.sdxl import GenerationPrompt
from .common.trace import Timer


class BaseSignalHolder(QObject): 
    done = pyqtSignal()


class BaseOperation(object):
    signals: BaseSignalHolder = BaseSignalHolder()
    process_manager: ProcessManager
    model_path: str

    def __init__(self):
        self.process_manager = None
        self.model_path = None
        self.start_process()

    def start_process(self):
        if self.process_manager:
            del self.process_manager
        self.process_manager = ProcessManager(self.run)

    def is_new_process_need(self, model_path: str) -> bool:
        return model_path != self.model_path

    @classmethod
    def run(cls, connection: Connection):
        """Run in child process"""
        raise NotImplementedError

    def exec(self, message: GenerationPrompt) -> bool:
        if self.is_new_process_need(message.model_path):
            self.start_process()
            self.model_path = message.model_path

        return self.process_manager.send(message)


class OperationWorker(QObject):
    finished = pyqtSignal()  # Worker is finished and starts to close.
    # done = pyqtSignal()  # Worker is done with the generation task.
    error = pyqtSignal(str)  # Worker encountered an error.

    operation: BaseOperation
    queue: Queue    # Incoming buffer

    def __init__(self, operation: BaseOperation, parent=None):
        super().__init__(parent)

        self.queue = Queue()
        self.operation = operation

    def run(self):
        """Run in thread"""
        print("starting")

        while True:
            message = self.queue.get()
            print(f"processing {message}")
            i = 0
            while not self.operation.exec(message):
                if i > 10:
                    raise TimeoutError("Operation timed out")
                sleep(0.1)
            sleep(0.1)

        self.finished.emit()
        
    def stop(self):
        print("stopping")
        self.finished.emit()


class ImageGenerationSignalHolder(BaseSignalHolder):
    progress_preview = pyqtSignal(bytes, int, int, int, int, datetime.timedelta)


class ImageGenerationOperation(BaseOperation):
    signals: ImageGenerationSignalHolder = ImageGenerationSignalHolder()

    def run(self, connection: Connection):
        while True:
            if connection.closed:
                print("Connection closed")
                break
                
            is_data_exist = connection.poll()
            if is_data_exist:
                msg = connection.recv()
                self.generate_image(msg)

    def generate_image(self, prompt: GenerationPrompt):
        from .generator.sdxl import generate
        
        prompt.callback = self.progress_callback

        with Timer("Image generation") as timer:
            image = generate(prompt)

        # TODO: interrupt

        print(f"Image generated in {timer.delta}")
        self.signals.progress_preview.emit(
            image.tobytes(),
            prompt.inference_steps,
            prompt.inference_steps,
            image.width,
            image.height,
            timer.delta
        )
        self.signals.done.emit()
        
    def progress_callback(self, image: Image.Image, step, total_steps):
        print(f"Progress: {step}/{total_steps}")
        self.signals.progress_preview.emit(
            image.tobytes(),
            step,
            total_steps,
            image.width,
            image.height,
            datetime.timedelta()
        )
