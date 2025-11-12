from queue import Empty, Queue
from multiprocessing.connection import Connection
from time import sleep
import datetime
from functools import partial
from contextlib import suppress
from collections import OrderedDict
from typing import Dict, Optional
import traceback

from PyQt6.QtCore import QObject, pyqtSignal, pyqtBoundSignal
from PIL import Image
from diffusers import SchedulerMixin
from diffusers.configuration_utils import FrozenDict

from .process_manager import ProcessManager
from .generator.sdxl import GenerationPrompt, get_scheduler, ModelSchedulerConfig
from .common.trace import Timer


def get_gpu_memory_info() -> Optional[Dict]:
    """Get GPU memory information from torch.cuda"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        # Get memory info
        device = torch.cuda.current_device()
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        used_mem = total_mem - free_mem

        # Get memory summary (as string)
        memory_summary = torch.cuda.memory_summary(device=device)

        return {
            'device': device,
            'free_mem': free_mem,
            'total_mem': total_mem,
            'used_mem': used_mem,
            'memory_summary': memory_summary,
            'allocated': torch.cuda.memory_allocated(device),
            'reserved': torch.cuda.memory_reserved(device),
            'max_allocated': torch.cuda.max_memory_allocated(device),
            'max_reserved': torch.cuda.max_memory_reserved(device)
        }
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return None


def signal_send(conn: Connection, signal_name: str, *args):
    fixed_args = []
    for arg in args:
        if isinstance(arg, FrozenDict):
            modified_dict = OrderedDict(
                (
                    k, tuple(v)
                    if isinstance(v, list)
                    else v
                ) for k, v in arg.items()
            )
            arg = frozenset(modified_dict.items())
        fixed_args.append(arg)
    fixed_args = tuple(fixed_args)
    msg = {"signal": signal_name, "args": fixed_args}
    conn.send(msg)


class BaseSignalHolder(QObject):
    done = pyqtSignal()

    def emit(self, name: str, *args, **kwargs):
        """Emit signal by attr name"""
        signal = getattr(self, name, None)
        if type(signal) is pyqtBoundSignal:
            signal.emit(*args, **kwargs)


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
            self.process_manager.stop()
            del self.process_manager
        self.process_manager = ProcessManager(self.run)

    def is_new_process_need(self, model_path: str) -> bool:
        """See `process_manager.py` for more info"""
        return (
            not model_path
            or model_path != self.model_path
            or not self.process_manager
        )

    def run(self, connection: Connection, back_connection: Connection):
        """Run in child process"""
        raise NotImplementedError

    def exec(self, message: GenerationPrompt) -> bool:
        if self.is_new_process_need(message.model_path):
            self.start_process()
            self.model_path = message.model_path

        return self.process_manager.send(message)

    def cleanup(self):
        """Clean up the process manager and resources."""
        if self.process_manager:
            try:
                self.process_manager.stop()
            except Exception as e:
                print(f"Error stopping process manager: {e}")
            finally:
                self.process_manager = None

    def __del__(self):
        """Destructor to ensure cleanup on object destruction."""
        try:
            self.cleanup()
        except Exception:
            pass


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
        self._running = False

    def run(self):
        """Run in thread"""
        print("starting")
        self._running = True

        try:
            while self._running:
                with suppress(Empty):
                    message = self.queue.get(block=False)
                    i = 0
                    while not self.operation.exec(message):
                        if i > 10:
                            raise TimeoutError("Operation timed out")
                        if not self._running:  # Check if we should stop
                            break
                        sleep(0.1)
                        i += 1

                if not self._running:  # Check if we should stop
                    break

                message = self.operation.process_manager.recv(timeout=0.1)
                if isinstance(message, dict):
                    if "signal" in message.keys():
                        self.operation.signals.emit(message["signal"], *message["args"])
                sleep(0.1)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Worker error: {e}\n{tb}")
            self.error.emit(str(e))
        finally:
            self._cleanup()
            self.finished.emit()

    def stop(self):
        print("stopping worker")
        self._running = False

    def _cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self.operation, 'process_manager') and self.operation.process_manager:
                print("Cleaning up process manager...")
                self.operation.process_manager.stop()
        except Exception as e:
            print(f"Error during cleanup: {e}")


class ImageGenerationSignalHolder(BaseSignalHolder):
    progress_preview = pyqtSignal(bytes, int, int, int, int, datetime.timedelta)
    scheduler_config = pyqtSignal(frozenset)
    gpu_memory_info = pyqtSignal(dict)


class ImageGenerationOperation(BaseOperation):
    signals = ImageGenerationSignalHolder()

    def run(self, connection: Connection, back_connection: Connection):
        while True:
            if connection.closed:
                print("Connection closed")
                break

            is_data_exist = connection.poll()

            if is_data_exist:
                msg = connection.recv()

                match msg:
                    case obj if obj is None:
                        print("No data")
                        break
                    case obj if isinstance(obj, ModelSchedulerConfig):
                        self.get_scheduler_config(msg, connection)
                    case obj if isinstance(obj, GenerationPrompt):
                        self.generate_image(msg, connection)
                    case _:
                        print(f"Unknown message type: {msg}")
            sleep(0.1)

    def generate_image(self, prompt: GenerationPrompt, back_connection: Connection):
        from .generator.sdxl import generate

        prompt.callback = partial(self.progress_callback, back_connection)

        with Timer("Image generation") as timer:
            image = generate(prompt)

        # TODO: interrupt

        print(f"Image generated in {timer.delta}")

        gpu_info = get_gpu_memory_info()
        signal_send(
            back_connection, "progress_preview",
            image.tobytes(),
            prompt.inference_steps,
            prompt.inference_steps,
            image.width, image.height,
            timer.delta
        )

        if gpu_info:
            signal_send(back_connection, "gpu_memory_info", gpu_info)

        signal_send(back_connection, "done")

    @staticmethod
    def progress_callback(back_connection: Connection, image: Image.Image, step, total_steps):
        signal_send(
            back_connection, "progress_preview",
            image.tobytes(),
            step,
            total_steps,
            image.width, image.height,
            datetime.timedelta()
        )

        gpu_info = get_gpu_memory_info()
        if gpu_info:
            signal_send(back_connection, "gpu_memory_info", gpu_info)

    def get_scheduler_config(self, command: ModelSchedulerConfig, back_connection: Connection):
        scheduler: SchedulerMixin = get_scheduler(command)
        signal_send(back_connection, "scheduler_config", scheduler.config)
