from __future__ import annotations

import datetime
from abc import ABC, abstractmethod, ABCMeta
from typing import Any, Callable, Optional, TYPE_CHECKING
from multiprocessing.connection import Connection

from PyQt6.QtCore import QObject, pyqtSignal

from .process_manager import ProcessManager

if TYPE_CHECKING:
    from .generator.sdxl import GenerationPrompt


class QObjectMeta(type(QObject), ABCMeta):
    """Metaclass that combines QObject's metaclass with ABCMeta."""
    pass


class BaseOperation(QObject, ABC, metaclass=QObjectMeta):
    """Abstract base class for operations that run in separate processes using ProcessManager.

    This class provides a common interface for operations like image generation and ADetailer
    processing, handling process management, communication, and Qt signal emission.
    """

    # Qt Signals
    started = pyqtSignal()
    finished = pyqtSignal()
    success = pyqtSignal(object)  # Emitted with result data
    error = pyqtSignal(str)  # Emitted with error message
    progress = pyqtSignal(int, int, dict)  # progress, total, extra_data

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._process_manager: Optional[ProcessManager] = None
        self._is_running = False

    @abstractmethod
    def create_worker_function(self) -> Callable[[Connection], None]:
        """Create the worker function that will run in the child process.

        Returns:
            A function that accepts a Connection object and processes tasks
        """
        pass

    def start(self) -> bool:
        """Start the operation process.

        Returns:
            True if started successfully, False otherwise
        """
        if self._is_running:
            return True

        try:
            worker_func = self.create_worker_function()
            self._process_manager = ProcessManager(worker_func)
            self._is_running = True
            self.started.emit()
            return True
        except Exception as e:
            self.error.emit(f"Failed to start operation: {e}")
            return False

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the operation process.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self._is_running:
            return

        try:
            if self._process_manager:
                self._process_manager.stop(timeout)
            self._is_running = False
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Error stopping operation: {e}")
        finally:
            self._process_manager = None

    def send_task(self, task_data: Any) -> bool:
        """Send a task to the operation process.

        Args:
            task_data: Data to send for processing

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._is_running or not self._process_manager:
            return False

        try:
            return self._process_manager.send(task_data)
        except Exception as e:
            self.error.emit(f"Failed to send task: {e}")
            return False

    def check_messages(self) -> None:
        """Check for messages from the process and emit appropriate signals."""
        if not self._is_running or not self._process_manager:
            return

        try:
            while self._process_manager.poll():
                message = self._process_manager.recv()
                self._handle_message(message)
        except Exception as e:
            self.error.emit(f"Error checking messages: {e}")

    def _handle_message(self, message: Any) -> None:
        """Handle a message received from the child process.

        Args:
            message: Message received from child process
        """
        if not isinstance(message, dict):
            return

        msg_type = message.get('type')

        if msg_type == 'progress':
            progress = message.get('progress', 0)
            total = message.get('total', 0)
            extra_data = message.get('data', {})
            self.progress.emit(progress, total, extra_data)

        elif msg_type == 'result':
            if message.get('success', False):
                self.success.emit(message.get('data'))
            else:
                self.error.emit(message.get('error', 'Unknown error'))

        elif msg_type == 'error':
            self.error.emit(message.get('message', 'Unknown error'))

    def is_alive(self) -> bool:
        """Check if the operation process is alive."""
        if not self._process_manager:
            return False
        return self._process_manager.is_alive()

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop()
        except Exception:
            pass


class ImageGenerationOperation(BaseOperation):
    """Operation for image generation using SDXL models."""

    # Additional signals specific to image generation
    preview_image = pyqtSignal(bytes, int, int, int, int, object)  # image_data, step, steps, width, height, gen_time
    generation_complete = pyqtSignal(object)  # final image

    def create_worker_function(self) -> Callable[[Connection], None]:
        """Create the image generation worker function."""

        def image_generation_worker(conn: Connection) -> None:
            """Worker function for image generation that runs in child process."""
            def send_message(msg_type: str, **kwargs):
                try:
                    message = {'type': msg_type, **kwargs}
                    conn.send(message)
                except Exception:
                    pass

            def create_progress_callback():
                def progress_callback(image: Image.Image, step: int, gen_time: datetime.timedelta = None):
                    try:
                        image_data = image.tobytes() if image else b''
                        width = image.width if image else 0
                        height = image.height if image else 0
                        send_message('progress', progress=step, total=0,
                                   data={'preview_data': image_data, 'width': width, 'height': height, 'generation_time': gen_time})
                    except Exception as e:
                        send_message('error', message=f"Progress callback error: {e}")
                return progress_callback

            def process_generation_task(task_data, generate, load_pipeline, PIPELINE_CACHE, Image, BytesIO):
                if not hasattr(task_data, 'inference_steps'):
                    send_message('error', message="Invalid task data: missing inference_steps")
                    return

                prompt: GenerationPrompt = task_data
                prompt.callback = create_progress_callback()
                send_message('progress', progress=0, total=prompt.inference_steps, data={'status': 'generation_started'})

                # Load input image if provided (for future use)
                if hasattr(prompt, 'image') and prompt.image:
                    try:
                        Image.open(BytesIO(prompt.image))
                    except Exception as e:
                        send_message('error', message=f"Failed to load input image: {e}")
                        return

                try:
                    result_image = generate(prompt)
                    pipeline = load_pipeline(prompt.model_path)

                    if hasattr(pipeline, '_interrupt') and pipeline._interrupt:
                        send_message('result', success=False, error="Generation was interrupted")
                        return

                    if result_image:
                        final_image_data = result_image.tobytes()
                        send_message('result', success=True, data={
                            'image': result_image, 'image_data': final_image_data,
                            'width': result_image.width, 'height': result_image.height,
                            'steps_completed': prompt.inference_steps
                        })
                    else:
                        send_message('result', success=False, error="No image generated")

                except Exception as e:
                    if "out of memory" in str(e).lower():
                        PIPELINE_CACHE.clear()
                        send_message('result', success=False, error=f"Out of memory: {e}")
                    else:
                        send_message('result', success=False, error=f"Generation failed: {e}")

            try:
                from generator.sdxl import generate, load_pipeline, PIPELINE_CACHE
                from PIL import Image
                from io import BytesIO

                print("Image generation worker started")
                send_message('progress', progress=0, total=1, data={'status': 'worker_started'})

                while True:
                    try:
                        if conn.poll(0.1):
                            task_data = conn.recv()
                            if task_data is None:
                                break
                            process_generation_task(task_data, generate, load_pipeline, PIPELINE_CACHE, Image, BytesIO)
                    except Exception as e:
                        send_message('error', message=f"Worker loop error: {e}")

            except Exception as e:
                try:
                    conn.send({'type': 'error', 'message': f"Worker initialization failed: {e}"})
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        return image_generation_worker

    def _handle_message(self, message: Any) -> None:
        """Handle messages specific to image generation."""
        super()._handle_message(message)

        if not isinstance(message, dict):
            return

        msg_type = message.get('type')

        if msg_type == 'progress':
            data = message.get('data', {})
            if 'preview_data' in data:
                # Emit preview image signal
                self.preview_image.emit(
                    data['preview_data'],
                    message.get('progress', 0),
                    message.get('total', 0),
                    data.get('width', 0),
                    data.get('height', 0),
                    data.get('generation_time')
                )
        elif msg_type == 'result' and message.get('success'):
            # Emit generation complete signal
            result_data = message.get('data', {})
            self.generation_complete.emit(result_data)


class ADetailerOperation(BaseOperation):
    """Operation for ADetailer face detection and inpainting."""

    # Additional signals specific to ADetailer
    face_detected = pyqtSignal(int, int, int, int)  # x, y, width, height
    inpainting_progress = pyqtSignal(int, int)  # current_face, total_faces
    adetailer_complete = pyqtSignal(object)  # processed image

    def create_worker_function(self) -> Callable[[Connection], None]:
        """Create the ADetailer worker function."""

        def adetailer_worker(conn: Connection) -> None:
            """Worker function for ADetailer processing that runs in child process."""
            def send_message(msg_type: str, **kwargs):
                try:
                    message = {'type': msg_type, **kwargs}
                    conn.send(message)
                except Exception:
                    pass

            def create_callbacks():
                def face_detection_callback(x: int, y: int, width: int, height: int):
                    send_message('progress', progress=0, total=1,
                               data={'face_detected': True, 'face_rect': (x, y, width, height)})

                def inpainting_progress_callback(step: int, total_steps: int):
                    send_message('progress', progress=step, total=total_steps,
                               data={'inpainting_step': True})

                return face_detection_callback, inpainting_progress_callback

            def load_image_from_task(task_data, Image, BytesIO):
                if not isinstance(task_data, dict):
                    send_message('error', message="Invalid task data format")
                    return None, None

                image_data = task_data.get('image_data')
                model_path = task_data.get('model_path', '')

                if not image_data:
                    send_message('error', message="No image data provided")
                    return None, None

                try:
                    if isinstance(image_data, bytes):
                        image = Image.open(BytesIO(image_data))
                    else:
                        image = image_data
                    return image, model_path
                except Exception as e:
                    send_message('error', message=f"Failed to load image: {e}")
                    return None, None

            def process_adetailer_task(task_data, fix_by_adetailer, Image, BytesIO):
                image, model_path = load_image_from_task(task_data, Image, BytesIO)
                if image is None:
                    return

                face_callback, progress_callback = create_callbacks()
                send_message('progress', progress=0, total=100, data={'status': 'adetailer_started'})

                try:
                    fixed_image = fix_by_adetailer(image, model_path, face_callback, progress_callback)

                    if fixed_image:
                        result_image_data = fixed_image.tobytes()
                        send_message('result', success=True, data={
                            'image': fixed_image, 'image_data': result_image_data,
                            'width': fixed_image.width, 'height': fixed_image.height
                        })
                    else:
                        send_message('result', success=False, error="ADetailer processing failed - no result image")

                except Exception as e:
                    send_message('result', success=False, error=f"ADetailer processing failed: {e}")

            try:
                from generator.sdxl import fix_by_adetailer
                from PIL import Image
                from io import BytesIO

                print("ADetailer worker started")
                send_message('progress', progress=0, total=1, data={'status': 'worker_started'})

                while True:
                    try:
                        if conn.poll(0.1):
                            task_data = conn.recv()
                            if task_data is None:
                                break
                            process_adetailer_task(task_data, fix_by_adetailer, Image, BytesIO)
                    except Exception as e:
                        send_message('error', message=f"Worker loop error: {e}")

            except Exception as e:
                try:
                    conn.send({'type': 'error', 'message': f"ADetailer worker initialization failed: {e}"})
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        return adetailer_worker

    def _handle_message(self, message: Any) -> None:
        """Handle messages specific to ADetailer."""
        super()._handle_message(message)

        if not isinstance(message, dict):
            return

        msg_type = message.get('type')

        if msg_type == 'progress':
            data = message.get('data', {})

            if data.get('face_detected'):
                # Emit face detected signal
                face_rect = data.get('face_rect', (0, 0, 0, 0))
                self.face_detected.emit(*face_rect)

            elif data.get('inpainting_step'):
                # Emit inpainting progress
                progress = message.get('progress', 0)
                total = message.get('total', 0)
                self.inpainting_progress.emit(progress, total)

        elif msg_type == 'result' and message.get('success'):
            # Emit ADetailer complete signal
            result_data = message.get('data', {})
            self.adetailer_complete.emit(result_data)


# Convenience functions for creating operations
def create_image_generation_operation(parent: Optional[QObject] = None) -> ImageGenerationOperation:
    """Create and return a new ImageGenerationOperation instance."""
    return ImageGenerationOperation(parent)


def create_adetailer_operation(parent: Optional[QObject] = None) -> ADetailerOperation:
    """Create and return a new ADetailerOperation instance."""
    return ADetailerOperation(parent)
