from __future__ import annotations


from abc import ABC, abstractmethod, ABCMeta
from typing import Any, Callable, Optional
from multiprocessing.connection import Connection

from PyQt6.QtCore import QObject, pyqtSignal

from .process_manager import ProcessManager




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
        self._worker_ready = False

    @abstractmethod
    def create_worker_function(self) -> Callable[[Connection], None]:
        """Create the worker function that will run in the child process.

        Returns:
            A function that accepts a Connection object and processes tasks
        """
        pass

    @staticmethod
    def _create_send_message(conn: Connection):
        """Create a reusable send_message function for worker processes."""
        def send_message(msg_type: str, **kwargs):
            try:
                message = {'type': msg_type, **kwargs}
                conn.send(message)
            except Exception:
                pass
        return send_message

    @abstractmethod
    def _get_worker_imports(self) -> dict:
        """Get the import statements needed for the worker function.

        Returns:
            Dict with import statements to execute in worker process
        """
        pass

    @abstractmethod
    def _get_process_task_function(self) -> Callable:
        """Get the function to process tasks in the worker process.

        This should return a static function that can be called without self reference.

        Returns:
            A function with signature: func(task_data, send_message, **imports)
        """
        pass

    def _create_base_worker(self) -> Callable[[Connection], None]:
        """Create a base worker function with common structure."""
        # Capture the methods we need as local variables to avoid self references in worker process
        import_specs = self._get_worker_imports()
        process_task_func = self._get_process_task_function()
        class_name = self.__class__.__name__

        if process_task_func is None:
            raise RuntimeError("process_task_func cannot be None")

        def base_worker(conn: Connection) -> None:
            send_message = BaseOperation._create_send_message(conn)

            try:
                # Get required imports
                imports = {}

                for name, import_path in import_specs.items():
                    try:
                        if '.' in import_path:
                            module_path, attr = import_path.rsplit('.', 1)
                            module = __import__(module_path, fromlist=[attr])
                            imports[name] = getattr(module, attr)
                        else:
                            imports[name] = __import__(import_path)
                    except ImportError as e:
                        send_message('error', message=f"Failed to import {import_path}: {e}")
                        return

                print(f"{class_name} worker started")
                send_message('progress', progress=0, total=1, data={'status': 'worker_started'})

                # Main processing loop
                while True:
                    try:
                        if conn.poll(0.1):
                            task_data = conn.recv()
                            if task_data is None:
                                break
                            process_task_func(task_data, send_message, **imports)
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

        return base_worker

    @staticmethod
    def _send_result_message(send_message, success: bool, data=None, error=None):
        """Send a standardized result message."""
        if success and data:
            if hasattr(data, 'tobytes') and hasattr(data, 'width') and hasattr(data, 'height'):
                # Handle PIL Image objects
                result_data = {
                    'image': data,
                    'image_data': data.tobytes(),
                    'width': data.width,
                    'height': data.height
                }
            else:
                result_data = data
            send_message('result', success=True, data=result_data)
        else:
            send_message('result', success=False, error=error or "Processing failed")

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
            self._worker_ready = False  # Reset ready flag
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
            self._worker_ready = False  # Reset ready flag
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Error stopping operation: {e}")
        finally:
            self._process_manager = None

    def send_task(self, task_data: Any, timeout: float = 30.0) -> bool:
        """Send a task to the operation process.

        Waits for the process to be ready before sending the task.

        Args:
            task_data: Data to send for processing
            timeout: Maximum time to wait for process readiness (seconds)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._is_running or not self._process_manager:
            print("Operation not running or process manager not available")
            return False

        # Wait for worker to be ready
        if not self._wait_for_worker_ready(timeout):
            self.error.emit("Worker process did not become ready within timeout")
            return False

        try:
            return self._process_manager.send(task_data)
        except Exception as e:
            self.error.emit(f"Failed to send task: {e}")
            return False

    def _wait_for_worker_ready(self, timeout: float) -> bool:
        """Wait for worker process to signal it's ready to accept tasks.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if worker is ready, False if timeout or error
        """
        import time

        start_time = time.time()

        # If we already received worker_started signal, we're ready
        if hasattr(self, '_worker_ready') and self._worker_ready:
            return True

        while time.time() - start_time < timeout:
            try:
                # Check for messages from worker
                if not self._process_manager:
                    return False

                while self._process_manager.poll(0.1):
                    message = self._process_manager.recv(0.1)
                    self._handle_message(message)

                    # Check if we received worker_started signal
                    if (isinstance(message, dict) and
                        message.get('type') == 'progress' and
                        isinstance(message.get('data'), dict) and
                        message.get('data', {}).get('status') == 'worker_started'):
                        self._worker_ready = True
                        return True

                # Small delay to avoid busy waiting
                time.sleep(0.01)

            except Exception as e:
                print(f"Error while waiting for worker ready: {e}")
                return False

        return False

    def check_messages(self) -> None:
        """Check for messages from the process and emit appropriate signals."""
        if not self._is_running or not self._process_manager:
            print("Check messages. Operation not running or process manager not available")
            return

        try:
            while self._process_manager.poll():
                message = self._process_manager.recv()
                self._handle_message(message)
        except Exception as e:
            self.error.emit(f"Error checking messages: {e}")

    def _handle_message(self, message: Any) -> None:
        """Handle a message received from the child process."""
        if not isinstance(message, dict):
            print("Invalid message received")
            return

        msg_type = message.get('type')
        print(f"Received message of type {msg_type}")

        if msg_type == 'progress':
            progress = message.get('progress', 0)
            total = message.get('total', 0)
            extra_data = message.get('data', {})

            # Check if this is a worker_started signal
            if (isinstance(extra_data, dict) and
                extra_data.get('status') == 'worker_started'):
                self._worker_ready = True
                print("Worker process is ready to accept tasks")

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

    def _get_worker_imports(self) -> dict:
        """Get the import statements needed for image generation worker."""
        return {
            'generate': '.generator.sdxl.generate',
            'load_pipeline': '.generator.sdxl.load_pipeline',
            'PIPELINE_CACHE': '.generator.sdxl.PIPELINE_CACHE',
            'Image': 'PIL.Image',
            'BytesIO': 'io.BytesIO'
        }

    @staticmethod
    def _create_progress_callback(send_message):
        """Create progress callback for image generation."""
        def progress_callback(image, step: int, gen_time=None):
            try:
                image_data = image.tobytes() if image else b''
                width = image.width if image else 0
                height = image.height if image else 0
                send_message('progress', progress=step, total=0,
                           data={'preview_data': image_data, 'width': width, 'height': height, 'generation_time': gen_time})
            except Exception as e:
                send_message('error', message=f"Progress callback error: {e}")
        return progress_callback

    @staticmethod
    def _process_image_generation_task(task_data, send_message, **imports):
        """Process image generation task."""
        generate = imports['generate']
        load_pipeline = imports['load_pipeline']
        PIPELINE_CACHE = imports['PIPELINE_CACHE']
        Image = imports['Image']
        BytesIO = imports['BytesIO']

        if not hasattr(task_data, 'inference_steps'):
            send_message('error', message="Invalid task data: missing inference_steps")
            return

        prompt = task_data
        prompt.callback = ImageGenerationOperation._create_progress_callback(send_message)
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
                BaseOperation._send_result_message(send_message, False, error="Generation was interrupted")
                return

            if result_image:
                result_data = {
                    'image': result_image,
                    'image_data': result_image.tobytes(),
                    'width': result_image.width,
                    'height': result_image.height,
                    'steps_completed': prompt.inference_steps
                }
                BaseOperation._send_result_message(send_message, True, data=result_data)
            else:
                BaseOperation._send_result_message(send_message, False, error="No image generated")

        except Exception as e:
            if "out of memory" in str(e).lower():
                PIPELINE_CACHE.clear()
                BaseOperation._send_result_message(send_message, False, error=f"Out of memory: {e}")
            else:
                BaseOperation._send_result_message(send_message, False, error=f"Generation failed: {e}")

    def _get_process_task_function(self) -> Callable:
        """Get the function to process image generation tasks."""
        return self._process_image_generation_task

    def create_worker_function(self) -> Callable[[Connection], None]:
        """Create the image generation worker function."""
        return self._create_base_worker()

    def _handle_message(self, message: Any) -> None:
        """Handle messages specific to image generation."""
        super()._handle_message(message)

        if not isinstance(message, dict):
            print(f"Received message of type {type(message)}")
            return

        msg_type = message.get('type')
        print(f"Received message of type {msg_type}")

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

    def _get_worker_imports(self) -> dict:
        """Get the import statements needed for ADetailer worker."""
        return {
            'fix_by_adetailer': 'generator.sdxl.fix_by_adetailer',
            'Image': 'PIL.Image',
            'BytesIO': 'io.BytesIO'
        }

    @staticmethod
    def _create_callbacks(send_message):
        """Create callbacks for ADetailer processing."""
        def face_detection_callback(x: int, y: int, width: int, height: int):
            send_message('progress', progress=0, total=1,
                       data={'face_detected': True, 'face_rect': (x, y, width, height)})

        def inpainting_progress_callback(step: int, total_steps: int):
            send_message('progress', progress=step, total=total_steps,
                       data={'inpainting_step': True})

        return face_detection_callback, inpainting_progress_callback

    @staticmethod
    def _load_image_from_task(task_data, send_message, Image, BytesIO):
        """Load image from task data."""
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

    @staticmethod
    def _process_adetailer_task(task_data, send_message, **imports):
        """Process ADetailer task."""
        fix_by_adetailer = imports['fix_by_adetailer']
        Image = imports['Image']
        BytesIO = imports['BytesIO']

        image, model_path = ADetailerOperation._load_image_from_task(task_data, send_message, Image, BytesIO)
        if image is None:
            return

        face_callback, progress_callback = ADetailerOperation._create_callbacks(send_message)
        send_message('progress', progress=0, total=100, data={'status': 'adetailer_started'})

        try:
            fixed_image = fix_by_adetailer(image, model_path, face_callback, progress_callback)

            if fixed_image:
                BaseOperation._send_result_message(send_message, True, data=fixed_image)
            else:
                BaseOperation._send_result_message(send_message, False, error="ADetailer processing failed - no result image")

        except Exception as e:
            BaseOperation._send_result_message(send_message, False, error=f"ADetailer processing failed: {e}")

    def _get_process_task_function(self) -> Callable:
        """Get the function to process ADetailer tasks."""
        return self._process_adetailer_task

    def create_worker_function(self) -> Callable[[Connection], None]:
        """Create the ADetailer worker function."""
        return self._create_base_worker()

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
