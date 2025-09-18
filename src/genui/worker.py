from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from .process_manager import ProcessManager, ImageGenerationProcessor, TaskResult

if TYPE_CHECKING:
    from .generator.sdxl import GenerationPrompt


class Worker(QObject):
    """Worker runs a generation task in a separate process using ProcessManager."""

    finished = pyqtSignal()  # Worker is finished and starts to close.
    done = pyqtSignal()  # Worker is done with the generation task.
    error = pyqtSignal(str)  # Worker encountered an error.
    progress_preview = pyqtSignal(bytes, int, int, int, int, datetime.timedelta)
    progress_adetailer = pyqtSignal(int, int)
    show_adetailer_rect = pyqtSignal(int, int, int, int)

    # Internal signals for message checking control
    _check_messages_signal = pyqtSignal()

    poll_timeout = 0.3  # Poll timeout for checking data availability

    def __init__(self):
        super().__init__()
        self._started = False
        self.step = 0  # Current step of the current generation process.
        self.steps = 0  # Total steps of the current generation process.

        # Process manager will be created when needed
        self._process_manager = None

        # Connect internal signals
        self._check_messages_signal.connect(self._check_messages)

    def _create_process_manager(self):
        """Create and configure a new process manager."""
        # Ensure complete cleanup of any existing process
        self._force_cleanup_existing_process()

        self._process_manager = ProcessManager(
            processor_factory=lambda: ImageGenerationProcessor(),
            poll_timeout=self.poll_timeout
        )

        # Set up callbacks
        self._process_manager.set_result_callback(self._on_result)
        self._process_manager.set_progress_callback(self._on_progress)
        self._process_manager.set_error_callback(self._on_error)
        self._process_manager.set_finished_callback(self._on_finished)

    def _force_cleanup_existing_process(self):
        """Force cleanup of any existing process."""
        if self._process_manager is not None:
            try:
                # First try graceful stop
                self._process_manager.stop(timeout=1.0)

                # If process still exists, force kill it
                if (self._process_manager._process and
                    self._process_manager._process.is_alive()):

                    import os
                    import signal
                    import time

                    pid = self._process_manager._process.pid
                    if pid is not None:
                        print(f"Force killing existing process {pid}")

                        try:
                            os.kill(pid, signal.SIGKILL)
                            time.sleep(0.1)
                        except (OSError, ProcessLookupError):
                            pass  # Process already dead

            except Exception as e:
                print(f"Warning: Error during force cleanup: {e}")
            finally:
                self._process_manager = None

    def _on_result(self, result: TaskResult):
        """Handle task result from the process."""
        if result.success:
            # Task completed successfully
            self.done.emit()
        else:
            # Task failed
            self.error.emit(result.error or "Unknown error")

    def _on_progress(self, progress: int, total: int, kwargs: dict):
        """Handle progress updates from the process."""
        self.step = progress
        self.steps = total

        # Handle different types of progress updates
        if 'preview_data' in kwargs:
            # Preview image update
            gen_time = kwargs.get('generation_time', datetime.timedelta())
            width = kwargs.get('width', 0)
            height = kwargs.get('height', 0)
            self.progress_preview.emit(kwargs['preview_data'], progress, total, width, height, gen_time)

        if kwargs.get('adetailer_step', False):
            # ADetailer progress update
            self.progress_adetailer.emit(progress, total)

        if 'adetailer_rect' in kwargs:
            # ADetailer rectangle update
            rect = kwargs['adetailer_rect']
            self.show_adetailer_rect.emit(*rect)

    def _on_error(self, error_msg: str):
        """Handle errors from the process manager."""
        self.error.emit(error_msg)

    def _on_finished(self):
        """Handle process finished event."""
        self.finished.emit()

    def _check_messages(self):
        """Check for messages from the managed process and schedule next check."""
        if self._started and self._process_manager:
            self._process_manager.check_messages()
            # Schedule next check using QTimer.singleShot for thread safety
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(int(self.poll_timeout * 1000), self._check_messages_signal.emit)

    def run(self):
        """Start the worker process.

        This method now uses ProcessManager instead of running in a thread.
        """
        if self._started:
            return

        self._started = True
        print("Starting worker process")

        # Ensure complete cleanup before starting
        self._force_cleanup_existing_process()

        # Create new process manager
        self._create_process_manager()

        # Start the process manager
        if self._process_manager and self._process_manager.start():
            # Start message checking using signal-based approach
            self._check_messages_signal.emit()
        else:
            self._started = False
            self.error.emit("Failed to start worker process")

    def send_task(self, prompt: 'GenerationPrompt') -> bool:
        """Send a generation task to the worker process.

        Args:
            prompt: The generation prompt to process

        Returns:
            True if the task was sent successfully, False otherwise
        """
        if not self._started or not self._process_manager or not self._process_manager.is_running():
            return False

        return self._process_manager.send_task(prompt)

    @property
    def parent_conn(self):
        """Legacy property for backward compatibility."""
        return self._process_manager._parent_conn if self._process_manager else None

    @property
    def child_conn(self):
        """Legacy property for backward compatibility."""
        return None  # Child connection is managed internally by ProcessManager

    def stop(self):
        """Stop the worker process."""
        if not self._started:
            return

        print("Stopping worker process")
        self._started = False

        # Stop the process manager if it exists
        if self._process_manager is not None:
            try:
                self._process_manager.stop(timeout=2.0)  # Shorter timeout

                # Additional cleanup to ensure process is really dead
                import time
                import os
                import signal

                if self._process_manager._process and self._process_manager._process.is_alive():
                    pid = self._process_manager._process.pid
                    if pid is not None:
                        print(f"Force killing stubborn process {pid}")

                        try:
                            os.kill(pid, signal.SIGKILL)
                            time.sleep(0.2)

                            # Check if it's really dead
                            try:
                                os.kill(pid, 0)  # Check if process exists
                                print(f"WARNING: Process {pid} still exists after SIGKILL!")
                            except OSError:
                                print(f"Process {pid} successfully terminated")
                        except OSError as e:
                            print(f"Error killing process: {e}")

            except Exception as e:
                print(f"Warning: Error during process cleanup: {e}")
            finally:
                self._process_manager = None
