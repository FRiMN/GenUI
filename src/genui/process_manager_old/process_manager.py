from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Callable, Generic, TypeVar
from multiprocessing.connection import Connection
from multiprocessing import Process

from .task_processor import ITaskProcessor, TaskResult, run_processor_loop


T = TypeVar('T', bound=ITaskProcessor)


class ProcessManager(Generic[T]):
    """Manages a separate process running a task processor.

    This class follows the Single Responsibility Principle by focusing solely
    on process lifecycle management and communication.
    """

    def __init__(self, processor_factory: Callable[[], T], poll_timeout: float = 0.3):
        """Initialize the process manager.

        Args:
            processor_factory: Factory function that creates the processor instance
            poll_timeout: Timeout for polling the pipe connection
        """
        self._processor_factory = processor_factory
        self._poll_timeout = poll_timeout

        self._parent_conn: Connection | None = None
        self._child_conn: Connection | None = None
        self._process: Process | None = None
        self._is_running = False

        # Callbacks for different events
        self._on_result: Callable[[TaskResult], None] | None = None
        self._on_progress: Callable[[int, int, dict], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._on_finished: Callable[[], None] | None = None

    def start(self) -> bool:
        """Start the managed process.

        Returns:
            True if the process was started successfully, False otherwise
        """
        if self._is_running:
            return True

        # Ensure clean state before starting
        self._cleanup()

        try:
            # Create communication pipe
            self._parent_conn, self._child_conn = mp.Pipe()

            # Create and start the process
            self._process = Process(
                target=self._run_processor_process,
                args=(self._child_conn,)
            )
            self._process.start()

            # Close child connection in parent process
            self._child_conn.close()
            self._child_conn = None

            self._is_running = True
            return True

        except Exception as e:
            self._cleanup()
            if self._on_error:
                self._on_error(f"Failed to start process: {e}")
            return False

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the managed process gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self._is_running:
            return

        try:
            # Send shutdown signal
            if self._parent_conn:
                try:
                    self._parent_conn.send(None)
                    # Give a moment for the signal to be received
                    time.sleep(0.1)
                except Exception:
                    pass  # Ignore send errors

            # Wait for process to finish gracefully
            if self._process and self._process.is_alive():
                self._process.join(timeout)

                # Terminate if still alive after graceful timeout
                if self._process.is_alive():
                    print(f"Process {self._process.pid} not responding, terminating...")
                    self._process.terminate()
                    self._process.join(2.0)  # Wait longer for terminate

                    # Force kill as last resort
                    if self._process.is_alive():
                        print(f"Process {self._process.pid} still alive, force killing...")
                        self._process.kill()
                        self._process.join(1.0)

                        # Final check
                        if self._process.is_alive():
                            print(f"WARNING: Process {self._process.pid} could not be stopped!")

        except Exception as e:
            print(f"Error stopping process: {e}")
            if self._on_error:
                self._on_error(f"Error stopping process: {e}")
        finally:
            self._cleanup()
            if self._on_finished:
                self._on_finished()

    def send_task(self, task_data: Any) -> bool:
        """Send a task to the managed process.

        Args:
            task_data: Data to send to the processor

        Returns:
            True if the task was sent successfully, False otherwise
        """
        if not self._is_running or not self._parent_conn:
            return False

        try:
            self._parent_conn.send(task_data)
            return True
        except Exception as e:
            if self._on_error:
                self._on_error(f"Failed to send task: {e}")
            return False

    def check_messages(self) -> None:
        """Check for messages from the managed process and handle them."""
        if not self._is_running or not self._parent_conn:
            return

        try:
            while self._parent_conn.poll():
                message = self._parent_conn.recv()
                self._handle_message(message)
        except Exception as e:
            if self._on_error:
                self._on_error(f"Error checking messages: {e}")

    def is_running(self) -> bool:
        """Check if the managed process is running."""
        if not self._is_running:
            return False

        if self._process:
            return self._process.is_alive()
        return False

    def set_result_callback(self, callback: Callable[[TaskResult], None]) -> None:
        """Set callback for task results."""
        self._on_result = callback

    def set_progress_callback(self, callback: Callable[[int, int, dict], None]) -> None:
        """Set callback for progress updates."""
        self._on_progress = callback

    def set_error_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for errors."""
        self._on_error = callback

    def set_finished_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for when process finishes."""
        self._on_finished = callback

    def _run_processor_process(self, child_conn: Connection) -> None:
        """Entry point for the child process."""
        import signal
        import sys

        def signal_handler(signum, frame):
            print(f"Child process {mp.current_process().pid} received signal {signum}, exiting...")
            sys.exit(0)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Create processor instance in child process
            processor = self._processor_factory()

            # Run the processor loop
            run_processor_loop(processor, child_conn, self._poll_timeout)

        except Exception as e:
            print(f"Child process error: {e}")
            try:
                # Send error back to parent
                error_result = TaskResult(success=False, error=f"Process initialization failed: {e}")
                child_conn.send(('result', error_result))
            except Exception:
                pass  # Connection might be closed
        finally:
            try:
                child_conn.close()
            except Exception:
                pass
            print(f"Child process {mp.current_process().pid} exiting...")

    def _handle_message(self, message: tuple) -> None:
        """Handle a message received from the child process."""
        if not isinstance(message, tuple) or len(message) < 2:
            return

        message_type = message[0]

        if message_type == 'result' and self._on_result:
            result = message[1]
            self._on_result(result)

        elif message_type == 'progress' and self._on_progress and len(message) >= 4:
            progress, total, kwargs = message[1], message[2], message[3]
            self._on_progress(progress, total, kwargs)

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_running = False

        if self._parent_conn:
            try:
                self._parent_conn.close()
            except Exception:
                pass
            self._parent_conn = None

        if self._child_conn:
            try:
                self._child_conn.close()
            except Exception:
                pass
            self._child_conn = None

        self._process = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
