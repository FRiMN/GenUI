from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol
from multiprocessing.connection import Connection
import multiprocessing as mp


class TaskResult:
    """Represents the result of a task execution."""

    def __init__(self, success: bool = True, data: Any = None, error: str | None = None):
        self.success = success
        self.data = data
        self.error = error


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""

    def __call__(self, progress: int, total: int, **kwargs) -> None:
        """Called to report progress updates."""
        ...


class ITaskProcessor(ABC):
    """Interface for task processors that run in separate processes."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize the processor. Called once when the process starts."""
        pass

    @abstractmethod
    def process_task(self, task_data: Any, progress_callback: ProgressCallback | None = None) -> TaskResult:
        """Process a single task and return the result."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources. Called once when the process is stopping."""
        pass

    @abstractmethod
    def should_continue(self) -> bool:
        """Return True if the processor should continue running."""
        pass


class BaseTaskProcessor(ITaskProcessor):
    """Base implementation of task processor with common functionality."""

    def __init__(self):
        self._running = False
        self._setup_done = False

    def setup(self) -> None:
        """Default setup implementation."""
        self._running = True
        self._setup_done = True

    def cleanup(self) -> None:
        """Default cleanup implementation."""
        self._running = False

    def should_continue(self) -> bool:
        """Default implementation returns the running state."""
        return self._running

    def stop(self) -> None:
        """Stop the processor."""
        self._running = False


def _create_progress_callback(conn: Connection):
    """Create progress callback that sends updates through the connection."""
    def progress_callback(progress: int, total: int, **kwargs):
        try:
            conn.send(('progress', progress, total, kwargs))
        except Exception:
            pass  # Ignore progress callback errors
    return progress_callback


def _handle_task(processor: ITaskProcessor, conn: Connection, task_data) -> bool:
    """Handle a single task."""
    if task_data is None:  # Shutdown signal
        return False

    progress_callback = _create_progress_callback(conn)
    result = processor.process_task(task_data, progress_callback)
    try:
        conn.send(('result', result))
    except (BrokenPipeError, EOFError):
        print("Parent connection closed, stopping task processing")
        return False
    return True


def _send_error_result(conn: Connection, error_msg: str) -> None:
    """Send error result, ignoring connection errors."""
    try:
        error_result = TaskResult(success=False, error=error_msg)
        conn.send(('result', error_result))
    except Exception:
        pass


def run_processor_loop(processor: ITaskProcessor, conn: Connection, poll_timeout: float = 0.3) -> None:
    """Main loop for running a task processor in a separate process."""
    import signal
    import sys

    shutdown_requested = False

    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        print(f"Processor received shutdown signal {signum}")
        shutdown_requested = True
        processor.stop()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        processor.setup()
        print(f"Processor loop started in process {mp.current_process().pid}")

        while processor.should_continue() and not shutdown_requested:
            try:
                if conn.poll(timeout=poll_timeout):
                    try:
                        task_data = conn.recv()
                        if task_data is None:  # Shutdown signal
                            print("Received shutdown signal via connection")
                            break
                        if not _handle_task(processor, conn, task_data):
                            break
                    except EOFError:
                        print("Connection closed by parent, shutting down")
                        break
                    except Exception as e:
                        if not shutdown_requested:
                            _send_error_result(conn, str(e))

            except KeyboardInterrupt:
                print("Processor interrupted")
                break
            except Exception as e:
                print(f"Processor loop error: {e}")
                break

    except Exception as e:
        print(f"Processor setup error: {e}")
        _send_error_result(conn, f"Processor error: {e}")
    finally:
        try:
            print("Processor cleanup starting...")
            processor.cleanup()
            print("Processor cleanup completed")
        except Exception as e:
            print(f"Processor cleanup error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print(f"Processor loop exiting from process {mp.current_process().pid}")
