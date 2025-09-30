"""
ProcessManager class for managing child processes with Pipe communication.

We need using child process for working with models, because CUDA driver does not release memory until the process is terminated. Every time a new model is loaded, it causes memory leaks.

From <https://github.com/tensorflow/tensorflow/issues/1727>:
> currently the Allocator in the GPUDevice belongs to the ProcessState, which is essentially a global singleton. The first session using GPU initializes it, and frees itself when the process shuts down.
> Yes, certain NVIDIA drivers had problems with releasing the memory.

Another solution with Numba: <https://stackoverflow.com/a/60354785>.

Also links:
    - https://discuss.pytorch.org/t/how-totally-remove-data-and-free-gpu/79272
    - https://github.com/pytorch/pytorch/issues/37664
    - https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution/51731238#51731238
"""

import multiprocessing as mp
from typing import Callable, Any, Optional
from multiprocessing.connection import Connection
from contextlib import suppress

import torch


# See:
#   - <https://github.com/pytorch/pytorch/issues/40403>
#   - <https://stackoverflow.com/questions/72779926/gunicorn-cuda-cannot-re-initialize-cuda-in-forked-subprocess>
#   - <https://stackoverflow.com/questions/61939952/mp-set-start-methodspawn-triggered-an-error-saying-the-context-is-already-be>
with suppress(RuntimeError):
   # mp.set_start_method('spawn', force=True)
   torch.multiprocessing.set_start_method("spawn", force=True)
   print("spawned")


class ProcessManager:
    """Simple ProcessManager that runs a function in a child process with Pipe communication."""

    def __init__(self, target_function: Callable[[Connection, Connection], None]):
        """Initialize and start the child process.

        Args:
            target_function: Function to run in child process.
                           Must accept a Connection object as first parameter.
        """
        self.target_function = target_function
        self.parent_conn: Optional[Connection] = None
        self.child_conn: Optional[Connection] = None
        self.process: Optional[mp.Process] = None
        self._is_running = False

        # Start the process immediately
        self.start()

    def start(self) -> bool:
        """Start the child process."""
        if self._is_running:
            return True

        try:
            # Create pipe for communication
            self.parent_conn, self.child_conn = mp.Pipe()

            # Create and start process
            self.process = mp.Process(
                target=self._run_target_function,
                args=(self.child_conn, self.parent_conn)
            )
            self.process.start()

            # Close child connection in parent process
            # self.child_conn.close()
            # self.child_conn = None

            self._is_running = True
            return True

        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to start process: {e}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the child process gracefully."""
        print("Stopping process")
        if not self._is_running:
            return

        try:
            # Send shutdown signal to child process first
            if self.parent_conn and not self.parent_conn.closed:
                with suppress(Exception):
                    self.parent_conn.send(None)

            # Give process a moment to shutdown gracefully
            if self.process and self.process.is_alive():
                self.process.join(timeout=min(timeout/2, 2.0))

            # Close parent connection after signaling
            if self.parent_conn and not self.parent_conn.closed:
                with suppress(Exception):
                    self.parent_conn.close()

            # Wait for process to finish
            self.finish_process(timeout)

        except Exception as e:
            print(f"Error stopping process: {e}")
        finally:
            self._cleanup()

    def finish_process(self, timeout: float = 5.0):
        if self.process and self.process.is_alive():
            remaining_timeout = max(timeout, 1.0)
            self.process.join(remaining_timeout)

            # Terminate if still alive
            if self.process.is_alive():
                print("Terminating process")
                self.process.terminate()
                self.process.join(2.0)

                # Kill as last resort
                if self.process.is_alive():
                    print("Killing process")
                    self.process.kill()
                    self.process.join(1.0)

            self.process = None

    def send(self, data: Any) -> bool:
        """Send data to the child process.

        Args:
            data: Data to send to child process

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._is_running or not self.parent_conn:
            print("Process not running or connection closed")
            return False

        try:
            self.parent_conn.send(data)
            return True
        except Exception:
            print("Failed to send data")
            return False

    def recv(self, timeout: Optional[float] = None) -> Any:
        """Receive data from the child process.

        Args:
            timeout: Timeout in seconds (None for blocking)

        Returns:
            Data from child process

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If process not running or connection closed
        """
        if not self._is_running or not self.parent_conn:
            raise RuntimeError("Process not running or connection closed")

        try:
            if timeout is not None:
                if self.parent_conn.poll(timeout):
                    return self.parent_conn.recv()
                else:
                    return None
            else:
                return self.parent_conn.recv()
        except Exception as e:
            raise RuntimeError(f"Failed to receive data: {e}")

    def poll(self, timeout: float = 0) -> bool:
        """Check if data is available to receive.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if data is available, False otherwise
        """
        if not self._is_running or not self.parent_conn:
            return False

        try:
            return self.parent_conn.poll(timeout)
        except Exception:
            return False

    def is_alive(self) -> bool:
        """Check if the child process is still running."""
        if not self._is_running or not self.process:
            return False
        return self.process.is_alive()

    def _run_target_function(self, child_conn: Connection, parent_conn: Connection):
        """Wrapper to run the target function in child process."""
        try:
            # Set up signal handler for graceful shutdown
            import signal
            import sys

            def signal_handler(signum, frame):
                print(f"Child process received signal {signum}, shutting down...")
                child_conn.close()
                parent_conn.close()
                sys.exit(0)

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            self.target_function(child_conn, parent_conn)
        except Exception as e:
            print(f"Error in child process: {e}")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_running = False

        # Close connections
        if self.parent_conn and not self.parent_conn.closed:
            with suppress(Exception):
                self.parent_conn.close()
        # self.parent_conn = None

        if self.child_conn and not self.child_conn.closed:
            with suppress(Exception):
                self.child_conn.close()
        # self.child_conn = None

        # Clean up process reference. Only join a child process!
        if self.process:
            try:
                # Force cleanup of any remaining process resources
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1.0)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join(timeout=0.5)
            except AttributeError:
                print("Warning: AttributeError during process cleanup")
                pass
            except Exception as e:
                print(f"Warning: Error during process cleanup: {e}")
                raise e
        print("Process cleaned up")
        self.process = None

    def __del__(self):
        """Stop the process when object is destroyed."""
        try:
            self.stop(timeout=1.0)
        except Exception:
            print("Error stopping process")

    # def __enter__(self):
    #     """Context manager entry."""
    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     """Context manager exit."""
    #     self.stop()

    def __bool__(self):
        """Return True if the process is running."""
        return self._is_running
