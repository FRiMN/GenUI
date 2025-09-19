"""
ProcessManager class for managing child processes with Pipe communication.

We need using child process for working with models, because CUDA driver does not release memory until the process is terminated.

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
import threading
from typing import Callable, Any, Optional
from multiprocessing.connection import Connection


class ProcessManager:
    """Simple ProcessManager that runs a function in a child process with Pipe communication."""

    def __init__(self, target_function: Callable[[Connection], None]):
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
        self._lock = threading.Lock()

        # Start the process immediately
        self.start()

    def start(self) -> bool:
        """Start the child process."""
        with self._lock:
            if self._is_running:
                return True

            try:
                # Create pipe for communication
                self.parent_conn, self.child_conn = mp.Pipe()

                # Create and start process
                self.process = mp.Process(
                    target=self._run_target_function,
                    args=(self.child_conn,)
                )
                self.process.start()

                # Close child connection in parent process
                self.child_conn.close()
                self.child_conn = None

                self._is_running = True
                return True

            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to start process: {e}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the child process gracefully."""
        with self._lock:
            if not self._is_running:
                return

            try:
                # Close parent connection to signal shutdown
                if self.parent_conn:
                    self.parent_conn.close()

                # Wait for process to finish
                if self.process and self.process.is_alive():
                    self.process.join(timeout)

                    # Terminate if still alive
                    if self.process.is_alive():
                        self.process.terminate()
                        self.process.join(2.0)

                        # Kill as last resort
                        if self.process.is_alive():
                            self.process.kill()
                            self.process.join(1.0)

            except Exception as e:
                print(f"Error stopping process: {e}")
            finally:
                self._cleanup()

    def send(self, data: Any) -> bool:
        """Send data to the child process.

        Args:
            data: Data to send to child process

        Returns:
            True if sent successfully, False otherwise
        """
        print(f"Sending data to child process: {data}")
        if not self._is_running or not self.parent_conn:
            print("Process not running or connection closed")
            return False

        try:
            self.parent_conn.send(data)
            print("Data sent successfully")
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
                    raise TimeoutError("Receive timeout exceeded")
            else:
                return self.parent_conn.recv()
        except Exception as e:
            if isinstance(e, TimeoutError):
                raise
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

    def _run_target_function(self, child_conn: Connection) -> None:
        """Wrapper to run the target function in child process."""
        try:
            self.target_function(child_conn)
        except Exception as e:
            print(f"Error in child process: {e}")
        finally:
            try:
                child_conn.close()
            except Exception:
                pass

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_running = False

        if self.parent_conn:
            try:
                self.parent_conn.close()
            except Exception:
                pass
            self.parent_conn = None

        if self.child_conn:
            try:
                self.child_conn.close()
            except Exception:
                pass
            self.child_conn = None

        self.process = None

    def __del__(self):
        """Stop the process when object is destroyed."""
        try:
            self.stop()
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
