from __future__ import annotations

"""
Example usage of the new ProcessManager architecture with Worker integration.

This example demonstrates how to use the refactored Worker class that now
leverages ProcessManager for running image generation tasks in separate processes.
"""

import sys
import datetime
from typing import TYPE_CHECKING
from PyQt6.QtCore import QCoreApplication, QTimer
from PyQt6.QtWidgets import QApplication

# Mock GenerationPrompt for demonstration
class MockGenerationPrompt:
    def __init__(self, prompt: str = "test prompt"):
        self.prompt = prompt
        self.inference_steps = 20
        self.model_path = "test_model"
        self.image = None
        self.use_adetailer = False
        self.callback = None

if TYPE_CHECKING:
    from ..generator.sdxl import GenerationPrompt
else:
    GenerationPrompt = MockGenerationPrompt


def example_worker_usage():
    """Example of using the refactored Worker class."""
    from ..worker import Worker

    app = QApplication(sys.argv)

    # Create worker instance
    worker = Worker()

    # Connect signals to handlers
    worker.finished.connect(lambda: print("Worker finished"))
    worker.done.connect(lambda: print("Task completed"))
    worker.error.connect(lambda err: print(f"Error: {err}"))
    worker.progress_preview.connect(
        lambda data, step, steps, w, h, time: print(f"Preview: step {step}/{steps}")
    )
    worker.progress_adetailer.connect(
        lambda step, steps: print(f"ADetailer: step {step}/{steps}")
    )
    worker.show_adetailer_rect.connect(
        lambda x, y, w, h: print(f"ADetailer rect: ({x}, {y}, {w}, {h})")
    )

    # Start the worker
    worker.run()

    # Send a test task
    prompt = GenerationPrompt("A beautiful sunset over mountains")
    if worker.send_task(prompt):
        print("Task sent successfully")
    else:
        print("Failed to send task")

    # Stop after 5 seconds for demo purposes
    QTimer.singleShot(5000, worker.stop)
    QTimer.singleShot(6000, app.quit)

    return app.exec()


def example_direct_process_manager():
    """Example of using ProcessManager directly."""
    from .process_manager import ProcessManager
    from .image_processor import ImageGenerationProcessor

    def on_result(result):
        if result.success:
            print(f"Task completed: {result.data}")
        else:
            print(f"Task failed: {result.error}")

    def on_progress(progress, total, kwargs):
        print(f"Progress: {progress}/{total}")
        if 'preview_data' in kwargs:
            print("  Preview image received")
        if 'adetailer_step' in kwargs:
            print("  ADetailer step")

    def on_error(error_msg):
        print(f"Process error: {error_msg}")

    def on_finished():
        print("Process finished")

    # Create and configure process manager
    manager = ProcessManager(
        processor_factory=lambda: ImageGenerationProcessor(),
        poll_timeout=0.1
    )

    manager.set_result_callback(on_result)
    manager.set_progress_callback(on_progress)
    manager.set_error_callback(on_error)
    manager.set_finished_callback(on_finished)

    # Start the process
    if manager.start():
        print("Process started successfully")

        # Send a test task
        prompt = GenerationPrompt("A majestic dragon in the clouds")
        if manager.send_task(prompt):
            print("Task sent")

        # Check messages for a few seconds
        import time
        for _ in range(50):  # 5 seconds with 0.1s intervals
            manager.check_messages()
            time.sleep(0.1)

        # Stop the process
        manager.stop()
    else:
        print("Failed to start process")


def example_context_manager():
    """Example using ProcessManager as a context manager."""
    from .process_manager import ProcessManager
    from .image_processor import ImageGenerationProcessor

    print("Using ProcessManager as context manager:")

    with ProcessManager(lambda: ImageGenerationProcessor()) as manager:
        manager.set_result_callback(lambda r: print(f"Result: {r.success}"))
        manager.set_progress_callback(lambda p, t, k: print(f"Progress: {p}/{t}"))

        prompt = GenerationPrompt("A serene lake at dawn")
        manager.send_task(prompt)

        # Process messages
        import time
        for _ in range(30):
            manager.check_messages()
            time.sleep(0.1)

    print("Context manager exited, process should be cleaned up")


if __name__ == "__main__":
    print("=== Worker Usage Example ===")
    # example_worker_usage()  # Uncomment to test with Qt

    print("\n=== Direct ProcessManager Example ===")
    example_direct_process_manager()

    print("\n=== Context Manager Example ===")
    example_context_manager()

    print("\nAll examples completed!")
