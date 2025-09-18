from __future__ import annotations

import datetime
from io import BytesIO
from typing import Any, TYPE_CHECKING

from PIL import Image
from torch import OutOfMemoryError

from .task_processor import BaseTaskProcessor, TaskResult, ProgressCallback
from ..common.trace import Timer

if TYPE_CHECKING:
    from ..generator.sdxl import GenerationPrompt


class ImageGenerationProcessor(BaseTaskProcessor):
    """Task processor for image generation using SDXL models.

    This processor handles image generation tasks in a separate process,
    following the Single Responsibility Principle by focusing solely on
    image generation logic.
    """

    def __init__(self):
        super().__init__()
        self._step = 0
        self._steps = 0

    def setup(self) -> None:
        """Initialize the image generation processor."""
        super().setup()
        print("Image generation processor initialized")

    def process_task(self, task_data: Any, progress_callback: ProgressCallback | None = None) -> TaskResult:
        """Process an image generation task.

        Args:
            task_data: GenerationPrompt object containing generation parameters
            progress_callback: Callback for progress updates

        Returns:
            TaskResult with the generated image or error information
        """
        try:
            from ..generator.sdxl import generate, load_pipeline, PIPELINE_CACHE
        except ImportError as e:
            return TaskResult(success=False, error=f"Failed to import generation modules: {e}")

        try:
            prompt: GenerationPrompt = task_data
            self._steps = prompt.inference_steps
            self._step = 0

            # Set up progress callback for the prompt
            if progress_callback:
                prompt.callback = lambda img, step, gen_time=None: self._handle_preview_callback(
                    img, step, gen_time, progress_callback
                )

            # Load input image if provided
            image: Image.Image | None = None
            if prompt.image:
                image = Image.open(BytesIO(prompt.image))

            # Generate image
            with Timer("Image generation") as timer:
                if not image:
                    image = generate(prompt)

            # Check if generation was interrupted
            pipe = load_pipeline(prompt.model_path)
            if pipe._interrupt:
                return TaskResult(success=False, error="Generation was interrupted")

            # Send final result
            if progress_callback and image:
                self._handle_preview_callback(image, self._steps, timer.delta, progress_callback)

            # Apply ADetailer if requested
            if prompt.use_adetailer and image:
                fixed_image = self._apply_adetailer(image, prompt, progress_callback)
                if fixed_image:
                    image = fixed_image
                    if progress_callback:
                        self._handle_preview_callback(image, self._steps, timer.delta, progress_callback)

            return TaskResult(success=True, data={
                'image': image,
                'generation_time': timer.delta,
                'steps': self._steps
            })

        except OutOfMemoryError as e:
            # Clear pipeline cache on OOM
            PIPELINE_CACHE.clear()
            return TaskResult(success=False, error=f"Out of memory: {str(e)}")

        except Exception as e:
            return TaskResult(success=False, error=str(e))

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        print("Image generation processor cleaned up")

    def _handle_preview_callback(
        self,
        image: Image.Image,
        step: int,
        gen_time: datetime.timedelta | None,
        progress_callback: ProgressCallback
    ) -> None:
        """Handle preview callback and forward to progress callback."""
        self._step = step
        gen_time = gen_time or datetime.timedelta()

        # Convert image to bytes for transmission
        image_data = image.tobytes()

        progress_callback(
            progress=step,
            total=self._steps,
            preview_data=image_data,
            width=image.width,
            height=image.height,
            generation_time=gen_time
        )

    def _apply_adetailer(
        self,
        image: Image.Image,
        prompt: 'GenerationPrompt',
        progress_callback: ProgressCallback | None
    ) -> Image.Image | None:
        """Apply ADetailer post-processing to the image."""
        from ..generator.sdxl import fix_by_adetailer

        def adetailer_progress_callback(step: int, steps: int):
            self._step = step if step > self._step else self._step + 1
            self._steps = steps
            if progress_callback:
                progress_callback(
                    progress=self._step,
                    total=self._steps,
                    adetailer_step=True
                )

        def adetailer_rect_callback(rect: tuple[int, int, int, int]):
            if progress_callback:
                progress_callback(
                    progress=self._step,
                    total=self._steps,
                    adetailer_rect=rect
                )

        # Reset step counter for ADetailer
        self._step = 0

        try:
            fixed_image = fix_by_adetailer(
                image,
                prompt.model_path,
                adetailer_rect_callback,
                adetailer_progress_callback
            )
            return fixed_image
        except Exception as e:
            print(f"ADetailer failed: {e}")
            return None


class ImageGenerationTask:
    """Data class for image generation tasks.

    This class encapsulates all the data needed for an image generation task,
    following the Data Transfer Object pattern.
    """

    def __init__(self, prompt: 'GenerationPrompt'):
        self.prompt = prompt
        self.timestamp = datetime.datetime.now()

    def __repr__(self) -> str:
        return f"ImageGenerationTask(prompt={self.prompt.prompt[:50]}...)"
