from __future__ import annotations
import gc
from typing import TYPE_CHECKING

from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

from ...utils import FIFODict
from ...common.trace import Timer
from .pipelines import GenUIStableDiffusionXLPipeline

if TYPE_CHECKING:
    import torch
    from collections.abc import Callable


def pipeline_cache_callback(cache: FIFODict):
    """We need to clear the CUDA cache because of a memory leak issue. Even after deleting the pipeline, the model data remains allocated in memory."""
    empty_cache()


IMAGE_CACHE = FIFODict(maxsize=5)
PIPELINE_CACHE = FIFODict(maxsize=1, remove_callback=pipeline_cache_callback)


def empty_cache():
    print("empty cache")
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    for x in ("npu", "xpu", "mps"):
        try:
            d = getattr(torch, x)
            d.empty_cache()
        except (AttributeError, RuntimeError):  # noqa: PERF203
            pass


def accelerate(pipe: GenUIStableDiffusionXLPipeline):
    import torch

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    # FIXME: In fact, not a significant acceleration of generation. Do really need it?
    # WARN: Change generated image.
    # pipe.enable_xformers_memory_efficient_attention()

    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    # pipe.unet.to(memory_format=torch.channels_last)
    # pipe.vae.to(memory_format=torch.channels_last)

    # pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
    # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
    # pipe.to("cuda")


def load_pipeline(model_path: str) -> GenUIStableDiffusionXLPipeline:
    if model_path in PIPELINE_CACHE:
        return PIPELINE_CACHE[model_path]

    print("start load pipeline")
    import torch

    with Timer("Pipeline loading"):
        pipe = GenUIStableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            # device_map="auto",
            # max_memory={0: "512MB", 1: "8GB"},
        )
        # pipe.to("cuda")

    accelerate(pipe)

    PIPELINE_CACHE[model_path] = pipe
    return pipe


def latents_to_rgb(latents: torch.Tensor) -> Image.Image:
    """Converts latents to RGB image.

    From <https://huggingface.co/docs/diffusers/using-diffusers/callback>.
    """
    import torch

    weights = (
        (60, -60, 25, -70),
        (60, -5, 15, -50),
        (60, 10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = (torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor)
                  + biases_tensor.unsqueeze(-1).unsqueeze(-1))
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)


def latents_to_rgb_vae(latents: torch.Tensor, pipe: GenUIStableDiffusionXLPipeline) -> Image.Image:
    """Converts latents to RGB image."""
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]

    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def callback_factory(callback: Callable) -> Callable:
    """Factory function to create a callback function for decoding tensors.

    Args:
        callback: The callback function to be called with the decoded image.

    Returns:
        A callback function that decodes the tensors and calls the provided callback function.
    """
    def callback_wrap(
            pipe: GenUIStableDiffusionXLPipeline,
            step: int,
            timestep: torch.Tensor,
            callback_kwargs: dict
    ) -> dict:
        if pipe.is_step_cached(step): return callback_kwargs
        if pipe._interrupt: return callback_kwargs

        latents = callback_kwargs["latents"]
        total_steps = pipe._num_timesteps

        image = latents_to_rgb(latents[0])
        callback(image, step + 1, total_steps)

        return callback_kwargs

    return callback_wrap


def callback_adetailer_factory(callback: Callable) -> Callable:
    def callback_wrap(
            pipe: StableDiffusionXLInpaintPipeline,
            step: int,
            timestep: torch.Tensor,
            callback_kwargs: dict
    ) -> dict:
        steps = pipe._num_timesteps
        callback(step + 1, steps)
        return callback_kwargs

    return callback_wrap
