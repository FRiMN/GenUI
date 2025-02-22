from __future__ import annotations

import gc
from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import TYPE_CHECKING

from DeepCache import DeepCacheSDHelper

# from diffusers.utils.testing_utils import enable_full_determinism
from diffusers import StableDiffusionXLPipeline

if TYPE_CHECKING:
    from diffusers.configuration_utils import FrozenDict
    from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
    import torch
    from collections.abc import Callable

import PIL
from PIL import Image

from utils import Timer


""" 
Docs: 
    - <https://huggingface.co/docs/diffusers/stable_diffusion> 
    - <https://huggingface.co/docs/diffusers/using-diffusers/sdxl>
"""


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


class CachedStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    @cached_property
    def deep_cache(self):
        return DeepCacheSDHelper(pipe=self)

    def __call__(self, *args, **kwargs):
        self.deep_cache.set_params(cache_interval=3)
        self.deep_cache.enable()
        res = super().__call__(*args, **kwargs)
        self.deep_cache.disable()
        return res

def accelerate(pipe: StableDiffusionXLPipeline):
    print("start accelerate enabling")
    import torch

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    # FIXME: In fact, not a significant acceleration of generation. Do really need it?
    pipe.enable_xformers_memory_efficient_attention()

    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.unet.to(memory_format=torch.channels_last)

@lru_cache(maxsize=1)
def load_pipeline(model_path: str) -> CachedStableDiffusionXLPipeline:
    print("start load pipeline")
    import torch

    with Timer("Pipeline loading"):
        pipe = CachedStableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        # pipe.to("cuda")

    accelerate(pipe)

    return pipe


@dataclass(unsafe_hash=True)
class GenerationPrompt:
    """Dataclass for Stable Diffusion XL pipeline.

    Args:
        model_path: Path to the model file.
        scheduler_name: Name of the scheduler to use.
        prompt: Prompt to generate.
        neg_prompt: Negative prompt to generate.
        seed: Seed for random number generator.
        size: Size of the generated image.
        guidance_scale: Guidance scale for the generation.
        callback: Callback function to be called with the decoded image.

    Returns:
        GenerationPrompt object.
    """
    model_path: str
    scheduler_name: str
    prompt: str
    neg_prompt: str
    seed: int
    size: tuple[int, int]
    guidance_scale: int
    callback: Callable | None = None


@lru_cache(maxsize=1)
def get_scheduler_config(model_path: str) -> dict:
    """Get the scheduler configuration for the given model path.

    Cached for using only original scheduler configuration.

    Args:
        model_path: Path to the model file.

    Returns:
        Scheduler configuration dictionary.
    """

    pipeline = load_pipeline(model_path)
    conf = pipeline.scheduler.config
    d = {
        "beta_schedule": conf["beta_schedule"],
        "beta_start": conf["beta_start"],
        "beta_end": conf["beta_end"],
        "num_train_timesteps": conf["num_train_timesteps"],
        "steps_offset": conf["steps_offset"],
    }
    return d


@lru_cache(maxsize=3)
def generate(
    prompt: GenerationPrompt
) -> PIL.Image.Image:
    import torch

    # enable_full_determinism()

    generator = torch.manual_seed(prompt.seed)
    pipeline = load_pipeline(prompt.model_path)

    scheduler_config = {
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "beta_end": 0.014,
        "num_train_timesteps": 1100,
        "steps_offset": 1,
    }

    set_scheduler(
        prompt.model_path,
        prompt.scheduler_name,
        get_scheduler_config(prompt.model_path),
    )

    # We prepare latents for reproducible (bug in diffusers lib?).
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        1,
        num_channels_latents,
        prompt.size[1],
        prompt.size[0],
        torch.float16,
        pipeline._execution_device,
        generator,
        None,
    )

    data = dict(
        prompt=prompt.prompt,
        negative_prompt=prompt.neg_prompt,
        num_inference_steps=20,
        width=prompt.size[0],
        height=prompt.size[1],
        callback_on_step_end=callback_factory(prompt.callback),
        callback_on_step_end_tensor_inputs=["latents"],
        generator=generator,
        latents=latents,
    )
    if prompt.guidance_scale:
        data["guidance_scale"] = prompt.guidance_scale

    with torch.inference_mode():
        image = pipeline(**data).images[0]

    return image


def interrupt(model_path: str):
    pipeline = load_pipeline(model_path)
    pipeline._interrupt = True


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


def callback_factory(callback: callable) -> callable:
    """Factory function to create a callback function for decoding tensors.

    Args:
        callback: The callback function to be called with the decoded image.

    Returns:
        A callback function that decodes the tensors and calls the provided callback function.
    """
    def callback_wrap(
            pipe: CachedStableDiffusionXLPipeline,
            step: int,
            timestep: torch.Tensor,
            callback_kwargs: dict
    ) -> dict:
        latents = callback_kwargs["latents"]

        image = latents_to_rgb(latents[0])
        callback(image, step + 1)

        steps = pipe.num_timesteps
        limit = int(steps * 0.45)
        if step > limit:
            pipe.deep_cache.set_params()

        return callback_kwargs

    return callback_wrap


@lru_cache(maxsize=1)
def get_schedulers_map() -> dict:
    from diffusers.schedulers import KarrasDiffusionSchedulers
    from diffusers import schedulers as diffusers_schedulers
    from diffusers import SchedulerMixin

    result = {}

    karras_schedulers = [e.name for e in KarrasDiffusionSchedulers]
    schedulers: list[str] = dir(diffusers_schedulers)
    schedulers: list[SchedulerMixin] = [
        getattr(diffusers_schedulers, x) for x in schedulers
        if x.endswith("Scheduler")
           and x in karras_schedulers
    ]

    for s in schedulers:
        name: str = s.__name__ if hasattr(s, "__name__") else s.__class__.__name__
        name = name.removesuffix("Scheduler")
        result[name] = s

    return result


def set_scheduler(
    model_path: str,
    scheduler_name: str,
    scheduler_config: FrozenDict,
) -> None:
    pipeline: DiffusionPipeline = load_pipeline(model_path)
    schedulers_map = get_schedulers_map()
    scheduler_class = schedulers_map[scheduler_name]

    is_same_scheduler = isinstance(pipeline.scheduler, scheduler_class)
    # is_same_config = scheduler_config == pipeline.scheduler.config

    # print(f"{is_same_scheduler=}; {scheduler_class=}; {pipeline.scheduler=}")
    # print(f"{is_same_config=}; {scheduler_config=}; {pipeline.scheduler.config=}")

    if is_same_scheduler:
        return

    print(f"Set new scheduler {scheduler_class} with {scheduler_config=}")
    # scheduler_config["prediction_type"] = "v_prediction"
    pipeline.scheduler = scheduler_class.from_config(scheduler_config)
