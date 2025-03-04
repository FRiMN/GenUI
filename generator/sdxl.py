from __future__ import annotations

import gc
from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import TYPE_CHECKING

from DeepCache import DeepCacheSDHelper
from compel import Compel, ReturnedEmbeddingsType

# from diffusers.utils.testing_utils import enable_full_determinism
from diffusers import StableDiffusionXLPipeline

from settings import settings

if TYPE_CHECKING:
    from diffusers.configuration_utils import FrozenDict
    from diffusers import DiffusionPipeline
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


class CompelStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """
    Pipeline that uses Compel to condition the Stable Diffusion XL model.

    Syntax docs: <https://github.com/damian0815/compel/blob/main/doc/syntax.md>.
    """

    @cached_property
    def compel(self):
        return Compel(
            tokenizer=[self.tokenizer, self.tokenizer_2] ,
            text_encoder=[self.text_encoder, self.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

    def __call__(self, *args, **kwargs):
        """
        Diffusers lib do not allow to mix and match `prompt_2` and `prompt_embeds`.
        See: <https://github.com/huggingface/diffusers/issues/5718>.
        """
        prompt = kwargs.pop("prompt")
        conditioning, pooled = self.compel(prompt)

        neg_prompt = kwargs.pop("negative_prompt")
        neg_conditioning, neg_pooled = self.compel(neg_prompt)

        return super().__call__(
            *args,
            **kwargs,
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_conditioning,
            negative_pooled_prompt_embeds=neg_pooled,

        )


class CachedStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    deep_cache_enabled: bool = False

    @cached_property
    def deep_cache(self):
        return DeepCacheSDHelper(pipe=self)

    def stop_caching(self):
        self.deep_cache.set_params()
        self.deep_cache.disable()

    def start_caching(self):
        s = settings.deep_cache
        self.deep_cache.set_params(
            cache_interval=s.cache_interval,
            cache_branch_id=s.cache_branch_id,
            skip_mode=s.skip_mode,
        )
        self.deep_cache.enable()

    def is_step_cached(self, step: int) -> bool:
        if not self.deep_cache_enabled: return False

        deep_cache_interval = self.deep_cache.params['cache_interval']
        return step % deep_cache_interval != 0

    def __call__(self, *args, **kwargs):
        if not self.deep_cache_enabled:
            return super().__call__(*args, **kwargs)

        self.start_caching()
        res = super().__call__(*args, **kwargs)
        self.stop_caching()

        return res


class GenUIStableDiffusionXLPipeline(
    CachedStableDiffusionXLPipeline,
    CompelStableDiffusionXLPipeline
):
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
    pipe.unet.to(memory_format=torch.channels_last)

@lru_cache(maxsize=1)
def load_pipeline(model_path: str) -> GenUIStableDiffusionXLPipeline:
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
        inference_steps: Number of inference steps.
        deepcache_enabled: Whether to enable deepcache.
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
    inference_steps: int
    deepcache_enabled: bool
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

    # scheduler_config = {
    #     "beta_schedule": "scaled_linear",
    #     "beta_start": 0.00085,
    #     "beta_end": 0.014,
    #     "num_train_timesteps": 1100,
    #     "steps_offset": 1,
    # }

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
        num_inference_steps=prompt.inference_steps,
        width=prompt.size[0],
        height=prompt.size[1],
        callback_on_step_end=callback_factory(prompt.callback),
        callback_on_step_end_tensor_inputs=["latents"],
        generator=generator,
        latents=latents,
    )
    if prompt.guidance_scale:
        data["guidance_scale"] = prompt.guidance_scale

    pipeline.deep_cache_enabled = prompt.deepcache_enabled

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


def latents_to_rgb_vae(latents: torch.Tensor, pipe: GenUIStableDiffusionXLPipeline) -> Image.Image:
    """Converts latents to RGB image.
    """
    latents = (latents / pipe.vae.config.scaling_factor)
    image = pipe.vae.decode(latents, return_dict=False)[0]

    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def callback_factory(callback: callable) -> callable:
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

        image = latents_to_rgb(latents[0])
        callback(image, step + 1)

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
