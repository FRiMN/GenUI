from __future__ import annotations

import gc
from dataclasses import dataclass, field
from functools import lru_cache, cached_property
from typing import TYPE_CHECKING

from DeepCache import DeepCacheSDHelper
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType

# from diffusers.utils.testing_utils import enable_full_determinism
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin

from ..settings import settings
from ..common.trace import Timer
from ..utils import FIFODict

if TYPE_CHECKING:
    import torch
    from collections.abc import Callable


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


def pipeline_cache_callback(cache: FIFODict):
    """We need to clear the CUDA cache because of a memory leak issue. Even after deleting the pipeline, the model data remains allocated in memory."""
    empty_cache()


IMAGE_CACHE = FIFODict(maxsize=3)
PIPELINE_CACHE = FIFODict(maxsize=1, remove_callback=pipeline_cache_callback)


class CompelStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """
    Pipeline that uses Compel to condition the Stable Diffusion XL model.

    Syntax docs: <https://github.com/damian0815/compel/blob/main/doc/syntax.md>.
    """

    @cached_property
    def compel(self):
        return Compel(
            tokenizer=[self.tokenizer, self.tokenizer_2],
            text_encoder=[self.text_encoder, self.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
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


class LoraStableDiffusionXLPipeline(StableDiffusionXLLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__loras = set([])

    def load_lora_weights(self, *args, **kwargs) -> None:
        adapter_name = kwargs["adapter_name"]
        if adapter_name in self.__loras:
            return

        lora_filepath = args[0]
        print(f"Loading LoRA {adapter_name} ({lora_filepath})...")
        self.__loras.add(adapter_name)
        return super().load_lora_weights(*args, **kwargs)

    def delete_adapters(self, adapter_names: list[str] | str) -> None:
        an = adapter_names if isinstance(adapter_names, list) else [adapter_names]
        for a in an:
            print(f"Deleting LoRA {a}...")
        self.__loras.remove(*an)
        return super().delete_adapters(adapter_names)

    def get_all_adapters(self):
        return self.__loras.copy()


class GenUIStableDiffusionXLPipeline(
    CachedStableDiffusionXLPipeline,
    CompelStableDiffusionXLPipeline,
    LoraStableDiffusionXLPipeline,
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
    pipe.vae.to(memory_format=torch.channels_last)

    # pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
    # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)


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


@dataclass(unsafe_hash=True)
class LoRASettings:
    name: str
    filepath: str
    weight: float
    active: bool


def load_loras(pipe: GenUIStableDiffusionXLPipeline, loras: frozenset[LoRASettings]) -> None:
    for lora in loras:
        pipe.load_lora_weights(lora.filepath, adapter_name=lora.name)

    # Delete old LoRAs
    lora_names = [l.name for l in loras]
    for adapter in pipe.get_all_adapters():
        if adapter not in lora_names:
            pipe.delete_adapters(adapter)


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
        use_karras_sigmas: Whether to use Karras sigmas.
        use_vpred: Whether to use v-prediction.
        callback: Callback function to be called with the decoded image.
        loras: Set of used LoRAs.

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
    use_karras_sigmas: bool
    use_vpred: bool
    callback: Callable | None = None
    loras: frozenset[LoRASettings] = field(default_factory=frozenset)


@lru_cache(maxsize=1)
def get_scheduler_config(model_path: str) -> frozenset[tuple]:
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
    return frozenset(d.items())


def generate(
    prompt: GenerationPrompt
) -> Image.Image:
    import torch

    if prompt in IMAGE_CACHE:
        return IMAGE_CACHE[prompt]

    with torch.inference_mode():
    # with torch.no_grad():
        # enable_full_determinism()

        generator = torch.manual_seed(prompt.seed)
        pipeline = load_pipeline(prompt.model_path)
        load_loras(pipeline, prompt.loras)

        active_loras = [l for l in prompt.loras if l.active]
        if not active_loras:
            pipeline.disable_lora()
        else:
            pipeline.enable_lora()
            pipeline.set_adapters(
                [l.name for l in active_loras],
                [l.weight for l in active_loras],
            )

        # scheduler_config = {
        #     "beta_schedule": "scaled_linear",
        #     "beta_start": 0.00085,
        #     "beta_end": 0.014,
        #     "num_train_timesteps": 1100,
        #     "steps_offset": 1,
        # }

        pipeline.scheduler = get_scheduler(
            prompt.scheduler_name,
            get_scheduler_config(prompt.model_path),
            use_karras_sigmas=prompt.use_karras_sigmas,
            use_vpred=prompt.use_vpred,
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

        image = pipeline(**data).images[0]

    if not pipeline._interrupt:
        IMAGE_CACHE[prompt] = image

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
    """Converts latents to RGB image."""
    latents = latents / pipe.vae.config.scaling_factor
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
    schedulers_: list[str] = dir(diffusers_schedulers)
    schedulers: list[SchedulerMixin] = [
        getattr(diffusers_schedulers, x) for x in schedulers_
        if x.endswith("Scheduler")
           and x in karras_schedulers
    ]

    for s in schedulers:
        name: str = s.__name__ if hasattr(s, "__name__") else s.__class__.__name__
        name = name.removesuffix("Scheduler")
        result[name] = s

    return result


@lru_cache(maxsize=1)
def get_scheduler(
    scheduler_name: str,
    scheduler_config: frozenset[tuple],
    *,
    use_karras_sigmas: bool,
    use_vpred: bool,
):
    schedulers_map = get_schedulers_map()
    scheduler_class = schedulers_map[scheduler_name]

    # frozenset convert to dict
    scheduler_config = {k: v for k, v in scheduler_config}
    if use_vpred:
        scheduler_config["prediction_type"] = "v_prediction"

    print(f"Get new scheduler {scheduler_class} with {scheduler_config=} and {use_karras_sigmas=} and {use_vpred=}")
    return scheduler_class.from_config(scheduler_config, use_karras_sigmas=use_karras_sigmas)
