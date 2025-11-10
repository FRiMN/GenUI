"""
Docs:
    - <https://huggingface.co/docs/diffusers/stable_diffusion>
    - <https://huggingface.co/docs/diffusers/using-diffusers/sdxl>
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from PIL import Image

from ...settings import settings, ADetailerSettings
from .loras import LoRASettings, load_loras
from .schedulers import get_scheduler, ModelSchedulerConfig
from .utils import IMAGE_CACHE, load_pipeline, callback_factory, callback_adetailer_factory

if TYPE_CHECKING:
    from collections.abc import Callable


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
        use_adetailer: Whether to use adetailer.
        callback: Callback function to be called with the decoded image.
        loras: Set of used LoRAs.
        neg_condition_divider: Divider level for negative condition (0-4).

    Returns:
        GenerationPrompt object.
    """

    model_path: str
    scheduler_name: str
    prompt: str
    neg_prompt: str
    seed: int
    size: tuple[int, int]
    guidance_scale: float
    inference_steps: int
    deepcache_enabled: bool
    use_karras_sigmas: bool
    use_vpred: bool
    use_adetailer: bool = False
    callback: Callable | None = None
    loras: frozenset[LoRASettings] = field(default_factory=frozenset)
    neg_condition_divider: int = 0
    image: bytes | None = None


def generate(
    prompt: GenerationPrompt
) -> Image.Image:
    import torch

    cache_key = replace(prompt, use_adetailer=False, image=None)

    if cache_key in IMAGE_CACHE:
        return IMAGE_CACHE[cache_key]

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
            ModelSchedulerConfig(
                name=prompt.scheduler_name,
                model_path=prompt.model_path,
                use_karras_sigmas=prompt.use_karras_sigmas,
                use_vpred=prompt.use_vpred,
            )
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

        # if prompt.neg_condition_divider:
        #     data["negative_original_size"] = (
        #         int(prompt.size[0] / prompt.neg_condition_divider),
        #         int(prompt.size[1] / prompt.neg_condition_divider)
        #     )
        #     data["negative_target_size"] = prompt.size

        pipeline.deep_cache_enabled = prompt.deepcache_enabled

        image = pipeline(**data).images[0]

    if not pipeline._interrupt:
        IMAGE_CACHE[cache_key] = image

    return image


def interrupt(model_path: str):
    pipeline = load_pipeline(model_path)
    pipeline._interrupt = True


def fix_by_adetailer(
    image: Image.Image,
    model_path: str,
    callback: Callable,
    callback_step: Callable,
) -> Image.Image | None:
    import torch
    from adetailer_sdxl.asdff.base import AdPipelineBase, ADOutput

    s: ADetailerSettings = settings.adetailer

    pipe = load_pipeline(model_path)
    ad_components = pipe.components
    ad_pipe = AdPipelineBase(**ad_components)

    common = {
        "prompt": s.prompt,
        "n_prompt" : s.n_prompt,
        "num_inference_steps": s.inference_steps,
        # "target_size" : image.size
        "target_size" : s.target_size
    }
    inpaint_only = {'strength': s.inpaint_strength}

    with torch.inference_mode():
        result: ADOutput = ad_pipe(
            common=common,
            inpaint_only=inpaint_only,
            images=[image],
            mask_dilation=s.mask_dilation,
            mask_blur=s.mask_blur,
            mask_padding=s.mask_padding,
            model_path=str(s.yolov_model_path),
            rect_callback=callback,
            callback=callback_adetailer_factory(callback_step),
        )
    return result.images[0] if result.images else None
