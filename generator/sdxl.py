from __future__ import annotations
from typing import TYPE_CHECKING
import gc
from functools import lru_cache

if TYPE_CHECKING:
    from diffusers.configuration_utils import FrozenDict

from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

import PIL
from PIL import Image
import torch

from utils import Timer


""" Docs: <https://huggingface.co/docs/diffusers/stable_diffusion> """


def empty_cache():
    print("empty cache")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    for x in ("npu", "xpu", "mps"):
        try:
            d = getattr(torch, x)
            d.empty_cache()
        except (AttributeError, RuntimeError):
            pass


@lru_cache(maxsize=1)
def load_pipeline(model_path: str) -> StableDiffusionXLPipeline:
    print("start load pipeline")
    empty_cache()
    with Timer("Pipeline loading") as t:
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        pipe.to("cuda")
        # torch.device('cuda')
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        pipe.unet.to(memory_format=torch.channels_last)

        # helper = DeepCacheSDHelper(pipe=pipe)
        # helper.set_params(
        #     cache_interval=3,
        #     cache_branch_id=0,
        # )
        # helper.enable()

    print(f"{pipe.scheduler}")
    # print(pipe.scheduler.compatibles)

    return pipe

def generate(
        model_path: str,
        scheduler_name: str,
        prompt: str,
        neg_prompt: str,
        seed: int,
        size: tuple[int, int],
        clip_skip: int,
        callback: callable,
) -> PIL.Image.Image:
    # max seed is 2147483647 ?
    torch.Generator("cuda").manual_seed(seed)

    set_scheduler(
        model_path,
        scheduler_name,
        get_scheduler_config(model_path)
    )

    pipeline = load_pipeline(model_path)

    # with torch.inference_mode():
    image = pipeline(
        prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=20,
        guidance_scale=5,
        width=size[0],
        height=size[1],
        # clip_skip=clip_skip,
        callback_on_step_end=callback_factory(callback),
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
    # Empty cache corrupt image?
    # empty_cache()
    return image

def latents_to_rgb(latents):
    # https://huggingface.co/docs/diffusers/using-diffusers/callback
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = (torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor)
                  + biases_tensor.unsqueeze(-1).unsqueeze(-1))
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)

def callback_factory(callback: callable) -> callable:
    def decode_tensors(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        image = latents_to_rgb(latents[0])
        callback(image, step+1)

        return callback_kwargs

    return decode_tensors

@lru_cache(maxsize=1)
def get_schedulers_map() -> dict:
    from diffusers import schedulers as diffusers_schedulers

    result = {}

    karras_schedulers = [e.name for e in KarrasDiffusionSchedulers]
    schedulers = dir(diffusers_schedulers)
    schedulers = [
        getattr(diffusers_schedulers, x) for x in schedulers if x.endswith("Scheduler")
        and x in karras_schedulers
    ]

    for s in schedulers:
        name = s.__name__ if hasattr(s, "__name__") else s.__class__.__name__
        if name.endswith("Scheduler"):
            name = name[:-9]
        result[name] = s

    print(f"{result.keys()=}")
    return result

@lru_cache(maxsize=1)
def get_scheduler_config(model_path: str):
    pipe: DiffusionPipeline = load_pipeline(model_path)
    return pipe.scheduler.config

def set_scheduler(
        model_path: str,
        scheduler_name: str,
        scheduler_config: FrozenDict,
) -> None:
    pipeline: DiffusionPipeline = load_pipeline(model_path)
    schedulers_map = get_schedulers_map()
    scheduler_class = schedulers_map[scheduler_name]

    is_same_scheduler = isinstance(pipeline.scheduler, scheduler_class)
    is_same_config = scheduler_config == pipeline.scheduler.config

    print(f"{is_same_scheduler=}; {scheduler_class=}; {pipeline.scheduler=}")
    print(f"{is_same_config=}; {scheduler_config=}; {pipeline.scheduler.config=}")

    if is_same_scheduler and is_same_config:
        return

    print(f"Set new scheduler {scheduler_class} with {scheduler_config=}")
    # scheduler_config["prediction_type"] = "v_prediction"
    pipeline.scheduler = scheduler_class.from_config(scheduler_config)
