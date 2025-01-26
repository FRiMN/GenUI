import gc

import PIL
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

from utils import Timer


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


def load_pipline(model_path: str) -> DiffusionPipeline:
    print("start load pipline")
    empty_cache()
    with Timer("Pipline loading") as t:
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
        pipe.unet.to(memory_format=torch.channels_last)

        # helper = DeepCacheSDHelper(pipe=pipe)
        # helper.set_params(
        #     cache_interval=3,
        #     cache_branch_id=0,
        # )
        # helper.enable()

    print(f"{pipe.scheduler}")
    print(pipe.scheduler.compatibles)

    return pipe

def generate(
        pipline: StableDiffusionXLPipeline,
        prompt: str,
        neg_prompt: str,
        seed: int,
        size: tuple[int, int],
        clip_skip: int,
        callback,
) -> PIL.Image.Image:
    empty_cache()

    # max seed is 2147483647 ?
    torch.manual_seed(seed)

    # with torch.inference_mode():
    image = pipline(
        prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=20,
        width=size[0],
        height=size[1],
        clip_skip=clip_skip,
        callback_on_step_end=callback_factory(callback),
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
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

def callback_factory(callback):
    def decode_tensors(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        image = latents_to_rgb(latents[0])
        callback(image, step+1)

        return callback_kwargs

    return decode_tensors

def get_schedulers_map(pipe: DiffusionPipeline) -> dict:
    result = {}
    schedulers = pipe.scheduler.compatibles
    for s in schedulers:
        name = s.__name__ if hasattr(s, "__name__") else s.__class__.__name__
        if name.endswith("Scheduler"):
            name = name[:-9]
        if s == pipe.scheduler.__class__:
            name = f"{name} (Default)"
        result[name] = s

    print(f"{result.keys()=}")
    return result

def get_scheduler_config(pipe: DiffusionPipeline):
    return pipe.scheduler.config

def set_scheduler(
        pipe: DiffusionPipeline,
        scheduler_name: str,
        schedulers_map: dict,
        scheduler_config,
) -> None:
    print(f"{schedulers_map.keys()=}")
    scheduler = schedulers_map[scheduler_name]
    pipe.scheduler = scheduler.from_config(scheduler_config)
