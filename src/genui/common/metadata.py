from __future__ import annotations

import dataclasses
import json

from ..__version__ import __version__
from ..generator.sdxl import GenerationPrompt, load_pipeline
from ..settings import settings


def get_metadata_from_prompt(prompt: GenerationPrompt) -> dict:
    """Return metadata from a GenerationPrompt object.
    
    This function extracts relevant information from the prompt object and returns it as a dictionary.
    We use the XMP metadata format to store the prompt information.
    """
    # from safetensors import safe_open
    
    d = dataclasses.asdict(prompt)
    d.pop("callback")
    d.pop("image")
    
    model_path = d.pop("model_path")
    model_name = model_path.split("/")[-1]
    
    pipeline = load_pipeline(model_path)
    scheduler_config = pipeline.scheduler.config
    
    # with safe_open(model_path, framework="pt", device="cuda") as f:
    #     for key, value in f.metadata().items():
    #         print(key, value)
    
    metadata = {
        "Xmp.genui.prompt": json.dumps(d),
        "Xmp.genui.generator": "Genui",
        "Xmp.genui.generator_version": __version__,
        "Xmp.genui.model_name": model_name,
        "Xmp.genui.scheduler_config": json.dumps(scheduler_config),
        "Xmp.genui.deepcache": settings.deep_cache.json(),
    }
    return metadata
    

def get_prompt_from_metadata(metadata: dict) -> GenerationPrompt:
    """Return a GenerationPrompt object from metadata.
    
    This function extracts relevant information from the metadata dictionary and returns it as a GenerationPrompt object.
    """
   
    metadata = { 
        k: v 
        for k, v in metadata.items() 
        if k.startswith("Xmp.genui.") 
    }
    prompt = json.loads(metadata["Xmp.genui.prompt"])
    prompt["model_path"] = metadata["Xmp.genui.model_name"]
    prompt["size"] = tuple(prompt["size"])
    return GenerationPrompt(**prompt)
