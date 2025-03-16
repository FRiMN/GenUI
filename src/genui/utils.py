from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
import collections
import time
from functools import lru_cache
import dataclasses
import json

from .settings import settings
from .__version__ import __version__

if TYPE_CHECKING:
    from .generator.sdxl import GenerationPrompt


BACKGROUND_COLOR_HEX = "#1e1e1e"
TOOLBAR_MARGIN = (3, 0, 3, 0)


class FIFODict(collections.OrderedDict):
    def __init__(self, maxsize=128):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        else:
            if len(self) + 1 > self.maxsize:
                self.popitem(last=False)
        super().__setitem__(key, value)
        
        
def generate_image_filepath() -> Path:
    """Generate a unique image filepath."""
    t = int(time.time())
    return settings.autosave_image.path / Path(f"{t}.jpg")
    
    
def get_metadata_from_prompt(prompt: GenerationPrompt) -> dict:
    """Return metadata from a GenerationPrompt object.
    
    This function extracts relevant information from the prompt object and returns it as a dictionary.
    We use the XMP metadata format to store the prompt information.
    """
    from .generator.sdxl import load_pipeline
    # from safetensors import safe_open
    
    d = dataclasses.asdict(prompt)
    d.pop("callback")
    
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
