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
    
    
@lru_cache(maxsize=1)
def get_file_hash(filename, algorithm='sha256') -> str:
    """Compute the hash of the given file using the specified algorithm."""
    import hashlib
    
    hasher = hashlib.new(algorithm)
    
    with open(filename, 'rb') as f:
        while True:
            data = f.read(65536)  # Read in chunks of 64KB for efficiency
            if not data:
                break
            hasher.update(data)
    
    # Return the hexadecimal representation of the hash
    return hasher.hexdigest()
    
    
def get_metadata_from_prompt(prompt: GenerationPrompt) -> dict:
    """Return metadata from a GenerationPrompt object.
    
    This function extracts relevant information from the prompt object and returns it as a dictionary.
    We use the XMP metadata format to store the prompt information.
    We compress some fields using Zstandard compression for saving space.
    """
    from .generator.sdxl import load_pipeline
    
    d = dataclasses.asdict(prompt)
    d.pop("callback")
    
    model_path = d.pop("model_path")
    model_name = model_path.split("/")[-1]
    
    pipeline = load_pipeline(model_path)
    scheduler_config = pipeline.scheduler.config
    
    metadata = {
        "Xmp.genui.prompt": json.dumps(d),
        "Xmp.genui.generator": "Genui",
        "Xmp.genui.generator_version": __version__,
        "Xmp.genui.model_name": model_name,
        # "Xmp.genui.model_hash_sha256": get_file_hash(model_path),
        "Xmp.genui.scheduler_config": json.dumps(scheduler_config),
        "Xmp.genui.deepcache": settings.deep_cache.json(),
    }
    return metadata
