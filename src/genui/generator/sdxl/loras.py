from dataclasses import dataclass

from .pipelines import GenUIStableDiffusionXLPipeline


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
