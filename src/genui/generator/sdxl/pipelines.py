from functools import cached_property

from compel import Compel, ReturnedEmbeddingsType
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin

from ...settings import settings


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
    # Bug in diffusers library.
    _interrupt = False
