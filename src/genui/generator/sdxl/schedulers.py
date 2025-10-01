from functools import lru_cache
from dataclasses import dataclass

from .utils import load_pipeline


@dataclass(unsafe_hash=True)
class ModelSchedulerConfig:
    name: str
    model_path: str
    use_karras_sigmas: bool
    use_vpred: bool


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


@lru_cache(maxsize=1)
def get_scheduler(
    sc: ModelSchedulerConfig,
):
    schedulers_map = get_schedulers_map()
    scheduler_class = schedulers_map[sc.name]
    config = get_scheduler_config(sc.model_path)

    # frozenset convert to dict
    scheduler_config = {k: v for k, v in config}
    if sc.use_vpred:
        scheduler_config["prediction_type"] = "v_prediction"

    print(f"Get new scheduler {scheduler_class} with {scheduler_config=} and {sc.use_karras_sigmas=} and {sc.use_vpred=}")
    return scheduler_class.from_config(scheduler_config, use_karras_sigmas=sc.use_karras_sigmas)


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
