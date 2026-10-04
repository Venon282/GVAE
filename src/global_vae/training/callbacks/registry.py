"""Self-registration registry for training callbacks. Mirrors `fusion/registry.py`.

Scope note (ADR 0022): this registry is for `TrainerCallback` strategies a
training run opts into by name from `TrainingConfig.callbacks` (checkpointing,
early stopping, learning-rate scheduling, and any caller's own
`@registerCallback(...)`-decorated class). Experiment loggers
(`training/loggers/`, `AbstractExperimentLogger`) are a separate, pre-existing
subsystem with their own `registerLogger`/`getLoggerClass` registry and their
own `TrainingConfig.loggers` config field: a `Logger` is a journalling
service, not a callback selection, even though `AbstractExperimentLogger`
happens to subclass `TrainerCallback` for the hook plumbing (ADR 0008). Do not
register logger classes here.
"""

from collections.abc import Callable

from global_vae.training.callbacks.base import TrainerCallback

_CALLBACK_REGISTRY: dict[str, type[TrainerCallback]] = {}


def registerCallback(name: str) -> Callable[[type[TrainerCallback]], type[TrainerCallback]]:
    """Class decorator registering a callback implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"early_stopping"`,
            `"best_checkpoint"`), referenced from the keys of
            `TrainingConfig.callbacks` (spec §9).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[TrainerCallback]) -> type[TrainerCallback]:
        if name in _CALLBACK_REGISTRY:
            raise ValueError(f"Callback '{name}' is already registered.")
        _CALLBACK_REGISTRY[name] = cls
        return cls

    return decorator


def getCallbackClass(name: str) -> type[TrainerCallback]:
    """Look up a registered callback class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The callback class registered under `name`.

    Raises:
        KeyError: If no callback is registered under `name`.
    """
    if name not in _CALLBACK_REGISTRY:
        available = ", ".join(sorted(_CALLBACK_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown callback '{name}'. Available: {available}")
    return _CALLBACK_REGISTRY[name]


def listRegisteredCallbacks() -> list[str]:
    """Return all currently registered callback names.

    Returns:
        Sorted list of registered callback names.
    """
    return sorted(_CALLBACK_REGISTRY)
