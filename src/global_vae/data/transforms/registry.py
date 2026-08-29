"""Self-registration registry for data transforms. Mirrors `fusion/registry.py`."""

from collections.abc import Callable

from global_vae.data.transforms.base import AbstractTransform

_TRANSFORM_REGISTRY: dict[str, type[AbstractTransform]] = {}


def registerTransform(name: str) -> Callable[[type[AbstractTransform]], type[AbstractTransform]]:
    """Class decorator registering a transform implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"log"`, `"standardize"`,
            `"resample"`), referenced from `DataConfig.transforms`
            (spec §9).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractTransform]) -> type[AbstractTransform]:
        if name in _TRANSFORM_REGISTRY:
            raise ValueError(f"Transform '{name}' is already registered.")
        _TRANSFORM_REGISTRY[name] = cls
        return cls

    return decorator


def getTransformClass(name: str) -> type[AbstractTransform]:
    """Look up a registered transform class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The transform class registered under `name`.

    Raises:
        KeyError: If no transform is registered under `name`.
    """
    if name not in _TRANSFORM_REGISTRY:
        available = ", ".join(sorted(_TRANSFORM_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown transform '{name}'. Available: {available}")
    return _TRANSFORM_REGISTRY[name]


def listRegisteredTransforms() -> list[str]:
    """Return all currently registered transform names.

    Returns:
        Sorted list of registered transform names.
    """
    return sorted(_TRANSFORM_REGISTRY)
