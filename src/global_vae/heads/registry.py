"""Self-registration registry for latent heads. Mirrors `fusion/registry.py`."""

from collections.abc import Callable

from global_vae.heads.base import AbstractLatentHead

_HEAD_REGISTRY: dict[str, type[AbstractLatentHead]] = {}


def registerHead(name: str) -> Callable[[type[AbstractLatentHead]], type[AbstractLatentHead]]:
    """Class decorator registering a latent head under `name`.

    Args:
        name: Unique registry key (e.g. `"identity"`, `"linear"`).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractLatentHead]) -> type[AbstractLatentHead]:
        if name in _HEAD_REGISTRY:
            raise ValueError(f"Latent head '{name}' is already registered.")
        _HEAD_REGISTRY[name] = cls
        return cls

    return decorator


def getHeadClass(name: str) -> type[AbstractLatentHead]:
    """Look up a registered latent head class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The latent head class registered under `name`.

    Raises:
        KeyError: If no latent head is registered under `name`.
    """
    if name not in _HEAD_REGISTRY:
        available = ", ".join(sorted(_HEAD_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown latent head '{name}'. Available: {available}")
    return _HEAD_REGISTRY[name]


def listRegisteredHeads() -> list[str]:
    """Return all currently registered latent head names.

    Returns:
        Sorted list of registered latent head names.
    """
    return sorted(_HEAD_REGISTRY)
