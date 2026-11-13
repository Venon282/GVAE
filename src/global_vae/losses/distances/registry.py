"""Self-registration registry for latent distances. Mirrors `losses/terms/registry.py`."""

from collections.abc import Callable

from global_vae.losses.distances.base import AbstractLatentDistance

_LATENT_DISTANCE_REGISTRY: dict[str, type[AbstractLatentDistance]] = {}


def registerLatentDistance(
    name: str,
) -> Callable[[type[AbstractLatentDistance]], type[AbstractLatentDistance]]:
    """Class decorator registering a latent distance implementation under `name`.

    Args:
        name: Unique registry key, the name a distance has in YAML (`mean_mse`,
            `wasserstein2_diag`, ...).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractLatentDistance]) -> type[AbstractLatentDistance]:
        if name in _LATENT_DISTANCE_REGISTRY:
            raise ValueError(f"Latent distance '{name}' is already registered.")
        _LATENT_DISTANCE_REGISTRY[name] = cls
        return cls

    return decorator


def getLatentDistanceClass(name: str) -> type[AbstractLatentDistance]:
    """Look up a registered latent distance class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The latent distance class registered under `name`.

    Raises:
        KeyError: If no latent distance is registered under `name`.
    """
    if name not in _LATENT_DISTANCE_REGISTRY:
        available = ", ".join(sorted(_LATENT_DISTANCE_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown latent distance '{name}'. Available: {available}")
    return _LATENT_DISTANCE_REGISTRY[name]


def listRegisteredLatentDistances() -> list[str]:
    """Return all currently registered latent distance names.

    Returns:
        Sorted list of registered latent distance names.
    """
    return sorted(_LATENT_DISTANCE_REGISTRY)
