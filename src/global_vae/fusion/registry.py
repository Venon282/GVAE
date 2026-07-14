"""Self-registration registry for fusion strategies. Mirrors `encoders/registry.py`."""

from collections.abc import Callable

from global_vae.fusion.base import AbstractFusion

_FUSION_REGISTRY: dict[str, type[AbstractFusion]] = {}


def registerFusion(name: str) -> Callable[[type[AbstractFusion]], type[AbstractFusion]]:
    """Class decorator registering a fusion strategy under `name`.

    Args:
        name: Unique registry key (e.g. `"poe"`, `"moe"`,
            `"concat_mlp"`, `"cross_attention"`).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractFusion]) -> type[AbstractFusion]:
        if name in _FUSION_REGISTRY:
            raise ValueError(f"Fusion strategy '{name}' is already registered.")
        _FUSION_REGISTRY[name] = cls
        return cls

    return decorator


def getFusionClass(name: str) -> type[AbstractFusion]:
    """Look up a registered fusion strategy class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The fusion class registered under `name`.

    Raises:
        KeyError: If no fusion strategy is registered under `name`.
    """
    if name not in _FUSION_REGISTRY:
        available = ", ".join(sorted(_FUSION_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown fusion strategy '{name}'. Available: {available}")
    return _FUSION_REGISTRY[name]


def listRegisteredFusions() -> list[str]:
    """Return all currently registered fusion strategy names.

    Returns:
        Sorted list of registered fusion strategy names.
    """
    return sorted(_FUSION_REGISTRY)
