"""Self-registration registry for latent regularizers. Mirrors `fusion/registry.py`."""

from collections.abc import Callable

from global_vae.losses.regularizers.base import AbstractLatentRegularizer

_REGULARIZER_REGISTRY: dict[str, type[AbstractLatentRegularizer]] = {}


def registerRegularizer(
    name: str,
) -> Callable[[type[AbstractLatentRegularizer]], type[AbstractLatentRegularizer]]:
    """Class decorator registering a latent regularizer implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"kl_standard_normal"`, `"mmd"`,
            `"free_bits_kl"`).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(
        cls: type[AbstractLatentRegularizer],
    ) -> type[AbstractLatentRegularizer]:
        if name in _REGULARIZER_REGISTRY:
            raise ValueError(f"Latent regularizer '{name}' is already registered.")
        _REGULARIZER_REGISTRY[name] = cls
        return cls

    return decorator


def getRegularizerClass(name: str) -> type[AbstractLatentRegularizer]:
    """Look up a registered latent regularizer class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The regularizer class registered under `name`.

    Raises:
        KeyError: If no regularizer is registered under `name`.
    """
    if name not in _REGULARIZER_REGISTRY:
        available = ", ".join(sorted(_REGULARIZER_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown latent regularizer '{name}'. Available: {available}")
    return _REGULARIZER_REGISTRY[name]


def listRegisteredRegularizers() -> list[str]:
    """Return all currently registered latent regularizer names.

    Returns:
        Sorted list of registered latent regularizer names.
    """
    return sorted(_REGULARIZER_REGISTRY)
