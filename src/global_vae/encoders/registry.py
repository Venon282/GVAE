"""Self-registration registry for encoders.

Mirrors the pattern used for decoders, fusion strategies, and
assemblers: concrete implementations register themselves by name via a
decorator, so `models/global_vae.py` can instantiate any encoder from a
config string without importing the concrete class directly.
"""

from collections.abc import Callable

from global_vae.encoders.base import AbstractEncoder

_ENCODER_REGISTRY: dict[str, type[AbstractEncoder]] = {}


def registerEncoder(name: str) -> Callable[[type[AbstractEncoder]], type[AbstractEncoder]]:
    """Class decorator registering an encoder implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"signal_cnn_v1"`), referenced
            from config files (see spec §9).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractEncoder]) -> type[AbstractEncoder]:
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' is already registered.")
        _ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def getEncoderClass(name: str) -> type[AbstractEncoder]:
    """Look up a registered encoder class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The encoder class registered under `name`.

    Raises:
        KeyError: If no encoder is registered under `name`.
    """
    if name not in _ENCODER_REGISTRY:
        available = ", ".join(sorted(_ENCODER_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown encoder '{name}'. Available: {available}")
    return _ENCODER_REGISTRY[name]


def listRegisteredEncoders() -> list[str]:
    """Return all currently registered encoder names.

    Returns:
        Sorted list of registered encoder names.
    """
    return sorted(_ENCODER_REGISTRY)
