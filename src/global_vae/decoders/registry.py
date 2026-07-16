"""Self-registration registry for decoders. Mirrors `encoders/registry.py`."""

from collections.abc import Callable

from global_vae.decoders.base import AbstractDecoder

_DECODER_REGISTRY: dict[str, type[AbstractDecoder]] = {}


def registerDecoder(name: str) -> Callable[[type[AbstractDecoder]], type[AbstractDecoder]]:
    """Class decorator registering a decoder implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"signal_cnn_v1"`), referenced
            from config files (see spec §9).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractDecoder]) -> type[AbstractDecoder]:
        if name in _DECODER_REGISTRY:
            raise ValueError(f"Decoder '{name}' is already registered.")
        _DECODER_REGISTRY[name] = cls
        return cls

    return decorator


def getDecoderClass(name: str) -> type[AbstractDecoder]:
    """Look up a registered decoder class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The decoder class registered under `name`.

    Raises:
        KeyError: If no decoder is registered under `name`.
    """
    if name not in _DECODER_REGISTRY:
        available = ", ".join(sorted(_DECODER_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown decoder '{name}'. Available: {available}")
    return _DECODER_REGISTRY[name]


def listRegisteredDecoders() -> list[str]:
    """Return all currently registered decoder names.

    Returns:
        Sorted list of registered decoder names.
    """
    return sorted(_DECODER_REGISTRY)
