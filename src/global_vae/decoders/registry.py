"""Self-registration registry for decoders. Mirrors `encoders/registry.py`.

One registry holds both kinds of decoder: a one-modality `AbstractDecoder` and
an `AbstractSharedDecoder` that reconstructs several modalities (spec §2.1,
ADR 0027).
"""

from collections.abc import Callable
from typing import TypeVar

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.shared_base import AbstractSharedDecoder

DecoderClass = type[AbstractDecoder] | type[AbstractSharedDecoder]
"""A registrable decoder class: one-modality or shared."""

_DecoderClassT = TypeVar("_DecoderClassT", bound=DecoderClass)

_DECODER_REGISTRY: dict[str, DecoderClass] = {}


def registerDecoder(name: str) -> Callable[[_DecoderClassT], _DecoderClassT]:
    """Class decorator registering a decoder implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"1d_cnn_decoder_v1"`), referenced
            from config files (see spec §9).

    Returns:
        A decorator that registers the class, an `AbstractDecoder` or an
        `AbstractSharedDecoder` subclass, and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: _DecoderClassT) -> _DecoderClassT:
        if name in _DECODER_REGISTRY:
            raise ValueError(f"Decoder '{name}' is already registered.")
        _DECODER_REGISTRY[name] = cls
        return cls

    return decorator


def getDecoderClass(name: str) -> DecoderClass:
    """Look up a registered decoder class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The decoder class registered under `name`, an `AbstractDecoder` or
        an `AbstractSharedDecoder` subclass.

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
