"""Self-registration registry for encoders.

Mirrors the pattern used for decoders, fusion strategies, and
assemblers: concrete implementations register themselves by name via a
decorator, so `models/global_vae.py` can instantiate any encoder from a
config string without importing the concrete class directly.

One registry holds both kinds of encoder: a one-modality `AbstractEncoder`
and an `AbstractSharedEncoder` that ingests several modalities (spec §2.1,
ADR 0027). A config names either the same way, and `GlobalVae` tells them
apart with `isinstance`.
"""

from collections.abc import Callable
from typing import TypeVar

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.shared_base import AbstractSharedEncoder

EncoderClass = type[AbstractEncoder] | type[AbstractSharedEncoder]
"""A registrable encoder class: one-modality or shared."""

_EncoderClassT = TypeVar("_EncoderClassT", bound=EncoderClass)

_ENCODER_REGISTRY: dict[str, EncoderClass] = {}


def registerEncoder(name: str) -> Callable[[_EncoderClassT], _EncoderClassT]:
    """Class decorator registering an encoder implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"1d_cnn_encoder_v1"`), referenced
            from config files (see spec §9).

    Returns:
        A decorator that registers the class, an `AbstractEncoder` or an
        `AbstractSharedEncoder` subclass, and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: _EncoderClassT) -> _EncoderClassT:
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' is already registered.")
        _ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def getEncoderClass(name: str) -> EncoderClass:
    """Look up a registered encoder class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The encoder class registered under `name`, an `AbstractEncoder` or
        an `AbstractSharedEncoder` subclass.

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
