"""Self-registration registry for reconstruction losses. Mirrors `losses/terms/registry.py`."""

from collections.abc import Callable

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss

_RECONSTRUCTION_LOSS_REGISTRY: dict[str, type[AbstractReconstructionLoss]] = {}


def registerReconstructionLoss(
    name: str,
) -> Callable[[type[AbstractReconstructionLoss]], type[AbstractReconstructionLoss]]:
    """Class decorator registering a reconstruction loss implementation under `name`.

    Args:
        name: Unique registry key, the name the loss has in `training.reconstruction_loss`.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractReconstructionLoss]) -> type[AbstractReconstructionLoss]:
        if name in _RECONSTRUCTION_LOSS_REGISTRY:
            raise ValueError(f"Reconstruction loss '{name}' is already registered.")
        _RECONSTRUCTION_LOSS_REGISTRY[name] = cls
        return cls

    return decorator


def getReconstructionLossClass(name: str) -> type[AbstractReconstructionLoss]:
    """Look up a registered reconstruction loss class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The loss class registered under `name`.

    Raises:
        KeyError: If no loss is registered under `name`.
    """
    if name not in _RECONSTRUCTION_LOSS_REGISTRY:
        available = ", ".join(sorted(_RECONSTRUCTION_LOSS_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown reconstruction loss '{name}'. Available: {available}")
    return _RECONSTRUCTION_LOSS_REGISTRY[name]


def listRegisteredReconstructionLosses() -> list[str]:
    """Return all currently registered reconstruction loss names.

    Returns:
        Sorted list of registered reconstruction loss names.
    """
    return sorted(_RECONSTRUCTION_LOSS_REGISTRY)
