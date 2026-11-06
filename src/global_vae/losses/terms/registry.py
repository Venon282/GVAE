"""Self-registration registry for loss terms. Mirrors `losses/regularizers/registry.py`."""

from collections.abc import Callable

from global_vae.losses.terms.base import AbstractLossTerm

_LOSS_TERM_REGISTRY: dict[str, type[AbstractLossTerm]] = {}


def registerLossTerm(
    name: str,
) -> Callable[[type[AbstractLossTerm]], type[AbstractLossTerm]]:
    """Class decorator registering a loss term implementation under `name`.

    Args:
        name: Unique registry key, the name the term has in `training.loss_terms`, in the
            metrics (`loss/<name>`) and in the checkpoint. It must not be `total`,
            `reconstruction` or `regularization`, nor contain a dot (`Trainer` refuses both).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractLossTerm]) -> type[AbstractLossTerm]:
        if name in _LOSS_TERM_REGISTRY:
            raise ValueError(f"Loss term '{name}' is already registered.")
        _LOSS_TERM_REGISTRY[name] = cls
        return cls

    return decorator


def getLossTermClass(name: str) -> type[AbstractLossTerm]:
    """Look up a registered loss term class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The loss term class registered under `name`.

    Raises:
        KeyError: If no loss term is registered under `name`.
    """
    if name not in _LOSS_TERM_REGISTRY:
        available = ", ".join(sorted(_LOSS_TERM_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown loss term '{name}'. Available: {available}")
    return _LOSS_TERM_REGISTRY[name]


def listRegisteredLossTerms() -> list[str]:
    """Return all currently registered loss term names.

    Returns:
        Sorted list of registered loss term names.
    """
    return sorted(_LOSS_TERM_REGISTRY)
