"""Self-registration registry for beta schedules. Mirrors `losses/regularizers/registry.py`."""

from collections.abc import Callable

from global_vae.training.beta_schedules.base import AbstractBetaSchedule

_BETA_SCHEDULE_REGISTRY: dict[str, type[AbstractBetaSchedule]] = {}


def registerBetaSchedule(
    name: str,
) -> Callable[[type[AbstractBetaSchedule]], type[AbstractBetaSchedule]]:
    """Class decorator registering a beta schedule implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"constant"`, `"linear_warmup"`).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractBetaSchedule]) -> type[AbstractBetaSchedule]:
        if name in _BETA_SCHEDULE_REGISTRY:
            raise ValueError(f"Beta schedule '{name}' is already registered.")
        _BETA_SCHEDULE_REGISTRY[name] = cls
        return cls

    return decorator


def getBetaScheduleClass(name: str) -> type[AbstractBetaSchedule]:
    """Look up a registered beta schedule class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The beta schedule class registered under `name`.

    Raises:
        KeyError: If no beta schedule is registered under `name`.
    """
    if name not in _BETA_SCHEDULE_REGISTRY:
        available = ", ".join(sorted(_BETA_SCHEDULE_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown beta schedule '{name}'. Available: {available}")
    return _BETA_SCHEDULE_REGISTRY[name]


def listRegisteredBetaSchedules() -> list[str]:
    """Return all currently registered beta schedule names.

    Returns:
        Sorted list of registered beta schedule names.
    """
    return sorted(_BETA_SCHEDULE_REGISTRY)
