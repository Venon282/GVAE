"""Self-registration registry for experiment loggers. Mirrors `fusion/registry.py`."""

from collections.abc import Callable

from global_vae.training.loggers.base import AbstractExperimentLogger

_LOGGER_REGISTRY: dict[str, type[AbstractExperimentLogger]] = {}


def registerLogger(
    name: str,
) -> Callable[[type[AbstractExperimentLogger]], type[AbstractExperimentLogger]]:
    """Class decorator registering an experiment logger implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"csv"`, `"tensorboard"`).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractExperimentLogger]) -> type[AbstractExperimentLogger]:
        if name in _LOGGER_REGISTRY:
            raise ValueError(f"Experiment logger '{name}' is already registered.")
        _LOGGER_REGISTRY[name] = cls
        return cls

    return decorator


def getLoggerClass(name: str) -> type[AbstractExperimentLogger]:
    """Look up a registered experiment logger class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The logger class registered under `name`.

    Raises:
        KeyError: If no logger is registered under `name`.
    """
    if name not in _LOGGER_REGISTRY:
        available = ", ".join(sorted(_LOGGER_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown experiment logger '{name}'. Available: {available}")
    return _LOGGER_REGISTRY[name]


def listRegisteredLoggers() -> list[str]:
    """Return all currently registered experiment logger names.

    Returns:
        Sorted list of registered experiment logger names.
    """
    return sorted(_LOGGER_REGISTRY)
