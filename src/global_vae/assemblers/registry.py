"""Self-registration registry for assemblers. Mirrors `fusion/registry.py`."""

from collections.abc import Callable

from global_vae.assemblers.base import AbstractAssembler

_ASSEMBLER_REGISTRY: dict[str, type[AbstractAssembler]] = {}


def registerAssembler(name: str) -> Callable[[type[AbstractAssembler]], type[AbstractAssembler]]:
    """Class decorator registering an assembler implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"concat"`, `"sum"`, `"average"`).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractAssembler]) -> type[AbstractAssembler]:
        if name in _ASSEMBLER_REGISTRY:
            raise ValueError(f"Assembler '{name}' is already registered.")
        _ASSEMBLER_REGISTRY[name] = cls
        return cls

    return decorator


def getAssemblerClass(name: str) -> type[AbstractAssembler]:
    """Look up a registered assembler class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The assembler class registered under `name`.

    Raises:
        KeyError: If no assembler is registered under `name`.
    """
    if name not in _ASSEMBLER_REGISTRY:
        available = ", ".join(sorted(_ASSEMBLER_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown assembler '{name}'. Available: {available}")
    return _ASSEMBLER_REGISTRY[name]


def listRegisteredAssemblers() -> list[str]:
    """Return all currently registered assembler names.

    Returns:
        Sorted list of registered assembler names.
    """
    return sorted(_ASSEMBLER_REGISTRY)
