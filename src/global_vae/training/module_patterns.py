"""Select the modules of a model by name patterns (roadmap P6-5, ADR 0055).

`freeze_modules` (which modules stay fixed) and `training.param_groups` (which modules run at a
fraction of the learning rate) both point at parts of a model by the dotted names `nn.Module`
gives them: `encoders.signal`, `decoders.image`, `regularizers.z_signal`. They read those names
the same way, so the rule is written once here.

A pattern is a shell-style pattern (`fnmatch`, case sensitive) matched against the **whole**
dotted name of a module, and in it `*` stands for any characters, dots included:

- `encoders.signal` is that encoder, and every layer inside it;
- `decoders.*` is every decoder and every layer inside one (not the `decoders` container itself,
  which has no dot after it; `decoders` is that container, so every decoder too);
- `*` is the whole model, since the root module's name is the empty string.

A module stands for its own parameters and those of every module inside it, so a pattern needs to
name only the top of a part. This module holds no state and registers nothing.
"""

from collections.abc import Iterable, Sequence
from fnmatch import fnmatchcase

from torch import nn

_EXAMPLE_NAMES = 12


def checkPatterns(patterns: object, what: str) -> tuple[str, ...]:
    """Check that `patterns` is a non-empty sequence of non-empty strings.

    Args:
        patterns: What the caller passed. A YAML list arrives as a list or as an OmegaConf list,
            and both are accepted; a bare string is not (it would be read as its characters).
        what: What the patterns are for, e.g. `training.param_groups[0].modules`, for the message.

    Returns:
        The patterns, as a tuple, in the order given.

    Raises:
        ValueError: If `patterns` is a string, is not a sequence, is empty, or holds an entry
            that is not a non-empty string.
    """
    if isinstance(patterns, str) or not isinstance(patterns, Sequence):
        raise ValueError(
            f"{what} must be a list of module name patterns, e.g. [encoders.signal, decoders.*], "
            f"got {patterns!r}."
        )
    if len(patterns) == 0:
        raise ValueError(
            f"{what} is empty. List at least one module name pattern, e.g. [encoders.signal]."
        )
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(
                f"{what}[{index}] must be a non-empty module name pattern, got {pattern!r}."
            )
    return tuple(patterns)


def matchModules(model: nn.Module, patterns: Sequence[str], what: str) -> dict[str, nn.Module]:
    """Return the modules of `model` whose dotted name matches any of `patterns`.

    Args:
        model: The model; its own name is the empty string, so only `*` matches it.
        patterns: Shell-style patterns, as `checkPatterns` returned them.
        what: What the patterns are for, for the message of an unmatched pattern.

    Returns:
        Dotted name -> module, each module once, in the order `model.named_modules()` gives them
        (a parent before its children).

    Raises:
        ValueError: If a pattern matches no module. A pattern that matches nothing is a typo or
            a pattern written for another model, and silently doing nothing would change the
            run without a trace. The message lists some module names that exist.
    """
    named = list(model.named_modules())
    hits: set[str] = set()
    for pattern in patterns:
        found = {name for name, _ in named if fnmatchcase(name, pattern)}
        if not found:
            examples = [name for name, _ in named if name and name.count(".") <= 1]
            shown = ", ".join(examples[:_EXAMPLE_NAMES])
            more = ", ..." if len(examples) > _EXAMPLE_NAMES else ""
            raise ValueError(
                f"{what}: the pattern '{pattern}' matches no module of the model. A pattern is "
                f"matched against the whole dotted name of a module, and `*` stands for any "
                f"characters, dots included (modules: {shown}{more})."
            )
        hits |= found
    return {name: module for name, module in named if name in hits}


def parametersOf(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    """Return the parameters of `modules` and of every module inside them, each once.

    Args:
        modules: Modules, typically `matchModules(...).values()`; one may contain another.

    Returns:
        The parameters in order of first appearance. A parameter shared by two modules is listed
        once.
    """
    seen: set[int] = set()
    parameters: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters
