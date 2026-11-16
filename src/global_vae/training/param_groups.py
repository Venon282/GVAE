"""Learning-rate groups of the optimizer, by module (roadmap P6-5, ADR 0055).

Pretrained modules, the encoders and decoders that a pair stage starts from, are often better
trained with a fraction of the rate that the rest of the model uses, so that the stage does not
undo what the solo stages learned. `training.param_groups` says which modules, and by how much:

    param_groups:
      - modules: [encoders.signal, decoders.signal]
        lr_factor: 0.1

Each entry becomes one parameter group of the optimizer, with `lr = lr_factor * the optimizer's
lr`. The parameters that no entry names stay together in one more group, last, named `default`,
at the optimizer's own rate: so a run with no `param_groups` has one group, as before, and a run
with them has one group per entry, plus that one if anything is left (parameters of the loss
terms and of a reconstruction loss are left, since they belong to no module of the model).

Three rules keep this unambiguous:

- a parameter belongs to the **first** entry whose patterns match it (`module_patterns.py` gives
  the rule), so an entry cannot reach a parameter that an earlier one took;
- an entry that ends up with no parameter is an error, whether its patterns match no module, only
  modules that hold no parameter, or only parameters an earlier entry took: it would otherwise
  have no effect, and the run would not be what its config says;
- a factor is above zero. A factor of zero is not a way to freeze a module (the optimizer would
  still move its running statistics and its weight decay state), and `freeze_modules` is.

Every group carries its `name` and its `lr_factor` next to `params` and `lr`, which is how they
reach the checkpoint (`optimizer.state_dict()` keeps every key of a group but its parameters), and
how a resume finds out that it was given other groups than the run it continues
(`training/checkpoint.py`). The schedule callbacks of `training/callbacks/` scale each group from
its own starting rate, so a group keeps its factor under a warm-up or a cosine decay.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from global_vae.training.module_patterns import checkPatterns, matchModules, parametersOf

DEFAULT_GROUP_NAME = "default"
"""Name of the group that holds the parameters no entry names."""

NAME_KEY = "name"
LR_FACTOR_KEY = "lr_factor"


@dataclass(frozen=True)
class ParamGroupSpec:
    """One entry of `training.param_groups`: some modules and a factor on the learning rate.

    Attributes:
        modules: Module name patterns (`module_patterns.py`). A list is accepted and kept as a
            tuple.
        lr_factor: The group's learning rate is this times the optimizer's. Above zero and
            finite: `0.1` runs the modules at a tenth of the rate, `1.0` is a group of its own
            at the same rate (useful to log it apart), above `1.0` is allowed.
        name: What the group is called in the checkpoint and in messages. `None` (default) names
            it after its patterns, `encoders.signal+decoders.signal`. Not `default`, which is the
            group of what no entry names.

    Raises:
        ValueError: At construction, if `modules` is not a non-empty list of non-empty strings, if
            `lr_factor` is not a finite number above zero (a message says what to use to stop a
            module from training), or if `name` is empty or reserved.
    """

    modules: tuple[str, ...]
    lr_factor: float
    name: str | None = None

    def __post_init__(self) -> None:
        """Check the entry and keep `modules` as a tuple (see the class docstring for the rules)."""
        object.__setattr__(self, "modules", checkPatterns(self.modules, "param_groups modules"))
        factor = self.lr_factor
        if (
            isinstance(factor, bool)
            or not isinstance(factor, int | float)
            or not math.isfinite(factor)
            or factor <= 0.0
        ):
            raise ValueError(
                f"param_groups lr_factor must be a finite number above zero, got {factor!r}. "
                f"A module that should not train at all is frozen with the `freeze_modules` "
                f"callback; a factor of zero would still move its running statistics."
            )
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"param_groups name must be a non-empty string, got {self.name!r}.")
        if self.name == DEFAULT_GROUP_NAME:
            raise ValueError(
                f"param_groups name '{DEFAULT_GROUP_NAME}' is the name of the group that holds "
                f"the parameters no entry names. Choose another."
            )

    @property
    def label(self) -> str:
        """The group's name: `name`, or its patterns joined by `+`."""
        return self.name if self.name is not None else "+".join(self.modules)


def buildParamGroups(
    model: nn.Module,
    specs: Sequence[ParamGroupSpec],
    extra_parameters: Iterable[nn.Parameter] = (),
) -> list[dict[str, Any]]:
    """Split the parameters to train into the groups `specs` describe.

    Args:
        model: The model whose modules the patterns are matched against.
        specs: The entries, in the order that decides which one takes a parameter two of them
            match, and the order of the groups.
        extra_parameters: Parameters that belong to no module of `model` and are trained all the
            same (those of the loss terms), which go to the `default` group.

    Returns:
        One dict per group, ready to pass to an optimizer's constructor: `params`, `name` and
        `lr_factor`, with no `lr`, so that the optimizer's own rate applies until
        `applyLrFactors` scales it. The entries come first, in order, then the `default` group
        if any parameter is left.

    Raises:
        ValueError: If two entries have the same name, if a pattern matches no module, or if an
            entry has no parameter of its own (it matches modules with no parameter, or only
            parameters an earlier entry took). The message names the entry by its index.
    """
    labels = [spec.label for spec in specs]
    duplicated = sorted({label for label in labels if labels.count(label) > 1})
    if duplicated:
        raise ValueError(
            f"training.param_groups: the name(s) {duplicated} are used by more than one entry. "
            f"Give each entry a distinct `name`."
        )

    owner: dict[int, str] = {}
    groups: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        where = f"training.param_groups[{index}] ('{spec.label}')"
        modules = matchModules(model, spec.modules, where)
        matched = parametersOf(modules.values())
        own = [parameter for parameter in matched if id(parameter) not in owner]
        if not own:
            if not matched:
                raise ValueError(
                    f"{where}: the modules it matches ({sorted(modules)[:5]}) have no parameter "
                    f"to give a learning rate to."
                )
            taken = sorted({owner[id(parameter)] for parameter in matched})
            raise ValueError(
                f"{where}: every parameter it matches is already in the group(s) {taken}. A "
                f"parameter belongs to the first entry that matches it, so this entry has "
                f"nothing left: put it before the other, or drop it."
            )
        for parameter in own:
            owner[id(parameter)] = spec.label
        groups.append({"params": own, NAME_KEY: spec.label, LR_FACTOR_KEY: spec.lr_factor})

    rest: list[nn.Parameter] = []
    seen = set(owner)
    for parameter in (*model.parameters(), *extra_parameters):
        if id(parameter) not in seen:
            seen.add(id(parameter))
            rest.append(parameter)
    if rest:
        groups.append({"params": rest, NAME_KEY: DEFAULT_GROUP_NAME, LR_FACTOR_KEY: 1.0})
    return groups


def applyLrFactors(groups: Iterable[dict[str, Any]]) -> None:
    """Multiply the `lr` of every group by its `lr_factor`, in place.

    Called once, on the parameter groups of a freshly built optimizer, whose groups all hold the
    optimizer's own rate (the `lr` it was given, or its default), whatever that optimizer is.

    Args:
        groups: `optimizer.param_groups`; each must hold `lr` and `lr_factor`. The dicts are
            changed in place, since that is what an optimizer hands out.
    """
    for group in groups:
        group["lr"] = group["lr"] * group[LR_FACTOR_KEY]
