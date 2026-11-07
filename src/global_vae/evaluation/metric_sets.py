"""Which metrics `evaluate()` reports for each modality (roadmap P5-3, ADR 0048).

`evaluate(reconstruction_metrics=...)` is one of:

- `None` (or an empty one of the others): the default set of every modality, below;
- a dict of name -> function, for every modality (the form `evaluate` always took);
- a sequence of registered metric names (`evaluation.metrics.listRegisteredMetrics()`), for every
  modality;
- a mapping from a reconstruction name to a sequence of names or to a dict of functions, for the
  reconstructions it names. The others get their default set.

The default set of a modality follows the loss the model is trained with, given as
`reconstruction_loss_fn`, the first that applies:

1. the `default_metrics` of its `AbstractReconstructionLoss`, when it declares them (the cloud
   scores for `chamfer`, the token accuracy for `token_cross_entropy`);
2. the five default metrics in their masked form, **reported under the plain names** so that a table
   keeps its columns, when its loss `reads_nan_as_unobserved`;
3. the five default metrics.

A loss that is a plain function says nothing, so it gets the five.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from global_vae.evaluation.metrics import DEFAULT_METRIC_NAMES, MetricFn, getMetric
from global_vae.losses.reconstruction import AbstractReconstructionLoss, LossFn

MetricsSetting = Mapping[str, Any] | Sequence[str] | None


@dataclass(frozen=True)
class MetricSpec:
    """One metric of a report.

    Attributes:
        name: The name it is registered under in `evaluation.metrics`, or `None` for a function a
            caller passed in. Only a registered name can be accumulated batch by batch.
        function: What the pooled path calls, `(reconstruction, target) -> float`.
    """

    name: str | None
    function: MetricFn


def _fromNames(
    names: Sequence[Any], defaults: Mapping[str, MetricFn], label: str
) -> dict[str, MetricSpec]:
    """Build the specs of a sequence of registered names, in order."""
    specs: dict[str, MetricSpec] = {}
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"{label}: expected a metric name, got {name!r}.")
        if name in specs:
            raise ValueError(f"{label}: the metric '{name}' is listed twice.")
        try:
            function = defaults[name] if name in defaults else getMetric(name)
        except KeyError as error:
            raise KeyError(f"{label}: {error.args[0]}") from error
        specs[name] = MetricSpec(name, function)
    return specs


def _fromFunctions(functions: Mapping[str, MetricFn]) -> dict[str, MetricSpec]:
    """Build the specs of a dict of caller functions."""
    return {str(name): MetricSpec(None, function) for name, function in functions.items()}


def _lossFor(loss_fn: LossFn | Mapping[str, LossFn], modality: str) -> LossFn | None:
    """Return the loss of a reconstruction, or `None` if a mapping does not have it."""
    if isinstance(loss_fn, Mapping):
        return loss_fn.get(modality)
    return loss_fn


def defaultMetricSpecs(
    loss: LossFn | None, defaults: Mapping[str, MetricFn]
) -> dict[str, MetricSpec]:
    """Return the default metrics of a modality whose loss is `loss`.

    Args:
        loss: The reconstruction loss of the modality, or `None` if unknown.
        defaults: The five default metrics, `name -> function`
            (`DEFAULT_RECONSTRUCTION_METRICS`).

    Returns:
        Report key -> `MetricSpec`, as the module docstring orders the three cases.
    """
    if isinstance(loss, AbstractReconstructionLoss):
        if loss.default_metrics is not None:
            return _fromNames(
                loss.default_metrics, defaults, f"default metrics of {type(loss).__name__}"
            )
        if loss.reads_nan_as_unobserved:
            return {
                name: MetricSpec(f"masked_{name}", getMetric(f"masked_{name}"))
                for name in DEFAULT_METRIC_NAMES
            }
    return {name: MetricSpec(name, function) for name, function in defaults.items()}


def resolveMetricSets(
    setting: MetricsSetting,
    loss_fn: LossFn | Mapping[str, LossFn],
    modalities: Sequence[str],
    defaults: Mapping[str, MetricFn],
    field_name: str = "reconstruction_metrics",
) -> dict[str, dict[str, MetricSpec]]:
    """Resolve the `reconstruction_metrics` of `evaluate()` into one metric set per modality.

    Args:
        setting: The value the caller gave, in one of the forms of the module docstring.
        loss_fn: The `reconstruction_loss_fn` of `evaluate()`, which chooses the default set.
        modalities: The reconstruction names of the model.
        defaults: The five default metrics, `name -> function`.
        field_name: How the setting is called in error messages.

    Returns:
        Modality -> report key -> `MetricSpec`, for every modality.

    Raises:
        KeyError: If a metric name is not registered.
        ValueError: If the setting is none of the forms, a mapping names a reconstruction the model
            does not produce, or a mapping mixes functions and per-modality sets.
    """
    if setting is None or len(setting) == 0:
        return {
            modality: defaultMetricSpecs(_lossFor(loss_fn, modality), defaults)
            for modality in modalities
        }
    if isinstance(setting, str):
        raise ValueError(
            f"{field_name}: pass a list of metric names, not the string {setting!r} (a string "
            f"would be read one letter at a time)."
        )
    if not isinstance(setting, Mapping):
        shared = _fromNames(list(setting), defaults, field_name)
        return {modality: dict(shared) for modality in modalities}
    if all(callable(value) for value in setting.values()):
        shared = _fromFunctions(setting)
        return {modality: dict(shared) for modality in modalities}
    if any(callable(value) for value in setting.values()):
        raise ValueError(
            f"{field_name}: a mapping is either name -> function for every modality or "
            f"reconstruction name -> metrics, not a mix of the two."
        )
    unknown = sorted(str(key) for key in set(setting) - set(modalities))
    if unknown:
        raise ValueError(
            f"{field_name} names reconstruction(s) {unknown} that the model does not produce "
            f"(reconstruction names: {sorted(modalities)}). A modality left out keeps its default "
            f"metrics, so a mistyped name would change nothing and say nothing."
        )
    resolved: dict[str, dict[str, MetricSpec]] = {}
    for modality in modalities:
        if modality not in setting:
            resolved[modality] = defaultMetricSpecs(_lossFor(loss_fn, modality), defaults)
            continue
        value = setting[modality]
        label = f"{field_name}['{modality}']"
        if isinstance(value, Mapping) and all(callable(item) for item in value.values()):
            resolved[modality] = _fromFunctions(value)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            resolved[modality] = _fromNames(list(value), defaults, label)
        else:
            raise ValueError(
                f"{label}: expected a list of metric names or a dict of name -> function, "
                f"got {value!r}."
            )
    return resolved
