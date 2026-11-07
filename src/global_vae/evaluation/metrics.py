"""Reconstruction quality metrics (spec: "MSE at minimum") and their registry.

Metrics are a name -> function table with a declared default set (roadmap P5-3, ADR 0048), **not**
a registry of mutually exclusive strategies like encoders, fusion, regularizers, beta schedules or
experiment loggers: those are picked one at a time for a specific role, while a caller normally
wants *several* metrics computed together for a full picture. `registerMetric(name)` adds a
function to the table, `getMetric(name)` reads it, `listRegisteredMetrics()` lists it, and
`DEFAULT_METRIC_NAMES` is the set a report has when nothing says otherwise: the five metrics every
report always had. `DEFAULT_RECONSTRUCTION_METRICS` is those five as a plain
`dict[str, MetricFn]`, matching the "shared dict, override or extend by passing your own" pattern
`losses.reconstruction.computeTotalReconstructionLoss` uses for `loss_fn`.

The metrics that describe one kind of data (point clouds, graphs, text) are registered by
`evaluation/modality_metrics.py`. A metric whose name starts with `masked_` is the plain one
computed over the values whose target is not `NaN`.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]

_METRIC_REGISTRY: dict[str, MetricFn] = {}


def registerMetric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Function decorator registering a metric under `name`.

    Args:
        name: Unique key of the metric, the name it has in a report and in `evaluate()`.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(metric: MetricFn) -> MetricFn:
        if name in _METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        _METRIC_REGISTRY[name] = metric
        return metric

    return decorator


def getMetric(name: str) -> MetricFn:
    """Look up a registered metric by name.

    Args:
        name: Key used at registration time.

    Returns:
        The metric function, `(reconstruction, target) -> float`.

    Raises:
        KeyError: If no metric is registered under `name`.
    """
    if name not in _METRIC_REGISTRY:
        available = ", ".join(sorted(_METRIC_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")
    return _METRIC_REGISTRY[name]


def listRegisteredMetrics() -> list[str]:
    """Return all currently registered metric names.

    Returns:
        Sorted list of registered metric names.
    """
    return sorted(_METRIC_REGISTRY)


def maskUnobserved(metric: MetricFn) -> MetricFn:
    """Turn a metric into the same metric over the values whose target is not `NaN`.

    The five plain metrics pool every element of the tensors into one number, so the masked form is
    the plain metric of the observed values flattened, and equals the plain one on data with no gap.

    Args:
        metric: A metric function.

    Returns:
        A function `(reconstruction, target) -> float` that returns `nan` when no target value is
        observed, and raises `ValueError` when the two shapes differ.
    """

    def masked(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"reconstruction and target must have one shape, got {tuple(reconstruction.shape)} "
                f"and {tuple(target.shape)}."
            )
        observed = ~torch.isnan(target)
        if not bool(observed.any()):
            return float("nan")
        return metric(reconstruction[observed], target[observed])

    masked.__doc__ = f"{metric.__doc__}\n\n    Computed over the values whose target is not NaN."
    return masked


@registerMetric("mse")
def computeMse(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error.

    Args:
        reconstruction: Model output.
        target: Ground truth, same shape as `reconstruction`.

    Returns:
        Scalar MSE.
    """
    return F.mse_loss(reconstruction, target).item()


@registerMetric("rmse")
def computeRmse(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error (same units as the data, often more interpretable than MSE).

    Args:
        reconstruction: Model output.
        target: Ground truth, same shape as `reconstruction`.

    Returns:
        Scalar RMSE.
    """
    return F.mse_loss(reconstruction, target).sqrt().item()


@registerMetric("mae")
def computeMae(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error (more robust to outliers than MSE/RMSE).

    Args:
        reconstruction: Model output.
        target: Ground truth, same shape as `reconstruction`.

    Returns:
        Scalar MAE.
    """
    return F.l1_loss(reconstruction, target).item()


@registerMetric("r2")
def computeR2(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination: `1 - SS_res / SS_tot`.

    Normalizes MSE against the variance of the target itself, so unlike raw MSE it is
    comparable across datasets/scales: `1.0` is a perfect reconstruction, `0.0` is "no
    better than always predicting the target's mean", negative is worse than that.

    The baseline "always predict the mean" uses a single scalar `target.mean()` over
    every element pooled together (matching `computeMse`'s own flat pooling across the
    whole tensor), not a per-feature/per-position mean (the convention e.g.
    scikit-learn's `r2_score` uses for multi-output regression). For a reconstructed
    series, individual positions do not have an independently meaningful baseline the
    way separate regression targets would, so one pooled number is the more natural
    fit here.

    Args:
        reconstruction: Model output.
        target: Ground truth, same shape as `reconstruction`.

    Returns:
        Scalar R^2, or `nan` if `target` is exactly constant (undefined: `SS_tot == 0`).
    """
    residual_sum_of_squares = (target - reconstruction).pow(2).sum()
    total_sum_of_squares = (target - target.mean()).pow(2).sum()
    if total_sum_of_squares.item() == 0.0:
        return float("nan")
    return (1.0 - residual_sum_of_squares / total_sum_of_squares).item()


@registerMetric("pearson_r")
def computePearsonR(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson correlation coefficient between the flattened reconstruction and target.

    Useful as a "does the model capture the right shape/pattern" check independent of
    any constant offset or scale mismatch, which R^2 (unlike correlation) penalizes.

    Args:
        reconstruction: Model output.
        target: Ground truth, same shape as `reconstruction`.

    Returns:
        Scalar Pearson r in `[-1, 1]`, or `nan` if either tensor is exactly constant
        (undefined: zero variance).
    """
    reconstruction_flat = reconstruction.flatten().double()
    target_flat = target.flatten().double()
    if reconstruction_flat.std() == 0.0 or target_flat.std() == 0.0:
        return float("nan")
    return torch.corrcoef(torch.stack([reconstruction_flat, target_flat]))[0, 1].item()


DEFAULT_METRIC_NAMES: tuple[str, ...] = ("mse", "rmse", "mae", "r2", "pearson_r")
"""The metrics of a report when nothing says otherwise: the five it always had."""

DEFAULT_RECONSTRUCTION_METRICS: dict[str, MetricFn] = {
    name: getMetric(name) for name in DEFAULT_METRIC_NAMES
}

for _name in DEFAULT_METRIC_NAMES:
    registerMetric(f"masked_{_name}")(maskUnobserved(getMetric(_name)))
