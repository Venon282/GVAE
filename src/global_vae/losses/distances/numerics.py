"""Small numerical helpers shared by the latent distances (roadmap P6-3, ADR 0053)."""

import torch

DIMENSION_REDUCTIONS = ("sum", "mean")


def checkDimensionReduction(dimension_reduction: str) -> str:
    """Validate the `dimension_reduction` option of a distance that sums over latent dimensions.

    Args:
        dimension_reduction: `"sum"` (the footing of the KL term, ADR 0029) or `"mean"`.

    Returns:
        `dimension_reduction`, unchanged.

    Raises:
        ValueError: If it is neither `"sum"` nor `"mean"`.
    """
    if dimension_reduction not in DIMENSION_REDUCTIONS:
        raise ValueError(
            f"dimension_reduction must be one of {list(DIMENSION_REDUCTIONS)}, "
            f"got {dimension_reduction!r}."
        )
    return dimension_reduction


def reduceOverDimensions(per_dimension: torch.Tensor, dimension_reduction: str) -> torch.Tensor:
    """Reduce a `(batch, dim)` tensor over its last axis, by sum or by mean.

    Args:
        per_dimension: The value of every latent dimension of every sample, `(batch, dim)`.
        dimension_reduction: `"sum"` (per-sample sum, the scale of the KL term, ADR 0029) or
            `"mean"` (the sum divided by the width, for a weight that does not depend on it).

    Returns:
        One value per sample, shape `(batch,)`.
    """
    if dimension_reduction == "mean":
        return per_dimension.mean(dim=-1)
    return per_dimension.sum(dim=-1)


def safeSqrt(values: torch.Tensor) -> torch.Tensor:
    """Square root of a non-negative tensor whose derivative is 0, and not infinite, at 0.

    The exact square root has an infinite derivative at 0, so the gradient of a distance that
    takes one is `nan` on two identical inputs (`inf * 0`). Here the value is exact everywhere
    and the derivative is taken as 0 where the input is 0. Both branches of the `where` are
    finite, so no `nan` reaches the backward pass.

    Args:
        values: A tensor of values that are at least 0.

    Returns:
        `sqrt(values)`, same shape.
    """
    positive = values > 0
    safe = torch.where(positive, values, torch.ones_like(values))
    return torch.where(positive, safe.sqrt(), torch.zeros_like(values))


def clampLogvar(logvar: torch.Tensor, min_logvar: float, max_logvar: float) -> torch.Tensor:
    """Clamp a log-variance to `[min_logvar, max_logvar]`.

    Args:
        logvar: Posterior log-variance.
        min_logvar: Lower bound. A variance below `exp(min_logvar)` makes a divergence that
            divides by it explode.
        max_logvar: Upper bound.

    Returns:
        The clamped tensor. A value outside the range has no gradient, which is the intended
        effect on a variance that is out of range.
    """
    return logvar.clamp(min=min_logvar, max=max_logvar)


def checkLogvarRange(min_logvar: float, max_logvar: float) -> None:
    """Validate the `min_logvar` and `max_logvar` options.

    Args:
        min_logvar: Lower bound of the log-variance.
        max_logvar: Upper bound of the log-variance.

    Raises:
        ValueError: If `min_logvar` is not strictly below `max_logvar`.
    """
    if not min_logvar < max_logvar:
        raise ValueError(
            f"min_logvar must be below max_logvar, got min_logvar={min_logvar} "
            f"and max_logvar={max_logvar}."
        )
