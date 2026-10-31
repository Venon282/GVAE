"""Building blocks of the point cloud encoder and decoder (roadmap P3-8, ADR 0044).

A point cloud is a set: `(batch, points, dim)` with no meaningful order of the rows. A row of `NaN`
is a point that does not exist, which is how a cloud with fewer points than the batch tensor holds
is padded (the same convention as the loss of `point_cloud_metrics.py`, so the batch tensor is the
one thing that both the encoder and the loss read).
"""

import torch

POOLING_CHOICES = ("max_mean", "max", "mean")
"""Every value `poolPoints` accepts: the max, the mean, or both side by side."""


def splitPoints(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a batch of clouds into its values, with the missing points zeroed, and a mask.

    Args:
        x: A float tensor `(batch, points, dim)`. A row with a `NaN` anywhere is not a point.
        dim: The expected number of coordinates (and features) per point.

    Returns:
        `(filled, valid)`: `x` with every `NaN` replaced by `0`, and the boolean `(batch, points)`
        mask that is `True` at the rows that are points.

    Raises:
        ValueError: If `x` is not a floating point tensor of shape `(batch, points, dim)`.
    """
    if not x.is_floating_point() or x.dim() != 3 or x.shape[-1] != dim:
        raise ValueError(
            f"A batch of point clouds is a float tensor (batch, points, {dim}), got "
            f"{tuple(x.shape)} of dtype {x.dtype}."
        )
    valid = ~torch.isnan(x).any(dim=-1)
    return torch.nan_to_num(x, nan=0.0), valid


def poolPoints(features: torch.Tensor, valid: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool the features of the points of each cloud into one vector, leaving missing points out.

    The max and the mean are symmetric functions, so the result does not depend on the order of the
    points, nor on how many rows of padding follow them. A cloud with no point at all pools to
    zeros, so it stays finite.

    Args:
        features: `(batch, points, width)`, one feature vector per row.
        valid: Boolean `(batch, points)`, `True` where the row is a point.
        pooling: `"max"`, `"mean"` or `"max_mean"` (the two concatenated, twice the width).

    Returns:
        `(batch, width)`, or `(batch, 2 * width)` for `"max_mean"`.

    Raises:
        ValueError: If `pooling` is not one of `POOLING_CHOICES`.
    """
    if pooling not in POOLING_CHOICES:
        raise ValueError(f"Unknown pooling '{pooling}'. Expected one of {POOLING_CHOICES}.")
    present = valid.unsqueeze(-1)
    pooled: list[torch.Tensor] = []
    if pooling in ("max", "max_mean"):
        biggest = features.masked_fill(~present, float("-inf")).max(dim=1).values
        pooled.append(biggest.masked_fill(~valid.any(dim=1, keepdim=True), 0.0))
    if pooling in ("mean", "max_mean"):
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(features.dtype)
        pooled.append((features * present).sum(dim=1) / count)
    return torch.cat(pooled, dim=-1)
