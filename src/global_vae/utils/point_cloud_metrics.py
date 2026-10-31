"""Losses and scores for sets of points (roadmap P3-8, ADR 0044).

A point cloud has no order, so the position-by-position losses of a training config (`mse`, `l1`,
...) compare a point with whichever point the file happened to list at the same index. The
Chamfer distance compares each point with its nearest neighbour in the other cloud instead, so it
does not depend on the order. It is a plain function of `(reconstruction, target)`, so it is passed
to `Trainer(reconstruction_loss_fn=...)` with no change to the framework.

A cloud with fewer points than the batch tensor holds is padded with rows of `NaN`: a row is a point
that does not exist. Every function here leaves such a row out, on the target side.
"""

import torch

DEFAULT_THRESHOLD = 0.1
"""Distance under which `computeCloudScores` counts a point as found, in the units of the data."""


def _checkClouds(reconstruction: torch.Tensor, target: torch.Tensor) -> None:
    """Raise if the two tensors are not batches of point sets of the same dimension.

    Args:
        reconstruction: Candidate `(batch, points, dim)` tensor.
        target: Candidate `(batch, points, dim)` tensor.

    Raises:
        ValueError: If either is not 3-dimensional, if the batch sizes or the point dimensions
            differ, or if some cloud of `target` has no point that is not `NaN`.
    """
    if reconstruction.dim() != 3 or target.dim() != 3:
        raise ValueError(
            "A batch of point clouds is (batch, points, dim), got "
            f"{tuple(reconstruction.shape)} and {tuple(target.shape)}."
        )
    if reconstruction.shape[0] != target.shape[0] or reconstruction.shape[2] != target.shape[2]:
        raise ValueError(
            f"The batch size and the point dimension must match, got {tuple(reconstruction.shape)} "
            f"and {tuple(target.shape)}. The number of points may differ."
        )
    if not (~torch.isnan(target).any(dim=-1)).any(dim=1).all():
        raise ValueError("Every cloud of the target needs at least one point that is not NaN.")


def _computeNearestDistances(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squared distance from each point to the nearest point of the other cloud.

    Args:
        reconstruction: `(batch, points, dim)`.
        target: `(batch, target_points, dim)`, rows that are `NaN` are padding.

    Returns:
        `(to_target, to_reconstruction, valid)`: for every reconstructed point the squared distance
        to the nearest real target point `(batch, points)`, for every target point the squared
        distance to the nearest reconstructed point `(batch, target_points)` (meaningless where
        `valid` is `False`), and the boolean mask of the real target points.
    """
    valid = ~torch.isnan(target).any(dim=-1)
    filled = torch.nan_to_num(target, nan=0.0)
    # |a|^2 + |b|^2 - 2 a.b, so that memory is batch * points * target_points and the gradient is
    # finite where two points coincide (a plain `cdist` is not).
    squared = (
        reconstruction.pow(2).sum(-1).unsqueeze(2)
        + filled.pow(2).sum(-1).unsqueeze(1)
        - 2.0 * reconstruction @ filled.transpose(1, 2)
    ).clamp_min(0.0)
    to_target = squared.masked_fill(~valid.unsqueeze(1), float("inf")).min(dim=2).values
    to_reconstruction = squared.min(dim=1).values
    return to_target, to_reconstruction, valid


def computeChamferDistance(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the Chamfer distance of a batch of clouds, summed over points, averaged over clouds.

    For each cloud it is the sum of two terms: the squared distance from every reconstructed point
    to its nearest real target point, and the squared distance from every real target point to its
    nearest reconstructed point. A cloud that is reproduced as a set, in any order, scores `0`.
    Summed over points, it has the scale of a summed squared error, so a weight on it reads like
    the weight of the `sum` reduction of `mse` (roadmap decision D-6, ADR 0029). Pass it as
    `Trainer(reconstruction_loss_fn=computeChamferDistance)` with the default `mean` reduction,
    which leaves a scalar alone; the `sum` reduction would multiply it by the number of values
    again.

    Args:
        reconstruction: `(batch, points, dim)`, the decoder's output.
        target: `(batch, target_points, dim)`, the batch tensor. A row of `NaN` is a point that
            does not exist and is left out, and `target_points` may differ from `points`.

    Returns:
        A scalar tensor.

    Raises:
        ValueError: If the tensors are not batches of point sets of the same dimension, or a cloud
            of `target` has no point that is not `NaN`.
    """
    _checkClouds(reconstruction, target)
    to_target, to_reconstruction, valid = _computeNearestDistances(reconstruction, target)
    backward = torch.where(valid, to_reconstruction, torch.zeros_like(to_reconstruction))
    per_cloud: torch.Tensor = to_target.sum(dim=1) + backward.sum(dim=1)
    return per_cloud.mean()


def computeCloudScores(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Score reconstructed clouds against real ones, per point and in the units of the data.

    Args:
        reconstruction: `(batch, points, dim)`, the decoder's output.
        target: `(batch, target_points, dim)`, rows that are `NaN` are padding and are left out.
        threshold: Distance under which a point counts as found, in the units of the data.

    Returns:
        A dict with `chamfer` (the mean squared distance to the nearest point of the other cloud,
        in both directions, added, averaged over clouds, so it does not grow with the number of
        points), `precision` (the share of reconstructed points within `threshold` of a real
        point), `recall` (the share of real points within `threshold` of a reconstructed point),
        `f_score` (their harmonic mean, `0` if both are `0`) and `num_points` (the number of real
        points scored).

    Raises:
        ValueError: If the tensors are not batches of point sets of the same dimension, a cloud of
            `target` has no point that is not `NaN`, or `threshold` is not positive.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    _checkClouds(reconstruction, target)
    to_target, to_reconstruction, valid = _computeNearestDistances(reconstruction, target)
    num_valid = valid.sum(dim=1).to(to_target.dtype)
    forward_mean = to_target.mean(dim=1)
    backward_mean = (
        torch.where(valid, to_reconstruction, torch.zeros_like(to_reconstruction)).sum(dim=1)
        / num_valid
    )
    squared_threshold = threshold**2
    precision = float((to_target <= squared_threshold).to(to_target.dtype).mean())
    found = (to_reconstruction <= squared_threshold) & valid
    recall = float(found.sum() / valid.sum())
    f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "chamfer": float((forward_mean + backward_mean).mean()),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "num_points": float(valid.sum()),
    }
