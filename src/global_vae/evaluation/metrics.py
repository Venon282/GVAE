"""Reconstruction quality metrics (spec: "MSE at minimum").

Deliberately plain functions plus a lookup dict, not a registry (unlike encoders,
fusion, regularizers, beta schedules, or experiment loggers): those are mutually
exclusive strategies picked one at a time for a specific role. Evaluation metrics are
the opposite: a caller normally wants *all* of them computed together for a full
picture, not to pick exactly one. `DEFAULT_RECONSTRUCTION_METRICS` is a plain
`dict[str, MetricFn]`, matching the "shared dict, override or extend by passing your
own" pattern `losses.reconstruction.computeTotalReconstructionLoss` already uses for
`loss_fn`.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


def computeMse(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error.

    Args:
        reconstruction: Model output.
        target: Ground truth, same shape as `reconstruction`.

    Returns:
        Scalar MSE.
    """
    return F.mse_loss(reconstruction, target).item()


def computeRmse(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error (same units as the data, often more interpretable than MSE).

    Args:
        reconstruction: Model output.
        target: Ground truth, same shape as `reconstruction`.

    Returns:
        Scalar RMSE.
    """
    return F.mse_loss(reconstruction, target).sqrt().item()


def computeMae(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error (more robust to outliers than MSE/RMSE).

    Args:
        reconstruction: Model output.
        target: Ground truth, same shape as `reconstruction`.

    Returns:
        Scalar MAE.
    """
    return F.l1_loss(reconstruction, target).item()


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


DEFAULT_RECONSTRUCTION_METRICS: dict[str, MetricFn] = {
    "mse": computeMse,
    "rmse": computeRmse,
    "mae": computeMae,
    "r2": computeR2,
    "pearson_r": computePearsonR,
}
