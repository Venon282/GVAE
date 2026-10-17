"""Latent diagnostics: how many latent dimensions of a space carry information (roadmap P2-1).

Two numbers per latent dimension, both computed from the encoder's `(mu, logvar)` over the whole
evaluation set, so they do not depend on which regularizer trained the model:

- the **KL per dimension**: the mean over samples of that dimension's KL divergence to a standard
  normal, in nats. Summed over the dimensions it is the `kl_standard_normal` value of
  `EvaluationResults.regularization_metrics`. A dimension whose KL is close to zero has an
  approximate posterior equal to the prior, so it carries no information about the input.
- the **variance of the posterior mean** across samples. A dimension is an **active unit** when
  this variance is above a threshold (`DEFAULT_ACTIVE_UNIT_THRESHOLD`, 0.01), the criterion of
  Burda, Grosse and Salakhutdinov, "Importance Weighted Autoencoders" (2016). A dimension whose
  mean does not move with the input is not used by the decoder, whatever its `logvar` is.

The two views can disagree, and that is informative: a dimension with a KL above zero but a flat
mean spends its KL on a constant offset of the mean or on a variance that differs from one, not
on the input.

The point of reporting both is the loss scale convention (`docs/adr/0029-loss-scale-convention.md`):
a `beta` that is too large for the data shape shows up as a number of active units that is lower
than the latent size, which no loss curve says directly.
"""

import math
from dataclasses import dataclass, field

import torch

from global_vae.losses.regularizers.kl_standard_normal import computeKlPerDimension

DEFAULT_ACTIVE_UNIT_THRESHOLD = 0.01
"""Variance of the posterior mean above which a dimension counts as active (Burda et al., 2016)."""


@dataclass
class LatentDiagnostics:
    """The diagnostics of one latent space.

    Attributes:
        num_dimensions: The width of the latent space.
        active_threshold: The threshold on `mu_variance_per_dimension` that was used, kept in the
            report so that the JSON file says how `active_units` was obtained.
        active_units: How many dimensions have a posterior-mean variance above
            `active_threshold`. Equal to `num_dimensions` when every dimension is used.
        kl_per_dimension: Mean over samples of each dimension's KL to a standard normal, in nats,
            one value per dimension in dimension order.
        mu_variance_per_dimension: Variance over samples of each dimension's posterior mean (the
            population variance, so a single sample gives 0), one value per dimension.
    """

    num_dimensions: int
    active_threshold: float
    active_units: int
    kl_per_dimension: list[float] = field(default_factory=list)
    mu_variance_per_dimension: list[float] = field(default_factory=list)


def validateActiveUnitThreshold(threshold: float) -> None:
    """Check that an active-unit threshold is a finite, non-negative number.

    Args:
        threshold: The value to check.

    Raises:
        ValueError: If `threshold` is negative, infinite or NaN.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"active_unit_threshold must be finite and >= 0, got {threshold}.")


def computeLatentDiagnostics(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    active_threshold: float = DEFAULT_ACTIVE_UNIT_THRESHOLD,
) -> LatentDiagnostics:
    """Compute the per-dimension KL and the active units of one latent space.

    Args:
        mu: Posterior mean of every sample, shape `(N, dim)`.
        logvar: Posterior log-variance of every sample, shape `(N, dim)`.
        active_threshold: A dimension is active when the variance of its posterior mean over the
            `N` samples is strictly above this value. Defaults to 0.01.

    Returns:
        A `LatentDiagnostics`.

    Raises:
        ValueError: If `mu` is not two-dimensional, if `mu` and `logvar` differ in shape, if
            there is no sample or no dimension, or if `active_threshold` is not a finite number
            that is at least zero.
    """
    validateActiveUnitThreshold(active_threshold)
    if mu.ndim != 2:
        raise ValueError(f"mu must have shape (samples, dim), got {tuple(mu.shape)}.")
    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu and logvar must have the same shape, got {tuple(mu.shape)} and "
            f"{tuple(logvar.shape)}."
        )
    if mu.numel() == 0:
        raise ValueError("computeLatentDiagnostics received an empty mu/logvar.")

    kl_per_dimension = computeKlPerDimension(mu, logvar).mean(dim=0)
    mu_variance = mu.var(dim=0, unbiased=False)
    return LatentDiagnostics(
        num_dimensions=mu.shape[1],
        active_threshold=active_threshold,
        active_units=int((mu_variance > active_threshold).sum().item()),
        kl_per_dimension=[float(value) for value in kl_per_dimension.tolist()],
        mu_variance_per_dimension=[float(value) for value in mu_variance.tolist()],
    )
