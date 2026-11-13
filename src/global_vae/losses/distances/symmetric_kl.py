"""Symmetric KL divergence between two diagonal Gaussian posteriors (roadmap P6-3, ADR 0053)."""

from collections.abc import Sequence

import torch

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.numerics import (
    checkDimensionReduction,
    checkLogvarRange,
    clampLogvar,
    reduceOverDimensions,
)
from global_vae.losses.distances.registry import registerLatentDistance


@registerLatentDistance("symmetric_kl")
class SymmetricKlDistance(AbstractLatentDistance):
    """`KL(a || b) + KL(b || a)` between two diagonal Gaussians, in nats.

    The sum of the two directions (the Jeffreys divergence), not their average. The logarithms of
    the variances cancel, which leaves, per dimension,

        `1/2 * [(var_a + d^2) / var_b + (var_b + d^2) / var_a - 2]`, with `d = mu_a - mu_b`.

    It explodes when one variance is tiny, because it divides by it, so both log-variances are
    clamped to `[min_logvar, max_logvar]` first. A value outside that range has no gradient.
    For `N(0, 1)` and `N(1, 4)` (variances 1 and 4) it is 1.75.
    """

    def __init__(
        self,
        dimension_reduction: str = "sum",
        min_logvar: float = -10.0,
        max_logvar: float = 10.0,
    ) -> None:
        """Build the distance.

        Args:
            dimension_reduction: `"sum"` (default, the footing of the KL term, ADR 0029) or
                `"mean"`.
            min_logvar: Lower clamp of both log-variances.
            max_logvar: Upper clamp of both log-variances.

        Raises:
            ValueError: If `dimension_reduction` is neither `"sum"` nor `"mean"`, or
                `min_logvar` is not below `max_logvar`.
        """
        super().__init__()
        self.dimension_reduction = checkDimensionReduction(dimension_reduction)
        checkLogvarRange(min_logvar, max_logvar)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
        """Per-sample symmetric KL divergence between the two diagonal Gaussians.

        Args:
            posteriors: The `(mu, logvar)` of the two latent spaces.

        Returns:
            Shape `(batch,)`.
        """
        (mu_a, logvar_a), (mu_b, logvar_b) = posteriors
        var_a = clampLogvar(logvar_a, self.min_logvar, self.max_logvar).exp()
        var_b = clampLogvar(logvar_b, self.min_logvar, self.max_logvar).exp()
        squared_gap = (mu_a - mu_b).pow(2)
        per_dimension = 0.5 * ((var_a + squared_gap) / var_b + (var_b + squared_gap) / var_a - 2.0)
        return reduceOverDimensions(per_dimension, self.dimension_reduction)
