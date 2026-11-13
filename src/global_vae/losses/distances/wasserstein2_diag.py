"""2-Wasserstein distance between two diagonal Gaussian posteriors (roadmap P6-3, ADR 0053)."""

from collections.abc import Sequence

import torch

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.numerics import (
    checkDimensionReduction,
    reduceOverDimensions,
    safeSqrt,
)
from global_vae.losses.distances.registry import registerLatentDistance


@registerLatentDistance("wasserstein2_diag")
class Wasserstein2DiagDistance(AbstractLatentDistance):
    """Closed-form 2-Wasserstein distance between `N(mu_a, diag(sigma_a^2))` and `N(mu_b, ...)`.

    For two diagonal Gaussians the squared 2-Wasserstein distance is the squared gap of the means
    plus the squared gap of the standard deviations, summed over the dimensions:

        `W2^2 = sum_d [(mu_a - mu_b)^2 + (sigma_a - sigma_b)^2]`, with `sigma = exp(logvar / 2)`.

    This is the distribution alignment loss of CADA-VAE. The published form is the root of that
    sum, whose derivative is infinite at 0, so the squared form is the default (roadmap D-13) and
    `squared: false` gives the root, computed so that its gradient stays finite on identical
    inputs. For `N(0, 1^2)` and `N(1, 2^2)` the distance is `sqrt(2)` and its square is 2.
    """

    def __init__(self, squared: bool = True, dimension_reduction: str = "sum") -> None:
        """Build the distance.

        Args:
            squared: `True` (default) returns `W2^2`; `False` returns `W2`, the square root of
                the value after the dimensions are combined.
            dimension_reduction: How the dimensions are combined: `"sum"` (default, the footing
                of the KL term, ADR 0029) or `"mean"`. With `squared: false` the root is taken
                after this reduction.

        Raises:
            ValueError: If `dimension_reduction` is neither `"sum"` nor `"mean"`.
        """
        super().__init__()
        self.squared = bool(squared)
        self.dimension_reduction = checkDimensionReduction(dimension_reduction)

    def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
        """Per-sample 2-Wasserstein distance between the two diagonal Gaussians.

        Args:
            posteriors: The `(mu, logvar)` of the two latent spaces.

        Returns:
            Shape `(batch,)`: `W2^2`, or `W2` when `squared` is `False`.
        """
        (mu_a, logvar_a), (mu_b, logvar_b) = posteriors
        sigma_gap = torch.exp(0.5 * logvar_a) - torch.exp(0.5 * logvar_b)
        value = reduceOverDimensions(
            (mu_a - mu_b).pow(2) + sigma_gap.pow(2), self.dimension_reduction
        )
        return value if self.squared else safeSqrt(value)
