"""Squared distance between the posterior means (roadmap P6-3, ADR 0053)."""

from collections.abc import Sequence

import torch

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.numerics import checkDimensionReduction, reduceOverDimensions
from global_vae.losses.distances.registry import registerLatentDistance


@registerLatentDistance("mean_mse")
class MeanMseDistance(AbstractLatentDistance):
    """Squared Euclidean distance between the posterior means of two latent spaces.

    The simplest distance: `sum_d (mu_a - mu_b)^2` per sample, so with `dimension_reduction:
    mean` it is the mean squared error of the two means. It ignores the variances, which makes
    it the ablation of the distances that use them (`wasserstein2_diag`, `symmetric_kl`).
    """

    def __init__(self, dimension_reduction: str = "sum") -> None:
        """Build the distance.

        Args:
            dimension_reduction: How the squared gaps of the latent dimensions are combined:
                `"sum"` (default, the footing of the KL term, ADR 0029) or `"mean"`.

        Raises:
            ValueError: If `dimension_reduction` is neither `"sum"` nor `"mean"`.
        """
        super().__init__()
        self.dimension_reduction = checkDimensionReduction(dimension_reduction)

    def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
        """Per-sample squared distance between the two posterior means.

        Args:
            posteriors: The `(mu, logvar)` of the two latent spaces. `logvar` is not used.

        Returns:
            Shape `(batch,)`.
        """
        (mu_a, _), (mu_b, _) = posteriors
        return reduceOverDimensions((mu_a - mu_b).pow(2), self.dimension_reduction)
