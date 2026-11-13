"""Cosine distance between the posterior means (roadmap P6-3, ADR 0053)."""

from collections.abc import Sequence

import torch
import torch.nn.functional as functional

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.registry import registerLatentDistance


@registerLatentDistance("cosine")
class CosineDistance(AbstractLatentDistance):
    """`1 - cos(mu_a, mu_b)` per sample, in `[0, 2]`.

    It compares the direction of the posterior means, so it is blind to their scale and to the
    variances. It has no sum over the latent dimensions, hence no `dimension_reduction`, and its
    scale is its own: it is not on the footing of the KL term (ADR 0053).

    A zero mean has no direction. It counts as orthogonal to everything, distance 1, so two
    identical zero means are not at distance 0. The norms are floored at `eps`, so that the
    gradient is finite there (and large, the usual convention).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Build the distance.

        Args:
            eps: The floor of a norm in the normalization.

        Raises:
            ValueError: If `eps` is not positive.
        """
        super().__init__()
        if not eps > 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = float(eps)

    def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
        """Per-sample cosine distance between the two posterior means.

        Args:
            posteriors: The `(mu, logvar)` of the two latent spaces. `logvar` is not used.

        Returns:
            Shape `(batch,)`, between 0 and 2.
        """
        (mu_a, _), (mu_b, _) = posteriors
        direction_a = functional.normalize(mu_a, dim=-1, eps=self.eps)
        direction_b = functional.normalize(mu_b, dim=-1, eps=self.eps)
        return 1.0 - (direction_a * direction_b).sum(dim=-1)
