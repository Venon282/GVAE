"""InfoNCE contrastive distance between the posterior means (roadmap P6-3, ADR 0053)."""

from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn.functional as functional

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.registry import registerLatentDistance


@registerLatentDistance("info_nce")
class InfoNceDistance(AbstractLatentDistance):
    """Symmetric InfoNCE of the posterior means, one value per sample.

    Row `i` of the first space and row `i` of the second are a positive pair, and the other rows
    of the batch are the negatives. With `s_ij = cos(mu_a_i, mu_b_j) / temperature`, the value of
    sample `i` is the mean of two cross-entropies: row `i` of `s` against the label `i` (which
    sample of the second space matches this one) and column `i` of `s` against the label `i`
    (the other way round). It is the objective that finding I-9 found missing: it asks that a
    sample be closer to its own pair than to the other samples, not merely close.

    The variances are not used. The negatives are the other samples of the same call, so with
    gradient accumulation each micro-batch has its own, and a batch of one sample has none (a
    `ValueError`). The minimum is above 0 (for identical, mutually orthogonal directions it is
    `log(1 + (batch - 1) * exp(-1 / temperature))`), and the value at chance is `log(batch)`.
    There is no sum over the latent dimensions, hence no `dimension_reduction`, and the scale is
    its own (ADR 0053).
    """

    is_zero_on_identical: ClassVar[bool] = False
    min_batch_size: ClassVar[int] = 2

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8) -> None:
        """Build the distance.

        Args:
            temperature: The divisor of the cosine similarities. Smaller values make the
                softmax sharper and the negatives count more.
            eps: The floor of a norm in the normalization.

        Raises:
            ValueError: If `temperature` or `eps` is not positive.
        """
        super().__init__()
        if not temperature > 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if not eps > 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
        """Per-sample symmetric InfoNCE between the two latent spaces.

        Args:
            posteriors: The `(mu, logvar)` of the two latent spaces. `logvar` is not used.

        Returns:
            Shape `(batch,)`, never negative.
        """
        (mu_a, _), (mu_b, _) = posteriors
        direction_a = functional.normalize(mu_a, dim=-1, eps=self.eps)
        direction_b = functional.normalize(mu_b, dim=-1, eps=self.eps)
        logits = direction_a @ direction_b.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        first_against_second = functional.cross_entropy(logits, labels, reduction="none")
        second_against_first = functional.cross_entropy(logits.T, labels, reduction="none")
        return 0.5 * (first_against_second + second_against_first)
