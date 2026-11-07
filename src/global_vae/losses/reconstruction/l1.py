"""Mean absolute error (roadmap P5-3, ADR 0048)."""

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss


@registerReconstructionLoss("l1")
class L1Loss(AbstractReconstructionLoss):
    """`torch.nn.functional.l1_loss`, unchanged: a mean over every value of the batch.

    More robust to outliers than `mse`: the likelihood of a Laplace decoder of fixed scale.

    Same shape for the reconstruction and the target, no `NaN` handling, and one term per value of
    a sample, so `reduction: sum` multiplies it by the number of values of a sample (ADR 0029).
    """

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: The decoder's output, `(batch, ...)`.
            target: The batch tensor, the same shape.

        Returns:
            A scalar tensor, what `torch.nn.functional.l1_loss` returns.
        """
        return F.l1_loss(reconstruction, target)
