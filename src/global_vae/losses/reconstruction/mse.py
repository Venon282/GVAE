"""Mean squared error (roadmap P5-3, ADR 0048)."""

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss


@registerReconstructionLoss("mse")
class MseLoss(AbstractReconstructionLoss):
    """`torch.nn.functional.mse_loss`, unchanged: a mean over every value of the batch.

    Equivalent, up to an additive constant, to the likelihood of a Gaussian decoder with a fixed
    variance, and the default of every modality that names no loss.

    Same shape for the reconstruction and the target, no `NaN` handling, and one term per value of
    a sample, so `reduction: sum` multiplies it by the number of values of a sample (ADR 0029).
    """

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: The decoder's output, `(batch, ...)`.
            target: The batch tensor, the same shape.

        Returns:
            A scalar tensor, what `torch.nn.functional.mse_loss` returns.
        """
        return F.mse_loss(reconstruction, target)
