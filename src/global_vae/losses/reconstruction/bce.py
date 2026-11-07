"""Binary cross-entropy on probabilities (roadmap P5-3, ADR 0048)."""

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss


@registerReconstructionLoss("bce")
class BceLoss(AbstractReconstructionLoss):
    """`torch.nn.functional.binary_cross_entropy`, unchanged: a mean over every value of the batch.

    The reconstruction must be a probability in `[0, 1]` (a decoder with a sigmoid output) and the
    target a value in `[0, 1]`. For raw scores use `bce_with_logits`.

    Same shape for the reconstruction and the target, no `NaN` handling, and one term per value of
    a sample, so `reduction: sum` multiplies it by the number of values of a sample (ADR 0029).
    """

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: The decoder's output, `(batch, ...)`.
            target: The batch tensor, the same shape.

        Returns:
            A scalar tensor, what `torch.nn.functional.binary_cross_entropy` returns.
        """
        return F.binary_cross_entropy(reconstruction, target)
