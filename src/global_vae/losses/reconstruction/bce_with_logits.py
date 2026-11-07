"""Binary cross-entropy on raw scores (roadmap P5-3, ADR 0048)."""

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss


@registerReconstructionLoss("bce_with_logits")
class BceWithLogitsLoss(AbstractReconstructionLoss):
    """`torch.nn.functional.binary_cross_entropy_with_logits`, unchanged: a mean over every value.

    The reconstruction is the raw score of a decoder built without a sigmoid, and the sigmoid is
    taken inside the loss, which is more stable than a sigmoid followed by `bce`.

    Same shape for the reconstruction and the target, no `NaN` handling, and one term per value of
    a sample, so `reduction: sum` multiplies it by the number of values of a sample (ADR 0029).
    """

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: The decoder's output, `(batch, ...)`.
            target: The batch tensor, the same shape.

        Returns:
            A scalar tensor, what `torch.nn.functional.binary_cross_entropy_with_logits` returns.
        """
        return F.binary_cross_entropy_with_logits(reconstruction, target)
