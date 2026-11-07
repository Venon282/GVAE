"""Smooth L1 (Huber) loss (roadmap P5-3, ADR 0048)."""

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss


@registerReconstructionLoss("smooth_l1")
class SmoothL1Loss(AbstractReconstructionLoss):
    """`torch.nn.functional.smooth_l1_loss`: quadratic near zero, linear beyond `beta`.

    With the default `beta` it is what the name gave before the registry. Same shape for the
    reconstruction and the target, no `NaN` handling, one term per value of a sample.
    """

    def __init__(self, beta: float = 1.0) -> None:
        """Build the loss.

        Args:
            beta: The error at which the loss goes from quadratic to linear, as in
                `torch.nn.functional.smooth_l1_loss`. Default `1.0`, its default.

        Raises:
            ValueError: If `beta` is negative.
        """
        super().__init__()
        if beta < 0:
            raise ValueError(f"beta must be at least 0, got {beta}.")
        self.beta = float(beta)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: The decoder's output, `(batch, ...)`.
            target: The batch tensor, the same shape.

        Returns:
            A scalar tensor, the mean over every value.
        """
        return F.smooth_l1_loss(reconstruction, target, beta=self.beta)
