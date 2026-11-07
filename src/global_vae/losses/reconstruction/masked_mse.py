"""Squared error over the observed values of a series (roadmap P5-3, ADR 0048)."""

import math
from typing import ClassVar

import torch

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss
from global_vae.utils.series_blocks import computeMaskedSquaredError


@registerReconstructionLoss("masked_mse")
class MaskedMseLoss(AbstractReconstructionLoss):
    """Squared error that leaves out what was not observed (a `NaN` in the target).

    It wraps `computeMaskedSquaredError`, the sum of the squared errors over the observed values of
    a series averaged over the batch, divided by the number of values of a sample, so that it is a
    mean over the terms like every loss here. With `reduction: sum` it is exactly
    `computeMaskedSquaredError`, and on a series with no gap it is the `mse` of the same data. With
    `mean` an unobserved value counts as a term with no error, which is the reading that keeps the
    `sum` exact (ADR 0048). The reconstruction receives no gradient where the target is `NaN`.
    """

    reads_nan_as_unobserved: ClassVar[bool] = True

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: The decoder's output, `(batch, ...)`.
            target: The batch tensor, the same shape, `NaN` where unobserved.

        Returns:
            A scalar tensor.

        Raises:
            ValueError: If the two shapes differ, or `reconstruction` has no batch axis.
        """
        loss: torch.Tensor = computeMaskedSquaredError(reconstruction, target) / math.prod(
            reconstruction.shape[1:]
        )
        return loss
