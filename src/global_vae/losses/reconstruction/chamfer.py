"""Chamfer distance between two sets of points (roadmap P5-3, ADR 0048)."""

from typing import ClassVar

import torch

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss
from global_vae.utils.point_cloud_metrics import computeChamferDistance


@registerReconstructionLoss("chamfer")
class ChamferLoss(AbstractReconstructionLoss):
    """The Chamfer distance of a batch of point clouds, as `computeChamferDistance` returns it.

    For each cloud, the squared distance from every reconstructed point to its nearest real point
    plus the squared distance from every real point to its nearest reconstructed point, summed over
    the points and averaged over the clouds. A cloud reproduced as a set, in any order, scores `0`.
    The two clouds may have different numbers of points, and a row of `NaN` in the target is a
    point that does not exist.

    It is already a sum over points, so it counts as a single term: `mean` and `sum` give the same
    number, and its weight reads like the weight of a summed squared error.
    """

    requires_matching_shapes: ClassVar[bool] = False
    reads_nan_as_unobserved: ClassVar[bool] = True
    default_metrics: ClassVar[tuple[str, ...] | None] = (
        "chamfer",
        "precision",
        "recall",
        "f_score",
    )

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: `(batch, points, dim)`, the decoder's output.
            target: `(batch, target_points, dim)`, the batch tensor; `target_points` may differ
                from `points`, and a row of `NaN` is a point that does not exist.

        Returns:
            A scalar tensor.

        Raises:
            ValueError: If the tensors are not batches of point sets of the same dimension, or a
                cloud of `target` has no point that is not `NaN`.
        """
        return computeChamferDistance(reconstruction, target)

    def countTerms(self, reconstruction: torch.Tensor, target: torch.Tensor) -> int:
        """Return `1`: the loss is a sum over points already.

        Args:
            reconstruction: The decoder's output.
            target: The batch tensor.

        Returns:
            `1`.
        """
        return 1
