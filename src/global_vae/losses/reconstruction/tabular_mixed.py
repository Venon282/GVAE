"""Squared error on numeric columns, cross-entropy on categorical ones (roadmap P5-3, ADR 0048)."""

from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn.functional as functional

from global_vae.losses.reconstruction.base import AbstractReconstructionLoss
from global_vae.losses.reconstruction.registry import registerReconstructionLoss

_MIN_PROBABILITY = 1e-12
"""A probability is clamped to this before its logarithm, so a block that puts no mass on the true
category gives a large finite loss and not an infinite one."""


@registerReconstructionLoss("tabular_mixed")
class TabularMixedLoss(AbstractReconstructionLoss):
    """The loss of a table with numeric and categorical columns, a row at a time.

    The layout is the one of a tabular decoder built with `categorical_sizes`: the numeric columns
    first, then one block per categorical column, each the one-hot encoding of that column. A
    numeric column adds its squared error, and a categorical block adds its cross-entropy against
    the one-hot target. It is the mean of these terms over the numeric columns and the blocks, so
    with `reduction: sum` it is the squared error of the numeric part plus the negative
    log-likelihood of the categorical part, the scale of the KL term.

    With no categorical block it is the `mse` of the table.
    """

    decoder_arguments: ClassVar[tuple[str, ...]] = ("categorical_sizes",)

    def __init__(self, categorical_sizes: Sequence[int] = (), probabilities: bool = True) -> None:
        """Build the loss.

        Args:
            categorical_sizes: The number of categories of each one-hot block, the last
                `sum(categorical_sizes)` columns, in order. The config layer fills it in from the
                decoder when the YAML does not give it. Empty means every column is numeric.
            probabilities: Whether the blocks of the reconstruction are probabilities, which is
                what a decoder built with `categorical_sizes` returns (a softmax inside each
                block). `False` for the raw scores of a decoder built without it, and the softmax
                is taken here.

        Raises:
            ValueError: If a size is below `2`.
        """
        super().__init__()
        sizes = tuple(int(size) for size in categorical_sizes)
        if any(size < 2 for size in sizes):
            raise ValueError(
                f"categorical_sizes must all be at least 2 (a column with one category carries "
                f"no information), got {list(sizes)}."
            )
        self.categorical_sizes = sizes
        self.probabilities = bool(probabilities)

    def _numericWidth(self, num_features: int) -> int:
        """Return the number of numeric columns of a row of `num_features` columns.

        Raises:
            ValueError: If the blocks are wider than the row.
        """
        numeric = num_features - sum(self.categorical_sizes)
        if numeric < 0:
            raise ValueError(
                f"categorical_sizes {list(self.categorical_sizes)} add up to "
                f"{sum(self.categorical_sizes)} one-hot columns, more than the "
                f"{num_features} columns of a row."
            )
        return numeric

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: `(batch, num_features)`: numeric columns, then the categorical blocks.
            target: The batch tensor, the same shape: numeric values, then one-hot blocks.

        Returns:
            A scalar tensor: the mean over the numeric columns and the blocks, averaged over the
            batch.

        Raises:
            ValueError: If the tensors are not `(batch, num_features)` of one shape, or the blocks
                are wider than the row.
        """
        if reconstruction.dim() != 2 or reconstruction.shape != target.shape:
            raise ValueError(
                "tabular_mixed needs a reconstruction and a target of one shape (batch, "
                f"num_features), got {tuple(reconstruction.shape)} and {tuple(target.shape)}."
            )
        batch, num_features = reconstruction.shape
        numeric = self._numericWidth(num_features)
        total = (reconstruction[:, :numeric] - target[:, :numeric]).pow(2).sum()
        start = numeric
        for size in self.categorical_sizes:
            block = reconstruction[:, start : start + size]
            log_probabilities = (
                block.clamp_min(_MIN_PROBABILITY).log()
                if self.probabilities
                else functional.log_softmax(block, dim=1)
            )
            total = total - (target[:, start : start + size] * log_probabilities).sum()
            start += size
        loss: torch.Tensor = total / (batch * (numeric + len(self.categorical_sizes)))
        return loss

    def countTerms(self, reconstruction: torch.Tensor, target: torch.Tensor) -> int:
        """Return the number of numeric columns plus the number of categorical blocks.

        Args:
            reconstruction: `(batch, num_features)`.
            target: The batch tensor.

        Returns:
            One term per numeric column and one per block.

        Raises:
            ValueError: If the blocks are wider than the row.
        """
        return self._numericWidth(reconstruction.shape[1]) + len(self.categorical_sizes)
