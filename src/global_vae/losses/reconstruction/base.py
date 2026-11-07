"""Abstract interface for reconstruction losses (roadmap P5-3, ADR 0048).

A reconstruction loss compares what a decoder produced with the batch tensor of the same name and
returns one scalar. It used to be a plain function picked from a fixed table of four names. It is
now a registered class, like the other pluggable components (spec §10, §12), so that a loss that
`torch.nn.functional` does not have (a cross-entropy over tokens, a Chamfer distance, a squared
error that skips what was not observed) can be named in a YAML file.

A loss is an `nn.Module`: one that owns parameters (a learned noise scale, roadmap P7-1) is held by
the `Trainer`, in its optimizer and its checkpoint (`Trainer.reconstruction_losses`).

Three declarations say what the aggregate and the evaluation may assume. They are read from the
class, through the registry, so nothing has to build a loss to ask (the pattern of
`requires_matching_dims` on the assemblers):

- `requires_matching_shapes`: the reconstruction and the target must have the same shape;
- `reads_nan_as_unobserved`: a `NaN` in the target is a value that was not observed;
- `countTerms`: the number of terms one sample's value is the mean of, which the `sum` reduction
  multiplies by (`docs/adr/0029-loss-scale-convention.md`).

Concrete subclasses must self-register via `@registerReconstructionLoss(name)` (see
`registry.py`).
"""

import math
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import nn


class AbstractReconstructionLoss(nn.Module, ABC):
    """Base class for every reconstruction loss.

    `forward` returns a **mean over the terms** of a sample, averaged over the batch (ADR 0029), so
    that `reduction: sum` means the same thing for every loss: the per-sample sum, averaged over
    the batch.

    Attributes:
        requires_matching_shapes: Class-level flag, `True` by default. Whether the reconstruction
            and the target must have the same shape. `computeTotalReconstructionLoss` checks it
            before calling the loss (the check of roadmap item P0-4(g), whose message names the
            two shapes), so a loss that sets it to `False` (token ids against scores, point sets
            of different sizes) checks what it needs itself.
        reads_nan_as_unobserved: Class-level flag, `False` by default. Whether a `NaN` in the
            target means "not observed" (a gap in a series, a point that does not exist) and adds
            nothing to the loss. `evaluate()` reads it to choose metrics that ignore the same
            values (ADR 0048).
        default_metrics: Class-level, `None` by default. The registered metric names
            (`evaluation.metrics`) that describe this loss, in the order to report them, for a
            loss whose modality the five plain metrics do not describe (a point cloud).
            `evaluate()` uses them for a modality without an explicit metric set.
        decoder_arguments: Class-level, empty by default. Names of constructor arguments that the
            config layer fills in from the attribute of the same name of the decoder, when the
            YAML does not give them (`tabular_mixed` reads `categorical_sizes` that way).
    """

    requires_matching_shapes: ClassVar[bool] = True
    reads_nan_as_unobserved: ClassVar[bool] = False
    default_metrics: ClassVar[tuple[str, ...] | None] = None
    decoder_arguments: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of one batch.

        Args:
            reconstruction: The decoder's output for the batch, `(batch, ...)`.
            target: The batch tensor of the same name.

        Returns:
            A scalar tensor: the mean over the terms of a sample (`countTerms`), averaged over the
            batch.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    def countTerms(self, reconstruction: torch.Tensor, target: torch.Tensor) -> int:
        """Return the number of terms one sample's value of this loss is the mean of.

        The `sum` reduction multiplies the loss by it. The default is the number of values of a
        sample (the product of the axes after the batch), right for an element-wise loss. A loss
        that is a mean over something else says so: the tokens of a text, the columns of a table,
        `1` for a set loss that already sums over its points.

        Args:
            reconstruction: The decoder's output for the batch.
            target: The batch tensor of the same name.

        Returns:
            A positive integer.
        """
        return math.prod(reconstruction.shape[1:])
