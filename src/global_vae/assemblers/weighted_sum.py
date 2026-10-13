"""Learned weighted-sum assembler (spec §2.2)."""

from collections.abc import Sequence
from typing import ClassVar

import torch
from torch import nn

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import registerAssembler


@registerAssembler("weighted_sum")
class WeightedSumAssembler(AbstractAssembler):
    """Sums latent vectors elementwise, each scaled by a learned per-space weight.

    The dimension rule is the one of `sum`: all input latent spaces share one
    dimensionality (`requires_matching_dims`), and the output has that width.
    The only parameters are one scalar per latent space, so the strategy adds
    `len(input_dims)` values to the model.

    Two parametrisations, picked by `normalize`:

    - `normalize=True` (default): the stored values are logits and the weights
      are their softmax, so they are positive and sum to 1 over the latent
      spaces present in the pass. Every logit starts at 0, so the assembler
      starts as `average`. When a modality is left out (spec §5), the weights
      of the remaining spaces are renormalised, so the decoder input keeps the
      scale it was trained with.
    - `normalize=False`: the stored values are the weights themselves, used as
      they are, so they may be negative or larger than 1. Every weight starts
      at 1, so the assembler starts as `sum`. With a missing space the output
      is the weighted sum of the remaining ones, not renormalised.

    Attributes:
        normalize: Whether the weights are a softmax over the stored values.
        raw_weights: Learned parameter of shape `(len(input_dims),)`, one
            value per latent space, in `input_dims` order.
    """

    requires_matching_dims: ClassVar[bool] = True

    def __init__(self, input_dims: Sequence[int], normalize: bool = True) -> None:
        """Build the assembler and its per-space weights.

        Args:
            input_dims: Width of each latent vector. All must be equal.
            normalize: Use the softmax of the stored values as weights (start
                at `average`) instead of the stored values themselves (start at
                `sum`). See the class docstring.

        Raises:
            ValueError: If `input_dims` is empty, holds anything but positive
                integers, or has differing widths.
        """
        super().__init__(input_dims)
        self.normalize = normalize
        count = len(self.input_dims)
        self.raw_weights = nn.Parameter(torch.zeros(count) if normalize else torch.ones(count))

    @property
    def output_dim(self) -> int:
        """Width of the weighted sum, the shared width of the inputs.

        Returns:
            `input_dims[0]`, which every entry of `input_dims` equals.
        """
        return self.input_dims[0]

    def computeWeights(self, present: Sequence[int] | None = None) -> torch.Tensor:
        """The weights `forward` applies to the latent spaces of one pass.

        Args:
            present: Position in `input_dims` of each latent space taking part,
                or `None` for all of them. With `normalize`, the softmax runs
                over these spaces only.

        Returns:
            A tensor of shape `(len(present),)`, one weight per latent space
            in the order of `present`, still attached to `raw_weights` for
            autograd.
        """
        positions = list(range(len(self.input_dims))) if present is None else list(present)
        raw = self.raw_weights[positions]
        return torch.softmax(raw, dim=0) if self.normalize else raw

    def forward(
        self, latents: list[torch.Tensor], present: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Weight each latent vector and add them up.

        Args:
            latents: Already-sampled latent tensors, each of shape `(batch, dim)`,
                all sharing the same `dim`.
            present: Position of each tensor in `input_dims`, or `None` when
                `latents` holds every latent space. It selects which weights
                apply.

        Returns:
            The weighted sum, shape `(batch, dim)`.

        Raises:
            ValueError: If `present` is inconsistent with `latents`, or a tensor
                does not have the width `input_dims` gives.
        """
        positions = self._resolvePresent(latents, present)
        weights = self.computeWeights(positions)
        return (weights.view(-1, 1, 1) * torch.stack(latents, dim=0)).sum(dim=0)

    @property
    def handlesMissingLatents(self) -> bool:
        """The weighted sum is defined for any non-empty subset of the latent spaces.

        Returns:
            `True`: with some latent spaces missing, the decoder receives the
            weighted sum of the ones that are available.
        """
        return True
