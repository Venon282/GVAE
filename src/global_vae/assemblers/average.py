"""Elementwise-average assembler (spec §2.2)."""

from collections.abc import Sequence
from typing import ClassVar

import torch

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import registerAssembler


@registerAssembler("average")
class AverageAssembler(AbstractAssembler):
    """Averages latent vectors elementwise.

    Requires all input latent spaces to share the same dimensionality
    (`requires_matching_dims`: checked by `validateRoutingGraph` and again by
    the constructor). Parameter-free.
    """

    requires_matching_dims: ClassVar[bool] = True

    @property
    def output_dim(self) -> int:
        """Width of the mean, the shared width of the inputs.

        Returns:
            `input_dims[0]`, which every entry of `input_dims` equals.
        """
        return self.input_dims[0]

    def forward(
        self, latents: list[torch.Tensor], present: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Average the latent vectors elementwise.

        Args:
            latents: Already-sampled latent tensors, each of shape `(batch, dim)`,
                all sharing the same `dim`.
            present: Position of each tensor in `input_dims`; the mean does not
                depend on it, so it only serves the width check.

        Returns:
            The elementwise mean, shape `(batch, dim)`.

        Raises:
            ValueError: If a tensor does not have the width `input_dims` gives.
        """
        self._resolvePresent(latents, present)
        return torch.stack(latents, dim=0).mean(dim=0)

    @property
    def handlesMissingLatents(self) -> bool:
        """The mean is defined for any non-empty list of latent vectors.

        Returns:
            `True`: with some latent spaces missing, the decoder receives the
            mean of the ones that are available.
        """
        return True
