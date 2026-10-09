"""Concatenation assembler (spec §2.2)."""

import torch

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import registerAssembler


@registerAssembler("concat")
class ConcatAssembler(AbstractAssembler):
    """Concatenates latent vectors along the feature dimension.

    No dimensionality restriction across inputs.
    """

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate the latent vectors along the feature dimension.

        Args:
            latents: Already-sampled latent tensors, each of shape
                `(batch, dim_i)`; the `dim_i` may differ.

        Returns:
            The concatenation, shape `(batch, sum(dim_i))`.
        """
        return torch.cat(latents, dim=-1)
