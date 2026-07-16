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
        return torch.cat(latents, dim=-1)
