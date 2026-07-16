"""Elementwise-sum assembler (spec §2.2)."""

import torch

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import registerAssembler


@registerAssembler("sum")
class SumAssembler(AbstractAssembler):
    """Sums latent vectors elementwise.

    Requires all input latent spaces to share the same dimensionality
    (enforced at construction time by `validateRoutingGraph`).
    """

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(latents, dim=0).sum(dim=0)
