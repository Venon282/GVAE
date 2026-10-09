"""Elementwise-average assembler (spec §2.2)."""

import torch

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import registerAssembler


@registerAssembler("average")
class AverageAssembler(AbstractAssembler):
    """Averages latent vectors elementwise.

    Requires all input latent spaces to share the same dimensionality
    (enforced at construction time by `validateRoutingGraph`).
    """

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Average the latent vectors elementwise.

        Args:
            latents: Already-sampled latent tensors, each of shape `(batch, dim)`,
                all sharing the same `dim`.

        Returns:
            The elementwise mean, shape `(batch, dim)`.
        """
        return torch.stack(latents, dim=0).mean(dim=0)
