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
        """Sum the latent vectors elementwise.

        Args:
            latents: Already-sampled latent tensors, each of shape `(batch, dim)`,
                all sharing the same `dim`.

        Returns:
            The elementwise sum, shape `(batch, dim)`.
        """
        return torch.stack(latents, dim=0).sum(dim=0)

    @property
    def handlesMissingLatents(self) -> bool:
        """The sum is defined for any non-empty list of latent vectors.

        Returns:
            `True`: with some latent spaces missing, the decoder receives the
            sum of the ones that are available.
        """
        return True
