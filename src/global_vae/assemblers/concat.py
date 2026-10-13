"""Concatenation assembler (spec §2.2)."""

from collections.abc import Sequence

import torch

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import registerAssembler


@registerAssembler("concat")
class ConcatAssembler(AbstractAssembler):
    """Concatenates latent vectors along the feature dimension.

    No dimensionality restriction across inputs. Parameter-free. It needs
    every latent space: with one missing, the output would be narrower than
    the decoder's `latent_dim`, so `handlesMissingLatents` stays `False`.
    """

    @property
    def output_dim(self) -> int:
        """Width of the concatenation.

        Returns:
            The sum of `input_dims`.
        """
        return sum(self.input_dims)

    def forward(
        self, latents: list[torch.Tensor], present: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Concatenate the latent vectors along the feature dimension.

        Args:
            latents: Already-sampled latent tensors, each of shape
                `(batch, dim_i)`; the `dim_i` may differ. All of them, in
                `input_dims` order.
            present: Must be `None` or name every position: `concat` has no
                way to fill the width of a missing latent space.

        Returns:
            The concatenation, shape `(batch, sum(dim_i))`.

        Raises:
            ValueError: If `latents` is not the complete list `input_dims`
                describes, or a tensor has the wrong width.
        """
        self._resolvePresent(latents, present)
        return torch.cat(latents, dim=-1)
