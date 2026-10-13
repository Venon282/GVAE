"""Cross-attention assembler (spec §2.2)."""

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import registerAssembler


@registerAssembler("attention")
class AttentionAssembler(AbstractAssembler):
    """Pools latent vectors of any widths with a learned cross-attention query.

    No fixed-dimensionality requirement: each latent space `i` gets its own
    linear projection `nn.Linear(input_dims[i], output_dim)` that turns its
    vector into one token of width `output_dim`. A single learned query then
    attends over those tokens (`nn.MultiheadAttention`, queries from the
    parameter, keys and values from the tokens), and the attended vector is
    the decoder input. The per-space projections, with their biases, tell the
    attention which space a token comes from.

    A missing latent space (spec §5) just contributes no token, so the
    attention is taken over the tokens that are present and the output keeps
    the same width: `handlesMissingLatents` is `True`.

    The parameters are the per-space projections, the query and the attention
    block's own projections. Initialisation: the query is drawn from a normal
    distribution of standard deviation `output_dim ** -0.5` (a unit-scale
    query), the projections use PyTorch's default `nn.Linear` initialisation
    and the attention block its default Xavier one.

    Attributes:
        num_heads: Number of attention heads.
        projections: One `nn.Linear(input_dims[i], output_dim)` per latent space.
        query: Learned query of shape `(1, 1, output_dim)`.
        attention: The `nn.MultiheadAttention` block (`batch_first`).
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        output_dim: int | None = None,
        num_heads: int = 1,
    ) -> None:
        """Build the per-space projections, the query and the attention block.

        Args:
            input_dims: Width of each latent vector. They may all differ.
            output_dim: Width of the tokens and of the output, which becomes the
                decoder's `latent_dim`. Must be a multiple of `num_heads`.
                Defaults to `max(input_dims)`.
            num_heads: Number of attention heads. Defaults to 1, which divides
                every width; raise it together with `output_dim`.

        Raises:
            ValueError: If `input_dims` is empty or holds anything but positive
                integers, `output_dim` or `num_heads` is not a positive integer,
                or `output_dim` is not a multiple of `num_heads`.
        """
        super().__init__(input_dims)
        width = max(self.input_dims) if output_dim is None else output_dim
        for label, value in (("output_dim", width), ("num_heads", num_heads)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{label}` must be a positive integer, got {value!r}.")
        if width % num_heads != 0:
            raise ValueError(
                f"`output_dim` ({width}) must be a multiple of `num_heads` ({num_heads}). "
                f"Set `output_dim` explicitly to a multiple of `num_heads`."
            )
        self._output_dim = width
        self.num_heads = num_heads
        self.projections = nn.ModuleList(nn.Linear(dim, width) for dim in self.input_dims)
        self.query = nn.Parameter(torch.randn(1, 1, width) * width**-0.5)
        self.attention = nn.MultiheadAttention(width, num_heads, batch_first=True)

    @property
    def output_dim(self) -> int:
        """Width of the attended vector.

        Returns:
            The `output_dim` given at construction, or `max(input_dims)` when
            none was given.
        """
        return self._output_dim

    def forward(
        self, latents: list[torch.Tensor], present: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Project each latent vector to a token and pool the tokens by attention.

        Args:
            latents: Already-sampled latent tensors, each of shape
                `(batch, dim_i)`; the `dim_i` may differ.
            present: Position of each tensor in `input_dims`, or `None` when
                `latents` holds every latent space. It selects which projection
                each tensor goes through.

        Returns:
            The attended vector, shape `(batch, output_dim)`.

        Raises:
            ValueError: If `present` is inconsistent with `latents`, or a tensor
                does not have the width `input_dims` gives.
        """
        positions = self._resolvePresent(latents, present)
        tokens = torch.stack(
            [
                cast(torch.Tensor, self.projections[position](latent))
                for position, latent in zip(positions, latents, strict=True)
            ],
            dim=1,
        )
        query = self.query.expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        return cast(torch.Tensor, attended).squeeze(1)

    @property
    def handlesMissingLatents(self) -> bool:
        """The attention is defined over any non-empty subset of the latent spaces.

        Returns:
            `True`: with some latent spaces missing, the decoder receives the
            attention over the ones that are available.
        """
        return True
