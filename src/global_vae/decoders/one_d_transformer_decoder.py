"""1D transformer decoder (spec §6, roadmap P3-5)."""

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.transformer_blocks import (
    PatchUnembed1d,
    SeriesPositions,
    buildTransformerStack,
    computeNumPatches,
)


@registerDecoder("1d_transformer_decoder_v1")
class OneDTransformerDecoder(AbstractDecoder):
    """Small transformer reconstructing a fixed-length series from a latent vector.

    The series is `ceil(output_length / patch_size)` patches long. There is one token per patch:
    its position vector plus a linear projection of the latent vector, the same for every token,
    so the tokens differ only by position and the layers of self-attention are what turn one
    latent vector into a series. Each output token is projected to the values of its patch, the
    patches are laid side by side, and the last one is cropped to `output_length`.

    Like every decoder here it reconstructs one fixed length, never a resized one.
    """

    def __init__(
        self,
        latent_dim: int,
        output_length: int,
        out_channels: int = 1,
        patch_size: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        positional_encoding: str = "sinusoidal",
        modality_name: str = "vector",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the latent vector it receives.
            output_length: Length of the series to reconstruct.
            out_channels: Number of channels of the series (`1` for a plain scalar series).
            patch_size: Number of values per patch.
            d_model: Token width.
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.
            positional_encoding: `"sinusoidal"` or `"learned"`. The number of tokens is fixed by
                `output_length`, so a learned table has exactly that many vectors.
            modality_name: Name of the modality this decoder reconstructs.

        Raises:
            ValueError: If `num_heads` does not divide `d_model`, `positional_encoding` is not a
                known choice, or another argument is out of range (the message names it).
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        self._modality_name = modality_name
        num_tokens = computeNumPatches(output_length, patch_size)
        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.positions = SeriesPositions(d_model, positional_encoding, num_tokens)
        self.transformer = buildTransformerStack(
            d_model, num_heads, num_layers, feedforward_dim, dropout, activation
        )
        self.unembed = PatchUnembed1d(d_model, patch_size, out_channels, output_length)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of fixed-length series from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Reconstructed series. Shape `(batch, output_length)` for the default single-channel
            case (matching the encoder's plain-series input convention); `(batch, out_channels,
            output_length)` if `out_channels` was set above `1`.
        """
        num_tokens = self.unembed.num_tokens
        latent = self.latent_projection(z).unsqueeze(1)
        tokens = self.positions(num_tokens, latent).unsqueeze(0) + latent
        series: torch.Tensor = self.unembed(self.transformer(tokens))
        return series.squeeze(1) if series.shape[1] == 1 else series

    @property
    def modality_name(self) -> str:
        """Name of the modality this decoder reconstructs."""
        return self._modality_name
