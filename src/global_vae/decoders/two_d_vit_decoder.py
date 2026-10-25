"""2D Vision Transformer decoder (spec §6, roadmap P3-5)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.transformer_blocks import (
    ImagePositions,
    PatchUnembed2d,
    buildTransformerStack,
    computeNumPatches,
    resolvePatchSize2d,
)


@registerDecoder("2d_vit_decoder_v1")
class TwoDVitDecoder(AbstractDecoder):
    """Vision Transformer reconstructing a fixed-size image from a latent vector.

    The grid has `ceil(height / patch_height)` by `ceil(width / patch_width)` patches, one token
    each: its position vector plus a linear projection of the latent vector, the same for every
    token, so the tokens differ only by position and the layers of self-attention are what turn
    one latent vector into an image. Each output token is projected to the pixels of its patch,
    the patches are tiled in row-major order, and the bottom and right edges are cropped to
    `output_shape`.

    Like every decoder here it reconstructs one fixed shape, never a resized one.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: Sequence[int],
        out_channels: int = 1,
        patch_size: int | Sequence[int] = 8,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        positional_encoding: str = "sinusoidal",
        modality_name: str = "image",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the latent vector it receives.
            output_shape: `(height, width)` of the image to reconstruct.
            out_channels: Number of channels of the image (`1` for a grayscale image).
            patch_size: Side of a square patch, or `(height, width)` of a rectangular one.
            d_model: Token width. A multiple of `4` for sinusoidal positions.
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.
            positional_encoding: `"sinusoidal"` or `"learned"`. The grid is fixed by
                `output_shape`, so a learned table has exactly one vector per patch of it.
            modality_name: Name of the modality this decoder reconstructs.

        Raises:
            ValueError: If `output_shape` is not two positive integers, `num_heads` does not
                divide `d_model`, `positional_encoding` is not a known choice, or another
                argument is out of range (the message names it).
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if len(output_shape) != 2:
            raise ValueError(f"output_shape must be (height, width), got {list(output_shape)}.")
        shape = (int(output_shape[0]), int(output_shape[1]))
        patch = resolvePatchSize2d(patch_size)
        self._modality_name = modality_name
        self.latent_projection = nn.Linear(latent_dim, d_model)
        grid_shape = (computeNumPatches(shape[0], patch[0]), computeNumPatches(shape[1], patch[1]))
        self.positions = ImagePositions(d_model, positional_encoding, grid_shape)
        self.transformer = buildTransformerStack(
            d_model, num_heads, num_layers, feedforward_dim, dropout, activation
        )
        self.unembed = PatchUnembed2d(d_model, patch, out_channels, shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of fixed-size images from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Reconstructed image. Shape `(batch, height, width)` for the default single-channel
            case (matching the encoder's plain-image input convention); `(batch, out_channels,
            height, width)` if `out_channels` was set above `1`.
        """
        grid_height, grid_width = self.unembed.grid_shape
        latent = self.latent_projection(z).unsqueeze(1)
        tokens = self.positions(grid_height, grid_width, latent).unsqueeze(0) + latent
        image: torch.Tensor = self.unembed(self.transformer(tokens))
        return image.squeeze(1) if image.shape[1] == 1 else image

    @property
    def modality_name(self) -> str:
        """Name of the modality this decoder reconstructs."""
        return self._modality_name
