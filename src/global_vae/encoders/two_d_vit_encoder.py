"""2D Vision Transformer encoder (spec §6, roadmap P3-5)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.transformer_blocks import (
    ImagePositions,
    PatchEmbed2d,
    buildPoolingHead,
    buildTransformerStack,
    computeNumPatches,
    resolvePatchSize2d,
)

POOLING_CHOICES = ("cls", "mean")


@registerEncoder("2d_vit_encoder_v1")
class TwoDVitEncoder(AbstractEncoder):
    """Vision Transformer over an image, robust to variable input size.

    The image is cut into non-overlapping patches of `patch_size` pixels, each projected to a
    `d_model` token (`PatchEmbed2d`). Row-major position vectors are added, a learned `[CLS]`
    token is put in front (with `pooling="cls"`), and a stack of pre-norm transformer layers lets
    every patch attend to every other. The `[CLS]` token (or the mean of the patch tokens, with
    `pooling="mean"`) goes through an optional MLP and two linear layers to give `(mu, logvar)`.

    With the default sinusoidal positions the encoder has no size limit, as the CNN encoders have
    none through global pooling: images of any height and width give the same output shape. A
    side that is not a multiple of the patch side is padded at the bottom or right by repeating
    the edge pixels. Attention is quadratic in the number of patches, so a large image wants a
    larger `patch_size`.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 8,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: str = "cls",
        positional_encoding: str = "sinusoidal",
        image_size: Sequence[int] | None = None,
        head_hidden_dims: Sequence[int] = (),
        modality_name: str = "image",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            in_channels: Number of input channels (`1` for a grayscale image).
            patch_size: Side of a square patch, or `(height, width)` of a rectangular one.
            d_model: Token width. A multiple of `4` for sinusoidal positions (half of the vector
                encodes the row and half the column).
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.
            pooling: `"cls"` (a learned token read after the last layer) or `"mean"` (the mean
                of the patch tokens).
            positional_encoding: `"sinusoidal"` (no parameters, any size) or `"learned"` (one
                trained vector per patch of `image_size`).
            image_size: `(height, width)` of the images a learned position table is built for.
                Required for `positional_encoding="learned"`, which then refuses any image whose
                grid of patches differs; ignored otherwise.
            head_hidden_dims: Hidden widths of an optional MLP between the pooled token and the
                `mu` and `logvar` layers. Empty (default) goes straight to them.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If `pooling` is not a known choice, `num_heads` does not divide
                `d_model`, learned positions have no valid `image_size`, or another argument is
                out of range (the message names it).
        """
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"Unknown pooling '{pooling}'. Expected one of {POOLING_CHOICES}.")
        self._latent_dim = latent_dim
        self._modality_name = modality_name
        self._in_channels = in_channels
        self._pooling = pooling

        patch = resolvePatchSize2d(patch_size)
        grid_shape: tuple[int, int] | None = None
        if positional_encoding == "learned":
            if image_size is None or len(image_size) != 2:
                raise ValueError(
                    "positional_encoding='learned' needs image_size=(height, width), because it "
                    f"holds one trained vector per patch; got {image_size!r}."
                )
            grid_shape = (
                computeNumPatches(int(image_size[0]), patch[0]),
                computeNumPatches(int(image_size[1]), patch[1]),
            )
        self.embed = PatchEmbed2d(in_channels, d_model, patch)
        self.positions = ImagePositions(d_model, positional_encoding, grid_shape)
        self.cls_token: nn.Parameter | None = None
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.transformer = buildTransformerStack(
            d_model, num_heads, num_layers, feedforward_dim, dropout, activation
        )
        self.head, self.to_mu, self.to_logvar = buildPoolingHead(
            d_model, head_hidden_dims, latent_dim
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images.

        Args:
            x: Raw images, shape `(batch, height, width)` or `(batch, in_channels, height,
                width)`. A 3D input is treated as `(batch, height, width)` and given an explicit
                channel dimension of `1`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `x` is not 3D or 4D, has a channel count other than `in_channels`, or
                has a grid of patches that learned positions were not built for.
        """
        images = x.unsqueeze(1) if x.dim() == 3 else x
        if images.dim() != 4:
            raise ValueError(
                f"TwoDVitEncoder expects (batch, height, width) or (batch, channels, height, "
                f"width), got shape {tuple(x.shape)}."
            )
        if images.shape[1] != self._in_channels:
            raise ValueError(
                f"TwoDVitEncoder was built for in_channels={self._in_channels}, got "
                f"{images.shape[1]} channel(s) in shape {tuple(x.shape)}."
            )
        grid_height, grid_width = self.embed.gridShape(images.shape[-2], images.shape[-1])
        tokens = self.embed(images)
        tokens = tokens + self.positions(grid_height, grid_width, tokens).unsqueeze(0)
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        encoded = self.transformer(tokens)
        pooled = encoded[:, 0] if self._pooling == "cls" else encoded.mean(dim=1)
        pooled = self.head(pooled)
        mu: torch.Tensor = self.to_mu(pooled)
        logvar: torch.Tensor = self.to_logvar(pooled)
        return mu, logvar

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the `(mu, logvar)` output."""
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        """Name of the modality this encoder handles."""
        return self._modality_name

    @property
    def minimal_input_length(self) -> int:
        """Shortest side an input image accepts: `1`.

        An image smaller than a patch is padded to one, so no side collapses a feature map the
        way it can in a strided conv stack.

        Returns:
            `1`.
        """
        return 1
