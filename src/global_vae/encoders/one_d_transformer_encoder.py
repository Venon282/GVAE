"""1D transformer encoder (spec §6, roadmap P3-5)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.transformer_blocks import (
    PatchEmbed1d,
    SeriesPositions,
    buildPoolingHead,
    buildTransformerStack,
    computeNumPatches,
)

POOLING_CHOICES = ("cls", "mean")


@registerEncoder("1d_transformer_encoder_v1")
class OneDTransformerEncoder(AbstractEncoder):
    """Small transformer over a 1D series, robust to variable input length.

    The series is cut into non-overlapping patches of `patch_size` values, each projected to a
    `d_model` token (`PatchEmbed1d`). Position vectors are added, a learned `[CLS]` token is put
    in front (with `pooling="cls"`), and a stack of pre-norm transformer layers lets every token
    attend to every other. The `[CLS]` token (or the mean of the tokens, with `pooling="mean"`)
    goes through an optional MLP and two linear layers to give `(mu, logvar)`.

    With the default sinusoidal positions the encoder has no length limit, as the CNN encoders
    have none through adaptive pooling: series of any length give the same output shape. A length
    that is not a multiple of `patch_size` is padded at the end by repeating the last value.
    Attention is quadratic in the number of tokens, so a long series wants a larger `patch_size`
    and not more layers.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        patch_size: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: str = "cls",
        positional_encoding: str = "sinusoidal",
        max_length: int | None = None,
        head_hidden_dims: Sequence[int] = (),
        modality_name: str = "vector",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            in_channels: Number of input channels (`1` for a plain scalar series).
            patch_size: Number of values per patch, which is also the stride between patches.
            d_model: Token width.
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.
            pooling: `"cls"` (a learned token read after the last layer) or `"mean"` (the mean
                of the patch tokens).
            positional_encoding: `"sinusoidal"` (no parameters, any length) or `"learned"` (one
                trained vector per patch, up to `max_length`).
            max_length: Longest series, in values, that learned positions cover. Required for
                `positional_encoding="learned"` and ignored otherwise.
            head_hidden_dims: Hidden widths of an optional MLP between the pooled token and the
                `mu` and `logvar` layers. Empty (default) goes straight to them.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If `pooling` is not a known choice, `num_heads` does not divide
                `d_model`, a learned position table has no `max_length`, or another argument is
                out of range (the message names it).
        """
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"Unknown pooling '{pooling}'. Expected one of {POOLING_CHOICES}.")
        self._latent_dim = latent_dim
        self._modality_name = modality_name
        self._in_channels = in_channels
        self._pooling = pooling

        if positional_encoding == "learned" and max_length is None:
            raise ValueError(
                "positional_encoding='learned' needs max_length, the longest series it must "
                "cover, because it holds one trained vector per patch."
            )
        max_tokens = (
            computeNumPatches(max_length, patch_size)
            if positional_encoding == "learned" and max_length is not None
            else None
        )
        self.embed = PatchEmbed1d(in_channels, d_model, patch_size)
        self.positions = SeriesPositions(d_model, positional_encoding, max_tokens)
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
        """Encode a batch of 1D series.

        Args:
            x: Raw series, shape `(batch, length)` or `(batch, in_channels, length)`. A 2D input
                is treated as `(batch, length)` and given an explicit channel dimension of `1`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `x` is not 2D or 3D, has a channel count other than `in_channels`, or
                has more patches than a learned position table covers.
        """
        series = x.unsqueeze(1) if x.dim() == 2 else x
        if series.dim() != 3:
            raise ValueError(
                f"OneDTransformerEncoder expects (batch, length) or (batch, channels, length), "
                f"got shape {tuple(x.shape)}."
            )
        if series.shape[1] != self._in_channels:
            raise ValueError(
                f"OneDTransformerEncoder was built for in_channels={self._in_channels}, got "
                f"{series.shape[1]} channel(s) in shape {tuple(x.shape)}."
            )
        tokens = self.embed(series)
        tokens = tokens + self.positions(tokens.shape[1], tokens).unsqueeze(0)
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
        """Shortest input length this encoder accepts: `1`.

        A series shorter than a patch is padded to one, so no length collapses a feature map
        the way it can in a strided conv stack.

        Returns:
            `1`.
        """
        return 1
