"""Time series encoder (spec §6 and §10, roadmap P3-7 and P3-8, ADR 0042)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.series_blocks import ObservedPatchEmbed
from global_vae.utils.text_blocks import computeMaskedMean
from global_vae.utils.transformer_blocks import (
    SeriesPositions,
    buildPoolingHead,
    buildTransformerStack,
    computeNumPatches,
)

POOLING_CHOICES = ("mean", "cls")


@registerEncoder("time_series_encoder_v1")
class TimeSeriesEncoder(AbstractEncoder):
    """Small transformer over a multivariate series with gaps, robust to length and padding.

    A `NaN` in the input is a value that was not observed: a gap in the middle of a recording, or
    the padding of a short series at its end. The series is cut into non-overlapping patches of
    `patch_size` steps, and each patch, with its observed mask stacked as extra channels
    (`ObservedPatchEmbed`), is projected to a `d_model` token. Position vectors are added, and a
    stack of pre-norm transformer layers lets every token attend to every other, with the patches
    that hold no observed value skipped as attention keys. The mean of the tokens that hold data
    (or a learned `[CLS]` token, with `pooling="cls"`) goes through an optional MLP and two linear
    layers to give `(mu, logvar)`.

    So a series padded with `NaN` to a longer length gives the same `(mu, logvar)`, with the default
    sinusoidal positions, which do not depend on the length. A series that is unobserved
    everywhere stays finite. On a series with no gap the model is a patch transformer like
    `1d_transformer_encoder_v1`, whose mask channels are all ones. Attention is quadratic in the
    number of tokens, so a long series wants a larger `patch_size` and not more layers.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        patch_size: int = 8,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: str = "mean",
        positional_encoding: str = "sinusoidal",
        max_length: int | None = None,
        head_hidden_dims: Sequence[int] = (),
        modality_name: str = "time_series",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            in_channels: Number of channels of the series (`1` for a univariate one). Every
                channel is read at once, so co-measured channels are one multivariate series.
            patch_size: Number of time steps per patch, which is also the stride between patches.
            d_model: Token width.
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.
            pooling: `"mean"` (the mean of the patches that hold data) or `"cls"` (a learned token
                read after the last layer).
            positional_encoding: `"sinusoidal"` (no parameters, any length) or `"learned"` (one
                trained vector per patch, up to `max_length`).
            max_length: Longest series, in time steps, that learned positions cover. Required for
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
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if positional_encoding == "learned" and max_length is None:
            raise ValueError(
                "positional_encoding='learned' needs max_length, the longest series it must "
                "cover, because it holds one trained vector per patch."
            )
        self._latent_dim = latent_dim
        self._modality_name = modality_name
        self._pooling = pooling
        max_tokens = (
            computeNumPatches(max_length, patch_size)
            if positional_encoding == "learned" and max_length is not None
            else None
        )
        self.embed = ObservedPatchEmbed(in_channels, d_model, patch_size)
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
        """Encode a batch of series.

        Args:
            x: A float series, `(batch, length)` for one channel or `(batch, in_channels, length)`,
                with `NaN` where unobserved. `length` is at least `1`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `x` is not a float tensor of 2 or 3 dimensions, has a channel count other
                than `in_channels`, has no time step, or has more patches than a learned position
                table covers.
        """
        tokens, empty = self.embed(x)
        tokens = tokens + self.positions(tokens.shape[1], tokens).unsqueeze(0)
        key_mask = empty
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
            key_mask = torch.cat([torch.zeros_like(empty[:, :1]), empty], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=key_mask)
        if self._pooling == "cls":
            pooled = encoded[:, 0]
        else:
            pooled = computeMaskedMean(encoded, empty)
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
        """Shortest series this encoder accepts: `1` time step.

        A series shorter than a patch is padded to one, so no length collapses a feature map the way
        it can in a strided conv stack.

        Returns:
            `1`.
        """
        return 1
