"""Shared transformer encoder: one token stream for several modalities (spec §2.1, roadmap P3-5)."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from global_vae.encoders.registry import registerEncoder
from global_vae.encoders.shared_base import AbstractSharedEncoder
from global_vae.utils.transformer_blocks import (
    IMAGE_KIND,
    SERIES_KIND,
    ImagePositions,
    PatchEmbed1d,
    PatchEmbed2d,
    SeriesPositions,
    buildPoolingHead,
    buildTransformerStack,
    checkModalitySpecs,
    resolvePatchSize2d,
)

POOLING_CHOICES = ("cls", "mean")
_ALLOWED_KEYS = {
    SERIES_KIND: ("patch_size", "in_channels"),
    IMAGE_KIND: ("patch_size", "in_channels"),
}
_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {SERIES_KIND: (), IMAGE_KIND: ()}
_DEFAULT_SERIES_PATCH = 16
_DEFAULT_IMAGE_PATCH = 8


@registerEncoder("shared_transformer_encoder_v1")
class SharedTransformerEncoder(AbstractSharedEncoder):
    """One transformer over the patch tokens of every modality (the `E1` rows, ADR 0027).

    Each modality is cut into patches and projected to `d_model` tokens by its own patch embedding
    (`PatchEmbed1d` for a series, `PatchEmbed2d` for an image). A position vector (1D or 2D,
    sinusoidal) and a learned embedding of the modality are added to each token, so the shared
    layers can tell which modality and which place a token comes from. The tokens of all the
    modalities present are concatenated into one stream, a learned `[CLS]` token is put in front
    (with `pooling="cls"`), and one stack of pre-norm transformer layers lets every token attend
    to every other, across modalities. The `[CLS]` token (or the mean of all the tokens, with
    `pooling="mean"`) gives `(mu, logvar)`.

    A missing modality (spec §5) is simply a shorter stream: its tokens are not there, and
    attention and pooling are defined for any number of tokens, so any non-empty subset of the
    modalities is encoded with no mask and no placeholder. This is what a shared encoder is for:
    the cross-modal fusion happens inside the attention, with no fusion strategy.
    """

    def __init__(
        self,
        latent_dim: int,
        modalities: Mapping[str, Mapping[str, Any]],
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: str = "cls",
        head_hidden_dims: Sequence[int] = (),
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            modalities: Modality name -> its settings. The names are the batch keys, in the order
                of `input_modalities`. Each has `kind`, `"series"` or `"image"`, and optionally
                `patch_size` (an integer; for an image also `[height, width]`; default `16` for a
                series and `8` for an image) and `in_channels` (default `1`).
            d_model: Token width. A multiple of `4` if any modality is an image.
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.
            pooling: `"cls"` (a learned token read after the last layer) or `"mean"` (the mean
                of all the patch tokens present).
            head_hidden_dims: Hidden widths of an optional MLP between the pooled token and the
                `mu` and `logvar` layers. Empty (default) goes straight to them.

        Raises:
            ValueError: If `modalities` is empty or a modality's settings are invalid (no
                `kind`, a key its kind does not take), `pooling` is not a known choice,
                `num_heads` does not divide `d_model`, or another argument is out of range.
        """
        super().__init__()
        checkModalitySpecs(modalities, _ALLOWED_KEYS, _REQUIRED_KEYS, "SharedTransformerEncoder")
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"Unknown pooling '{pooling}'. Expected one of {POOLING_CHOICES}.")
        self._latent_dim = latent_dim
        self._pooling = pooling
        self._input_modalities = tuple(modalities)
        self._in_channels: dict[str, int] = {}

        embeds: dict[str, nn.Module] = {}
        positions: dict[str, nn.Module] = {}
        for name, spec in modalities.items():
            in_channels = int(spec.get("in_channels", 1))
            self._in_channels[name] = in_channels
            if spec["kind"] == SERIES_KIND:
                patch = int(spec.get("patch_size", _DEFAULT_SERIES_PATCH))
                embeds[name] = PatchEmbed1d(in_channels, d_model, patch)
                positions[name] = SeriesPositions(d_model)
            else:
                patch_2d = resolvePatchSize2d(spec.get("patch_size", _DEFAULT_IMAGE_PATCH))
                embeds[name] = PatchEmbed2d(in_channels, d_model, patch_2d)
                positions[name] = ImagePositions(d_model)
        self.embeds = nn.ModuleDict(embeds)
        self.positions = nn.ModuleDict(positions)
        self.modality_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(d_model) * 0.02) for name in modalities}
        )
        self.cls_token: nn.Parameter | None = None
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.transformer = buildTransformerStack(
            d_model, num_heads, num_layers, feedforward_dim, dropout, activation
        )
        self.head, self.to_mu, self.to_logvar = buildPoolingHead(
            d_model, head_hidden_dims, latent_dim
        )

    def _tokenize(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Turn one modality into tokens, with its positions and modality embedding added.

        Args:
            name: The modality, one of `input_modalities`.
            x: Its batch: `(batch, length)` or `(batch, channels, length)` for a series,
                `(batch, height, width)` or `(batch, channels, height, width)` for an image.

        Returns:
            `(batch, num_tokens, d_model)`.

        Raises:
            ValueError: If `x` has the wrong number of dimensions or channels for the modality.
        """
        embed = self.embeds[name]
        positions = self.positions[name]
        if isinstance(embed, PatchEmbed1d) and isinstance(positions, SeriesPositions):
            series = x.unsqueeze(1) if x.dim() == 2 else x
            self._checkInput(name, x, series, expected_dim=3)
            tokens = embed(series)
            tokens = tokens + positions(tokens.shape[1], tokens).unsqueeze(0)
        elif isinstance(embed, PatchEmbed2d) and isinstance(positions, ImagePositions):
            image = x.unsqueeze(1) if x.dim() == 3 else x
            self._checkInput(name, x, image, expected_dim=4)
            grid_height, grid_width = embed.gridShape(image.shape[-2], image.shape[-1])
            tokens = embed(image)
            tokens = tokens + positions(grid_height, grid_width, tokens).unsqueeze(0)
        else:  # pragma: no cover  (the constructor builds only the two pairs above)
            raise TypeError(f"Modality '{name}' has an unexpected tokenizer.")
        embedded: torch.Tensor = tokens + self.modality_embeddings[name]
        return embedded

    def _checkInput(
        self, name: str, original: torch.Tensor, shaped: torch.Tensor, *, expected_dim: int
    ) -> None:
        """Check the dimensions and channel count of one modality's batch.

        Args:
            name: The modality.
            original: The tensor as received, for the message.
            shaped: The tensor with its channel axis made explicit.
            expected_dim: Number of dimensions `shaped` must have.

        Raises:
            ValueError: If `shaped` has the wrong number of dimensions or channels.
        """
        if shaped.dim() != expected_dim:
            raise ValueError(
                f"SharedTransformerEncoder: modality '{name}' expects {expected_dim - 1} or "
                f"{expected_dim} dimensions, got shape {tuple(original.shape)}."
            )
        if shaped.shape[1] != self._in_channels[name]:
            raise ValueError(
                f"SharedTransformerEncoder: modality '{name}' was built for "
                f"in_channels={self._in_channels[name]}, got {shaped.shape[1]} channel(s) in "
                f"shape {tuple(original.shape)}."
            )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode whichever of the modalities are present.

        Args:
            inputs: Modality name -> batch, a non-empty subset of `input_modalities`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `inputs` is empty, names a modality this encoder does not ingest, or a
                batch has the wrong shape for its modality.
        """
        if not inputs:
            raise ValueError("SharedTransformerEncoder received no modality.")
        unknown = sorted(set(inputs) - set(self._input_modalities))
        if unknown:
            raise ValueError(
                f"SharedTransformerEncoder received modality(ies) {unknown} it does not ingest. "
                f"Its input_modalities: {list(self._input_modalities)}."
            )
        streams = [
            self._tokenize(name, inputs[name]) for name in self._input_modalities if name in inputs
        ]
        tokens = torch.cat(streams, dim=1)
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        encoded = self.transformer(tokens)
        pooled = encoded[:, 0] if self._pooling == "cls" else encoded.mean(dim=1)
        pooled = self.head(pooled)
        mu: torch.Tensor = self.to_mu(pooled)
        logvar: torch.Tensor = self.to_logvar(pooled)
        return mu, logvar

    @property
    def input_modalities(self) -> tuple[str, ...]:
        """The modalities this encoder ingests, in the order the config lists them."""
        return self._input_modalities

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the `(mu, logvar)` output."""
        return self._latent_dim
