"""Text transformer encoder (spec §6 and §10, roadmap P3-8, ADR 0041)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.text_blocks import TokenEmbedding, computeMaskedMean, computePaddingMask
from global_vae.utils.transformer_blocks import (
    SeriesPositions,
    buildPoolingHead,
    buildTransformerStack,
)

POOLING_CHOICES = ("mean", "cls")


@registerEncoder("text_transformer_encoder_v1")
class TextTransformerEncoder(AbstractEncoder):
    """Small transformer over a sequence of tokens, robust to length and to padding.

    Every token becomes a `d_model` vector from one embedding table (`TokenEmbedding`), a position
    vector is added, and a stack of pre-norm transformer layers lets every token attend to every
    other. The mean of the tokens (or a learned `[CLS]` token, with `pooling="cls"`) goes through
    an optional MLP and two linear layers to give `(mu, logvar)`.

    The input is a one-hot tensor `(batch, length, vocab_size)`, which is what a text has to be to
    be a batch tensor and a loss target at once (`utils/text_blocks.py`), or integer ids `(batch,
    length)`, which give the same output and need far less memory for a large vocabulary. When
    `pad_token_id` is set, padded positions are skipped as attention keys and left out of the
    mean, so a text padded to a longer length gives the same `(mu, logvar)` (with sinusoidal
    positions, which do not depend on the length). With the default sinusoidal positions there is
    no length limit, as for the other transformer encoders.
    """

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: str = "mean",
        positional_encoding: str = "sinusoidal",
        max_length: int | None = None,
        pad_token_id: int | None = None,
        head_hidden_dims: Sequence[int] = (),
        modality_name: str = "text",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            vocab_size: Number of distinct tokens, padding included. It is data-dependent, so the
                config layer never fills it in, and it must equal the last axis of the one-hot
                input and the decoder's `vocab_size`.
            d_model: Token width.
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.
            pooling: `"mean"` (the mean of the unpadded tokens) or `"cls"` (a learned token read
                after the last layer).
            positional_encoding: `"sinusoidal"` (no parameters, any length) or `"learned"` (one
                trained vector per position, up to `max_length`).
            max_length: Longest text, in tokens, that learned positions cover. Required for
                `positional_encoding="learned"` and ignored otherwise.
            pad_token_id: The id of the padding token, or `None` (default) when the batch has no
                padding. It is a token of the vocabulary like any other, so a decoder predicts it.
            head_hidden_dims: Hidden widths of an optional MLP between the pooled token and the
                `mu` and `logvar` layers. Empty (default) goes straight to them.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If `pooling` is not a known choice, `num_heads` does not divide
                `d_model`, a learned position table has no `max_length`, `pad_token_id` is not a
                token of the vocabulary, or another argument is out of range (the message names
                it).
        """
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"Unknown pooling '{pooling}'. Expected one of {POOLING_CHOICES}.")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if positional_encoding == "learned" and max_length is None:
            raise ValueError(
                "positional_encoding='learned' needs max_length, the longest text it must "
                "cover, because it holds one trained vector per position."
            )
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError(f"pad_token_id must be in [0, {vocab_size}), got {pad_token_id}.")
        self._latent_dim = latent_dim
        self._modality_name = modality_name
        self._vocab_size = vocab_size
        self._pad_token_id = pad_token_id
        self._pooling = pooling

        self.embed = TokenEmbedding(vocab_size, d_model)
        self.positions = SeriesPositions(d_model, positional_encoding, max_length)
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
        """Encode a batch of texts.

        Args:
            x: One-hot (or soft) tensor `(batch, length, vocab_size)`, or integer ids `(batch,
                length)`. `length` is at least `1`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `x` is neither layout (a float `(batch, length)` tensor, ids of the
                wrong rank, a last axis that is not `vocab_size`), is empty along `length`, or is
                longer than a learned position table covers.
        """
        tokens = self.embed(x)
        if tokens.shape[1] < 1:
            raise ValueError(f"A text needs at least one token, got shape {tuple(x.shape)}.")
        padding_mask = computePaddingMask(x, self._pad_token_id, self._vocab_size)
        tokens = tokens + self.positions(tokens.shape[1], tokens).unsqueeze(0)
        key_mask = padding_mask
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
            if key_mask is not None:
                no_pad = torch.zeros_like(key_mask[:, :1])
                key_mask = torch.cat([no_pad, key_mask], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=key_mask)
        if self._pooling == "cls":
            pooled = encoded[:, 0]
        else:
            pooled = computeMaskedMean(encoded, padding_mask)
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
        """Shortest text this encoder accepts: `1` token.

        Nothing is strided or patched, so no length collapses a feature map.

        Returns:
            `1`.
        """
        return 1
