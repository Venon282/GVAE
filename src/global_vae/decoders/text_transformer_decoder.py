"""Text transformer decoder (spec §6 and §10, roadmap P3-8, ADR 0041)."""

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.transformer_blocks import SeriesPositions, buildTransformerStack

OUTPUT_ACTIVATIONS = ("softmax", None)


@registerDecoder("text_transformer_decoder_v1")
class TextTransformerDecoder(AbstractDecoder):
    """Small transformer reconstructing a fixed-length text from a latent vector.

    There is one token per output position: its position vector plus a linear projection of the
    latent vector, the same for every position, so the tokens differ only by position and the
    layers of self-attention are what turn one latent vector into a text. Each output token is
    projected to `vocab_size` scores, and the default `softmax` turns them into a distribution over
    the vocabulary at every position, which is the same shape as the one-hot input.

    The whole text is produced in one pass from the latent vector, with no token feeding the next
    (no autoregression). That is a choice and not a limit of the interface only in one direction:
    a decoder that reads the previous tokens needs the target in `forward`, which the
    `AbstractDecoder` contract (`forward(z)`) does not carry, and it is the usual way a text VAE
    ends up ignoring its latent vector (posterior collapse). This one cannot, because the latent
    vector is the only input it has.

    The padding token is a token like any other here: the decoder predicts it, and that is how it
    learns the length of a text. Like every decoder here it reconstructs one fixed length.
    """

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        output_length: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        positional_encoding: str = "sinusoidal",
        output_activation: str | None = "softmax",
        modality_name: str = "text",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the latent vector it receives.
            vocab_size: Number of distinct tokens, padding included. It is data-dependent, so the
                config layer never fills it in, and it must equal the encoder's.
            output_length: Number of tokens of the text to reconstruct, padding included. It must
                match the loader's `sequence_length` of the modality.
            d_model: Token width.
            num_heads: Number of attention heads. Must divide `d_model`.
            num_layers: Number of transformer layers.
            feedforward_dim: Hidden width of each layer's feed-forward block. `None` (default) is
                `4 * d_model`.
            dropout: Dropout probability inside the transformer layers.
            activation: `"gelu"` or `"relu"`, inside the layers.
            positional_encoding: `"sinusoidal"` or `"learned"`. The number of tokens is fixed by
                `output_length`, so a learned table has exactly that many vectors.
            output_activation: `"softmax"` (default, probabilities that sum to `1` over the
                vocabulary at each position, which the `mse` and `bce` losses of a YAML file
                compare with a one-hot target) or `None` (raw scores, for a loss written in Python
                that applies its own softmax, such as a cross-entropy).
            modality_name: Name of the modality this decoder reconstructs.

        Raises:
            ValueError: If `num_heads` does not divide `d_model`, `output_activation` or
                `positional_encoding` is not a known choice, or another argument is out of range
                (the message names it).
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be at least 2, got {vocab_size}.")
        if output_length < 1:
            raise ValueError(f"output_length must be at least 1, got {output_length}.")
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"Unknown output_activation {output_activation!r}. "
                f"Expected one of {OUTPUT_ACTIVATIONS}."
            )
        self._modality_name = modality_name
        self._output_length = output_length
        self._output_activation = output_activation
        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.positions = SeriesPositions(d_model, positional_encoding, output_length)
        self.transformer = buildTransformerStack(
            d_model, num_heads, num_layers, feedforward_dim, dropout, activation
        )
        self.to_vocabulary = nn.Linear(d_model, vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of fixed-length texts from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            `(batch, output_length, vocab_size)`: a distribution over the vocabulary at every
            position (`output_activation="softmax"`), or the raw scores (`None`). The most likely
            token ids are `reconstruction.argmax(dim=-1)`.
        """
        latent = self.latent_projection(z).unsqueeze(1)
        tokens = self.positions(self._output_length, latent).unsqueeze(0) + latent
        scores: torch.Tensor = self.to_vocabulary(self.transformer(tokens))
        if self._output_activation == "softmax":
            return torch.softmax(scores, dim=-1)
        return scores

    @property
    def modality_name(self) -> str:
        """Name of the modality this decoder reconstructs."""
        return self._modality_name
