"""Tabular MLP decoder (spec §6, roadmap P3-8)."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.mlp_blocks import buildMlp

DEFAULT_HIDDEN_DIMS = (64, 128)


@registerDecoder("tabular_mlp_decoder_v1")
class TabularMlpDecoder(AbstractDecoder):
    """Fully connected decoder reconstructing one row of a table from a latent vector.

    A stack of `Linear` layers (each with an optional normalization, an activation and an optional
    dropout) and a final linear layer of `num_features` outputs. The output layout is the one the
    encoder reads: the numeric columns first, linear and unbounded, then one block per categorical
    column, each the one-hot encoding of that column. With `categorical_sizes` set, a softmax is
    applied inside each block, so a block is a probability vector over the categories and the
    default `mse` reconstruction loss against the one-hot target is the Brier score of the
    column, a proper scoring rule. Without it every output is linear, which is the right choice
    for an all-numeric table.

    Like every decoder here it reconstructs one fixed width, never a resized one.
    """

    def __init__(
        self,
        latent_dim: int,
        num_features: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        activation: str = "relu",
        normalization: str | None = None,
        dropout: float = 0.0,
        categorical_sizes: Sequence[int] = (),
        modality_name: str = "tabular",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the latent vector it receives.
            num_features: Width of the row to reconstruct. Must match the encoder's, and the
                width of the data the caller's `loader_factory` yields. Data-dependent, so the
                config layer never fills it in.
            hidden_dims: Width of each hidden layer, from the latent vector outwards. Empty goes
                straight from `z` to the output layer.
            activation: Activation after each hidden layer, one of
                `global_vae.utils.activations.listSupportedActivations()`.
            normalization: `None` (default), `"layer"` or `"batch"`.
            dropout: Dropout probability after each hidden layer.
            categorical_sizes: Number of categories of each one-hot encoded column, in the order
                of the blocks, which are the last `sum(categorical_sizes)` outputs. Empty
                (default) means every column is numeric.
            modality_name: Name of the modality this decoder reconstructs.

        Raises:
            ValueError: If `latent_dim` or `num_features` is below `1`, a categorical size is
                below `2`, the blocks are wider than `num_features`, or an argument of the stack
                is out of range (the message names it).
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if num_features < 1:
            raise ValueError(f"num_features must be at least 1, got {num_features}.")
        if any(size < 2 for size in categorical_sizes):
            raise ValueError(
                f"categorical_sizes must all be at least 2 (a column with one category carries "
                f"no information), got {list(categorical_sizes)}."
            )
        if sum(categorical_sizes) > num_features:
            raise ValueError(
                f"categorical_sizes {list(categorical_sizes)} add up to {sum(categorical_sizes)} "
                f"one-hot columns, more than num_features={num_features}."
            )
        self._modality_name = modality_name
        self._num_features = num_features
        self._categorical_sizes = tuple(categorical_sizes)
        self._num_numeric = num_features - sum(self._categorical_sizes)
        self.body, width = buildMlp(latent_dim, hidden_dims, activation, normalization, dropout)
        self.to_output = nn.Linear(width, num_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of table rows from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Reconstructed rows, shape `(batch, num_features)`: numeric columns first, then the
            one-hot blocks (probabilities if `categorical_sizes` is set).
        """
        out: torch.Tensor = self.to_output(self.body(z))
        if not self._categorical_sizes:
            return out
        blocks = torch.split(out[:, self._num_numeric :], list(self._categorical_sizes), dim=1)
        return torch.cat([out[:, : self._num_numeric], *(F.softmax(b, dim=1) for b in blocks)], 1)

    @property
    def modality_name(self) -> str:
        """Name of the modality this decoder reconstructs."""
        return self._modality_name
