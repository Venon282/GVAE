"""Tabular MLP encoder (spec §6, roadmap P3-8)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.mlp_blocks import buildMlp

DEFAULT_HIDDEN_DIMS = (128, 64)


@registerEncoder("tabular_mlp_encoder_v1")
class TabularMlpEncoder(AbstractEncoder):
    """Fully connected encoder for one row of a table, given as a vector of `num_features` floats.

    The row goes through a stack of `Linear` layers (each with an optional normalization, an
    activation and an optional dropout) and two linear layers give `(mu, logvar)`. With an empty
    `hidden_dims` the encoder is just those two linear layers.

    The input is one flat float tensor, `(batch, num_features)`: the encoder knows nothing about
    columns. Turning a table into that tensor is the caller's `loader_factory` (spec §6.2):
    numeric columns standardized (the `standardize` transform, with one mean and one standard
    deviation per column as tensors) and categorical columns one-hot encoded, numeric columns
    first. Missing values are the caller's to impute, since a NaN would make every output NaN.
    """

    def __init__(
        self,
        latent_dim: int,
        num_features: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        activation: str = "relu",
        normalization: str | None = None,
        dropout: float = 0.0,
        modality_name: str = "tabular",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            num_features: Width of a row: the number of numeric columns plus the width of every
                one-hot block. Data-dependent, so the config layer never fills it in.
            hidden_dims: Width of each hidden layer, from the input outwards. Empty goes straight
                from the row to `(mu, logvar)`.
            activation: Activation after each hidden layer, one of
                `global_vae.utils.activations.listSupportedActivations()`.
            normalization: `None` (default), `"layer"` or `"batch"`, after each hidden layer's
                linear map. `"batch"` fails in training on a batch of one sample.
            dropout: Dropout probability after each hidden layer.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If `latent_dim` or `num_features` is below `1`, or an argument of the
                stack is out of range (the message names it).
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if num_features < 1:
            raise ValueError(f"num_features must be at least 1, got {num_features}.")
        self._latent_dim = latent_dim
        self._num_features = num_features
        self._modality_name = modality_name
        self.body, width = buildMlp(num_features, hidden_dims, activation, normalization, dropout)
        self.to_mu = nn.Linear(width, latent_dim)
        self.to_logvar = nn.Linear(width, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of table rows.

        Args:
            x: Float tensor of shape `(batch, num_features)`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `x` is not 2D or its last dimension is not `num_features`.
        """
        if x.dim() != 2 or x.shape[1] != self._num_features:
            raise ValueError(
                f"TabularMlpEncoder expects (batch, {self._num_features}) rows, got shape "
                f"{tuple(x.shape)}."
            )
        features = self.body(x)
        mu: torch.Tensor = self.to_mu(features)
        logvar: torch.Tensor = self.to_logvar(features)
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
        """Shortest row this encoder accepts, which is also the only one: `num_features`.

        A table row has a fixed width, so no length collapses a feature map and a different one
        is refused rather than resized.

        Returns:
            `num_features`.
        """
        return self._num_features
