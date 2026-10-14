"""Dummy shared encoder and decoder for the `E1` and `D1` rows (spec §2.1, ADR 0027).

Not a test module (the leading underscore keeps pytest from collecting it): importing it registers
the modules below, so any test can name them in a config. They are deliberately small. What they
exercise is the machinery around a shared module (routing, modality dropout, the cross-modal
reports, the losses), not a real backbone, which is roadmap P3-5.

* `dummy_shared_encoder_v1` projects each modality it receives to one token with its own linear
  layer, averages the tokens it received, and maps the mean to `(mu, logvar)`. Averaging over the
  tokens present is what lets it accept any non-empty subset of its modalities, the contract of
  `AbstractSharedEncoder` (spec §5).
* `dummy_shared_decoder_v1` maps the latent vector to one tensor per modality, each through its
  own linear layer.
* `dummy_shared_encoder_misdeclared_v1` and `dummy_shared_decoder_misdeclared_v1` break the
  contract on purpose, so the construction and forward guards of `GlobalVae` have something to
  refuse.

Registry names carry the `dummy_` prefix and a `_v1` suffix, unique across the suite.
`sharedEncoderConfig` and `sharedDecoderConfig` build the `ModelConfig` entries that name them.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from global_vae.config.model import DecoderConfig, EncoderConfig
from global_vae.decoders.registry import registerDecoder
from global_vae.decoders.shared_base import AbstractSharedDecoder
from global_vae.encoders.registry import registerEncoder
from global_vae.encoders.shared_base import AbstractSharedEncoder

SHARED_ENCODER = "dummy_shared_encoder_v1"
SHARED_DECODER = "dummy_shared_decoder_v1"
MISDECLARED_ENCODER = "dummy_shared_encoder_misdeclared_v1"
MISDECLARED_DECODER = "dummy_shared_decoder_misdeclared_v1"


@registerEncoder(SHARED_ENCODER)
class DummySharedEncoder(AbstractSharedEncoder):
    """One linear token per modality, mean over the tokens present, then `mu` and `logvar`."""

    def __init__(self, input_dims: Mapping[str, int], latent_dim: int, token_dim: int = 16) -> None:
        """Build the per-modality projections.

        Args:
            input_dims: Modality name -> number of values per sample once flattened. Its order is
                the order of `input_modalities`.
            latent_dim: Width of `mu` and `logvar`.
            token_dim: Width of each modality's token.
        """
        super().__init__()
        self._input_modalities = tuple(input_dims)
        self._latent_dim = latent_dim
        self.tokenizers = nn.ModuleDict(
            {name: nn.Linear(width, token_dim) for name, width in input_dims.items()}
        )
        self.to_mu = nn.Linear(token_dim, latent_dim)
        self.to_logvar = nn.Linear(token_dim, latent_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode whichever of the modalities are present.

        Args:
            inputs: Modality name -> tensor, a non-empty subset of `input_modalities`.

        Returns:
            `(mu, logvar)`, each `(batch, latent_dim)`.

        Raises:
            ValueError: If `inputs` is empty.
            KeyError: If a key is not one of `input_modalities`.
        """
        if not inputs:
            raise ValueError("DummySharedEncoder received no modality.")
        tokens = [self.tokenizers[name](tensor.flatten(1)) for name, tensor in inputs.items()]
        pooled = torch.stack(tokens, dim=0).mean(dim=0)
        return self.to_mu(pooled), self.to_logvar(pooled)

    @property
    def input_modalities(self) -> tuple[str, ...]:
        """The modalities this encoder ingests."""
        return self._input_modalities

    @property
    def latent_dim(self) -> int:
        """Width of the distribution parameters."""
        return self._latent_dim


@registerDecoder(SHARED_DECODER)
class DummySharedDecoder(AbstractSharedDecoder):
    """One linear layer per modality, reshaped to that modality's shape."""

    def __init__(self, output_shapes: Mapping[str, Sequence[int]], latent_dim: int) -> None:
        """Build the per-modality projections.

        Args:
            output_shapes: Modality name -> shape of one sample, without the batch axis. Its order
                is the order of `output_modalities`.
            latent_dim: Width of the tensor the decoder receives.
        """
        super().__init__()
        self._output_modalities = tuple(output_shapes)
        self._shapes = {
            name: tuple(int(size) for size in shape) for name, shape in output_shapes.items()
        }
        self.projections = nn.ModuleDict(
            {name: nn.Linear(latent_dim, math.prod(shape)) for name, shape in self._shapes.items()}
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode `z` into one tensor per modality.

        Args:
            z: Latent tensor, `(batch, latent_dim)`.

        Returns:
            Modality name -> `(batch, *shape)` tensor.
        """
        return {
            name: self.projections[name](z).reshape(z.shape[0], *shape)
            for name, shape in self._shapes.items()
        }

    @property
    def output_modalities(self) -> tuple[str, ...]:
        """The modalities this decoder reconstructs."""
        return self._output_modalities


@registerEncoder(MISDECLARED_ENCODER)
class MisdeclaredSharedEncoder(AbstractSharedEncoder):
    """A shared encoder whose declared modalities are whatever the test hands it."""

    def __init__(self, declared: Any, latent_dim: int = 4) -> None:
        """Store the declaration without checking it.

        Args:
            declared: What `input_modalities` returns, valid or not.
            latent_dim: Width of the (constant) distribution parameters.
        """
        super().__init__()
        self._declared = declared
        self._latent_dim = latent_dim
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a constant posterior of the right batch size.

        Args:
            inputs: Any non-empty modality dict.

        Returns:
            `(mu, logvar)`, each `(batch, latent_dim)`.
        """
        batch = next(iter(inputs.values())).shape[0]
        mu = self.anchor.expand(batch, self._latent_dim)
        return mu, mu

    @property
    def input_modalities(self) -> tuple[str, ...]:
        """The declaration, as given."""
        declared: tuple[str, ...] = self._declared
        return declared

    @property
    def latent_dim(self) -> int:
        """Width of the distribution parameters."""
        return self._latent_dim


@registerDecoder(MISDECLARED_DECODER)
class MisdeclaredSharedDecoder(AbstractSharedDecoder):
    """A shared decoder that declares one set of modalities and returns another."""

    def __init__(
        self,
        declared: Any,
        returned: Sequence[str] | None = None,
        latent_dim: int = 4,
    ) -> None:
        """Store the declaration and the keys actually returned.

        Args:
            declared: What `output_modalities` returns, valid or not.
            returned: The keys `forward` returns. `None` returns exactly `declared`.
            latent_dim: Width of the tensor the decoder receives.
        """
        super().__init__()
        self._declared = declared
        self._returned = tuple(declared if returned is None else returned)
        self.project = nn.Linear(latent_dim, 3)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return one `(batch, 3)` tensor per key in `returned`.

        Args:
            z: Latent tensor, `(batch, latent_dim)`.

        Returns:
            Key -> tensor.
        """
        return {name: self.project(z) for name in self._returned}

    @property
    def output_modalities(self) -> tuple[str, ...]:
        """The declaration, as given."""
        declared: tuple[str, ...] = self._declared
        return declared


def sharedEncoderConfig(shapes: Mapping[str, Sequence[int]]) -> EncoderConfig:
    """The `encoders` entry of a dummy shared encoder over the given modalities.

    Args:
        shapes: Modality name -> shape of one sample, without the batch axis, in the order the
            encoder declares them. `latent_dim` is left out: the config layer fills it in.

    Returns:
        An `EncoderConfig` naming `SHARED_ENCODER`.
    """
    return EncoderConfig(
        name=SHARED_ENCODER,
        kwargs={"input_dims": {name: math.prod(shape) for name, shape in shapes.items()}},
    )


def sharedDecoderConfig(shapes: Mapping[str, Sequence[int]]) -> DecoderConfig:
    """The `decoders` entry of a dummy shared decoder over the given modalities.

    Args:
        shapes: Modality name -> shape of one sample, without the batch axis, in the order the
            decoder declares them. `latent_dim` is left out: the config layer fills it in.

    Returns:
        A `DecoderConfig` naming `SHARED_DECODER`.
    """
    return DecoderConfig(
        name=SHARED_DECODER,
        kwargs={"output_shapes": {name: list(shape) for name, shape in shapes.items()}},
    )
