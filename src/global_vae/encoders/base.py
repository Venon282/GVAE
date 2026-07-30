"""Abstract interface for all encoders in the Global VAE framework."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractEncoder(nn.Module, ABC):
    """Base class for every modality-specific (or shared) encoder.

    An encoder maps a modality's raw input to distribution parameters
    `(mu, logvar)` of a Gaussian. When used upstream of a Fusion module
    (spec §4), these are per-modality "expert" parameters that Fusion
    will combine into a single posterior — not yet the model's final
    latent distribution.

    Concrete subclasses must self-register via `@registerEncoder(name)`
    (see `registry.py`) so that model-assembly code never needs to
    import or special-case any specific encoder class. This is what
    makes "add a modality without touching the core" (spec §10)
    operationally true.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of raw inputs.

        Args:
            x: Raw input tensor for this modality. Shape is
                modality-specific (e.g. `(batch, length)` for 1D
                signals, `(batch, channels, height, width)` for
                images).

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Name of the modality this encoder.

            Returns:
            The modality name (e.g. `"signal"`, `"image"`).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimensionality of the distribution parameters this encoder outputs.

        Returns:
            The latent (or pre-fusion feature) dimensionality.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def minimal_input_length(self) -> int:
        """Minimal input len that can receive the encoder to avoid to collapse.

        Returns:
            The minimal input len.
        """
        raise NotImplementedError
