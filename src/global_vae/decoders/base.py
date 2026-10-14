"""Abstract interface for all decoders in the Global VAE framework."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractDecoder(nn.Module, ABC):
    """Base class for every one-modality decoder.

    A decoder shared by several modalities (the `D1` rows of spec §2.1)
    subclasses `AbstractSharedDecoder` instead (ADR 0027).

    A decoder maps a latent vector back to a modality-specific
    reconstruction. If this decoder consumes more than one latent
    space (spec §2.2), the vector it receives has already been merged
    by an Assembler (`latent/assembler.py`). The decoder itself
    stays agnostic to how many latent spaces fed into it.

    Concrete subclasses must self-register via `@registerDecoder(name)`
    (see `registry.py`).
    """

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent vectors into a reconstruction.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)` (or
                `(batch, assembled_dim)` if this decoder consumes
                several latent spaces through an Assembler).

        Returns:
            Reconstruction tensor, shape modality-specific.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Name of the modality this decoder reconstructs.

        Returns:
            The modality name (e.g. `"signal"`, `"image"`).
        """
        raise NotImplementedError
