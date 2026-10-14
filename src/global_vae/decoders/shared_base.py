"""Abstract interface for a decoder shared by several modalities (spec §2.1, ADR 0027)."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractSharedDecoder(nn.Module, ABC):
    """Base class for a decoder that reconstructs several modalities at once (the `D1` rows).

    `AbstractDecoder` maps a latent vector to one tensor, the reconstruction of one modality. A
    shared decoder maps it to a dict of tensors, one per modality, the "one stream out" of spec
    §2.1 split back into modalities. `GlobalVae` flattens that dict into
    `"reconstructions"` by modality name, so the losses, the evaluation and the figures, which
    walk the reconstructions by name, see several one-modality decoders and need no change.

    It is a sibling of `AbstractDecoder`, not a subclass, because `forward` returns a different
    type. Both kinds register in the same registry (`registerDecoder`), and `GlobalVae` tells them
    apart with `isinstance`. Like any decoder, it receives one tensor: when it consumes several
    latent spaces, an assembler has already merged them (spec §2.2).

    Concrete subclasses must self-register via `@registerDecoder(name)` (see `registry.py`), and
    accept a `latent_dim` constructor argument, which the config layer fills in.
    """

    @abstractmethod
    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a batch of latent vectors into one reconstruction per modality.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)` (or `(batch, assembled_dim)` if this
                decoder consumes several latent spaces through an assembler).

        Returns:
            Modality name -> reconstruction tensor, with exactly the keys of
            `output_modalities`.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_modalities(self) -> tuple[str, ...]:
        """Names of the modalities this decoder reconstructs.

        Each is a key of `"reconstructions"` and the batch key of its reconstruction target, and
        no other decoder of the same model may reconstruct it. The tuple is not empty and has no
        repeated name.

        Returns:
            The modality names, for example `("signal", "image")`.
        """
        raise NotImplementedError
