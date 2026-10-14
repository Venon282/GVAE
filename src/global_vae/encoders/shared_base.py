"""Abstract interface for an encoder shared by several modalities (spec §2.1, ADR 0027)."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn


class AbstractSharedEncoder(nn.Module, ABC):
    """Base class for an encoder that ingests several modalities at once (the `E1` rows).

    `AbstractEncoder` maps one tensor, the raw input of one modality, to a `(mu, logvar)` pair.
    A shared encoder maps a dict of tensors, one per modality, to one `(mu, logvar)` pair: the
    modalities are "tokenized/concatenated into one stream" (spec §2.1), and how that stream is
    built is the concrete module's own business (tokens, a concatenated vector, a sum of
    per-modality projections). `GlobalVae` never looks inside it.

    It is a sibling of `AbstractEncoder`, not a subclass: `forward` takes a different argument,
    and `modality_name` and `minimal_input_length` mean nothing for several modalities. Both kinds
    register in the same registry (`registerEncoder`), and `GlobalVae` tells them apart with
    `isinstance`.

    Missing modalities (spec §5). With one encoder there is no fusion strategy to tolerate a
    missing modality, so the module must. `forward` receives a non-empty subset of
    `input_modalities`, with the modalities that are absent from the batch omitted from the
    dict (modality dropout hides them during training, a cross-modal report hides them on
    purpose), and must work for every such subset.

    Concrete subclasses must self-register via `@registerEncoder(name)` (see `registry.py`), and
    accept a `latent_dim` constructor argument, which the config layer fills in.
    """

    @abstractmethod
    def forward(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch holding some or all of this encoder's modalities.

        Args:
            inputs: Modality name -> raw input tensor, batch first, all with the same batch
                size. The keys are a non-empty subset of `input_modalities`, in
                `input_modalities` order. A modality missing from the batch is omitted.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def input_modalities(self) -> tuple[str, ...]:
        """Names of the modalities this encoder ingests.

        Each is a key of the per-modality batch (spec §6.1), and no other encoder of the same
        model may ingest it. The tuple is not empty, has no repeated name, and its order is the
        order in which `forward` receives the modalities.

        Returns:
            The modality names, for example `("signal", "image")`.
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
