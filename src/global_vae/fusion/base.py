"""Abstract interface for fusion strategies (encoder outputs -> single posterior)."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractFusion(nn.Module, ABC):
    """Base class for every fusion strategy (spec §4).

    A Fusion module combines the distribution parameters produced by
    several encoders into a single set of distribution parameters,
    *before* sampling. It operates on `(mu, logvar)` pairs, not on
    already-sampled latent vectors — combining realized vectors across
    independent latent spaces is the job of an Assembler
    (`latent/assembler.py`), not Fusion (spec §2.2).

    Concrete subclasses must self-register via `@registerFusion(name)`
    (see `registry.py`), e.g. `poe`, `moe`, `concat_mlp`,
    `cross_attention`.
    """

    @abstractmethod
    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse per-modality distribution parameters into one posterior.

        Args:
            params: Mapping from modality name to that modality's
                `(mu, logvar)` pair. Modalities absent from this dict
                are treated as missing for this forward pass (spec
                §5); strategies that are natively subset-tolerant
                (PoE, MoE, cross-attention) must handle an arbitrary
                non-empty subset of keys.

        Returns:
            The fused `(mu, logvar)` pair, shape `(batch, latent_dim)`.

        Raises:
            NotImplementedError: If called on the abstract base class.
            ValueError: If `params` is empty.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def handlesMissingModalities(self) -> bool:
        """Whether this strategy natively tolerates a partial `params` dict.

        Returns:
            `True` for PoE / MoE / cross-attention; `False` for
            concat+MLP, which needs an explicit imputation/masking
            scheme instead (spec §5) — a known, documented limitation
            rather than something to silently patch around.
        """
        raise NotImplementedError
