"""Abstract interface for latent-space regularization strategies (spec §2.3).

KL divergence to a standard normal prior is the default regularization
term for a latent space (`kl_standard_normal.py`), but it must not be
hardcoded as the only option inside the model class (spec §10). Each
latent space's regularization is a pluggable strategy, registered like
Fusion / Assembler, so alternatives (Maximum Mean Discrepancy, free-bits
KL, a learned/autoregressive prior, spec §7) can be added later without
touching `GlobalVae` or `LatentSpace`.

Concrete subclasses must self-register via `@registerRegularizer(name)`
(see `registry.py`), e.g. `kl_standard_normal`.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractLatentRegularizer(nn.Module, ABC):
    """Base class for every latent-space regularization strategy.

    A regularizer computes a per-sample scalar penalty from a latent
    space's distribution parameters, pulling the approximate posterior
    toward some prior or otherwise constraining it (spec §2.3). It
    operates on `(mu, logvar)`, the same distribution-parameter
    convention used by encoders and Fusion, not on realized (sampled)
    latent vectors, which is the Assembler's domain instead
    (`assemblers/base.py`).

    Declared as an `nn.Module` (not a plain function) so that
    strategies with learnable parameters (e.g. a learned prior, spec
    §7) are interchangeable with stateless ones like
    `kl_standard_normal` through the exact same registry and the same
    `GlobalVae.regularizers` `nn.ModuleDict`.
    """

    @abstractmethod
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute the per-sample regularization penalty.

        Args:
            mu: Posterior mean, shape `(batch, dim)`.
            logvar: Posterior log-variance, shape `(batch, dim)`. Some
                strategies (e.g. MMD against samples) may not use
                `logvar` at all; it remains part of the shared
                signature so every strategy stays interchangeable
                through the registry.

        Returns:
            Per-sample penalty, shape `(batch,)`. Callers sum across
            latent spaces and average across the batch (see
            `losses.regularization.computeTotalRegularizationLoss`).

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError
