"""KL divergence to a standard normal prior (spec §2.3 default strategy).

The default latent regularization strategy: penalizes the approximate
posterior `N(mu, exp(logvar))` by its KL divergence to a standard
normal prior `N(0, I)`. This is the plain VAE / beta-VAE regularization
term. It used to live as `LatentSpace.klDivergence` (`latent/base.py`);
it now lives here, as one pluggable strategy among others (spec §10),
so it is never the model class's only regularization option.
"""

import torch

from global_vae.losses.regularizers.base import AbstractLatentRegularizer
from global_vae.losses.regularizers.registry import registerRegularizer


@registerRegularizer("kl_standard_normal")
class KlStandardNormalRegularizer(AbstractLatentRegularizer):
    """KL divergence of `N(mu, exp(logvar))` to `N(0, I)`.

    Stateless: this strategy has no learnable parameters, but is still
    an `nn.Module` so it is interchangeable with strategies that do
    (e.g. a learned prior, spec §7) through the same registry.
    """

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-sample KL divergence to a standard normal prior.

        Args:
            mu: Posterior mean, shape `(batch, dim)`.
            logvar: Posterior log-variance, shape `(batch, dim)`.

        Returns:
            Per-sample KL divergence, shape `(batch,)`.
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
