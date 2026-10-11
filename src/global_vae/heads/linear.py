"""Linear latent head (spec §2.2).

A small learned projection between an encoder output and a latent space whose width, or
purpose, differs. It is the head to use when one encoder fans out to several latent
spaces (a shared space and private ones, for example): each edge gets its own `linear`
head, with its own parameters, so the spaces do not collapse to the same posterior.
"""

import torch
from torch import nn

from global_vae.heads.base import AbstractLatentHead
from global_vae.heads.registry import registerHead


@registerHead("linear")
class LinearHead(AbstractLatentHead):
    """Projects `mu` and `logvar` each through its own `nn.Linear(in_dim, out_dim)`.

    `mu` and `logvar` have separate projections, one per output, rather than one shared
    matrix: a mean and a log-variance are different quantities, and a shared projection
    would force the adapted log-variance to be a function of the same mixing of features
    as the adapted mean.

    Initialisation (documented because it decides where training starts): both weight
    matrices use Xavier (Glorot) uniform initialisation with gain 1, and both biases are
    zero. Xavier keeps the scale of the activations roughly unchanged from input to
    output, so the adapted `mu` and `logvar` start on the scale of the encoder's. Zero
    biases make the head map `(mu, logvar) = (0, 0)`, the standard normal prior, to
    itself, so a latent that starts at the prior stays there. The weights are random, not
    the identity, so two heads on the same encoder start different and can describe
    different latent spaces. This replaces PyTorch's default `nn.Linear` initialisation
    (Kaiming uniform with a nonzero bias).

    Attributes:
        to_mu: The projection applied to `mu`.
        to_logvar: The projection applied to `logvar`.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Build the two projections and initialise them.

        Args:
            in_dim: Width of the encoder output.
            out_dim: Width of the latent space.

        Raises:
            ValueError: If `in_dim` or `out_dim` is not a positive integer.
        """
        super().__init__(in_dim, out_dim)
        self.to_mu = nn.Linear(in_dim, out_dim)
        self.to_logvar = nn.Linear(in_dim, out_dim)
        for projection in (self.to_mu, self.to_logvar):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project `mu` and `logvar` to `out_dim`.

        Args:
            mu: Posterior mean, shape `(batch, in_dim)`.
            logvar: Posterior log-variance, shape `(batch, in_dim)`.

        Returns:
            The projected `(mu, logvar)` pair, each of shape `(batch, out_dim)`.

        Raises:
            ValueError: If `mu` or `logvar` does not have shape `(batch, in_dim)`, or
                the two shapes differ.
        """
        self._checkParams(mu, logvar)
        return self.to_mu(mu), self.to_logvar(logvar)
