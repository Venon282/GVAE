"""Identity latent head, the default (spec §2.2).

An encoder that feeds one latent space needs no adaptation: the head on that edge is the
identity and the encoder's `(mu, logvar)` becomes the latent space's own parameters
unchanged. It is also what keeps a graph without heads running exactly as it did before
heads existed.
"""

import torch

from global_vae.heads.base import AbstractLatentHead
from global_vae.heads.registry import registerHead


@registerHead("identity")
class IdentityHead(AbstractLatentHead):
    """Passes `(mu, logvar)` through unchanged.

    Stateless: it has no learnable parameter, and it returns the very tensors it
    received, not copies, so it adds nothing to the autograd graph. Because nothing can
    change the width, it requires `in_dim == out_dim`.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Build the head.

        Args:
            in_dim: Width of the encoder output.
            out_dim: Width of the latent space. Must equal `in_dim`.

        Raises:
            ValueError: If `in_dim` or `out_dim` is not a positive integer, or if
                `in_dim != out_dim`. Use the `linear` head to change the width.
        """
        super().__init__(in_dim, out_dim)
        if in_dim != out_dim:
            raise ValueError(
                f"IdentityHead requires in_dim == out_dim, got in_dim={in_dim} and "
                f"out_dim={out_dim}. Use the 'linear' head to change the width."
            )

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `mu` and `logvar` unchanged.

        Args:
            mu: Posterior mean, shape `(batch, in_dim)`.
            logvar: Posterior log-variance, shape `(batch, in_dim)`.

        Returns:
            The same `(mu, logvar)` pair, the same tensor objects.

        Raises:
            ValueError: If `mu` or `logvar` does not have shape `(batch, in_dim)`, or
                the two shapes differ.
        """
        self._checkParams(mu, logvar)
        return mu, logvar
