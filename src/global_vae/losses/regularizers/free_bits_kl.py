"""Free-bits KL divergence to a standard normal prior (spec §2.3, §11 candidate).

Plain KL-to-standard-normal (`kl_standard_normal.py`) gives every
latent dimension unconditional gradient pressure toward the prior,
including dimensions that have already reached zero KL. This is the
usual mechanism behind posterior collapse: once a dimension's KL hits
zero, the regularization term keeps pushing on it exactly as hard as
on any other dimension, with nothing to counterbalance that pressure
if the decoder has not yet learned to rely on that dimension for
reconstruction.

Free bits (Kingma et al., 2016, "Improved Variational Inference with
Inverse Autoregressive Flow") fixes this by giving each latent
dimension a small, fixed KL budget (`free_bits` nats) it is never
penalized for using: the regularizer only pushes a dimension toward
the prior once its own KL exceeds that budget. Dimensions within
budget contribute a constant (zero-gradient, since `torch.clamp`'s
minimum branch does not depend on the input there) term instead of a
shrinking-but-never-zero one.
"""

import torch

from global_vae.losses.regularizers.base import AbstractLatentRegularizer
from global_vae.losses.regularizers.registry import registerRegularizer


@registerRegularizer("free_bits_kl")
class FreeBitsKlRegularizer(AbstractLatentRegularizer):
    """KL divergence to `N(0, I)`, clipped from below by a free-bits budget.

    At `free_bits=0` this is mathematically identical to
    `KlStandardNormalRegularizer`: each per-dimension term of a
    diagonal-covariance Gaussian's KL to a standard normal is already
    non-negative on its own (the multivariate KL is an additive sum of
    independent per-dimension 1D KLs), so clamping at a `0` minimum is
    a no-op. This is verified by
    `tests/integration/test_regularizers.py`.
    """

    def __init__(self, free_bits: float = 0.5, per_dimension: bool = True) -> None:
        """Initialize the regularizer.

        Args:
            free_bits: KL budget (in nats), below which a dimension
                receives no penalty. The standard formulation (Kingma
                et al., 2016) applies this per dimension; a single
                scalar shared by every dimension of every latent space
                using this strategy, since a per-dimension budget is
                itself already a per-dimension, not per-latent-space,
                concept, and spec §2.3's per-latent-space granularity
                is expressed by assigning a *different instance* of
                this regularizer (with its own `free_bits`) to each
                latent space via `regularizer_strategies`, exactly like
                every other strategy in this registry.
            per_dimension: If `True` (default, the standard
                formulation), the budget is applied independently to
                each latent dimension, which is what prevents any
                single dimension from collapsing while others still
                carry KL. If `False`, the budget is instead applied
                once to the summed per-sample KL, a coarser variant
                that only prevents the *aggregate* KL from reaching
                zero, not each dimension individually. `False` is
                provided for parity/ablation purposes; `True` is the
                formulation that actually addresses per-dimension
                posterior collapse.

        Raises:
            ValueError: If `free_bits` is negative.
        """
        super().__init__()
        if free_bits < 0:
            raise ValueError(f"free_bits must be non-negative, got {free_bits}.")
        self.free_bits = free_bits
        self.per_dimension = per_dimension

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-sample free-bits KL divergence to a standard normal prior.

        Args:
            mu: Posterior mean, shape `(batch, dim)`.
            logvar: Posterior log-variance, shape `(batch, dim)`.

        Returns:
            Per-sample penalty, shape `(batch,)`.
        """
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.per_dimension:
            return torch.clamp(kl_per_dim, min=self.free_bits).sum(dim=-1)
        return torch.clamp(kl_per_dim.sum(dim=-1), min=self.free_bits)
