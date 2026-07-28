"""Product-of-Experts fusion strategy (spec §4).

Combines each modality's Gaussian "expert" `(mu, logvar)` by
multiplying their densities together, MVAE-style (Wu & Goodman, 2018).
Natively subset-tolerant: a missing modality simply omits its expert's
term from the product, which is what gives PoE-based fusion its
missing-modality robustness (spec §5).
"""

import torch

from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion


@registerFusion("poe")
class ProductOfExperts(AbstractFusion):
    """Product-of-Experts fusion (MVAE-style).

    Each encoder's `(mu, logvar)` is treated as one Gaussian "expert".
    Experts are combined by multiplying their densities, which reduces
    to a precision-weighted average of the means::

        precision_i = exp(-logvar_i)
        mu_fused = sum_i(precision_i * mu_i) / sum_i(precision_i)
        var_fused = 1 / sum_i(precision_i)

    Following Wu & Goodman (2018), the standard-normal prior is
    included as one extra expert (`mu=0`, `logvar=0`) so the fused
    posterior stays well-regularized even when only one modality is
    present, not only when several are.
    """

    def __init__(self, eps: float = 1e-8, include_prior_expert: bool = True) -> None:
        """Initialize the fusion module.

        Args:
            eps: Numerical-stability constant added to the summed
                precision before inverting it.
            include_prior_expert: Whether to add the standard-normal
                prior as an extra expert in the product (the original
                MVAE formulation, spec §4). Disabling it is
                occasionally useful for ablations.
        """
        super().__init__()
        self.eps = eps
        self.include_prior_expert = include_prior_expert

    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse per-modality experts via a product of Gaussians.

        Args:
            params: Modality name -> `(mu, logvar)`. Any non-empty
                subset of the model's modalities is accepted (spec §5).

        Returns:
            The fused `(mu, logvar)` pair, shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `params` is empty.
        """
        if not params:
            raise ValueError("ProductOfExperts received an empty `params` dict.")

        first_mu, _ = next(iter(params.values()))
        weighted_mu_sum = torch.zeros_like(first_mu)
        precision_sum = torch.zeros_like(first_mu)

        if self.include_prior_expert:
            # Prior expert: mu=0, logvar=0 -> precision=1. Contributes
            # nothing to weighted_mu_sum (mu=0) but 1 to precision_sum.
            precision_sum = precision_sum + 1.0

        for mu, logvar in params.values():
            precision = torch.exp(-logvar)
            weighted_mu_sum = weighted_mu_sum + precision * mu
            precision_sum = precision_sum + precision

        precision_sum = precision_sum + self.eps
        fused_mu = weighted_mu_sum / precision_sum
        fused_logvar = -torch.log(precision_sum)
        return fused_mu, fused_logvar

    @property
    def handlesMissingModalities(self) -> bool:
        """PoE natively tolerates a partial `params` dict (spec §4, §5).

        Returns:
            `True`.
        """
        return True
