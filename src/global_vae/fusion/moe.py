"""Mixture-of-Experts fusion strategy (spec §4).

Combines each modality's Gaussian "expert" `(mu, logvar)` as a mixture
(MMVAE-style, Shi et al., 2018), instead of PoE's product of densities:
where PoE's precision-weighted average always *sharpens* the fused
posterior as more experts agree, MoE instead reflects how much experts
*disagree*. Natively subset-tolerant: a missing modality simply drops
its component from the mixture, which is what gives MoE-based fusion
its own missing-modality robustness (spec §5), independent from and
complementary to PoE's.

Adaptation to this codebase's `AbstractFusion` interface (documented,
not hidden, exactly as `losses/regularizers/mmd.py`'s own docstring
documents its adaptation to the regularizer interface): a mixture of
Gaussians is not itself a Gaussian, but every downstream consumer of a
Fusion module's output (`LatentSpace.reparameterize`,
`AbstractLatentRegularizer.forward`) expects a single `(mu, logvar)`
pair. This module reduces the mixture to the single Gaussian that
shares its first two moments (mean and variance) via closed-form
moment matching, rather than a stochastic component draw. This keeps
Fusion's contract (a deterministic function of `params`, no sampling
inside Fusion itself) exactly as PoE's, and still gives the intended
MoE behaviour: the fused variance grows with how much the experts'
means disagree, on top of each expert's own variance, unlike PoE where
disagreement is not represented at all.
"""

import torch

from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion


@registerFusion("moe")
class MixtureOfExperts(AbstractFusion):
    """Mixture-of-Experts fusion via closed-form moment matching (MMVAE-style).

    Each encoder's `(mu, logvar)` is treated as one Gaussian mixture
    component. The fused `(mu, logvar)` is the single Gaussian that
    matches the mixture's first two moments::

        var_i = exp(logvar_i)
        mu_fused = sum_i(weight_i * mu_i)
        var_fused = sum_i(weight_i * (var_i + mu_i^2)) - mu_fused^2

    With a single active component, this reduces exactly to that
    component's own `(mu, logvar)` (the mixture has nothing to
    disagree with itself about), so a modality dropping in and out
    across training steps never introduces a discontinuity beyond the
    component itself changing.
    """

    def __init__(self, modality_weights: dict[str, float] | None = None, eps: float = 1e-8) -> None:
        """Initialize the fusion module.

        Args:
            modality_weights: Optional modality name -> non-negative
                mixture weight. Renormalized over whichever modalities
                are actually active in a given `forward` call, so this
                stays well-defined under missing modalities (spec §5).
                A modality active at `forward` time but absent from
                this dict falls back to a weight of `1.0` before
                renormalization, so a caller only needs to override the
                modalities whose relative weight should differ from
                the rest. `None` (default) weighs every active
                modality equally, the standard MMVAE convention.
            eps: Numerical-stability floor applied to the fused
                variance before taking its log, guarding against a
                tiny negative value from floating-point cancellation
                when every active component agrees almost exactly.

        Raises:
            ValueError: If `modality_weights` is given and contains a
                negative weight.
        """
        super().__init__()
        if modality_weights is not None and any(weight < 0 for weight in modality_weights.values()):
            raise ValueError(
                f"MixtureOfExperts: modality_weights must be non-negative, got {modality_weights}."
            )
        self.modality_weights = dict(modality_weights) if modality_weights is not None else None
        self.eps = eps

    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse per-modality mixture components via closed-form moment matching.

        Args:
            params: Modality name -> `(mu, logvar)`. Any non-empty
                subset of the model's modalities is accepted (spec §5).

        Returns:
            The fused `(mu, logvar)` pair, shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `params` is empty, or if every resolved
                mixture weight is zero (nothing to renormalize against).
        """
        if not params:
            raise ValueError("MixtureOfExperts received an empty `params` dict.")

        names = sorted(params)
        first_mu, _ = params[names[0]]
        raw_weights = [
            self.modality_weights.get(name, 1.0) if self.modality_weights is not None else 1.0
            for name in names
        ]
        weight_sum = sum(raw_weights)
        if weight_sum <= 0.0:
            active_weights = dict(zip(names, raw_weights, strict=True))
            raise ValueError(
                f"MixtureOfExperts: the active modalities' weights {active_weights} "
                f"sum to {weight_sum}, so there is nothing to renormalize against."
            )
        normalized_weights = [weight / weight_sum for weight in raw_weights]

        weighted_mu_sum = torch.zeros_like(first_mu)
        weighted_second_moment_sum = torch.zeros_like(first_mu)
        for name, weight in zip(names, normalized_weights, strict=True):
            mu, logvar = params[name]
            variance = torch.exp(logvar)
            weighted_mu_sum = weighted_mu_sum + weight * mu
            weighted_second_moment_sum = weighted_second_moment_sum + weight * (
                variance + mu.pow(2)
            )

        fused_mu = weighted_mu_sum
        fused_variance = (weighted_second_moment_sum - fused_mu.pow(2)).clamp(min=self.eps)
        fused_logvar = torch.log(fused_variance)
        return fused_mu, fused_logvar

    @property
    def handlesMissingModalities(self) -> bool:
        """MoE natively tolerates a partial `params` dict (spec §4, §5).

        Returns:
            `True`.
        """
        return True
