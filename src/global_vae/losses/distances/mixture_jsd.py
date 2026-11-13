"""Jensen-Shannon divergence of the posteriors against their mixture (roadmap P6-3, ADR 0053)."""

import math
from collections.abc import Sequence
from typing import ClassVar

import torch

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.numerics import (
    checkDimensionReduction,
    checkLogvarRange,
    clampLogvar,
)
from global_vae.losses.distances.registry import registerLatentDistance


@registerLatentDistance("mixture_jsd")
class MixtureJsdDistance(AbstractLatentDistance):
    """Jensen-Shannon divergence of `K` diagonal Gaussian posteriors, by Monte Carlo.

    Each posterior `q_k` is compared with the mixture `m = 1/K * sum_j q_j` of all of them, and
    the value is `1/K * sum_k KL(q_k || m)`, in nats, between 0 and `log K`. The mixture of
    Gaussians has no closed-form KL, so each `KL(q_k || m)` is estimated from `num_samples`
    reparameterized draws `z ~ q_k`:

        `KL(q_k || m) = E_z[ log K - logsumexp_j (log q_j(z) - log q_k(z)) ]`.

    The gradient flows through the draws and through the densities. The rate term of MMVM
    (Sutter et al., NeurIPS 2024) is `M` times the Jensen-Shannon divergence of the unimodal
    posteriors, which `sum_over_spaces: true` returns. It takes any number of spaces.

    The estimate is unbiased and **not clamped**: for posteriors that nearly coincide one sample
    can come out slightly below 0, and the mean over the batch is what estimates the divergence.
    On identical posteriors every log-density ratio is 0, so the value is 0 whatever the draws are.
    It draws random numbers (`is_stochastic`), from the global generator, which the trainer saves
    in its checkpoints.
    """

    is_stochastic: ClassVar[bool] = True
    max_spaces: ClassVar[int | None] = None

    def __init__(
        self,
        num_samples: int = 1,
        dimension_reduction: str = "sum",
        sum_over_spaces: bool = False,
        min_logvar: float = -10.0,
        max_logvar: float = 10.0,
    ) -> None:
        """Build the distance.

        Args:
            num_samples: Draws from each posterior per sample of the batch. More draws lower the
                variance of the estimate and cost memory in proportion.
            dimension_reduction: `"sum"` (default, the natural scale of a divergence, the footing
                of the KL term, ADR 0029) or `"mean"` (the divergence divided by the width).
            sum_over_spaces: `False` (default) returns the Jensen-Shannon divergence, the mean
                over the spaces of `KL(q_k || m)`. `True` returns their sum, `K` times that, the
                form of the MMVM rate term.
            min_logvar: Lower clamp of the log-variances, which keeps the densities finite.
            max_logvar: Upper clamp of the log-variances.

        Raises:
            ValueError: If `num_samples` is not a positive integer, `dimension_reduction` is
                neither `"sum"` nor `"mean"`, or `min_logvar` is not below `max_logvar`.
        """
        super().__init__()
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(f"num_samples must be an integer of at least 1, got {num_samples!r}.")
        self.num_samples = num_samples
        self.dimension_reduction = checkDimensionReduction(dimension_reduction)
        self.sum_over_spaces = bool(sum_over_spaces)
        checkLogvarRange(min_logvar, max_logvar)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
        """Per-sample Monte Carlo estimate of the Jensen-Shannon divergence.

        Args:
            posteriors: The `(mu, logvar)` of the `K >= 2` latent spaces.

        Returns:
            Shape `(batch,)`. Each value is an unbiased estimate, between `-inf` and `log K` for
            one draw, and the same posteriors give exactly 0.
        """
        count = len(posteriors)
        mu = torch.stack([mu_k for mu_k, _ in posteriors])  # (K, B, D)
        logvar = torch.stack(
            [clampLogvar(logvar_k, self.min_logvar, self.max_logvar) for _, logvar_k in posteriors]
        )
        width = mu.shape[-1]

        # z[s, k] is a draw from posterior k, for every sample of the batch.
        noise = torch.randn(
            (self.num_samples, *mu.shape), dtype=mu.dtype, device=mu.device
        )  # (S, K, B, D)
        draws = mu.unsqueeze(0) + torch.exp(0.5 * logvar).unsqueeze(0) * noise

        # log_q[s, k, j, b] = log q_j(z[s, k, b]), up to a constant that cancels in the ratios.
        gap = draws.unsqueeze(2) - mu[None, None]  # (S, K, K, B, D)
        log_q = -0.5 * (gap.pow(2) * torch.exp(-logvar)[None, None] + logvar[None, None]).sum(-1)
        own = log_q[:, torch.arange(count), torch.arange(count)]  # (S, K, B)
        kl_to_mixture = math.log(count) - torch.logsumexp(log_q - own.unsqueeze(2), dim=2)
        value = kl_to_mixture.mean(dim=0)  # (K, B): the estimate of KL(q_k || m)

        value = value.sum(dim=0) if self.sum_over_spaces else value.mean(dim=0)
        return value / width if self.dimension_reduction == "mean" else value
