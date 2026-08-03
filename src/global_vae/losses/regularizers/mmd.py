"""Maximum Mean Discrepancy regularizer (spec §2.3, §11 candidate; WAE/InfoVAE-style).

Unlike KL-based strategies, MMD does not compare distribution
parameters in closed form: it compares the *empirical* distribution of
a batch of posterior samples against a batch of prior samples, via a
kernel mean embedding (Gretton et al., 2012). This is the WAE-MMD
(Tolstikhin et al., 2018) / InfoVAE (Zhao et al., 2017) regularization
term: it does not require the posterior to be pointwise close to the
prior (as KL does for every single sample), only that the *aggregate*
posterior distribution over the whole data set looks like the prior,
which is the property VAE generation actually depends on. This makes
MMD an alternative to (not just a variant of) KL-to-standard-normal:
it is known to be less prone to posterior collapse, precisely because
it stops rewarding a degenerate posterior that collapses onto the
prior sample-by-sample.

Reparameterized samples, not `(mu, logvar)` directly, are what the
kernel operates on, so this strategy draws its own posterior sample
here (via the same reparameterization trick used everywhere else in
this codebase, e.g. `LatentSpace.reparameterize`) rather than
receiving an already-sampled `z`. This mirrors the constraint already
noted on `AbstractLatentRegularizer.forward`: the shared
`(mu, logvar)` signature is what keeps every strategy interchangeable
through the same registry, even ones (like this one) that need a
sample to do their work.
"""

import torch

from global_vae.losses.regularizers.base import AbstractLatentRegularizer
from global_vae.losses.regularizers.registry import registerRegularizer


def _pairwiseSquaredDistances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute all pairwise squared Euclidean distances between two sets of vectors.

    Args:
        x: Shape `(n, dim)`.
        y: Shape `(m, dim)`.

    Returns:
        Shape `(n, m)`, clamped to `>= 0` (guards against small
        negative values from floating-point cancellation in the
        `|x|^2 + |y|^2 - 2 x.y` expansion, which would otherwise
        occasionally push kernel values fractionally above their
        mathematical maximum for near-identical vectors).
    """
    x_sq = x.pow(2).sum(dim=-1, keepdim=True)
    y_sq = y.pow(2).sum(dim=-1, keepdim=True)
    distances = x_sq + y_sq.transpose(0, 1) - 2.0 * (x @ y.transpose(0, 1))
    return distances.clamp(min=0.0)


def _rbfKernel(x: torch.Tensor, y: torch.Tensor, scales: tuple[float, ...]) -> torch.Tensor:
    """Multi-scale (mixture of) RBF/Gaussian kernel, following common MMD-VAE practice.

    A mixture of several bandwidths is used instead of one fixed
    bandwidth so the estimate is not overly sensitive to a single
    bandwidth choice; each bandwidth is scaled by the ambient
    dimensionality (`2 * dim * scale`), the standard normalization
    that keeps the kernel well-behaved as `dim` changes.

    Args:
        x: Shape `(n, dim)`.
        y: Shape `(m, dim)`.
        scales: Bandwidth multipliers to mix over.

    Returns:
        Shape `(n, m)` kernel matrix.
    """
    dim = x.shape[-1]
    distances = _pairwiseSquaredDistances(x, y)
    total = torch.zeros_like(distances)
    for scale in scales:
        bandwidth = 2.0 * dim * scale
        total = total + torch.exp(-distances / bandwidth)
    return total


def _imqKernel(
    x: torch.Tensor, y: torch.Tensor, scales: tuple[float, ...], base_scale: float | None
) -> torch.Tensor:
    """Multi-scale inverse multiquadratic kernel (the WAE-MMD paper's preferred choice).

    Heavier-tailed than RBF, which the WAE-MMD paper (Tolstikhin et
    al., 2018) argues matches a standard normal prior in moderate to
    high latent dimensionality better than RBF's fast decay does.

    Args:
        x: Shape `(n, dim)`.
        y: Shape `(m, dim)`.
        scales: Multipliers applied to `base_scale` to form a mixture.
        base_scale: The kernel's characteristic scale `C` before the
            `scales` multipliers. Defaults (`None`) to `2 * dim`, the
            expected squared norm of a `dim`-dimensional standard
            normal sample, mirroring the RBF bandwidth's normalization.

    Returns:
        Shape `(n, m)` kernel matrix.
    """
    dim = x.shape[-1]
    resolved_base_scale = base_scale if base_scale is not None else 2.0 * dim
    distances = _pairwiseSquaredDistances(x, y)
    total = torch.zeros_like(distances)
    for scale in scales:
        c = resolved_base_scale * scale
        total = total + c / (c + distances)
    return total


@registerRegularizer("mmd")
class MmdRegularizer(AbstractLatentRegularizer):
    """Maximum Mean Discrepancy between reparameterized posterior samples and the prior.

    Note on the shared `(mu, logvar) -> (batch,)` regularizer
    interface: MMD is a *batch-level* statistic (it compares the
    empirical distribution of the whole batch of samples to the
    prior), not a sum of independent per-sample terms the way KL is.
    `forward` returns the same scalar MMD value repeated across every
    entry of the `(batch,)` output, so that `.mean()` (used by
    `losses.regularization.computeTotalRegularizationLoss` to
    aggregate every strategy the same way) recovers that scalar
    exactly, instead of accidentally rescaling it by `1 / batch_size`.
    This is a documented adaptation to the interface, not a
    per-sample decomposition of MMD: unlike KL, no individual sample
    in the batch "owns" a fraction of this penalty.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        scales: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
        imq_base_scale: float | None = None,
        num_prior_samples: int | None = None,
    ) -> None:
        """Initialize the regularizer.

        Args:
            kernel: `"rbf"` (Gaussian, the common MMD-VAE default) or
                `"imq"` (inverse multiquadratic, the original WAE-MMD
                paper's choice for its heavier tails).
            scales: Bandwidth/scale multipliers mixed together to form
                the kernel, rather than committing to a single
                bandwidth (spec §10: avoid hardcoding a single
                "significant" value where a sensible default with room
                to override is possible).
            imq_base_scale: Only used when `kernel="imq"`. The kernel's
                base characteristic scale before `scales` is applied.
                Defaults (`None`) to `2 * latent_dim`, matching the RBF
                bandwidth's own normalization.
            num_prior_samples: Number of prior samples to draw per
                `forward` call. Defaults (`None`) to the batch size,
                the standard one-prior-sample-per-posterior-sample
                convention; can be set higher for a lower-variance MMD
                estimate at extra compute cost.

        Raises:
            ValueError: If `kernel` is not `"rbf"` or `"imq"`.
        """
        super().__init__()
        if kernel not in ("rbf", "imq"):
            raise ValueError(f"Unknown kernel '{kernel}'. Expected 'rbf' or 'imq'.")
        self.kernel = kernel
        self.scales = tuple(scales)
        self.imq_base_scale = imq_base_scale
        self.num_prior_samples = num_prior_samples

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute the (batch-level) MMD penalty, broadcast to a per-sample shape.

        Args:
            mu: Posterior mean, shape `(batch, dim)`.
            logvar: Posterior log-variance, shape `(batch, dim)`.

        Returns:
            Shape `(batch,)`, every entry equal to the same scalar MMD
            estimate (see the class docstring).

        Raises:
            ValueError: If `batch < 2`. The unbiased MMD estimator
                used here needs at least 2 samples per side to form
                its within-group (off-diagonal) terms.
        """
        batch_size = mu.shape[0]
        if batch_size < 2:
            raise ValueError(
                f"MmdRegularizer needs at least 2 samples per batch to estimate MMD, got "
                f"{batch_size}."
            )

        std = torch.exp(0.5 * logvar)
        posterior_samples = mu + torch.randn_like(std) * std

        num_prior = self.num_prior_samples or batch_size
        prior_samples = torch.randn(num_prior, mu.shape[-1], device=mu.device, dtype=mu.dtype)

        if self.kernel == "rbf":
            k_pp = _rbfKernel(posterior_samples, posterior_samples, self.scales)
            k_qq = _rbfKernel(prior_samples, prior_samples, self.scales)
            k_pq = _rbfKernel(posterior_samples, prior_samples, self.scales)
        else:
            k_pp = _imqKernel(
                posterior_samples, posterior_samples, self.scales, self.imq_base_scale
            )
            k_qq = _imqKernel(prior_samples, prior_samples, self.scales, self.imq_base_scale)
            k_pq = _imqKernel(posterior_samples, prior_samples, self.scales, self.imq_base_scale)

        n = posterior_samples.shape[0]
        m = prior_samples.shape[0]
        # Unbiased estimator: exclude the diagonal (self-similarity) from the
        # within-group terms, which would otherwise bias the estimate toward
        # a spuriously low MMD (every vector is maximally similar to itself).
        pp_term = (k_pp.sum() - k_pp.diagonal().sum()) / (n * (n - 1))
        qq_term = (k_qq.sum() - k_qq.diagonal().sum()) / (m * (m - 1))
        pq_term = k_pq.sum() / (n * m)

        mmd = pp_term + qq_term - 2.0 * pq_term
        return mmd.expand(batch_size)
