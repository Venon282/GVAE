"""KL-divergence aggregation across one or more latent spaces (spec §2.2, §11).

`LatentSpace.klDivergence` (`latent/base.py`) computes the raw per-space
KL divergence to that space's own prior. This module is the policy
layer that sums the (optionally weighted) KL terms across every active
latent space into the model's total KL loss, so the weighting scheme
can change (spec §11: the precise beta schedule is still an open
question) without touching `LatentSpace` or the model class.
"""

import torch

from global_vae.latent.base import LatentSpace


def computeTotalKlLoss(
    latent_spaces: dict[str, LatentSpace],
    latent_params: dict[str, tuple[torch.Tensor, torch.Tensor]],
    beta: dict[str, float] | float = 1.0,
) -> torch.Tensor:
    """Sum the batch-averaged KL divergence across active latent spaces.

    Args:
        latent_spaces: Latent space name -> `LatentSpace`, as stored on
            the model.
        latent_params: Latent space name -> `(mu, logvar)` for whichever
            latent spaces were active in this forward pass. A subset of
            `latent_spaces` is accepted, e.g. when some latent spaces
            had no encoder input available (spec §5).
        beta: A single weight applied to every latent space, or a
            per-latent-space weight dict. Defaults to `1.0` (an
            unweighted sum, i.e. the plain multi-latent ELBO). The
            exact schedule this should follow is still open (spec §11).

    Returns:
        Scalar KL loss, summed across active latent spaces and
        averaged over the batch.

    Raises:
        ValueError: If `latent_params` is empty.
        KeyError: If `latent_params` references a name absent from
            `latent_spaces`.
    """
    if not latent_params:
        raise ValueError("computeTotalKlLoss received an empty `latent_params`.")

    total: torch.Tensor | None = None
    for latent_name, (mu, logvar) in latent_params.items():
        latent_space = latent_spaces[latent_name]
        weight = beta.get(latent_name, 1.0) if isinstance(beta, dict) else beta
        term = weight * latent_space.klDivergence(mu, logvar).mean()
        total = term if total is None else total + term

    assert total is not None  # guaranteed by the emptiness check above
    return total
