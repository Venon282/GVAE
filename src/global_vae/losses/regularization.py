"""Latent regularization aggregation across one or more latent spaces (spec §2.2, §2.3).

Supersedes `losses/kl.py`: each latent space's regularization strategy
(`AbstractLatentRegularizer`, `losses/regularizers/`) computes its own
raw per-space penalty. This module is the policy layer that sums the
(optionally weighted) penalties across every active latent space into
the model's total regularization loss, so the weighting scheme (beta,
spec §11, still open) and the aggregation itself stay independent of
which strategy any given latent space uses.

`regularizers` is typed as `nn.ModuleDict` rather than
`Mapping[str, AbstractLatentRegularizer]` (or `dict[...]`) on purpose:
`GlobalVae.regularizers` must be an `nn.ModuleDict` for PyTorch to
register each strategy's parameters, buffers, and `.to(device)`
behavior, and `nn.ModuleDict` is not a nominal subtype of `Mapping`/
`dict` under `mypy --strict` even though it behaves like one at
runtime. Indexing it below narrows each entry back down from `Module`
to a callable regularizer; the underlying `nn.Module.__call__` return
type is `Any` regardless, exactly like every other module lookup in
`models/global_vae.py` (`self.encoders[name](x)`,
`self.fusions[name](params)`, ...).
"""

import torch
from torch import nn


def computeTotalRegularizationLoss(
    regularizers: nn.ModuleDict,
    latent_params: dict[str, tuple[torch.Tensor, torch.Tensor]],
    beta: dict[str, float] | float = 1.0,
) -> torch.Tensor:
    """Sum the batch-averaged regularization penalty across active latent spaces.

    Args:
        regularizers: Latent space name -> `AbstractLatentRegularizer`
            module, as stored on the model (`GlobalVae.regularizers`).
            Every latent space has its own entry (spec §2.3): the
            default is `kl_standard_normal`, but any latent space may
            use a different registered strategy (MMD, free-bits KL, a
            learned prior, ...) without this function needing to know
            which.
        latent_params: Latent space name -> `(mu, logvar)` for whichever
            latent spaces were active in this forward pass. A subset of
            `regularizers` is accepted, e.g. when some latent spaces
            had no encoder input available this pass (spec §5).
        beta: A single weight applied to every latent space, or a
            per-latent-space weight dict. Defaults to `1.0` (an
            unweighted sum, i.e. the plain multi-latent ELBO). The
            exact schedule this should follow is still open (spec §11).

    Returns:
        Scalar regularization loss, summed across active latent spaces
        and averaged over the batch.

    Raises:
        ValueError: If `latent_params` is empty.
        KeyError: If `latent_params` references a name absent from
            `regularizers`.
    """
    if not latent_params:
        raise ValueError("computeTotalRegularizationLoss received an empty `latent_params`.")

    total: torch.Tensor | None = None
    for latent_name, (mu, logvar) in latent_params.items():
        if latent_name not in regularizers:
            available = ", ".join(sorted(regularizers.keys())) or "(none)"
            raise KeyError(
                f"No regularizer registered for latent space '{latent_name}'. "
                f"Available: {available}"
            )
        regularizer = regularizers[latent_name]
        weight = beta.get(latent_name, 1.0) if isinstance(beta, dict) else beta
        term = weight * regularizer(mu, logvar).mean()
        total = term if total is None else total + term

    assert total is not None  # guaranteed by the emptiness check above
    return total
