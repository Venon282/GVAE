"""Residual connection around any fusion strategy (spec §4: "a config flag alongside it").

Spec §4 says each fusion assignment "can optionally use residual connections (a config
flag alongside it)" and spec §9 shows `residual: true` next to `strategy: poe`, but neither
says what a residual connection means for a fusion. This module defines it once, for every
strategy at once, instead of once per strategy: a wrapper around an already-built
`AbstractFusion`, so a strategy added later gets the option for free (spec §10, §12).

**Definition.** Every fusion strategy in this codebase returns an *absolute* posterior
`(mu, logvar)`, not a correction to add to something. A literal `strategy(x) + x` would
therefore count the experts twice (two identical `poe` experts with mean `m` would fuse to
`2m`). Instead, the skip path is the parameter-free mean of the active experts, and the
strategy is treated as proposing a *correction relative to that skip*, scaled by a learned
gate::

    skip  = mean over the active experts of (mu_i, logvar_i)
    fused = skip + gate * (strategy(params) - skip)

with one scalar `gate` for `mu` and one for `logvar`, both initialized to `0.0` (the
ReZero / LayerScale initialization, Bachlechner et al., 2020; Touvron et al., 2021). This
gives the two properties that make a connection "residual":

- at initialization the fused posterior is exactly the mean of the active experts, whichever
  strategy is wrapped, so training starts from a sensible, strategy-independent baseline;
- `gate = 1` recovers the wrapped strategy exactly, so nothing is lost by enabling the flag.

Because the skip is a mean over whichever experts are present in `params`, it is defined for
any non-empty subset of modalities, so wrapping never changes `handlesMissingModalities`.

The wrapper needs every expert's `(mu, logvar)` to have the fused output's own shape, which
is what the single-latent architecture already guarantees (every encoder is built with the
latent dimensionality). A strategy that changes dimensionality (a `concat_mlp` whose
`latent_dim` differs from its experts' widths) has no well-defined skip path; that raises a
clear `ValueError` at `forward` instead of a shape error inside a broadcast.

Not registered under `@registerFusion`: like `data.transforms.compose.ComposeTransform`, this
is a combinator built from an already-resolved instance, not a named strategy selected by
string. `GlobalVae` applies it from the per-latent-space `fusion_residual` flag.

Enabling the flag changes the module tree (the wrapped strategy moves under `inner` and two
gate parameters are added), so a checkpoint saved with the flag off does not load into a
model built with it on, and vice versa: they are different architectures.
"""

import torch
from torch import nn

from global_vae.fusion.base import AbstractFusion


class ResidualFusion(AbstractFusion):
    """Wraps a fusion strategy with a learned-gate skip path from the mean of the experts.

    Attributes:
        inner: The wrapped fusion strategy.
        gate_mu: Learned scalar gating the correction applied to `mu`.
        gate_logvar: Learned scalar gating the correction applied to `logvar`.
    """

    def __init__(self, inner: AbstractFusion, initial_gate: float = 0.0) -> None:
        """Wrap `inner`.

        Args:
            inner: The fusion strategy to add a residual connection to. Any
                `AbstractFusion` works; the wrapper only calls its `forward`.
            initial_gate: Initial value of both gates. `0.0` (default) starts the fused
                posterior at exactly the mean of the experts (identity at initialization);
                `1.0` starts at the wrapped strategy's own output.
        """
        super().__init__()
        self.inner = inner
        self.gate_mu = nn.Parameter(torch.tensor(float(initial_gate)))
        self.gate_logvar = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse with the wrapped strategy, then blend toward the mean of the experts.

        Args:
            params: Modality name to `(mu, logvar)`, forwarded unchanged to the wrapped
                strategy, which decides which subsets it accepts.

        Returns:
            The fused `(mu, logvar)` pair, `skip + gate * (inner_output - skip)`, shape
            `(batch, latent_dim)`.

        Raises:
            ValueError: If `params` is empty, if the experts' `(mu, logvar)` do not all
                share one shape, or if that shape differs from the wrapped strategy's
                output shape (there is then no well-defined skip path).
        """
        if not params:
            raise ValueError("ResidualFusion received an empty `params` dict.")

        # The skip path is built first so that experts of mismatched shapes fail here, with
        # one clear message, instead of as whichever broadcast error the wrapped strategy
        # happens to raise first (each strategy would fail differently).
        skip_mu = self._meanOfExperts({name: mu for name, (mu, _) in params.items()}, "mu")
        skip_logvar = self._meanOfExperts(
            {name: logvar for name, (_, logvar) in params.items()}, "logvar"
        )
        fused_mu, fused_logvar = self.inner(params)
        if skip_mu.shape != fused_mu.shape:
            raise ValueError(
                f"ResidualFusion: the experts' shape {tuple(skip_mu.shape)} differs from the "
                f"wrapped {type(self.inner).__name__}'s output shape {tuple(fused_mu.shape)}, "
                f"so there is no skip path to add. A residual connection needs the experts and "
                f"the fused posterior to share one dimensionality."
            )

        return (
            skip_mu + self.gate_mu * (fused_mu - skip_mu),
            skip_logvar + self.gate_logvar * (fused_logvar - skip_logvar),
        )

    @staticmethod
    def _meanOfExperts(tensors: dict[str, torch.Tensor], what: str) -> torch.Tensor:
        """Average one parameter (`mu` or `logvar`) over the active experts.

        Args:
            tensors: Modality name to that modality's `mu` or `logvar`.
            what: `"mu"` or `"logvar"`, used only in the error message.

        Returns:
            The elementwise mean, same shape as each input.

        Raises:
            ValueError: If the experts' tensors do not all share one shape.
        """
        shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(
                f"ResidualFusion needs every expert's {what} to share one shape to average "
                f"them into a skip path, got {shapes}."
            )
        return torch.stack(list(tensors.values()), dim=0).mean(dim=0)

    @property
    def handlesMissingModalities(self) -> bool:
        """Mirrors the wrapped strategy: the mean skip path is defined for any subset.

        Returns:
            `self.inner.handlesMissingModalities`.
        """
        return self.inner.handlesMissingModalities
