"""Concatenation + MLP fusion strategy (spec §4).

The simplest baseline fusion: concatenate every modality's `(mu,
logvar)` into one vector and project it through a small MLP down to
the fused `(mu, logvar)`. Unlike PoE, MoE, and cross-attention, this
strategy is **not** natively subset-tolerant (spec §4, §5): a
concatenation has one fixed slot per modality, decided at construction
time (needed so the first `nn.Linear`'s input width is known), so a
missing modality cannot simply be omitted the way a dropped expert or
mixture component, or an omitted attention token, can.

Spec §5 requires that this limitation be handled by "an explicit
imputation/masking strategy", documented rather than silently patched
around. This module *does* implement such a scheme (so the strategy is
still usable when modalities go missing, e.g. under modality dropout,
spec §5's own recommended training technique) rather than merely
refusing: a missing modality's slot is filled with the standard-normal
prior's own parameters (`mu=0`, `logvar=0`), and, by default, a second
presence indicator (`1.0`/`0.0` per modality) is concatenated in too,
so the MLP can learn to distinguish "genuinely at the prior" from
"missing and imputed at the prior" instead of the two being
indistinguishable inputs. `handlesMissingModalities` still reports
`False`: this is an explicit, opt-out-able scheme this class provides,
not the free, automatic tolerance PoE/MoE/cross-attention have.
"""

from collections.abc import Callable, Mapping

import torch
from torch import nn

from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion


@registerFusion("concat_mlp")
class ConcatMlpFusion(AbstractFusion):
    """Concatenates every modality's `(mu, logvar)`, then projects through an MLP.

    Attributes:
        use_presence_mask: Whether a per-modality presence indicator is
            appended to the concatenated input (see the module
            docstring).
    """

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (128,),
        activation: Callable[[], nn.Module] | None = nn.ReLU,
        use_presence_mask: bool = True,
    ) -> None:
        """Build the fusion module.

        Args:
            modality_dims: Every modality this fusion may ever receive,
                mapped to that modality's own encoder output
                dimensionality (the width of its `mu`/`logvar`). Fixed
                at construction time: concatenation needs to know, up
                front, how many slots to reserve and in what order,
                unlike PoE/MoE/cross-attention, which read whichever
                subset of `params` is passed to a given `forward` call.
                Slot order is exactly this mapping's iteration order
                (insertion order for a plain `dict`), never re-sorted,
                so a caller controls it directly (e.g. matching a
                config file's own modality order) rather than it
                silently becoming alphabetical.
            latent_dim: Dimensionality of the fused `(mu, logvar)`
                output.
            hidden_dims: Hidden layer widths of the MLP between the
                concatenated input and the `to_mu`/`to_logvar` heads.
                Defaults to a single hidden layer of `128` units; pass
                `()` for a single linear layer straight from the
                concatenated input to each head.
            activation: Zero-argument factory returning a fresh
                activation module, applied after every hidden layer.
                Pass `None` to disable activation between hidden
                layers.
            use_presence_mask: See the module docstring and the class
                attribute of the same name.

        Raises:
            ValueError: If `modality_dims` is empty.
        """
        super().__init__()
        if not modality_dims:
            raise ValueError("ConcatMlpFusion requires at least one entry in `modality_dims`.")

        self._modality_order: list[str] = list(modality_dims)
        self._modality_dims: dict[str, int] = dict(modality_dims)
        self.use_presence_mask = use_presence_mask

        input_dim = sum(2 * dim for dim in modality_dims.values())
        if use_presence_mask:
            input_dim += len(modality_dims)

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if activation is not None:
                layers.append(activation())
            current_dim = hidden_dim
        self.mlp: nn.Module = nn.Sequential(*layers) if layers else nn.Identity()

        self.to_mu = nn.Linear(current_dim, latent_dim)
        self.to_logvar = nn.Linear(current_dim, latent_dim)

    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate every modality's `(mu, logvar)` (imputing missing ones) and project.

        Args:
            params: Modality name -> `(mu, logvar)`. Every key must
                have been declared in `modality_dims` at construction
                time; a modality declared there but absent from
                `params` is imputed (see the module docstring), not
                treated as an error.

        Returns:
            The fused `(mu, logvar)` pair, shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `params` is empty.
            KeyError: If `params` contains a modality name that was not
                declared in `modality_dims` at construction time.
        """
        if not params:
            raise ValueError("ConcatMlpFusion received an empty `params` dict.")
        unknown = set(params) - set(self._modality_dims)
        if unknown:
            raise KeyError(
                f"ConcatMlpFusion received modality(ies) {sorted(unknown)} that were not "
                f"declared in `modality_dims` at construction time. Declared modalities: "
                f"{sorted(self._modality_dims)}."
            )

        first_mu, _ = next(iter(params.values()))
        batch_size = first_mu.shape[0]
        device, dtype = first_mu.device, first_mu.dtype

        pieces: list[torch.Tensor] = []
        presence: list[torch.Tensor] = []
        for name in self._modality_order:
            dim = self._modality_dims[name]
            if name in params:
                mu, logvar = params[name]
                pieces.append(mu)
                pieces.append(logvar)
                if self.use_presence_mask:
                    presence.append(torch.ones(batch_size, 1, device=device, dtype=dtype))
            else:
                # Explicit, documented imputation for a missing modality (spec §5): the
                # standard-normal prior's own parameters, not an arbitrary sentinel value.
                pieces.append(torch.zeros(batch_size, dim, device=device, dtype=dtype))
                pieces.append(torch.zeros(batch_size, dim, device=device, dtype=dtype))
                if self.use_presence_mask:
                    presence.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))

        features = torch.cat([*pieces, *presence], dim=-1)
        hidden = self.mlp(features)
        return self.to_mu(hidden), self.to_logvar(hidden)

    @property
    def handlesMissingModalities(self) -> bool:
        """Concat+MLP needs an explicit imputation/masking scheme (spec §4, §5).

        Returns:
            `False`. See the module docstring: an explicit scheme is
            implemented and usable, but it is not the free, automatic
            tolerance PoE/MoE/cross-attention have.
        """
        return False
