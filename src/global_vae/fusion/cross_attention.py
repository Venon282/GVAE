"""Cross-attention / transformer fusion strategy (spec §4).

Treats each active modality's `(mu, logvar)` as one token and fuses the
resulting token sequence with a small stack of Transformer encoder
layers (self-attention), following spec §4's own description: "Treat
each modality's encoding as a token; a transformer block fuses them."
Natively subset-tolerant (spec §5): a missing modality is simply an
omitted token, and self-attention followed by mean-pooling over the
token axis is defined for any non-empty number of tokens, so nothing
about this strategy depends on how many modalities happen to be active
this forward pass. Spec §4 also names this the "most natural bridge
toward the long-term world model direction" (spec §7): it is the one
Fusion strategy already built from the same attention machinery that
direction would need more of.
"""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import registerFusion


@registerFusion("cross_attention")
class CrossAttentionFusion(AbstractFusion):
    """Fuses per-modality posteriors with a Transformer encoder over per-modality tokens.

    Each active modality's `(mu, logvar)` is concatenated and linearly
    projected into one `d_model`-wide token; an optional learned
    per-modality embedding is added so the fusion can tell tokens
    apart by *which* modality they came from, not only their content.
    The token sequence (one token per active modality, in whatever
    order `forward` receives them) passes through `num_layers` stacked
    `nn.TransformerEncoderLayer`s, is mean-pooled over the token axis
    (permutation-invariant and defined for any non-empty token count,
    which is exactly what makes this subset-tolerant), and the pooled
    vector is projected to the fused `(mu, logvar)`.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int | None = None,
        num_heads: int = 4,
        num_layers: int = 1,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        known_modalities: Sequence[str] | None = None,
    ) -> None:
        """Build the fusion module.

        Args:
            latent_dim: Dimensionality of every incoming `(mu, logvar)`
                pair, and of the fused output.
            d_model: Token width the transformer operates in. Defaults
                (`None`) to `latent_dim`. Each modality's own
                `(mu, logvar)` (concatenated to width `2 * latent_dim`)
                is linearly projected into this width before attention.
            num_heads: Number of self-attention heads. Must evenly
                divide `d_model` (or its default, `latent_dim`).
            num_layers: Number of stacked `nn.TransformerEncoderLayer`s.
            feedforward_dim: Hidden width of each transformer layer's
                feed-forward block. Defaults (`None`) to `4 * d_model`,
                the standard transformer ratio.
            dropout: Dropout probability inside the transformer layers.
            activation: `nn.TransformerEncoderLayer`'s own `activation`
                argument (`"relu"` or `"gelu"`).
            known_modalities: If given, every listed modality name gets
                its own learned embedding added to its token, so the
                fusion can distinguish tokens by modality identity, not
                only content. `None` (default) skips modality identity
                entirely: fusion is then purely content-based, which
                still works, but two modalities that happen to produce
                identical `(mu, logvar)` become indistinguishable
                tokens. A modality name encountered at `forward` time
                that is not in `known_modalities` (when given) raises
                `ValueError` there, rather than silently running
                without an embedding for it.

        Raises:
            ValueError: If `d_model` (or its default, `latent_dim`) is
                not evenly divisible by `num_heads`.
        """
        super().__init__()
        resolved_d_model = d_model if d_model is not None else latent_dim
        if resolved_d_model % num_heads != 0:
            raise ValueError(
                f"CrossAttentionFusion: d_model={resolved_d_model} is not evenly divisible "
                f"by num_heads={num_heads}."
            )

        self.latent_dim = latent_dim
        self.d_model = resolved_d_model
        self.known_modalities = (
            frozenset(known_modalities) if known_modalities is not None else None
        )

        self.input_proj = nn.Linear(2 * latent_dim, resolved_d_model)

        self.modality_embeddings: nn.ParameterDict | None
        if self.known_modalities is not None:
            self.modality_embeddings = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.zeros(resolved_d_model))
                    for name in self.known_modalities
                }
            )
            for embedding in self.modality_embeddings.values():
                nn.init.normal_(embedding, std=0.02)
        else:
            self.modality_embeddings = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resolved_d_model,
            nhead=num_heads,
            dim_feedforward=(
                feedforward_dim if feedforward_dim is not None else 4 * resolved_d_model
            ),
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_mu = nn.Linear(resolved_d_model, latent_dim)
        self.to_logvar = nn.Linear(resolved_d_model, latent_dim)

    def forward(
        self, params: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize, self-attend over, and mean-pool every active modality's posterior.

        Args:
            params: Modality name -> `(mu, logvar)`. Any non-empty
                subset of the model's modalities is accepted (spec §5).

        Returns:
            The fused `(mu, logvar)` pair, shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `params` is empty, or if it contains a
                modality name absent from `known_modalities` when that
                was given at construction time.
        """
        if not params:
            raise ValueError("CrossAttentionFusion received an empty `params` dict.")
        if self.known_modalities is not None:
            unknown = set(params) - self.known_modalities
            if unknown:
                raise ValueError(
                    f"CrossAttentionFusion received modality(ies) {sorted(unknown)} that "
                    f"were not listed in `known_modalities` at construction time. Known "
                    f"modalities: {sorted(self.known_modalities)}."
                )

        tokens: list[torch.Tensor] = []
        for name in sorted(params):
            mu, logvar = params[name]
            token = self.input_proj(torch.cat([mu, logvar], dim=-1))
            if self.modality_embeddings is not None:
                token = token + self.modality_embeddings[name]
            tokens.append(token)

        stacked_tokens = torch.stack(tokens, dim=1)  # (batch, num_active_modalities, d_model)
        fused_tokens = self.transformer(stacked_tokens)
        pooled = fused_tokens.mean(dim=1)
        return self.to_mu(pooled), self.to_logvar(pooled)

    @property
    def handlesMissingModalities(self) -> bool:
        """Cross-attention natively tolerates a partial `params` dict (spec §4, §5).

        Returns:
            `True`.
        """
        return True
