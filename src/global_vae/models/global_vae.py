"""Assembles a full model instance from encoders + fusion + latent + decoders.

Phase 1 implements the recommended default configuration (spec §2.1):
`EN-L1-DN` — per-modality encoders, a single fused latent space, and
per-modality decoders. The other 7 configurations in the matrix are
reachable through the same registries and the same `RoutingGraph`
validation (`latent/base.py`); they are not yet wired up in
`forward()`. Extending to them is the next milestone once
`tests/integration/` grows beyond the first of the 8 end-to-end
combinations called for in spec §10.
"""

from typing import Any

import torch
from torch import nn

from global_vae.decoders.registry import getDecoderClass
from global_vae.encoders.registry import getEncoderClass
from global_vae.fusion.registry import getFusionClass
from global_vae.latent.base import LatentSpace, RoutingGraph, validateRoutingGraph

class GlobalVae(nn.Module):
    """Config-driven multimodal VAE, `EN-L1-DN` configuration (spec §2.1).

    Attributes:
        encoders: Modality name -> encoder module.
        decoders: Modality name -> decoder module.
        fusion: Module combining per-modality `(mu, logvar)` into one
            fused posterior.
        latent: The single fused `LatentSpace`.
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, str]],
        fusion_strategy: str,
        latent_dim: int,
        encoder_kwargs: dict[str, dict[str, Any]] | None = None,
        decoder_kwargs: dict[str, dict[str, Any]] | None = None,
        fusion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Build a `GlobalVae` instance from a config.

        Args:
            modality_configs: Modality name -> `{"encoder": registry_name,
                "decoder": registry_name}` (mirrors spec §9's
                `model.modalities` block).
            fusion_strategy: Registry name of the fusion strategy
                (`"poe"`, `"moe"`, `"concat_mlp"`, or
                `"cross_attention"`).
            latent_dim: Dimensionality of the single fused latent space.
            encoder_kwargs: Optional per-modality constructor kwargs
                for encoders.
            decoder_kwargs: Optional per-modality constructor kwargs
                for decoders.
            fusion_kwargs: Optional constructor kwargs for the fusion
                module.

        Raises:
            ValueError: If the resulting routing graph is invalid
                (spec §2.2), e.g. an orphan latent space.
        """
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}
        fusion_kwargs = fusion_kwargs or {}

        self.encoders = nn.ModuleDict({
            modality: getEncoderClass(cfg["encoder"])(**encoder_kwargs.get(modality, {}))
            for modality, cfg in modality_configs.items()
        })
        self.decoders = nn.ModuleDict(
            {
                modality: getDecoderClass(cfg["decoder"])(**decoder_kwargs.get(modality, {}))
                for modality, cfg in modality_configs.items()
            }
        )
        self.fusion = getFusionClass(fusion_strategy(**fusion_kwargs))
        self.latent = LatentSpace(name="z_fused", dim=latent_dim)

        routing_graph = RoutingGraph(
            latent_specs={self.latent.name: self.latent},
            encoder_to_latents={modality: [self.latent.name] for modality in modality_configs},
            latent_to_decoders={self.latent.name: list(modality_configs)},
        )
        validateRoutingGraph(routing_graph)

    def forward(
            self, inputs: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
            """Run one encode -> fuse -> sample -> decode pass.

            Args:
                inputs: Modality name -> raw input tensor. Any non-empty
                    subset of the configured modalities is accepted; actual
                    missing-modality robustness depends on the fusion
                    strategy in use (spec §5).

            Returns:
                A dict with keys `"reconstructions"` (modality name ->
                reconstruction tensor), `"mu"`, `"logvar"`, and `"z"`.

            Raises:
                ValueError: If `inputs` is empty.
            """
            if not inputs:
                raise ValueError("GlobalVae.forward() requires at least one modality in `inputs`.")

            per_modality_params = {
                modality: self.encoders[modality](x) for modality, x in inputs.items()
            }
            mu, logvar = self.fusion(per_modality_params)
            z = self.latent.reparameterize(mu, logvar)
            reconstructions = {modality: decoder(z) for modality, decoder in self.decoders.items()}

            return {"reconstructions": reconstructions, "mu": mu, "logvar": logvar, "z": z}

    def computeKlLoss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Batch-averaged KL divergence for the fused latent space.

        Args:
            mu: Fused posterior mean, shape `(batch, latent_dim)`.
            logvar: Fused posterior log-variance, shape
                `(batch, latent_dim)`.

        Returns:
            Scalar KL loss, averaged over the batch.
        """
        return self.latent.klDivergence(mu, logvar).mean()
