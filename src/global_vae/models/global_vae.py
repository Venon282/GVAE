"""Assembles a full model instance from an explicit routing graph.

`GlobalVae` does not hardcode any single row of the configuration
matrix (spec §2.1). It is built from a `RoutingGraph` (spec §2.2): any
number of per-modality encoders, any number of independent latent
spaces, and any number of decoders, wired together by the graph.
`createSingleLatent()` is a convenience constructor for the `EN-L1-DN`
Phase-1 default (ADR 0001); it is not a separate code path, just a
`RoutingGraph` built from the `latent.single` preset before delegating
to `__init__`.

Fan-out from one encoder to several latent spaces is not yet
supported, whether that encoder is a single shared trunk (the `E1-*`
rows) or one of several per-modality encoders (an `EN-*` row where the
same encoder is also assigned to more than one latent space, as in the
shared-plus-private topology). `AbstractEncoder.forward` returns one
`(mu, logvar)` pair; reusing it as-is for more than one independent
latent space would give those spaces identical posteriors rather than
the independent ones the routing graph promises. `__init__` rejects
this case explicitly instead of silently producing the wrong shapes.
See `docs/adr/0002-generalize-global-vae-to-routing-graph.md`.

Each latent space also gets its own regularization strategy (spec
§2.3), selected per latent space exactly like Fusion, defaulting to
`"kl_standard_normal"` when unspecified. See
`docs/adr/0003-pluggable-latent-regularization.md`.
"""

from typing import Any

import torch
from torch import nn

from global_vae.assemblers.registry import getAssemblerClass
from global_vae.decoders.registry import getDecoderClass
from global_vae.encoders.registry import getEncoderClass
from global_vae.fusion.registry import getFusionClass
from global_vae.latent.base import RoutingGraph, validateRoutingGraph
from global_vae.latent.routing_graph_builders.single import buildSingleLatentRoutingGraph
from global_vae.losses.regularization import computeTotalRegularizationLoss
from global_vae.losses.regularizers.registry import getRegularizerClass


class GlobalVae(nn.Module):
    """Config-driven multimodal VAE assembled from an arbitrary routing graph.

    Attributes:
        routing_graph: The `RoutingGraph` this model was built from.
        latent_spaces: Latent space name -> `LatentSpace`, taken from
            `routing_graph.latent_specs`.
        encoders: Encoder name -> encoder module.
        decoders: Decoder name -> decoder module.
        fusions: Latent space name -> fusion module, one entry per
            latent space fed by more than one encoder.
        assemblers: Decoder name -> assembler module, one entry per
            decoder consuming more than one latent space.
        regularizers: Latent space name -> regularizer module (spec
            §2.3), one entry per latent space in `routing_graph`.
            Defaults to `"kl_standard_normal"` for any latent space not
            given an explicit entry in `regularizer_strategies`.
    """

    def __init__(
        self,
        encoder_configs: dict[str, str],
        decoder_configs: dict[str, str],
        routing_graph: RoutingGraph,
        fusion_strategies: dict[str, str] | None = None,
        regularizer_strategies: dict[str, str] | None = None,
        encoder_kwargs: dict[str, dict[str, Any]] | None = None,
        decoder_kwargs: dict[str, dict[str, Any]] | None = None,
        fusion_kwargs: dict[str, dict[str, Any]] | None = None,
        regularizer_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Build a `GlobalVae` instance from a routing graph.

        Args:
            encoder_configs: Encoder name -> registry name (spec §9).
                Encoder names must match the encoder names used in
                `routing_graph.encoder_to_latents`.
            decoder_configs: Decoder name -> registry name. Decoder
                names must match the decoder names used in
                `routing_graph.latent_to_decoders` /
                `routing_graph.decoder_assemblers`. Kept independent
                from `encoder_configs` so a single shared decoder (the
                `*-D1` rows of spec §2.1) can be registered under its
                own name instead of reusing a modality name.
            routing_graph: Encoder, latent, and decoder wiring (spec
                §2.2). Validated before the model is built.
            fusion_strategies: Latent space name -> fusion registry
                name. Required for every latent space fed by more than
                one encoder; ignored for latent spaces fed by exactly
                one.
            regularizer_strategies: Latent space name -> latent
                regularizer registry name (spec §2.3, `losses/
                regularizers/`). Every latent space in `routing_graph`
                gets its own regularizer module; any latent space
                absent from this dict defaults to
                `"kl_standard_normal"`, so the common case (every
                latent space regularized the same, default way) needs
                no entries here at all.
            encoder_kwargs: Optional per-encoder constructor kwargs.
            decoder_kwargs: Optional per-decoder constructor kwargs.
            fusion_kwargs: Optional per-latent-space constructor kwargs
                for fusion modules.
            regularizer_kwargs: Optional per-latent-space constructor
                kwargs for regularizer modules.

        Raises:
            ValueError: If `routing_graph` is invalid (spec §2.2), or
                if a latent space fed by more than one encoder has no
                entry in `fusion_strategies`.
            NotImplementedError: If any encoder in `routing_graph` is
                assigned to more than one latent space (fan-out is not
                yet supported).
            KeyError: If `regularizer_strategies` (or its default,
                `"kl_standard_normal"`) names a strategy absent from
                the regularizer registry.
        """
        super().__init__()
        validateRoutingGraph(routing_graph)
        for encoder_name, latent_names in routing_graph.encoder_to_latents.items():
            if len(latent_names) > 1:
                raise NotImplementedError(
                    f"Encoder '{encoder_name}' is assigned to {len(latent_names)} latent "
                    f"spaces {latent_names}. Encoder fan-out to several latent spaces is "
                    f"not yet supported: `AbstractEncoder.forward` returns a single "
                    f"`(mu, logvar)` pair, which cannot correctly serve as an independent "
                    f"posterior for more than one latent space. See "
                    f"docs/adr/0002-generalize-global-vae-to-routing-graph.md."
                )

        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}
        fusion_kwargs = fusion_kwargs or {}
        regularizer_kwargs = regularizer_kwargs or {}
        fusion_strategies = fusion_strategies or {}
        regularizer_strategies = regularizer_strategies or {}

        self.routing_graph = routing_graph
        self.latent_spaces = routing_graph.latent_specs
        self._feeding_encoders = self._encodersFeeding(routing_graph)

        self.encoders = nn.ModuleDict(
            {
                name: getEncoderClass(registry_name)(**encoder_kwargs.get(name, {}))
                for name, registry_name in encoder_configs.items()
            }
        )
        self.decoders = nn.ModuleDict(
            {
                name: getDecoderClass(registry_name)(**decoder_kwargs.get(name, {}))
                for name, registry_name in decoder_configs.items()
            }
        )

        fusions: dict[str, nn.Module] = {}
        for latent_name, encoder_names in self._feeding_encoders.items():
            if len(encoder_names) <= 1:
                continue
            if latent_name not in fusion_strategies:
                raise ValueError(
                    f"Latent space '{latent_name}' is fed by {len(encoder_names)} "
                    f"encoders {encoder_names}, but has no entry in `fusion_strategies`."
                )
            strategy = fusion_strategies[latent_name]
            fusions[latent_name] = getFusionClass(strategy)(**fusion_kwargs.get(latent_name, {}))
        self.fusions = nn.ModuleDict(fusions)

        self.regularizers = nn.ModuleDict(
            {
                latent_name: getRegularizerClass(
                    regularizer_strategies.get(latent_name, "kl_standard_normal")
                )(**regularizer_kwargs.get(latent_name, {}))
                for latent_name in routing_graph.latent_specs
            }
        )

        self.assemblers = nn.ModuleDict(
            {
                decoder_name: getAssemblerClass(assembler_name)()
                for decoder_name, assembler_name in routing_graph.decoder_assemblers.items()
            }
        )

    @classmethod
    def createSingleLatent(
        cls,
        modality_configs: dict[str, dict[str, str]],
        latent_dim: int,
        fusion_strategy: str | None = None,
        latent_name: str = "z_fused",
        regularizer_strategy: str = "kl_standard_normal",
        **kwargs: Any,
    ) -> "GlobalVae":
        """Convenience constructor for the `EN-L1-DN` Phase-1 default.

        Builds the single-latent routing graph via
        `latent.single.buildSingleLatentRoutingGraph` instead of
        re-deriving it, then delegates to `__init__`. Unlike the
        general constructor, this assumes one encoder and one decoder
        per modality, sharing the modality name as their key, which is
        exactly the `EN-L1-DN` case (per-modality encoders and
        decoders); it is not meant for `*-D1` topologies.

        Args:
            modality_configs: Modality name -> `{"encoder": registry_name,
                "decoder": registry_name}` (spec §9). A single entry
                (single-modality signal VAE, spec §6.1 milestone 1) is
                just as valid as several.
            latent_dim: Dimensionality of the single latent space.
            fusion_strategy: Fusion registry name used to combine every
                encoder into the single latent space. Only meaningful,
                and only required, when `modality_configs` has more
                than one entry (spec §4: a latent space fed by exactly
                one encoder needs no fusion strategy at all). Defaults
                to `None`, the correct value for the single-modality
                case; passing `None` with more than one modality
                delegates to `__init__`, which raises `ValueError`
                pointing at the missing `fusion_strategies` entry.
            latent_name: Identifier for the single latent space.
            regularizer_strategy: Latent regularizer registry name
                (spec §2.3) used for the single latent space. Defaults
                to `"kl_standard_normal"`, the plain VAE regularization
                term.
            **kwargs: Forwarded to `__init__` (`encoder_kwargs`,
                `decoder_kwargs`, `fusion_kwargs`, `regularizer_kwargs`).

        Returns:
            A `GlobalVae` instance wired as `EN-L1-DN` (or as the
            single-modality `signal -> z -> signal` case of spec §6.1
            milestone 1, when `modality_configs` has one entry).

        Raises:
            ValueError: If `modality_configs` has more than one entry
                and `fusion_strategy` is `None` (delegated to
                `__init__`; see its own `Raises` section).
        """
        routing_graph = buildSingleLatentRoutingGraph(
            encoder_names=list(modality_configs),
            decoder_names=list(modality_configs),
            latent_dim=latent_dim,
            latent_name=latent_name,
        )
        encoder_configs = {name: cfg["encoder"] for name, cfg in modality_configs.items()}
        decoder_configs = {name: cfg["decoder"] for name, cfg in modality_configs.items()}
        fusion_strategies = {latent_name: fusion_strategy} if fusion_strategy is not None else {}
        return cls(
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            routing_graph=routing_graph,
            fusion_strategies=fusion_strategies,
            regularizer_strategies={latent_name: regularizer_strategy},
            **kwargs,
        )

    @staticmethod
    def _encodersFeeding(routing_graph: RoutingGraph) -> dict[str, list[str]]:
        """Invert `encoder_to_latents` into latent name -> feeding encoders.

        Args:
            routing_graph: The routing graph to invert.

        Returns:
            Latent space name -> list of encoder names feeding it.
        """
        result: dict[str, list[str]] = {name: [] for name in routing_graph.latent_specs}
        for encoder_name, latent_names in routing_graph.encoder_to_latents.items():
            for latent_name in latent_names:
                result[latent_name].append(encoder_name)
        return result

    def forward(
        self, inputs: dict[str, torch.Tensor], use_mean: bool = False
    ) -> dict[str, dict[str, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Run one encode, fuse, sample, assemble, decode pass.

        Args:
            inputs: Modality name -> raw input tensor. Any non-empty
                subset of the configured modalities is accepted; actual
                missing-modality robustness depends on the fusion
                strategy in use for each latent space (spec §5).
            use_mean: If `False` (default, unchanged from every prior
                behavior of this method), each latent space is sampled
                via the reparameterization trick
                (`LatentSpace.reparameterize`), matching what training
                needs for the ELBO objective. If `True`, each latent
                space instead deterministically uses its posterior
                mean `mu` in place of a sample, everywhere a latent
                value is consumed downstream (assemblers, decoders, and
                `"latent_samples"` in the return value). Standard
                practice for reporting reconstruction quality
                (`evaluation.evaluate`, spec §10/§C8): it removes
                sampling noise as a source of variance between runs,
                which sampling deliberately keeps during training.

        Returns:
            A dict with keys `"reconstructions"` (decoder name ->
            reconstruction tensor), `"latent_params"` (latent space
            name -> `(mu, logvar)`), and `"latent_samples"` (latent
            space name -> sampled, or `mu` if `use_mean=True`, `z`). A
            latent space or decoder with no available input this pass
            is simply absent from the corresponding dict, not raised
            as an error.

        Raises:
            ValueError: If `inputs` is empty.
        """
        if not inputs:
            raise ValueError("GlobalVae.forward() requires at least one modality in `inputs`.")

        encoder_outputs = {
            encoder_name: self.encoders[encoder_name](x) for encoder_name, x in inputs.items()
        }

        latent_params: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        latent_samples: dict[str, torch.Tensor] = {}
        for latent_name, encoder_names in self._feeding_encoders.items():
            active = [name for name in encoder_names if name in encoder_outputs]
            if not active:
                continue
            if len(active) == 1:
                mu, logvar = encoder_outputs[active[0]]
            else:
                params = {name: encoder_outputs[name] for name in active}
                mu, logvar = self.fusions[latent_name](params)
            latent_params[latent_name] = (mu, logvar)
            latent_samples[latent_name] = (
                mu if use_mean else self.latent_spaces[latent_name].reparameterize(mu, logvar)
            )

        reconstructions: dict[str, torch.Tensor] = {}
        for decoder_name, decoder in self.decoders.items():
            consumed = [
                latent_name
                for latent_name, decoder_names in self.routing_graph.latent_to_decoders.items()
                if decoder_name in decoder_names and latent_name in latent_samples
            ]
            if not consumed:
                continue
            if len(consumed) == 1:
                z_in = latent_samples[consumed[0]]
            else:
                z_in = self.assemblers[decoder_name]([latent_samples[name] for name in consumed])
            reconstructions[decoder_name] = decoder(z_in)

        return {
            "reconstructions": reconstructions,
            "latent_params": latent_params,
            "latent_samples": latent_samples,
        }

    def computeRegularizationLoss(
        self,
        latent_params: dict[str, tuple[torch.Tensor, torch.Tensor]],
        beta: dict[str, float] | float = 1.0,
    ) -> torch.Tensor:
        """Batch-averaged regularization penalty, summed across active latent spaces.

        Delegates the per-space computation to each latent space's own
        `AbstractLatentRegularizer` (`self.regularizers`, spec §2.3)
        and the cross-space aggregation to
        `losses.regularization.computeTotalRegularizationLoss`, so
        neither the regularization strategy nor the weighting scheme
        (beta, spec §2.3) is hardcoded into this model class. Renamed
        from `computeKlLoss`: the default strategy is still
        KL-to-standard-normal, but this method is no longer
        KL-specific now that `self.regularizers` is pluggable.

        Args:
            latent_params: Latent space name -> `(mu, logvar)`, as
                returned by `forward()`.
            beta: A single weight applied to every latent space, or a
                per-latent-space weight dict, forwarded unchanged to
                `computeTotalRegularizationLoss`. Defaults to `1.0`
                (unweighted). A caller using pluggable beta schedules
                (`training/beta_schedules/`) resolves them for the
                current step via `resolveBetaSchedules(...)` and
                passes the result here, exactly as it would pass any
                hand-written `dict[str, float]` (spec §2.3, ADR 0004):
                this method needs no awareness that schedules exist.

        Returns:
            Scalar regularization loss, summed over latent spaces and
            averaged over the batch.

        Raises:
            ValueError: If `latent_params` is empty.
        """
        return computeTotalRegularizationLoss(self.regularizers, latent_params, beta=beta)
