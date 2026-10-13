"""Assembles a full model instance from an explicit routing graph.

`GlobalVae` does not hardcode any single row of the configuration
matrix (spec §2.1). It is built from a `RoutingGraph` (spec §2.2): any
number of per-modality encoders, any number of independent latent
spaces, and any number of decoders, wired together by the graph.
`createSingleLatent()` is a convenience constructor for the `EN-L1-DN`
Phase-1 default (ADR 0001); it is not a separate code path, just a
`RoutingGraph` built from the `latent.single` preset before delegating
to `__init__`.

An encoder may feed several latent spaces (the shared-plus-private
topology, or a single shared trunk over several latent spaces). Each
encoder -> latent edge then carries a latent head (`heads/`, spec §2.2)
that adapts the encoder's one `(mu, logvar)` pair to that latent space,
applied before fusion, so the spaces get their own posteriors instead of
the same one. The heads come from `RoutingGraph.encoder_heads`; an edge
with no entry is an identity edge and builds no module, so a graph
without heads builds and runs exactly as it did before heads existed.
`validateRoutingGraph` refuses an encoder with two identity edges, which
would give two spaces identical posteriors. See
`docs/adr/0024-encoder-fan-out-through-latent-heads.md`, which supersedes
the fan-out guard of
`docs/adr/0002-generalize-global-vae-to-routing-graph.md`.

Each latent space also gets its own regularization strategy (spec
§2.3), selected per latent space exactly like Fusion, defaulting to
`"kl_standard_normal"` when unspecified. See
`docs/adr/0003-pluggable-latent-regularization.md`.

`forward()`'s own `inputs` dict must be restricted to registered
encoder names; a raw per-modality batch that also carries decoder-only
reconstruction targets (a translation-style `image_in -> image_out`
model, where the target is not itself an encoder input) is filtered
down via `selectEncoderInputs` before it ever reaches `forward()`. See
`docs/adr/0019-decouple-encoder-inputs-from-decoder-targets.md`.
"""

import inspect
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import getAssemblerClass
from global_vae.decoders.registry import getDecoderClass
from global_vae.encoders.registry import getEncoderClass
from global_vae.fusion.base import AbstractFusion
from global_vae.fusion.registry import getFusionClass
from global_vae.fusion.residual import ResidualFusion
from global_vae.heads.base import AbstractLatentHead
from global_vae.heads.registry import getHeadClass
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
        heads: Encoder name -> latent space name -> latent head module,
            one entry per edge with a `HeadSpec` in
            `routing_graph.encoder_heads`. Empty for a graph without
            heads.
        fusions: Latent space name -> fusion module, one entry per
            latent space fed by more than one encoder.
        assemblers: Decoder name -> assembler module, one entry per
            decoder with an entry in `routing_graph.decoder_assemblers`
            (every decoder consuming more than one latent space has one).
            An assembler is built from the widths of the latent spaces its
            decoder consumes (`input_dims`), and its `output_dim` is the
            decoder's `latent_dim`.
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
        fusion_residual: dict[str, bool] | None = None,
        assembler_kwargs: dict[str, dict[str, Any]] | None = None,
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
            fusion_residual: Optional latent space name -> whether that
                latent space's fusion gets a residual connection (spec
                §4, `fusion.residual.ResidualFusion`). Absent or `False`
                leaves the strategy unwrapped, exactly as before. `True`
                is only valid for a latent space that actually has a
                fusion, i.e. one fed by more than one encoder.
            assembler_kwargs: Optional decoder name -> constructor kwargs
                for that decoder's assembler (`assemblers/`), next to the
                `input_dims` the model derives from the latent spaces the
                decoder consumes (`RoutingGraph.decoderInputs`), so
                `input_dims` is never given here. Absent means no kwargs,
                which is all `concat`, `sum` and `average` take and what
                `weighted_sum` and `attention` default to.

        Raises:
            ValueError: If `routing_graph` is invalid (spec §2.2, §10:
                this includes an encoder with two identity edges, and a
                head whose declared `out_dim` is not its latent space's
                `dim`), if a latent space fed by more than one encoder
                has no entry in `fusion_strategies`, if
                `fusion_residual` asks for a residual connection on a
                latent space with no fusion (unknown, or fed by fewer
                than two encoders), or if a latent head cannot be built:
                its encoder is absent from `encoder_configs`, exposes no
                integer `latent_dim`, or the head rejects its widths
                (`identity` needs the encoder's `latent_dim` to equal
                its latent space's `dim`), if an assembler cannot be built
                from its `assembler_kwargs` and the widths of its latent
                spaces, or if a decoder's `decoder_kwargs` set a `latent_dim`
                different from the width it receives (the latent space's
                `dim` for one latent space, the assembler's `output_dim`
                for several).
            KeyError: If `regularizer_strategies` (or its default,
                `"kl_standard_normal"`) names a strategy absent from
                the regularizer registry, if `routing_graph.
                encoder_heads` names a head absent from the head
                registry, or if `routing_graph.decoder_assemblers` names
                an assembler absent from the assembler registry.
        """
        super().__init__()
        validateRoutingGraph(routing_graph)

        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}
        fusion_kwargs = fusion_kwargs or {}
        regularizer_kwargs = regularizer_kwargs or {}
        fusion_strategies = fusion_strategies or {}
        regularizer_strategies = regularizer_strategies or {}
        assembler_kwargs = assembler_kwargs or {}

        self.routing_graph = routing_graph
        self.latent_spaces = routing_graph.latent_specs
        self._feeding_encoders = self._encodersFeeding(routing_graph)
        self._decoder_inputs = routing_graph.decoderInputs()
        # Built before the decoders, whose `latent_dim` is the assembler's `output_dim`, but
        # registered as `self.assemblers` after everything else, where it always was.
        assemblers = self._buildAssemblers(routing_graph, assembler_kwargs)

        fusion_residual = fusion_residual or {}
        fused_latent_names = {
            latent_name
            for latent_name, encoder_names in self._feeding_encoders.items()
            if len(encoder_names) > 1
        }
        no_fusion = sorted(
            name
            for name, enabled in fusion_residual.items()
            if enabled and name not in fused_latent_names
        )
        if no_fusion:
            raise ValueError(
                f"`fusion_residual` enables a residual connection for {no_fusion}, but a "
                f"residual connection wraps a fusion, and those latent spaces have none: "
                f"each is unknown or fed by fewer than two encoders. Latent spaces with a "
                f"fusion: {sorted(fused_latent_names)}."
            )

        self.encoders = nn.ModuleDict(
            {
                name: getEncoderClass(registry_name)(**encoder_kwargs.get(name, {}))
                for name, registry_name in encoder_configs.items()
            }
        )
        self.decoders = nn.ModuleDict(
            {
                name: self._buildDecoder(
                    name, registry_name, decoder_kwargs.get(name, {}), assemblers
                )
                for name, registry_name in decoder_configs.items()
            }
        )

        self.heads = nn.ModuleDict(self._buildHeads(routing_graph))

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
            fusion: AbstractFusion = getFusionClass(strategy)(**fusion_kwargs.get(latent_name, {}))
            if fusion_residual.get(latent_name, False):
                fusion = ResidualFusion(fusion)
            fusions[latent_name] = fusion
        self.fusions = nn.ModuleDict(fusions)

        self.regularizers = nn.ModuleDict(
            {
                latent_name: getRegularizerClass(
                    regularizer_strategies.get(latent_name, "kl_standard_normal")
                )(**regularizer_kwargs.get(latent_name, {}))
                for latent_name in routing_graph.latent_specs
            }
        )

        self.assemblers = nn.ModuleDict(assemblers)

    @classmethod
    def createSingleLatent(
        cls,
        modality_configs: dict[str, dict[str, str]],
        latent_dim: int,
        fusion_strategy: str | None = None,
        latent_name: str = "z_fused",
        regularizer_strategy: str = "kl_standard_normal",
        fusion_residual: bool = False,
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
            fusion_residual: Whether the single latent space's fusion gets
                a residual connection (spec §4). Only valid together with
                a fusion, i.e. more than one modality; `True` on a
                single-modality model raises `ValueError` (delegated to
                `__init__`).
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
            fusion_residual={latent_name: True} if fusion_residual else None,
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

    def _buildAssemblers(
        self,
        routing_graph: RoutingGraph,
        assembler_kwargs: dict[str, dict[str, Any]],
    ) -> dict[str, AbstractAssembler]:
        """Build one assembler per entry of `routing_graph.decoder_assemblers`.

        Each assembler is built as `getAssemblerClass(name)(input_dims,
        **kwargs)`, with `input_dims` the `dim` of each latent space its
        decoder consumes, in `RoutingGraph.decoderInputs` order, which is also
        the order `forward` hands the latent vectors over. An entry for a
        decoder that consumes no latent space is skipped: there is nothing to
        size it from, and it would never run.

        Args:
            routing_graph: The routing graph whose assemblers to build.
            assembler_kwargs: Decoder name -> constructor kwargs of its assembler.

        Returns:
            Decoder name -> assembler.

        Raises:
            ValueError: If an assembler cannot be built from its kwargs and
                the widths of its latent spaces.
            KeyError: If an assembler name is absent from the assembler registry.
        """
        assemblers: dict[str, AbstractAssembler] = {}
        for decoder_name, assembler_name in routing_graph.decoder_assemblers.items():
            latent_names = self._decoder_inputs.get(decoder_name)
            if not latent_names:
                continue
            input_dims = [routing_graph.latent_specs[name].dim for name in latent_names]
            try:
                assemblers[decoder_name] = getAssemblerClass(assembler_name)(
                    input_dims, **assembler_kwargs.get(decoder_name, {})
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Cannot build assembler '{assembler_name}' for decoder '{decoder_name}' "
                    f"(latent spaces {latent_names}, widths {input_dims}, kwargs "
                    f"{assembler_kwargs.get(decoder_name, {})}): {error}"
                ) from error
        return assemblers

    def _buildDecoder(
        self,
        decoder_name: str,
        registry_name: str,
        kwargs: Mapping[str, Any],
        assemblers: Mapping[str, AbstractAssembler],
    ) -> nn.Module:
        """Build one decoder, checking or filling in its `latent_dim`.

        The width a decoder receives is known from the routing graph: the `dim`
        of its latent space when it consumes one, the `output_dim` of its
        assembler otherwise. A `latent_dim` in `kwargs` must equal it, so a
        wrong value fails here, naming both widths, instead of at the first
        forward pass with a matrix shape error. When `kwargs` has no
        `latent_dim` and the decoder class takes one, it is filled in.

        Args:
            decoder_name: The decoder instance name.
            registry_name: The decoder registry name.
            kwargs: The decoder's constructor kwargs, not modified.
            assemblers: Decoder name -> assembler, as built by `_buildAssemblers`.

        Returns:
            The decoder module.

        Raises:
            ValueError: If `kwargs` sets a `latent_dim` different from the width
                the decoder receives.
            KeyError: If `registry_name` is absent from the decoder registry.
        """
        decoder_class = getDecoderClass(registry_name)
        resolved = dict(kwargs)
        latent_names = self._decoder_inputs.get(decoder_name)
        if latent_names:
            if decoder_name in assemblers:
                assembler = assemblers[decoder_name]
                width = assembler.output_dim
                source = (
                    f"the '{type(assembler).__name__}' assembler over latent spaces "
                    f"{latent_names} (widths {list(assembler.input_dims)})"
                )
            else:
                width = self.routing_graph.latent_specs[latent_names[0]].dim
                source = f"latent space '{latent_names[0]}'"
            explicit = resolved.get("latent_dim")
            if explicit is not None and explicit != width:
                raise ValueError(
                    f"Decoder '{decoder_name}' is built with latent_dim={explicit!r}, but it "
                    f"receives {width} features from {source}. Remove `latent_dim` from its "
                    f"kwargs to take it from the routing graph."
                )
            if explicit is None and "latent_dim" in inspect.signature(decoder_class).parameters:
                resolved["latent_dim"] = width
        return decoder_class(**resolved)

    def _buildHeads(self, routing_graph: RoutingGraph) -> dict[str, nn.Module]:
        """Build one latent head module per entry of `routing_graph.encoder_heads`.

        Each head is built as `getHeadClass(name)(in_dim=..., out_dim=...,
        **kwargs)`, with `in_dim` the feeding encoder's `latent_dim` and
        `out_dim` the target latent space's `dim`. An edge with no entry gets
        no module (identity edge). Call after `self.encoders` exists.

        Args:
            routing_graph: The routing graph whose heads to build.

        Returns:
            Encoder name -> `nn.ModuleDict` of latent space name -> head.

        Raises:
            ValueError: If a head's encoder is absent from `self.encoders`,
                exposes no integer `latent_dim`, or the head rejects the
                widths it is built with.
            KeyError: If a head name is absent from the head registry.
        """
        heads: dict[str, nn.Module] = {}
        for encoder_name, edge_specs in routing_graph.encoder_heads.items():
            if encoder_name not in self.encoders:
                raise ValueError(
                    f"Latent heads are declared for encoder '{encoder_name}', but that encoder "
                    f"is not in `encoder_configs` (encoders: {sorted(self.encoders)})."
                )
            in_dim = getattr(self.encoders[encoder_name], "latent_dim", None)
            if not isinstance(in_dim, int):
                raise ValueError(
                    f"Latent heads are declared for encoder '{encoder_name}', which exposes no "
                    f"integer `latent_dim`, so the heads' input width is unknown."
                )
            edge_heads: dict[str, nn.Module] = {}
            for latent_name, spec in edge_specs.items():
                out_dim = routing_graph.latent_specs[latent_name].dim
                head_class = getHeadClass(spec.name)
                try:
                    edge_heads[latent_name] = head_class(
                        in_dim=in_dim, out_dim=out_dim, **spec.kwargs
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Cannot build head '{spec.name}' on the edge '{encoder_name}' -> "
                        f"'{latent_name}' (encoder latent_dim={in_dim}, latent space "
                        f"dim={out_dim}): {error}"
                    ) from error
            heads[encoder_name] = nn.ModuleDict(edge_heads)
        return heads

    def getHead(self, encoder_name: str, latent_name: str) -> AbstractLatentHead | None:
        """Return the latent head on one encoder -> latent edge.

        Args:
            encoder_name: The feeding encoder.
            latent_name: The latent space the edge feeds.

        Returns:
            The head module, or `None` for an identity edge (one with no
            entry in `routing_graph.encoder_heads`, which builds no module).
        """
        if encoder_name not in self.heads:
            return None
        edge_heads = cast(nn.ModuleDict, self.heads[encoder_name])
        if latent_name not in edge_heads:
            return None
        return cast(AbstractLatentHead, edge_heads[latent_name])

    def _adaptThroughHead(
        self, encoder_name: str, latent_name: str, params: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the latent head of one encoder -> latent edge, if it has one.

        Args:
            encoder_name: The feeding encoder.
            latent_name: The latent space the edge feeds.
            params: The encoder's `(mu, logvar)` output.

        Returns:
            The head's `(mu, logvar)`, or `params` itself for an identity
            edge (one with no head module).
        """
        head = self.getHead(encoder_name, latent_name)
        if head is None:
            return params
        adapted: tuple[torch.Tensor, torch.Tensor] = head(*params)
        return adapted

    def selectEncoderInputs(
        self, batch: dict[str, torch.Tensor] | Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Restrict a per-modality batch to the keys this model can actually encode.

        A batch dict conventionally carries one entry per modality
        (spec §6.1), which is enough as long as every decoder shares
        its name with an encoder. That stops being true the moment an
        encoder's input and a decoder's reconstruction target are not
        the same modality (e.g. a translation-style `image_in ->
        image_out` model, ADR 0019): the batch then also carries
        decoder-only target keys `forward()` has no encoder for, and
        passing it straight through raises `KeyError` from
        `self.encoders[encoder_name]`. This is the one place that
        restriction happens, so every framework entry point that walks
        a raw per-modality batch (`Trainer.computeLosses`,
        `evaluation.evaluate.evaluate`, and the `visualization`
        collectors) can call `model.selectEncoderInputs(batch)` before
        `forward()` and stay correct regardless of whether inputs and
        targets share a name.

        Args:
            batch: Modality name -> tensor: encoder inputs and any
                decoder-only reconstruction targets alike. The caller
                keeps its own reference to the unfiltered `batch` for
                matching reconstructions against their targets; this
                method only ever returns a subset, never mutates its
                argument.

        Returns:
            The subset of `batch` whose key names one of `self.encoders`.

        Raises:
            ValueError: If no key of `batch` names one of this model's
                encoders, i.e. there would be nothing left to encode.
        """
        encoder_inputs = {name: tensor for name, tensor in batch.items() if name in self.encoders}
        if not encoder_inputs:
            raise ValueError(
                f"None of this batch's keys ({sorted(batch)}) match any of this model's "
                f"encoders ({sorted(self.encoders)}). At least one encoder input is required; "
                f"a decoder-only target with no matching encoder (ADR 0019) is fine as long as "
                f"at least one other key names a real encoder."
            )
        return encoder_inputs

    def forward(
        self, inputs: dict[str, torch.Tensor], use_mean: bool = False
    ) -> dict[str, dict[str, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Run one encode, fuse, sample, assemble, decode pass.

        Args:
            inputs: Modality name -> raw input tensor. Any non-empty
                subset of the configured modalities is accepted; actual
                missing-modality robustness depends on the fusion
                strategy in use for each latent space (spec §5). Every
                key must name a registered encoder (see
                `selectEncoderInputs` for restricting a raw batch that
                may also carry decoder-only target keys, ADR 0019).
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
            KeyError: If some key of `inputs` names no registered
                encoder (see `selectEncoderInputs`).
        """
        if not inputs:
            raise ValueError("GlobalVae.forward() requires at least one modality in `inputs`.")

        unknown_keys = sorted(set(inputs) - set(self.encoders))
        if unknown_keys:
            raise KeyError(
                f"GlobalVae.forward() received input key(s) {unknown_keys} with no matching "
                f"encoder (available: {sorted(self.encoders)}). If these are decoder-only "
                f"reconstruction targets rather than genuine encoder inputs (e.g. a "
                f"translation-style 'image_in -> image_out' model, ADR 0019), restrict "
                f"`inputs` first: model.selectEncoderInputs(batch)."
            )

        encoder_outputs = {
            encoder_name: self.encoders[encoder_name](x) for encoder_name, x in inputs.items()
        }

        latent_params: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        latent_samples: dict[str, torch.Tensor] = {}
        for latent_name, encoder_names in self._feeding_encoders.items():
            active = [name for name in encoder_names if name in encoder_outputs]
            if not active:
                continue
            edge_params = {
                name: self._adaptThroughHead(name, latent_name, encoder_outputs[name])
                for name in active
            }
            if len(active) == 1:
                mu, logvar = edge_params[active[0]]
            else:
                mu, logvar = self.fusions[latent_name](edge_params)
            latent_params[latent_name] = (mu, logvar)
            latent_samples[latent_name] = (
                mu if use_mean else self.latent_spaces[latent_name].reparameterize(mu, logvar)
            )

        reconstructions: dict[str, torch.Tensor] = {}
        for decoder_name, decoder in self.decoders.items():
            required = self._decoder_inputs.get(decoder_name, [])
            consumed = [latent_name for latent_name in required if latent_name in latent_samples]
            if not consumed:
                continue
            if decoder_name not in self.assemblers:
                # One latent space and no assembler: the sample is the decoder input.
                z_in = latent_samples[consumed[0]]
            else:
                assembler = cast(AbstractAssembler, self.assemblers[decoder_name])
                if len(consumed) < len(required) and not assembler.handlesMissingLatents:
                    # An incomplete input to an assembler that needs all of its latent
                    # spaces (`concat`): leave this decoder out of the pass (spec §5).
                    continue
                # Always through the assembler, even for a single remaining latent space:
                # `weighted_sum` and `attention` transform it, and the decoder was sized
                # for the assembler's `output_dim`. `present` says which spaces these are.
                present = (
                    None
                    if len(consumed) == len(required)
                    else [required.index(name) for name in consumed]
                )
                z_in = assembler([latent_samples[name] for name in consumed], present)
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
        (beta, spec §2.3) is hardcoded into this model class. The
        default strategy is still KL-to-standard-normal, but this
        method is not KL-specific: `self.regularizers` is pluggable,
        which is why it is named after regularization and not after
        the KL term (ADR 0003).

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
