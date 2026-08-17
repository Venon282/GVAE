"""Structured config schema for `GlobalVae` (spec §9, §10 "Config management"), plus
`buildModelFromConfig`, the one function that turns a validated `ModelConfig` into a
real `GlobalVae` instance.

Mirrors spec §9's illustrative single-latent example almost field-for-field
(`modalities.<name>.{encoder,decoder}`, `latent.mode`, `latent.fusion`), the difference
being that this is now an actual, validated schema instead of a comment saying "not
final". Only `latent_mode: "single"` is wired up to a real builder today, matching
`GlobalVae.createSingleLatent` (spec §6.1 milestone 1's single-modality signal VAE, and
the `EN-L1-DN` Phase-1 default, ADR 0001): `latent_mode: "several"` is accepted by the
schema (so a `RoutingGraph`-based config has a place to grow into later) but
`buildModelFromConfig` raises `NotImplementedError` for it today, the same "fail loudly
instead of guessing" stance `GlobalVae.__init__` itself already takes for encoder
fan-out (`docs/adr/0002-generalize-global-vae-to-routing-graph.md`). Building the
general multi-latent-space config schema before the single-modality milestone works
end to end would violate spec §6.1's own stated build order.

Every dataclass here is registered with Hydra's `ConfigStore` by
`global_vae.config.__init__` (importing that package is what makes the registration
happen, the same self-registration pattern used everywhere else in this codebase,
spec §10) so `configs/model/*.yaml` files are validated against these types at
composition time, not just passed through as untyed dicts.
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from global_vae.decoders.registry import getDecoderClass
from global_vae.encoders.registry import getEncoderClass
from global_vae.fusion.registry import getFusionClass  # noqa: F401  (imported for symmetry/clarity)
from global_vae.models.global_vae import GlobalVae


@dataclass
class EncoderConfig:
    """One encoder's registry name plus its constructor kwargs.

    Attributes:
        name: Registry key (spec §9), e.g. `"1d_cnn_encoder_v1"`. Required:
            there is no sensible default encoder.
        kwargs: Forwarded to the encoder's constructor. `latent_dim` may
            be omitted here: `buildModelFromConfig` fills it in from
            `SingleLatentConfig.dim` when absent (see that function's
            docstring), since the single-latent architecture requires
            every encoder to already agree on that dimensionality
            anyway. Any explicit value given here is left untouched.
    """

    name: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderConfig:
    """One decoder's registry name plus its constructor kwargs.

    Attributes:
        name: Registry key, e.g. `"1d_cnn_decoder_v1"`.
        kwargs: Forwarded to the decoder's constructor. `latent_dim` is
            auto-filled the same way as `EncoderConfig.kwargs`.
            Modality-specific, data-dependent arguments (e.g.
            `OneDCnnDecoder`'s `output_length`) are never auto-filled:
            this config domain deliberately knows nothing about the
            data domain (spec: data pipeline concerns are out of this
            framework's scope), so those must be given explicitly here,
            matching whatever `configs/data/*.yaml` describes.
    """

    name: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityConfig:
    """One modality's encoder and decoder pair, matching `GlobalVae.createSingleLatent`'s
    `modality_configs` shape (spec §9: `modalities.<name>.{encoder,decoder}`).
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class FusionConfig:
    """Fusion strategy for a latent space fed by more than one encoder (spec §4).

    Attributes:
        strategy: Registry key, e.g. `"poe"`.
        kwargs: Forwarded to the fusion module's constructor.
    """

    strategy: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegularizerConfig:
    """Latent regularization strategy for one latent space (spec §2.3).

    Attributes:
        strategy: Registry key, e.g. `"kl_standard_normal"` (the
            default, matching `GlobalVae.createSingleLatent`'s own
            default), `"free_bits_kl"`, or `"mmd"`.
        kwargs: Forwarded to the regularizer's constructor.
    """

    strategy: str = "kl_standard_normal"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SingleLatentConfig:
    """Configuration for the single-fused-latent case (`latent_mode: "single"`).

    Attributes:
        dim: Dimensionality of the single latent space. Required.
        name: Identifier for the latent space, matching
            `GlobalVae.createSingleLatent`'s own `latent_name` default.
        fusion: Fusion strategy combining every modality's encoder into
            the single latent space. Only meaningful, and only
            required, when `modalities` has more than one entry (spec
            §4): a latent space fed by exactly one encoder needs no
            fusion strategy at all, so this stays `None` for the
            single-modality signal VAE (spec §6.1 milestone 1).
        regularizer: Latent regularization strategy for the single
            latent space (spec §2.3). Defaults to `kl_standard_normal`.
    """

    dim: int = MISSING
    name: str = "z_fused"
    fusion: FusionConfig | None = None
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)


@dataclass
class ModelConfig:
    """Top-level model configuration (spec §9).

    Attributes:
        name: Free-form label for the model, not consumed by
            `buildModelFromConfig` itself; useful for logging/tracking
            (spec §10 "config snapshotted with every run").
        modalities: Modality name -> `ModalityConfig`. A single entry
            is the `signal -> z -> signal` case (spec §6.1 milestone
            1); several entries is `EN-L1-DN` (spec's Phase-1 default,
            ADR 0001), once `latent_mode` is `"single"`.
        latent_mode: `"single"` (the only mode `buildModelFromConfig`
            currently builds, via `GlobalVae.createSingleLatent`) or
            `"several"` (reserved for a future general `RoutingGraph`
            config, spec §2.2; accepted by this schema so config files
            do not need to change shape again once that lands, but
            `buildModelFromConfig` raises `NotImplementedError` for it
            today, per spec §6.1's build order).
        single_latent: Required when `latent_mode` is `"single"`;
            ignored otherwise.
    """

    name: str = "global_vae"
    modalities: dict[str, ModalityConfig] = field(default_factory=dict)
    latent_mode: str = "single"
    single_latent: SingleLatentConfig | None = None


def buildModelFromConfig(config: ModelConfig) -> GlobalVae:
    """Build a real `GlobalVae` instance from a validated `ModelConfig`.

    For `latent_mode: "single"`, this is a thin, config-shaped wrapper
    around `GlobalVae.createSingleLatent`: every field here maps
    directly onto one of that classmethod's parameters. The one piece
    of real logic is filling in `latent_dim` on every encoder/decoder
    that did not already specify one explicitly (see `EncoderConfig`/
    `DecoderConfig`'s own docstrings): since a single latent space
    architecturally requires every encoder and decoder to agree on its
    dimensionality, repeating `single_latent.dim` in every modality's
    `kwargs` in the YAML would be pure duplication (and a real risk of
    the copies silently drifting apart); auto-filling it here removes
    that duplication while still respecting an explicit override if
    one is given.

    Args:
        config: A `ModelConfig`, typically produced by
            `global_vae.config.experiment.loadExperimentConfig` or any
            other Hydra-composed, `OmegaConf.to_object`-materialized
            config.

    Returns:
        A `GlobalVae` instance, architecturally equivalent to calling
        `GlobalVae.createSingleLatent(...)` directly with the same
        values.

    Raises:
        NotImplementedError: If `config.latent_mode` is `"several"`
            (not yet wired to a builder; see this module's docstring).
        ValueError: If `config.latent_mode` is `"single"` but
            `config.single_latent` is `None`, if `config.modalities` is
            empty, or (delegated to `GlobalVae.__init__`) if more than
            one modality is given with no `fusion` configured.
        KeyError: If any encoder/decoder/fusion/regularizer `name`/
            `strategy` is not a registered registry name (delegated to
            the relevant `getXClass` lookup).
    """
    if config.latent_mode != "single":
        raise NotImplementedError(
            f"ModelConfig.latent_mode='{config.latent_mode}' is not yet supported by "
            f"buildModelFromConfig: only 'single' (GlobalVae.createSingleLatent) is "
            f"wired up today, matching the spec §6.1 milestone-1 build order. A general "
            f"RoutingGraph-based config for 'several' is a planned future extension, not "
            f"yet implemented; see global_vae/config/model.py's module docstring."
        )
    if config.single_latent is None:
        raise ValueError("ModelConfig.latent_mode='single' requires `single_latent` to be set.")
    if not config.modalities:
        raise ValueError("ModelConfig.modalities must contain at least one modality.")

    latent_dim = config.single_latent.dim
    modality_configs: dict[str, dict[str, str]] = {}
    encoder_kwargs: dict[str, dict[str, Any]] = {}
    decoder_kwargs: dict[str, dict[str, Any]] = {}
    for modality_name, modality in config.modalities.items():
        # Fail fast on an unregistered name here too, rather than only inside
        # GlobalVae.__init__: this gives a clear error pointing at the config
        # field before any module is even constructed.
        getEncoderClass(modality.encoder.name)
        getDecoderClass(modality.decoder.name)

        modality_configs[modality_name] = {
            "encoder": modality.encoder.name,
            "decoder": modality.decoder.name,
        }
        encoder_kwargs[modality_name] = {"latent_dim": latent_dim, **modality.encoder.kwargs}
        decoder_kwargs[modality_name] = {"latent_dim": latent_dim, **modality.decoder.kwargs}

    single_latent = config.single_latent
    fusion_strategy = single_latent.fusion.strategy if single_latent.fusion is not None else None
    fusion_kwargs = (
        {single_latent.name: single_latent.fusion.kwargs}
        if single_latent.fusion is not None
        else None
    )

    return GlobalVae.createSingleLatent(
        modality_configs=modality_configs,
        latent_dim=latent_dim,
        fusion_strategy=fusion_strategy,
        latent_name=single_latent.name,
        regularizer_strategy=single_latent.regularizer.strategy,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        fusion_kwargs=fusion_kwargs,
        regularizer_kwargs={single_latent.name: single_latent.regularizer.kwargs},
    )
