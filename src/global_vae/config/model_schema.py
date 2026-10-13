"""Structured config schema for `GlobalVae` (spec §9, §10 "Config management").

Holds every dataclass a `configs/model/*.yaml` file is validated against. The checks that need
more than a type (a name that must exist, a fusion that a latent space needs, widths that must
agree) live in `config/model_resolution.py`, and `config/model.py` turns a `ModelConfig` into a
real `GlobalVae`. All three are re-exported by `config/model.py`, so
`from global_vae.config.model import ModelConfig` keeps working.

A model is written in one of two ways, which can be mixed (ADR 0025):

- **`modalities`**, the common case. One entry per modality name carries one encoder and one
  decoder, and both take that name as their instance name. This is the shape of every model that
  reconstructs what it encodes.
- **`encoders` and `decoders`**, for everything else. Two sections keyed by instance name, with
  no tie between them: a decoder may have no encoder of the same name (the translation model of
  `examples/03_signal_image_to_image.py`, ADR 0019), and an encoder none of the same name.

Where the encoders and decoders go is set by `latent_mode`:

- **`single`**: `single_latent`. Every encoder feeds one latent space and every decoder consumes
  it (`GlobalVae.createSingleLatent`, the `EN-L1-DN` Phase-1 default).
- **`several`**: `latent`. Any number of latent spaces (`latent.spaces`), each fed by the
  encoders that `fed_by` names, and for every decoder the spaces it consumes and the assembler
  that combines them (`latent.decoders_consume`). This is the `RoutingGraph` of spec §2.2 written
  as YAML: several independent latent spaces, and encoder fan-out through latent heads (a
  mapping `encoder -> {head, kwargs}` in `fed_by`).

Beta weights are not part of this schema: they stay per latent space in
`TrainingConfig.beta_schedules`, under the latent space names declared here. Every dataclass is
registered with Hydra's `ConfigStore` through `ModelConfig` (see `config/__init__.py`).
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class EncoderConfig:
    """One encoder's registry name plus its constructor kwargs.

    Attributes:
        name: Registry key (spec §9), e.g. `"1d_cnn_encoder_v1"`. Required:
            there is no sensible default encoder.
        kwargs: Forwarded to the encoder's constructor. `latent_dim` may
            be omitted here: `buildModelFromConfig` fills it in when
            absent. In `single` mode it is `SingleLatentConfig.dim`. In
            `several` mode it is the `dim` of the latent space the encoder
            feeds through an identity edge (an edge with no head, or with
            the `identity` head); an encoder whose every edge goes through a
            head has no such space, so it needs an explicit `latent_dim`.
            Any explicit value given here is left untouched.
    """

    name: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderConfig:
    """One decoder's registry name plus its constructor kwargs.

    Attributes:
        name: Registry key, e.g. `"1d_cnn_decoder_v1"`.
        kwargs: Forwarded to the decoder's constructor. `latent_dim` is
            auto-filled the same way as `EncoderConfig.kwargs`: the single
            latent dimension in `single` mode, and in `several` mode the
            width of what the decoder's assembler produces from the latent
            spaces it consumes. Modality-specific, data-dependent
            arguments (e.g. `OneDCnnDecoder`'s `output_length`) are never
            auto-filled: this config domain deliberately knows nothing
            about the data domain (spec: data pipeline concerns are out of
            this framework's scope), so those must be given explicitly
            here, matching whatever `configs/data/*.yaml` describes.
    """

    name: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityConfig:
    """One modality's encoder and decoder pair.

    Matches `GlobalVae.createSingleLatent`'s `modality_configs` shape (spec §9:
    `modalities.<name>.{encoder,decoder}`). The modality name is the instance
    name of both the encoder and the decoder, so it is also the batch key the
    encoder reads and the decoder is trained against.
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class FusionConfig:
    """Fusion strategy for a latent space fed by more than one encoder (spec §4).

    Attributes:
        strategy: Registry key: `"poe"`, `"moe"`, `"concat_mlp"`, or
            `"cross_attention"` (or any further strategy registered in
            `fusion/`). See `docs/how-to/choose-a-fusion-strategy.md`.
        kwargs: Forwarded unchanged to the fusion module's constructor.
            Unlike `EncoderConfig.kwargs`/`DecoderConfig.kwargs`,
            `latent_dim` is not auto-filled here, since fusion strategies
            have unrelated signatures (`poe`/`moe` accept no `latent_dim`;
            `concat_mlp` and `cross_attention` require it, and
            `concat_mlp` also requires `modality_dims`): give it
            explicitly for the strategies that need it.
        residual: Whether to add a residual connection around the
            strategy (spec §4, §9: `residual: true` next to `strategy:`).
            Applies to every strategy alike, through
            `fusion.residual.ResidualFusion`: the fused posterior starts
            as the mean of the experts and a learned gate opens the
            strategy's own correction. Off by default.
    """

    strategy: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)
    residual: bool = False


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
        fusion: Fusion strategy combining every encoder into the single
            latent space. Only meaningful, and only required, when more than
            one encoder is declared (spec §4): a latent space fed by exactly
            one encoder needs no fusion strategy at all, so this stays
            `None` for the single-modality signal VAE (spec §6.1 milestone
            1).
        regularizer: Latent regularization strategy for the single
            latent space (spec §2.3). Defaults to `kl_standard_normal`.
    """

    dim: int = MISSING
    name: str = "z_fused"
    fusion: FusionConfig | None = None
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)


@dataclass
class EdgeHeadConfig:
    """The latent head on one encoder -> latent space edge (spec §2.2, ADR 0024).

    This is the value shape of a mapping `fed_by` (see `LatentSpaceConfig`).
    It maps onto `latent.base.HeadSpec`.

    Attributes:
        head: Head registry name (`heads/`), `"identity"` by default. The
            model builds the head as `getHeadClass(head)(in_dim=<the
            encoder's latent_dim>, out_dim=<the latent space's dim>,
            **kwargs)`.
        kwargs: Extra constructor arguments for the head. They never set
            `in_dim` or `out_dim`: the model derives both widths.
    """

    head: str = "identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatentSpaceConfig:
    """One independent latent space of a `latent_mode: "several"` model (spec §2.2).

    Attributes:
        dim: Dimensionality of the latent space. Required.
        fed_by: The encoders feeding this latent space, by instance name.
            Either a list of names, each an identity edge (the encoder's
            `(mu, logvar)` is used as it is), or a mapping `encoder name ->
            {head, kwargs}` (`EdgeHeadConfig`, or `null` for an identity
            edge), one latent head per edge. The mapping form is how encoder
            fan-out is written: an encoder that feeds several latent spaces
            gives all but one of its edges a non-identity head (ADR 0024).
            Typed `Any`, since OmegaConf cannot express "a list or a
            mapping"; `config/model_resolution.parseFedBy` checks the shape
            and names this field when it is wrong. Required.
        fusion: Fusion strategy combining the encoders in `fed_by`. Required
            when more than one encoder feeds this latent space (spec §4),
            and refused when exactly one does, since a fusion would never
            run.
        regularizer: Latent regularization strategy for this latent space
            alone (spec §2.3). Defaults to `kl_standard_normal`.
    """

    dim: int = MISSING
    fed_by: Any = MISSING
    fusion: FusionConfig | None = None
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)


@dataclass
class DecoderConsumptionConfig:
    """The latent spaces one decoder consumes, and how they are combined (spec §2.2).

    Attributes:
        spaces: Names of the latent spaces the decoder consumes. Required.
        assembler: `assemblers/` registry key combining the spaces into the
            decoder's input (`"concat"`, `"sum"`, `"average"`, ...).
            Required when `spaces` has more than one entry, and refused when
            it has exactly one (nothing to combine).
        kwargs: Forwarded to the assembler's constructor. Only meaningful
            together with `assembler`.
    """

    spaces: list[str] = MISSING
    assembler: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatentConfig:
    """The latent spaces of a `latent_mode: "several"` model and who consumes them.

    Attributes:
        spaces: Latent space name -> `LatentSpaceConfig`. At least one. The
            names are the keys of `TrainingConfig.beta_schedules`.
        decoders_consume: Decoder instance name -> `DecoderConsumptionConfig`.
            Every declared decoder has exactly one entry, since a decoder
            that consumes nothing would never run.
    """

    spaces: dict[str, LatentSpaceConfig] = field(default_factory=dict)
    decoders_consume: dict[str, DecoderConsumptionConfig] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Top-level model configuration (spec §9, ADR 0025).

    Attributes:
        name: Free-form label for the model, not consumed by
            `buildModelFromConfig` itself; useful for logging/tracking
            (spec §10 "config snapshotted with every run").
        modalities: Modality name -> `ModalityConfig`: one encoder and one
            decoder per name, both under that name. A single entry is the
            `signal -> z -> signal` case (spec §6.1 milestone 1); several
            entries is `EN-L1-DN` (spec's Phase-1 default, ADR 0001) in
            `single` mode.
        latent_mode: `"single"` (one latent space, `single_latent`) or
            `"several"` (any number of latent spaces, `latent`).
        single_latent: Required when `latent_mode` is `"single"`, refused
            otherwise.
        encoders: Encoder instance name -> `EncoderConfig`, for encoders
            that are not half of a modality pair. Added to the encoders of
            `modalities`; a name used in both is an error.
        decoders: Decoder instance name -> `DecoderConfig`, for decoders
            whose name is not a modality's (a decoder target that is not an
            encoder input, a decoder shared by several modalities). Added to
            the decoders of `modalities`; a name used in both is an error.
        latent: Required when `latent_mode` is `"several"`, refused
            otherwise.
    """

    name: str = "global_vae"
    modalities: dict[str, ModalityConfig] = field(default_factory=dict)
    latent_mode: str = "single"
    single_latent: SingleLatentConfig | None = None
    encoders: dict[str, EncoderConfig] = field(default_factory=dict)
    decoders: dict[str, DecoderConfig] = field(default_factory=dict)
    latent: LatentConfig | None = None