"""Model config and the builder that instantiates it (spec §9, §10 "Config management").

`buildModelFromConfig` is the one function that turns a validated `ModelConfig` into a real
`GlobalVae`. The schema itself lives in `config/model_schema.py` and the checks that go beyond a
type in `config/model_resolution.py`; both are re-exported here, so
`from global_vae.config.model import ModelConfig` is still the import to use.

Both latent modes build (ADR 0025):

- `latent_mode: "single"`: one latent space fed by every encoder and consumed by every decoder,
  the `EN-L1-DN` Phase-1 default (ADR 0001) and the single-modality signal VAE of spec §6.1
  milestone 1. Encoders and decoders come from `modalities` (one of each per name), from the
  `encoders` and `decoders` sections (independent names), or from both.
- `latent_mode: "several"`: any number of independent latent spaces wired to the encoders and
  decoders by the `latent` section, which `buildModelFromConfig` turns into a `RoutingGraph`
  (spec §2.2). This covers several independent latent spaces and encoder fan-out through latent
  heads (ADR 0024).

Every dataclass is registered with Hydra's `ConfigStore` by `global_vae.config.__init__`
(importing that package is what makes the registration happen, the same self-registration pattern
used everywhere else in this codebase, spec §10) so `configs/model/*.yaml` files are validated
against these types at composition time, not just passed through as untyped dicts.
"""

from global_vae.config.model_resolution import (
    LATENT_MODES,
    ResolvedModel,
    collectDecoders,
    collectEncoders,
    parseFedBy,
    resolveModelConfig,
    validateModelConfig,
)
from global_vae.config.model_schema import (
    DecoderConfig,
    DecoderConsumptionConfig,
    EdgeHeadConfig,
    EncoderConfig,
    FusionConfig,
    LatentConfig,
    LatentSpaceConfig,
    ModalityConfig,
    ModelConfig,
    RegularizerConfig,
    SingleLatentConfig,
)
from global_vae.models.global_vae import GlobalVae

__all__ = [
    "LATENT_MODES",
    "DecoderConfig",
    "DecoderConsumptionConfig",
    "EdgeHeadConfig",
    "EncoderConfig",
    "FusionConfig",
    "LatentConfig",
    "LatentSpaceConfig",
    "ModalityConfig",
    "ModelConfig",
    "RegularizerConfig",
    "ResolvedModel",
    "SingleLatentConfig",
    "buildModelFromConfig",
    "collectDecoders",
    "collectEncoders",
    "parseFedBy",
    "resolveModelConfig",
    "validateModelConfig",
]


def buildModelFromConfig(config: ModelConfig) -> GlobalVae:
    """Build a real `GlobalVae` instance from a `ModelConfig`.

    The config is validated first (`validateModelConfig`), then resolved
    (`resolveModelConfig`) into the arguments of `GlobalVae.__init__`, and the model is built
    from those. Resolution is where the config's omissions are filled in: the `latent_dim` of
    each encoder and decoder is not repeated in the YAML, because a number written in several
    places can drift apart. An explicit `latent_dim` in a `kwargs` mapping is never overwritten.

    - `single` mode: every encoder and decoder gets `single_latent.dim`. The model equals what
      `GlobalVae.createSingleLatent` builds from the same values, state dict included.
    - `several` mode: an encoder gets the `dim` of the latent space it feeds through an identity
      edge, and a decoder gets the width its assembler produces from the latent spaces it
      consumes (the latent dimension itself for one space).

    Args:
        config: A `ModelConfig`, typically produced by
            `global_vae.config.experiment.loadExperimentConfig` or any
            other Hydra-composed, `OmegaConf.to_object`-materialized
            config.

    Returns:
        A `GlobalVae` instance.

    Raises:
        ValueError: If the config is invalid. The message starts with the dotted path of the
            offending field (`model.latent.spaces.z_shared.fed_by`). See `validateModelConfig`
            for the list of checks; the routing graph's own rules (`validateRoutingGraph`) and
            the model's (a head whose widths do not fit, a fusion for a one-encoder space) are
            raised as they are.
        KeyError: If an encoder, decoder, fusion, regularizer, head or assembler name is not
            registered. The message starts with the path of the field that holds the name.
    """
    resolved = resolveModelConfig(config)
    return GlobalVae(
        encoder_configs=resolved.encoder_configs,
        decoder_configs=resolved.decoder_configs,
        routing_graph=resolved.routing_graph,
        fusion_strategies=resolved.fusion_strategies,
        regularizer_strategies=resolved.regularizer_strategies,
        encoder_kwargs=resolved.encoder_kwargs,
        decoder_kwargs=resolved.decoder_kwargs,
        fusion_kwargs=resolved.fusion_kwargs,
        regularizer_kwargs=resolved.regularizer_kwargs,
        fusion_residual=resolved.fusion_residual or None,
        assembler_kwargs=resolved.assembler_kwargs,
    )
