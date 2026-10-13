"""Validation and resolution of a `ModelConfig` (spec §9, §10, ADR 0025).

`config/model_schema.py` says what shape a model config has. This module says whether a config of
that shape describes a model that can be built, and turns it into the arguments of
`GlobalVae.__init__`:

- `validateModelConfig` checks everything that needs more than a type: the section a `latent_mode`
  requires, unique encoder and decoder names, every name used in `fed_by` and `decoders_consume`
  exists, a fusion exactly where several encoders feed a latent space, an assembler exactly where
  a decoder consumes several, no encoder or decoder left unused, and the widths an identity edge
  has to agree on. It does not look at registries, so a config that names a strategy nobody
  registered still validates.
- `resolveModelConfig` validates, then resolves registry names (an unknown one raises
  `KeyError`), builds the `RoutingGraph`, and fills in the widths the config leaves out
  (`latent_dim` of each encoder and decoder).

Every error starts with the dotted path of the offending field, in the form Hydra and OmegaConf
use (`model.latent.spaces.z_shared.fed_by`), so a failing run points at the line to edit. The
path prefix defaults to `model`, the key of `ExperimentConfig.model`.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
from omegaconf import MISSING

from global_vae.assemblers.registry import getAssemblerClass
from global_vae.config.model_schema import (
    DecoderConfig,
    DecoderConsumptionConfig,
    EdgeHeadConfig,
    EncoderConfig,
    LatentConfig,
    ModelConfig,
    SingleLatentConfig,
)
from global_vae.decoders.registry import getDecoderClass
from global_vae.encoders.registry import getEncoderClass
from global_vae.fusion.registry import getFusionClass
from global_vae.heads.registry import getHeadClass
from global_vae.latent.base import HeadSpec, LatentSpace, RoutingGraph, validateRoutingGraph
from global_vae.latent.routing_graph_builders.single import buildSingleLatentRoutingGraph
from global_vae.losses.regularizers.registry import getRegularizerClass

LATENT_MODES = ("single", "several")

_IDENTITY_HEAD = "identity"
_EDGE_KEYS = ("head", "kwargs")
_ClassT = TypeVar("_ClassT")


@dataclass
class ResolvedModel:
    """Everything `GlobalVae.__init__` needs, derived from one `ModelConfig`.

    Attributes:
        encoder_configs: Encoder instance name -> registry name.
        decoder_configs: Decoder instance name -> registry name.
        routing_graph: The wiring (spec §2.2), already validated.
        fusion_strategies: Latent space name -> fusion registry name.
        regularizer_strategies: Latent space name -> regularizer registry name.
        encoder_kwargs: Encoder instance name -> constructor kwargs, `latent_dim` filled in.
        decoder_kwargs: Decoder instance name -> constructor kwargs, `latent_dim` filled in.
        fusion_kwargs: Latent space name -> fusion constructor kwargs.
        regularizer_kwargs: Latent space name -> regularizer constructor kwargs.
        fusion_residual: Latent space name -> `True` for the spaces whose fusion is wrapped in a
            residual connection (spec §4). Only spaces with `residual: true` appear.
        assembler_kwargs: Decoder instance name -> assembler constructor kwargs.
    """

    encoder_configs: dict[str, str]
    decoder_configs: dict[str, str]
    routing_graph: RoutingGraph
    fusion_strategies: dict[str, str] = field(default_factory=dict)
    regularizer_strategies: dict[str, str] = field(default_factory=dict)
    encoder_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    decoder_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    fusion_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    regularizer_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    fusion_residual: dict[str, bool] = field(default_factory=dict)
    assembler_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class _Parsed:
    """A validated `ModelConfig`, with its instances collected and its edges parsed.

    Attributes:
        encoders: Encoder instance name -> `(config, path of that config)`, the encoders of
            `modalities` first, then those of `encoders`.
        decoders: Decoder instance name -> `(config, path of that config)`, likewise.
        single: The `single_latent` section, set in `single` mode only.
        latent: The `latent` section, set in `several` mode only.
        edges: `several` mode only. Latent space name -> encoder name -> its edge: an
            `EdgeHeadConfig` for a mapping `fed_by`, `None` for a plain identity edge (a list
            `fed_by`, which builds no head module).
    """

    encoders: dict[str, tuple[EncoderConfig, str]]
    decoders: dict[str, tuple[DecoderConfig, str]]
    single: SingleLatentConfig | None = None
    latent: LatentConfig | None = None
    edges: dict[str, dict[str, EdgeHeadConfig | None]] = field(default_factory=dict)


def _isMissing(value: Any) -> bool:
    """Whether `value` is OmegaConf's `MISSING` marker (`"???"`)."""
    return isinstance(value, str) and value == MISSING


def _requirePositiveInt(value: Any, path: str) -> int:
    """Return `value` if it is a positive integer.

    Args:
        value: The value read from the config.
        path: Dotted path of the field, for the error message.

    Returns:
        `value`.

    Raises:
        ValueError: If `value` is missing, not an integer (a `bool` is not one here), or not
            positive.
    """
    if _isMissing(value):
        raise ValueError(f"{path}: required, no value given.")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}: expected a positive integer, got {value!r}.")
    return value


def _lookup(getter: Callable[[str], _ClassT], name: str, path: str) -> _ClassT:
    """Resolve a registry name, naming the config field when it is unknown.

    Args:
        getter: A registry lookup (`getEncoderClass`, ...).
        name: The registry name read from the config.
        path: Dotted path of the field holding `name`.

    Returns:
        Whatever `getter` returns for `name`.

    Raises:
        KeyError: If `name` is not registered; the message starts with `path`.
    """
    try:
        return getter(name)
    except KeyError as error:
        raise KeyError(f"{path}: {error.args[0]}") from error


def collectEncoders(
    config: ModelConfig, prefix: str = "model"
) -> dict[str, tuple[EncoderConfig, str]]:
    """Gather every encoder of a model, from `modalities` and from `encoders`.

    Args:
        config: The model config.
        prefix: Dotted path of `config`, the start of every error path.

    Returns:
        Encoder instance name -> `(EncoderConfig, dotted path of that config)`. The encoders of
        `modalities` come first, in order, then those of `encoders`.

    Raises:
        ValueError: If an encoder has no registry name, or a name is used both under
            `modalities` and under `encoders`.
    """
    collected: dict[str, tuple[EncoderConfig, str]] = {}
    for name, modality in config.modalities.items():
        collected[name] = (modality.encoder, f"{prefix}.modalities.{name}.encoder")
    for name, encoder in config.encoders.items():
        if name in collected:
            raise ValueError(
                f"{prefix}.encoders.{name}: the encoder name '{name}' is already used by "
                f"{collected[name][1]}. Encoder names must be unique across `modalities` and "
                f"`encoders`."
            )
        collected[name] = (encoder, f"{prefix}.encoders.{name}")
    for _, (encoder, path) in collected.items():
        if _isMissing(encoder.name):
            raise ValueError(f"{path}.name: required, no encoder registry name given.")
    return collected


def collectDecoders(
    config: ModelConfig, prefix: str = "model"
) -> dict[str, tuple[DecoderConfig, str]]:
    """Gather every decoder of a model, from `modalities` and from `decoders`.

    Args:
        config: The model config.
        prefix: Dotted path of `config`, the start of every error path.

    Returns:
        Decoder instance name -> `(DecoderConfig, dotted path of that config)`. The decoders of
        `modalities` come first, in order, then those of `decoders`.

    Raises:
        ValueError: If a decoder has no registry name, or a name is used both under
            `modalities` and under `decoders`.
    """
    collected: dict[str, tuple[DecoderConfig, str]] = {}
    for name, modality in config.modalities.items():
        collected[name] = (modality.decoder, f"{prefix}.modalities.{name}.decoder")
    for name, decoder in config.decoders.items():
        if name in collected:
            raise ValueError(
                f"{prefix}.decoders.{name}: the decoder name '{name}' is already used by "
                f"{collected[name][1]}. Decoder names must be unique across `modalities` and "
                f"`decoders`."
            )
        collected[name] = (decoder, f"{prefix}.decoders.{name}")
    for _, (decoder, path) in collected.items():
        if _isMissing(decoder.name):
            raise ValueError(f"{path}.name: required, no decoder registry name given.")
    return collected


def _parseEdge(value: Any, path: str) -> EdgeHeadConfig:
    """Read the value of one `fed_by` mapping entry.

    Args:
        value: `None` (an identity edge), an `EdgeHeadConfig`, or a mapping with the keys `head`
            and `kwargs`, both optional.
        path: Dotted path of the entry, for the error message.

    Returns:
        The edge as an `EdgeHeadConfig`.

    Raises:
        ValueError: If `value` has another shape, an unknown key, a non-string `head` or a
            `kwargs` that is not a mapping with string keys.
    """
    if value is None:
        return EdgeHeadConfig()
    if isinstance(value, EdgeHeadConfig):
        head: Any = value.head
        kwargs: Any = value.kwargs
    elif isinstance(value, Mapping):
        unknown = [key for key in value if key not in _EDGE_KEYS]
        if unknown:
            raise ValueError(
                f"{path}.{unknown[0]}: unknown key. An edge takes {list(_EDGE_KEYS)} "
                f"(or `null` for an identity edge)."
            )
        head = value.get("head", _IDENTITY_HEAD)
        kwargs = value.get("kwargs", {})
    else:
        raise ValueError(
            f"{path}: expected a mapping with the keys {list(_EDGE_KEYS)} or `null`, got {value!r}."
        )
    if not isinstance(head, str) or not head or _isMissing(head):
        raise ValueError(f"{path}.head: expected a head registry name, got {head!r}.")
    if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
        raise ValueError(f"{path}.kwargs: expected a mapping with string keys, got {kwargs!r}.")
    return EdgeHeadConfig(head=head, kwargs=dict(kwargs))


def parseFedBy(raw: Any, path: str) -> dict[str, EdgeHeadConfig | None]:
    """Normalize the `fed_by` of one latent space into encoder name -> edge.

    Args:
        raw: Either a list of encoder names (identity edges), or a mapping `encoder name ->
            {head, kwargs}` (or `null`) with one latent head per edge.
        path: Dotted path of the `fed_by` field, for the error message.

    Returns:
        Encoder name -> `EdgeHeadConfig` for a mapping `fed_by`, `None` for the entries of a
        list `fed_by`. A `None` edge builds no head module, while an `EdgeHeadConfig` always
        builds one (an `identity` head checks that the two widths agree).

    Raises:
        ValueError: If `raw` is missing, empty, a bare string, neither a list nor a mapping, a
            list with a non-string or repeated entry, or a mapping with a malformed edge.
    """
    if raw is None or _isMissing(raw):
        raise ValueError(
            f"{path}: required. Give a list of encoder names, or a mapping "
            f"`encoder name -> {{head, kwargs}}`."
        )
    if isinstance(raw, str):
        raise ValueError(
            f"{path}: expected a list of encoder names or a mapping, got the string {raw!r}. "
            f"Write [{raw}] for one encoder."
        )
    if isinstance(raw, Mapping):
        mapped: dict[str, EdgeHeadConfig | None] = {}
        for encoder_name, value in raw.items():
            if not isinstance(encoder_name, str):
                raise ValueError(f"{path}: encoder names must be strings, got {encoder_name!r}.")
            mapped[encoder_name] = _parseEdge(value, f"{path}.{encoder_name}")
        if not mapped:
            raise ValueError(f"{path}: empty. A latent space needs at least one encoder.")
        return mapped
    if isinstance(raw, Sequence):
        listed: dict[str, EdgeHeadConfig | None] = {}
        for index, encoder_name in enumerate(raw):
            if not isinstance(encoder_name, str):
                raise ValueError(
                    f"{path}[{index}]: expected an encoder name, got {encoder_name!r}."
                )
            if encoder_name in listed:
                raise ValueError(f"{path}[{index}]: encoder '{encoder_name}' is listed twice.")
            listed[encoder_name] = None
        if not listed:
            raise ValueError(f"{path}: empty. A latent space needs at least one encoder.")
        return listed
    raise ValueError(
        f"{path}: expected a list of encoder names or a mapping, got {type(raw).__name__}."
    )


def _isIdentityEdge(edge: EdgeHeadConfig | None) -> bool:
    """Whether an edge hands the encoder's `(mu, logvar)` over unchanged."""
    return edge is None or edge.head == _IDENTITY_HEAD


def _checkSections(config: ModelConfig, prefix: str) -> None:
    """Check that a model has the section its `latent_mode` needs, and not the other one.

    Args:
        config: The model config.
        prefix: Dotted path of `config`.

    Raises:
        ValueError: If `latent_mode` is unknown, `single_latent` is absent in `single` mode,
            `latent` is absent in `several` mode, or the section of the other mode is set too
            (a section that would be ignored silently is refused instead).
    """
    if config.latent_mode not in LATENT_MODES:
        raise ValueError(
            f"{prefix}.latent_mode: expected one of {list(LATENT_MODES)}, got "
            f"{config.latent_mode!r}."
        )
    if config.latent_mode == "single":
        if config.single_latent is None:
            raise ValueError(
                f"{prefix}.single_latent: required when {prefix}.latent_mode is 'single'. Give "
                f"`single_latent: {{dim: ...}}`, or set latent_mode to 'several' and describe "
                f"{prefix}.latent."
            )
        if config.latent is not None:
            raise ValueError(
                f"{prefix}.latent: only used when {prefix}.latent_mode is 'several', and the "
                f"mode is 'single'. Remove it, or use {prefix}.single_latent."
            )
        return
    if config.latent is None:
        raise ValueError(
            f"{prefix}.latent: required when {prefix}.latent_mode is 'several'. Describe "
            f"`spaces` and `decoders_consume`, or set latent_mode to 'single'."
        )
    if config.single_latent is not None:
        raise ValueError(
            f"{prefix}.single_latent: only used when {prefix}.latent_mode is 'single', and the "
            f"mode is 'several'. Remove it, or use {prefix}.latent."
        )


def _validateSingle(
    single: SingleLatentConfig,
    encoders: dict[str, tuple[EncoderConfig, str]],
    prefix: str,
) -> None:
    """Check the `single_latent` section of a `latent_mode: single` model.

    Args:
        single: The `single_latent` section.
        encoders: The collected encoders of the model.
        prefix: Dotted path of the model config.

    Raises:
        ValueError: If `single_latent.dim` is not a positive integer, a fusion has no strategy,
            or several encoders feed the latent space without a fusion.
    """
    _requirePositiveInt(single.dim, f"{prefix}.single_latent.dim")
    if single.fusion is not None and _isMissing(single.fusion.strategy):
        raise ValueError(f"{prefix}.single_latent.fusion.strategy: required, no strategy given.")
    if len(encoders) > 1 and single.fusion is None:
        raise ValueError(
            f"{prefix}.single_latent.fusion: required, since {len(encoders)} encoders "
            f"{list(encoders)} feed the latent space '{single.name}' (spec §4). Choose a "
            f"strategy, for example `fusion: {{strategy: poe}}`."
        )


def _validateSpaces(
    latent: LatentConfig,
    encoders: dict[str, tuple[EncoderConfig, str]],
    prefix: str,
) -> dict[str, dict[str, EdgeHeadConfig | None]]:
    """Check `latent.spaces` of a `latent_mode: several` model.

    Args:
        latent: The `latent` section.
        encoders: The collected encoders of the model.
        prefix: Dotted path of the model config.

    Returns:
        Latent space name -> encoder name -> edge, as `parseFedBy` returns it.

    Raises:
        ValueError: If there is no latent space, a `dim` is not a positive integer, a `fed_by`
            is malformed or names an unknown encoder, a fusion is missing for a space fed by
            several encoders or given to one fed by a single encoder, an encoder feeds nothing,
            an encoder feeds a space through an identity edge of another width, or an encoder
            feeds several spaces through more than one identity edge.
    """
    if not latent.spaces:
        raise ValueError(f"{prefix}.latent.spaces: at least one latent space is required.")

    edges: dict[str, dict[str, EdgeHeadConfig | None]] = {}
    for space_name, space in latent.spaces.items():
        space_path = f"{prefix}.latent.spaces.{space_name}"
        _requirePositiveInt(space.dim, f"{space_path}.dim")
        space_edges = parseFedBy(space.fed_by, f"{space_path}.fed_by")
        for encoder_name in space_edges:
            if encoder_name not in encoders:
                raise ValueError(
                    f"{space_path}.fed_by: unknown encoder '{encoder_name}'. Declared encoders "
                    f"(under `modalities` or `encoders`): {sorted(encoders)}."
                )
        edges[space_name] = space_edges

        if space.fusion is not None and _isMissing(space.fusion.strategy):
            raise ValueError(f"{space_path}.fusion.strategy: required, no strategy given.")
        if len(space_edges) > 1 and space.fusion is None:
            raise ValueError(
                f"{space_path}.fusion: required, since {len(space_edges)} encoders "
                f"{list(space_edges)} feed this latent space (spec §4). Choose a strategy, for "
                f"example `fusion: {{strategy: poe}}`."
            )
        if len(space_edges) == 1 and space.fusion is not None:
            raise ValueError(
                f"{space_path}.fusion: this latent space is fed by one encoder "
                f"({next(iter(space_edges))}), so a fusion would never run. Remove it, or add "
                f"another encoder to `fed_by`."
            )

    fed = {name for space_edges in edges.values() for name in space_edges}
    for encoder_name, (_, encoder_path) in encoders.items():
        if encoder_name not in fed:
            raise ValueError(
                f"{encoder_path}: encoder '{encoder_name}' feeds no latent space. Name it in the "
                f"`fed_by` of a latent space under {prefix}.latent.spaces, or remove it."
            )

    for encoder_name, (encoder, encoder_path) in encoders.items():
        fed_spaces = [name for name, space_edges in edges.items() if encoder_name in space_edges]
        identity_spaces = [
            name for name in fed_spaces if _isIdentityEdge(edges[name][encoder_name])
        ]
        if len(fed_spaces) > 1 and len(identity_spaces) > 1:
            raise ValueError(
                f"{prefix}.latent.spaces.{identity_spaces[-1]}.fed_by: encoder '{encoder_name}' "
                f"feeds {len(fed_spaces)} latent spaces {fed_spaces}, and {len(identity_spaces)} "
                f"of those edges are identity edges {identity_spaces}: they would receive the "
                f"same `(mu, logvar)`, so those latent spaces would have identical posteriors "
                f"(ADR 0024). Give every edge but one a head, for example `fed_by: "
                f"{{{encoder_name}: {{head: linear}}}}`."
            )
        explicit = encoder.kwargs.get("latent_dim")
        for space_name in identity_spaces:
            space_dim = latent.spaces[space_name].dim
            if explicit is not None and explicit != space_dim:
                raise ValueError(
                    f"{encoder_path}.kwargs.latent_dim: {explicit!r} differs from dim={space_dim} "
                    f"of latent space '{space_name}', which encoder '{encoder_name}' feeds "
                    f"through an identity edge. Remove `latent_dim` (it is taken from the "
                    f"space), or give the edge a head, for example "
                    f"`{prefix}.latent.spaces.{space_name}.fed_by: {{{encoder_name}: "
                    f"{{head: linear}}}}`."
                )
    return edges


def _validateConsumption(
    latent: LatentConfig,
    decoders: dict[str, tuple[DecoderConfig, str]],
    prefix: str,
) -> None:
    """Check `latent.decoders_consume` of a `latent_mode: several` model.

    Args:
        latent: The `latent` section.
        decoders: The collected decoders of the model.
        prefix: Dotted path of the model config.

    Raises:
        ValueError: If an entry names an unknown decoder or latent space, lists no space or a
            space twice, lacks an assembler although it consumes several spaces, has an
            assembler or kwargs although it consumes one, a decoder has no entry, or a latent
            space is consumed by no decoder.
    """
    consumption_path = f"{prefix}.latent.decoders_consume"
    for decoder_name, consumption in latent.decoders_consume.items():
        path = f"{consumption_path}.{decoder_name}"
        if decoder_name not in decoders:
            raise ValueError(
                f"{path}: unknown decoder '{decoder_name}'. Declared decoders (under "
                f"`modalities` or `decoders`): {sorted(decoders)}."
            )
        if _isMissing(consumption.spaces) or not consumption.spaces:
            raise ValueError(f"{path}.spaces: required, give at least one latent space name.")
        for index, space_name in enumerate(consumption.spaces):
            if space_name not in latent.spaces:
                raise ValueError(
                    f"{path}.spaces[{index}]: unknown latent space '{space_name}'. Declared "
                    f"latent spaces: {sorted(latent.spaces)}."
                )
            if consumption.spaces.count(space_name) > 1:
                raise ValueError(f"{path}.spaces[{index}]: '{space_name}' is listed twice.")
        if len(consumption.spaces) > 1 and consumption.assembler is None:
            raise ValueError(
                f"{path}.assembler: required, since decoder '{decoder_name}' consumes "
                f"{len(consumption.spaces)} latent spaces {list(consumption.spaces)} (spec "
                f"§2.2). Choose one, for example `assembler: concat`."
            )
        if len(consumption.spaces) == 1 and (
            consumption.assembler is not None or consumption.kwargs
        ):
            raise ValueError(
                f"{path}: decoder '{decoder_name}' consumes one latent space "
                f"('{consumption.spaces[0]}'), so there is nothing to assemble. Remove "
                f"`assembler` and `kwargs`."
            )
        if consumption.assembler is None and consumption.kwargs:
            raise ValueError(f"{path}.kwargs: given without an `assembler` to receive them.")

    for decoder_name, (_, decoder_path) in decoders.items():
        if decoder_name not in latent.decoders_consume:
            raise ValueError(
                f"{decoder_path}: decoder '{decoder_name}' consumes no latent space. Add "
                f"`{decoder_name}: {{spaces: [...]}}` to {consumption_path}, or remove it."
            )

    consumed = {
        space_name
        for consumption in latent.decoders_consume.values()
        for space_name in consumption.spaces
    }
    for space_name in latent.spaces:
        if space_name not in consumed:
            raise ValueError(
                f"{prefix}.latent.spaces.{space_name}: no decoder consumes this latent space. "
                f"List it in the `spaces` of a decoder under {consumption_path}."
            )


def _validate(config: ModelConfig, prefix: str) -> _Parsed:
    """Validate a `ModelConfig` and collect what resolution needs from it.

    Args:
        config: The model config.
        prefix: Dotted path of `config`.

    Returns:
        The collected instances, the `single_latent` or `latent` section, and for `several` mode
        the parsed edges.

    Raises:
        ValueError: On any problem `validateModelConfig` lists.
    """
    _checkSections(config, prefix)
    encoders = collectEncoders(config, prefix)
    decoders = collectDecoders(config, prefix)
    if not encoders:
        raise ValueError(
            f"{prefix}.modalities: at least one modality is required (or declare encoders under "
            f"{prefix}.encoders)."
        )
    if not decoders:
        raise ValueError(
            f"{prefix}.modalities: at least one decoder is required (a modality brings one, or "
            f"declare decoders under {prefix}.decoders)."
        )

    if config.single_latent is not None and config.latent_mode == "single":
        _validateSingle(config.single_latent, encoders, prefix)
        return _Parsed(encoders=encoders, decoders=decoders, single=config.single_latent)
    if config.latent is None:
        raise ValueError(f"{prefix}.latent: required when {prefix}.latent_mode is 'several'.")
    edges = _validateSpaces(config.latent, encoders, prefix)
    _validateConsumption(config.latent, decoders, prefix)
    return _Parsed(encoders=encoders, decoders=decoders, latent=config.latent, edges=edges)


def validateModelConfig(config: ModelConfig, prefix: str = "model") -> None:
    """Check that a `ModelConfig` describes a model that can be built (ADR 0025).

    Looks at the config only: it does not import or instantiate anything, and it does not check
    that a registry name is registered (`resolveModelConfig` does). Safe to call on every
    shipped config file.

    Args:
        config: The model config.
        prefix: Dotted path of `config` in the experiment config, the start of every error path.

    Raises:
        ValueError: With the dotted path of the offending field first, if `latent_mode` is
            unknown; `single_latent` is missing (or `latent` is also set) in `single` mode;
            `latent` is missing (or `single_latent` is also set) in `several` mode; there is no
            encoder or no decoder; an encoder or decoder name is used twice; a name in
            `fed_by` or `decoders_consume` is unknown; a latent space has no fusion although
            several encoders feed it (or a fusion although one does); a decoder has no
            assembler although it consumes several latent spaces; an encoder or decoder is left
            unused; a latent space has no consumer; an encoder feeds several latent spaces
            through more than one identity edge; or an explicit `latent_dim` disagrees with the
            latent space an identity edge feeds.
    """
    _validate(config, prefix)


def _decoderInputWidth(consumption: DecoderConsumptionConfig, dims: list[int], path: str) -> int:
    """Width of the tensor a decoder receives from its latent spaces.

    Args:
        consumption: The decoder's `decoders_consume` entry.
        dims: The `dim` of each latent space it consumes, in `RoutingGraph.decoderInputs` order
            (the order `GlobalVae` sizes the assembler with).
        path: Dotted path of the entry, for the error message.

    Returns:
        The latent dimension itself for a single space; otherwise the `output_dim` of the
        assembler built from `dims` and `consumption.kwargs`, so any registered assembler works.
        The throwaway assembler is built with the random number generator state saved and
        restored, so resolving a config does not shift the initialisation of the real model.

    Raises:
        KeyError: If the assembler name is not registered.
        ValueError: If the assembler cannot be built from `kwargs` and these widths.
    """
    if len(dims) == 1:
        return dims[0]
    assembler_name = consumption.assembler or ""
    assembler_class = _lookup(getAssemblerClass, assembler_name, f"{path}.assembler")
    try:
        with torch.random.fork_rng(devices=[]):
            assembler = assembler_class(dims, **consumption.kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: assembler '{assembler_name}' cannot be built from `kwargs` "
            f"{consumption.kwargs} and latent spaces of widths {dims}: {error}"
        ) from error
    return assembler.output_dim


def _resolveSingle(parsed: _Parsed, single: SingleLatentConfig, prefix: str) -> ResolvedModel:
    """Resolve a validated `single` model.

    Args:
        parsed: The validated model.
        single: Its `single_latent` section.
        prefix: Dotted path of the model config.

    Returns:
        The arguments of `GlobalVae.__init__`. Every encoder and decoder receives
        `latent_dim=single.dim` unless it sets one itself.

    Raises:
        KeyError: If a registry name is unknown.
    """
    latent_dim = single.dim
    latent_path = f"{prefix}.single_latent"
    for _, (encoder, path) in parsed.encoders.items():
        _lookup(getEncoderClass, encoder.name, f"{path}.name")
    for _, (decoder, path) in parsed.decoders.items():
        _lookup(getDecoderClass, decoder.name, f"{path}.name")

    fusion = single.fusion
    if fusion is not None:
        _lookup(getFusionClass, fusion.strategy, f"{latent_path}.fusion.strategy")
    _lookup(getRegularizerClass, single.regularizer.strategy, f"{latent_path}.regularizer.strategy")

    return ResolvedModel(
        encoder_configs={name: encoder.name for name, (encoder, _) in parsed.encoders.items()},
        decoder_configs={name: decoder.name for name, (decoder, _) in parsed.decoders.items()},
        routing_graph=buildSingleLatentRoutingGraph(
            encoder_names=list(parsed.encoders),
            decoder_names=list(parsed.decoders),
            latent_dim=latent_dim,
            latent_name=single.name,
        ),
        fusion_strategies={single.name: fusion.strategy} if fusion is not None else {},
        regularizer_strategies={single.name: single.regularizer.strategy},
        encoder_kwargs={
            name: {"latent_dim": latent_dim, **encoder.kwargs}
            for name, (encoder, _) in parsed.encoders.items()
        },
        decoder_kwargs={
            name: {"latent_dim": latent_dim, **decoder.kwargs}
            for name, (decoder, _) in parsed.decoders.items()
        },
        fusion_kwargs={single.name: fusion.kwargs} if fusion is not None else {},
        regularizer_kwargs={single.name: single.regularizer.kwargs},
        fusion_residual={single.name: True} if fusion is not None and fusion.residual else {},
    )


def _resolveSeveral(parsed: _Parsed, latent: LatentConfig, prefix: str) -> ResolvedModel:
    """Resolve a validated `several` model: build its `RoutingGraph` and fill in the widths.

    Args:
        parsed: The validated model.
        latent: Its `latent` section.
        prefix: Dotted path of the model config.

    Returns:
        The arguments of `GlobalVae.__init__`.

    Raises:
        KeyError: If a registry name is unknown.
        ValueError: If the `RoutingGraph` is refused by `validateRoutingGraph`, an encoder whose
            every edge goes through a head has no `latent_dim`, an explicit decoder `latent_dim`
            differs from the width its assembler produces, or an assembler cannot be built.
    """
    spaces = latent.spaces
    encoder_to_latents: dict[str, list[str]] = {name: [] for name in parsed.encoders}
    encoder_heads: dict[str, dict[str, HeadSpec]] = {}
    for space_name, space_edges in parsed.edges.items():
        for encoder_name, edge in space_edges.items():
            encoder_to_latents[encoder_name].append(space_name)
            if edge is None:
                continue
            _lookup(
                getHeadClass,
                edge.head,
                f"{prefix}.latent.spaces.{space_name}.fed_by.{encoder_name}.head",
            )
            encoder_heads.setdefault(encoder_name, {})[space_name] = HeadSpec(
                name=edge.head, kwargs=dict(edge.kwargs)
            )

    latent_to_decoders: dict[str, list[str]] = {name: [] for name in spaces}
    for decoder_name, consumption in latent.decoders_consume.items():
        for space_name in consumption.spaces:
            latent_to_decoders[space_name].append(decoder_name)
        if consumption.assembler is not None:
            # Before `validateRoutingGraph`, which reads the assembler's class from the registry:
            # an unknown name is reported here, with the path of the field that holds it.
            _lookup(
                getAssemblerClass,
                consumption.assembler,
                f"{prefix}.latent.decoders_consume.{decoder_name}.assembler",
            )

    graph = RoutingGraph(
        latent_specs={name: LatentSpace(name, space.dim) for name, space in spaces.items()},
        encoder_to_latents=encoder_to_latents,
        latent_to_decoders=latent_to_decoders,
        decoder_assemblers={
            name: consumption.assembler
            for name, consumption in latent.decoders_consume.items()
            if consumption.assembler is not None
        },
        encoder_heads=encoder_heads,
    )
    try:
        validateRoutingGraph(graph)
    except ValueError as error:
        raise ValueError(f"{prefix}.latent: {error}") from error

    encoder_kwargs: dict[str, dict[str, Any]] = {}
    for encoder_name, (encoder, path) in parsed.encoders.items():
        _lookup(getEncoderClass, encoder.name, f"{path}.name")
        kwargs = dict(encoder.kwargs)
        if "latent_dim" not in kwargs:
            identity_dims = [
                spaces[space_name].dim
                for space_name in encoder_to_latents[encoder_name]
                if _isIdentityEdge(parsed.edges[space_name][encoder_name])
            ]
            if not identity_dims:
                raise ValueError(
                    f"{path}.kwargs.latent_dim: required. Every edge from encoder "
                    f"'{encoder_name}' goes through a head, so its output width cannot be taken "
                    f"from the dim of a latent space."
                )
            kwargs["latent_dim"] = identity_dims[0]
        encoder_kwargs[encoder_name] = kwargs

    decoder_kwargs: dict[str, dict[str, Any]] = {}
    for decoder_name, (decoder, path) in parsed.decoders.items():
        _lookup(getDecoderClass, decoder.name, f"{path}.name")
        consumption = latent.decoders_consume[decoder_name]
        width = _decoderInputWidth(
            consumption,
            [spaces[name].dim for name in graph.decoderInputs()[decoder_name]],
            f"{prefix}.latent.decoders_consume.{decoder_name}",
        )
        kwargs = dict(decoder.kwargs)
        explicit = kwargs.get("latent_dim")
        if explicit is not None and explicit != width:
            raise ValueError(
                f"{path}.kwargs.latent_dim: {explicit!r} differs from {width}, the width of what "
                f"decoder '{decoder_name}' receives from {list(consumption.spaces)}. Remove "
                f"`latent_dim`, it is taken from the latent spaces."
            )
        kwargs["latent_dim"] = width
        decoder_kwargs[decoder_name] = kwargs

    fusion_strategies: dict[str, str] = {}
    fusion_kwargs: dict[str, dict[str, Any]] = {}
    fusion_residual: dict[str, bool] = {}
    regularizer_strategies: dict[str, str] = {}
    regularizer_kwargs: dict[str, dict[str, Any]] = {}
    for space_name, space in spaces.items():
        space_path = f"{prefix}.latent.spaces.{space_name}"
        if space.fusion is not None:
            _lookup(getFusionClass, space.fusion.strategy, f"{space_path}.fusion.strategy")
            fusion_strategies[space_name] = space.fusion.strategy
            fusion_kwargs[space_name] = space.fusion.kwargs
            if space.fusion.residual:
                fusion_residual[space_name] = True
        _lookup(
            getRegularizerClass, space.regularizer.strategy, f"{space_path}.regularizer.strategy"
        )
        regularizer_strategies[space_name] = space.regularizer.strategy
        regularizer_kwargs[space_name] = space.regularizer.kwargs

    return ResolvedModel(
        encoder_configs={name: encoder.name for name, (encoder, _) in parsed.encoders.items()},
        decoder_configs={name: decoder.name for name, (decoder, _) in parsed.decoders.items()},
        routing_graph=graph,
        fusion_strategies=fusion_strategies,
        regularizer_strategies=regularizer_strategies,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        fusion_kwargs=fusion_kwargs,
        regularizer_kwargs=regularizer_kwargs,
        fusion_residual=fusion_residual,
        assembler_kwargs={
            name: dict(consumption.kwargs)
            for name, consumption in latent.decoders_consume.items()
            if consumption.assembler is not None
        },
    )


def resolveModelConfig(config: ModelConfig, prefix: str = "model") -> ResolvedModel:
    """Validate a `ModelConfig` and derive the arguments of `GlobalVae.__init__` (ADR 0025).

    Args:
        config: The model config.
        prefix: Dotted path of `config` in the experiment config, the start of every error path.

    Returns:
        A `ResolvedModel`: registry names, the `RoutingGraph`, and every kwargs mapping, with the
        `latent_dim` of each encoder and decoder filled in where the config omits it (an explicit
        value always wins, in `single` mode as before).

    Raises:
        ValueError: If `validateModelConfig` rejects the config, the `RoutingGraph` is refused by
            `validateRoutingGraph` (the message then starts with `<prefix>.latent`), an encoder
            whose every edge goes through a head has no `latent_dim`, or a decoder's explicit
            `latent_dim` differs from the width its assembler produces.
        KeyError: If an encoder, decoder, fusion, regularizer, head or assembler name is not
            registered; the message starts with the path of the field holding the name.
    """
    parsed = _validate(config, prefix)
    if parsed.latent is not None:
        return _resolveSeveral(parsed, parsed.latent, prefix)
    if parsed.single is not None:
        return _resolveSingle(parsed, parsed.single, prefix)
    raise ValueError(f"{prefix}.latent_mode: expected one of {list(LATENT_MODES)}.")
