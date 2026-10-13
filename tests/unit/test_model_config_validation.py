"""Validation and resolution of a `ModelConfig` (roadmap P1-3, ADR 0025).

Everything here works on dataclasses built in Python, with no YAML and no model: it pins what
`validateModelConfig` accepts and refuses, the shapes `parseFedBy` understands, and the arguments
`resolveModelConfig` derives for `GlobalVae.__init__` (widths, graph, kwargs). The one rule that
runs through all the refusals is that the message starts with the dotted path of the field to
edit, so most tests match on that path.
"""

from collections.abc import Sequence
from typing import Any

import pytest
import torch

import global_vae.assemblers.registry as assembler_registry
from global_vae.assemblers.base import AbstractAssembler
from global_vae.config.model import (
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
    collectDecoders,
    collectEncoders,
    parseFedBy,
    resolveModelConfig,
    validateModelConfig,
)
from global_vae.latent.base import HeadSpec

SIGNAL_ENCODER = "1d_cnn_encoder_v1"
IMAGE_ENCODER = "2d_cnn_encoder_v1"
IMAGE_DECODER = "2d_cnn_decoder_v1"
SIGNAL_DECODER = "1d_cnn_decoder_v1"


def _encoder(name: str = SIGNAL_ENCODER, **kwargs: Any) -> EncoderConfig:
    return EncoderConfig(name=name, kwargs=dict(kwargs))


def _decoder(name: str = IMAGE_DECODER, **kwargs: Any) -> DecoderConfig:
    return DecoderConfig(name=name, kwargs=dict(kwargs))


def _singleConfig(**overrides: Any) -> ModelConfig:
    """A valid `single` model: one signal modality, one latent space of 16."""
    fields: dict[str, Any] = {
        "modalities": {"signal": ModalityConfig(_encoder(), _decoder(SIGNAL_DECODER))},
        "latent_mode": "single",
        "single_latent": SingleLatentConfig(dim=16),
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def _severalConfig(latent: LatentConfig | None = None, **overrides: Any) -> ModelConfig:
    """A valid `several` model: two encoders, one latent space each, one joint decoder."""
    fields: dict[str, Any] = {
        "encoders": {"signal": _encoder(), "image": _encoder(IMAGE_ENCODER)},
        "decoders": {"image_out": _decoder()},
        "latent_mode": "several",
        "latent": latent
        or LatentConfig(
            spaces={
                "z_signal": LatentSpaceConfig(dim=16, fed_by=["signal"]),
                "z_image": LatentSpaceConfig(dim=16, fed_by=["image"]),
            },
            decoders_consume={
                "image_out": DecoderConsumptionConfig(
                    spaces=["z_signal", "z_image"], assembler="concat"
                )
            },
        ),
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def _sharedPrivateLatent() -> LatentConfig:
    """Shared plus private spaces over two encoders, with `modalities`-style decoder names."""
    return LatentConfig(
        spaces={
            "z_shared": LatentSpaceConfig(
                dim=32, fed_by=["signal", "image"], fusion=FusionConfig(strategy="poe")
            ),
            "z_signal_private": LatentSpaceConfig(dim=8, fed_by={"signal": {"head": "linear"}}),
            "z_image_private": LatentSpaceConfig(dim=8, fed_by={"image": {"head": "linear"}}),
        },
        decoders_consume={
            "signal": DecoderConsumptionConfig(
                spaces=["z_shared", "z_signal_private"], assembler="concat"
            ),
            "image": DecoderConsumptionConfig(
                spaces=["z_shared", "z_image_private"], assembler="concat"
            ),
        },
    )


def _sharedPrivateConfig() -> ModelConfig:
    return ModelConfig(
        modalities={
            "signal": ModalityConfig(_encoder(), _decoder(SIGNAL_DECODER)),
            "image": ModalityConfig(_encoder(IMAGE_ENCODER), _decoder()),
        },
        latent_mode="several",
        latent=_sharedPrivateLatent(),
    )


class TestParseFedBy:
    def test_a_list_gives_identity_edges_without_a_head(self) -> None:
        assert parseFedBy(["signal", "image"], "p") == {"signal": None, "image": None}

    def test_a_mapping_gives_one_edge_head_per_encoder(self) -> None:
        edges = parseFedBy({"signal": {"head": "linear", "kwargs": {"bias": False}}}, "p")
        assert edges == {"signal": EdgeHeadConfig(head="linear", kwargs={"bias": False})}

    def test_a_null_mapping_value_is_an_explicit_identity_head(self) -> None:
        assert parseFedBy({"signal": None}, "p") == {"signal": EdgeHeadConfig()}

    def test_an_empty_edge_mapping_defaults_to_the_identity_head(self) -> None:
        assert parseFedBy({"signal": {}}, "p") == {"signal": EdgeHeadConfig(head="identity")}

    def test_an_edge_head_config_instance_is_accepted(self) -> None:
        edge = EdgeHeadConfig(head="linear")
        assert parseFedBy({"signal": edge}, "p") == {"signal": edge}

    def test_the_parsed_kwargs_are_a_copy(self) -> None:
        kwargs = {"bias": True}
        edges = parseFedBy({"signal": {"head": "linear", "kwargs": kwargs}}, "p")
        edge = edges["signal"]
        assert edge is not None
        edge.kwargs["bias"] = False
        assert kwargs == {"bias": True}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("???", "p: required"),
            (None, "p: required"),
            ("signal", r"got the string 'signal'. Write \[signal\]"),
            ([], "p: empty"),
            ({}, "p: empty"),
            ([1], r"p\[0\]: expected an encoder name"),
            (["a", "a"], r"p\[1\]: encoder 'a' is listed twice"),
            (7, "p: expected a list of encoder names or a mapping, got int"),
            ({1: None}, "p: encoder names must be strings"),
            ({"a": "linear"}, "p.a: expected a mapping with the keys"),
            ({"a": {"heads": "linear"}}, r"p.a.heads: unknown key"),
            ({"a": {"head": 3}}, "p.a.head: expected a head registry name"),
            ({"a": {"head": ""}}, "p.a.head: expected a head registry name"),
            ({"a": {"kwargs": [1]}}, "p.a.kwargs: expected a mapping with string keys"),
            ({"a": {"kwargs": {1: 2}}}, "p.a.kwargs: expected a mapping with string keys"),
        ],
    )
    def test_a_malformed_fed_by_names_its_field(self, raw: Any, expected: str) -> None:
        with pytest.raises(ValueError, match=expected):
            parseFedBy(raw, "p")


class TestCollect:
    def test_encoders_and_decoders_of_modalities_come_before_the_sections(self) -> None:
        config = ModelConfig(
            modalities={"signal": ModalityConfig(_encoder(), _decoder(SIGNAL_DECODER))},
            encoders={"image_in": _encoder(IMAGE_ENCODER)},
            decoders={"image_out": _decoder()},
        )
        encoders = collectEncoders(config)
        decoders = collectDecoders(config)
        assert list(encoders) == ["signal", "image_in"]
        assert list(decoders) == ["signal", "image_out"]
        assert encoders["signal"][1] == "model.modalities.signal.encoder"
        assert encoders["image_in"][1] == "model.encoders.image_in"
        assert decoders["signal"][1] == "model.modalities.signal.decoder"

    def test_an_encoder_name_in_both_places_is_refused_with_both_paths(self) -> None:
        config = ModelConfig(
            modalities={"signal": ModalityConfig(_encoder(), _decoder(SIGNAL_DECODER))},
            encoders={"signal": _encoder()},
        )
        with pytest.raises(ValueError, match=r"model.encoders.signal: .*model.modalities.signal"):
            collectEncoders(config)

    def test_a_decoder_name_in_both_places_is_refused_with_both_paths(self) -> None:
        config = ModelConfig(
            modalities={"signal": ModalityConfig(_encoder(), _decoder(SIGNAL_DECODER))},
            decoders={"signal": _decoder()},
        )
        with pytest.raises(ValueError, match=r"model.decoders.signal: .*model.modalities.signal"):
            collectDecoders(config)

    def test_a_missing_encoder_name_is_refused_with_its_path(self) -> None:
        config = ModelConfig(encoders={"signal": EncoderConfig()})
        with pytest.raises(ValueError, match=r"model.encoders.signal.name: required"):
            collectEncoders(config)

    def test_a_missing_decoder_name_is_refused_with_its_path(self) -> None:
        config = ModelConfig(decoders={"image_out": DecoderConfig()})
        with pytest.raises(ValueError, match=r"model.decoders.image_out.name: required"):
            collectDecoders(config)

    def test_the_prefix_starts_every_path(self) -> None:
        config = ModelConfig(encoders={"signal": EncoderConfig()})
        with pytest.raises(ValueError, match=r"^experiment.model.encoders.signal.name"):
            collectEncoders(config, prefix="experiment.model")


class TestValidModels:
    def test_a_modalities_only_single_model_is_valid(self) -> None:
        validateModelConfig(_singleConfig())

    def test_a_single_model_may_declare_encoders_and_decoders(self) -> None:
        config = _singleConfig(
            modalities={},
            encoders={"signal": _encoder(), "image_in": _encoder(IMAGE_ENCODER)},
            decoders={"image_out": _decoder()},
            single_latent=SingleLatentConfig(dim=8, fusion=FusionConfig(strategy="poe")),
        )
        validateModelConfig(config)

    def test_an_independent_latents_model_is_valid(self) -> None:
        validateModelConfig(_severalConfig())

    def test_a_shared_private_model_is_valid(self) -> None:
        validateModelConfig(_sharedPrivateConfig())

    def test_validation_does_not_look_at_registries(self) -> None:
        config = _severalConfig(encoders={"signal": _encoder("nope"), "image": _encoder("nope")})
        validateModelConfig(config)


class TestSections:
    def test_an_unknown_latent_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"model.latent_mode: expected one of .*'multi'"):
            validateModelConfig(_singleConfig(latent_mode="multi"))

    def test_single_mode_requires_single_latent(self) -> None:
        with pytest.raises(ValueError, match=r"model.single_latent: required"):
            validateModelConfig(_singleConfig(single_latent=None))

    def test_single_mode_refuses_a_latent_section(self) -> None:
        with pytest.raises(ValueError, match=r"model.latent: only used when .* 'several'"):
            validateModelConfig(_singleConfig(latent=LatentConfig()))

    def test_several_mode_requires_latent(self) -> None:
        config = ModelConfig(
            encoders={"signal": _encoder()},
            decoders={"image_out": _decoder()},
            latent_mode="several",
        )
        with pytest.raises(ValueError, match=r"model.latent: required"):
            validateModelConfig(config)

    def test_several_mode_refuses_single_latent(self) -> None:
        config = _severalConfig(single_latent=SingleLatentConfig(dim=16))
        with pytest.raises(ValueError, match=r"model.single_latent: only used when .* 'single'"):
            validateModelConfig(config)

    def test_no_encoder_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"model.modalities: at least one modality"):
            validateModelConfig(_singleConfig(modalities={}))

    def test_no_decoder_is_refused(self) -> None:
        config = _singleConfig(modalities={}, encoders={"signal": _encoder()})
        with pytest.raises(ValueError, match=r"model.modalities: at least one decoder"):
            validateModelConfig(config)


class TestSingleLatent:
    @pytest.mark.parametrize("dim", [0, -4, 2.5, "8", True])
    def test_a_bad_dim_is_refused(self, dim: Any) -> None:
        config = _singleConfig(single_latent=SingleLatentConfig(dim=dim))
        with pytest.raises(ValueError, match=r"model.single_latent.dim: expected a positive"):
            validateModelConfig(config)

    def test_a_missing_dim_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"model.single_latent.dim: required"):
            validateModelConfig(_singleConfig(single_latent=SingleLatentConfig()))

    def test_several_encoders_need_a_fusion(self) -> None:
        config = _singleConfig(
            modalities={},
            encoders={"signal": _encoder(), "image": _encoder(IMAGE_ENCODER)},
            decoders={"image_out": _decoder()},
        )
        with pytest.raises(ValueError, match=r"model.single_latent.fusion: required, since 2"):
            validateModelConfig(config)

    def test_a_fusion_without_a_strategy_is_refused(self) -> None:
        config = _singleConfig(single_latent=SingleLatentConfig(dim=16, fusion=FusionConfig()))
        with pytest.raises(ValueError, match=r"model.single_latent.fusion.strategy: required"):
            validateModelConfig(config)

    def test_a_fusion_with_one_encoder_stays_accepted(self) -> None:
        validateModelConfig(
            _singleConfig(single_latent=SingleLatentConfig(dim=16, fusion=FusionConfig("poe")))
        )


def _replaceSpace(config: ModelConfig, name: str, space: LatentSpaceConfig) -> ModelConfig:
    assert config.latent is not None
    config.latent.spaces[name] = space
    return config


class TestSeveralSpaces:
    def test_no_space_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"model.latent.spaces: at least one"):
            validateModelConfig(_severalConfig(latent=LatentConfig()))

    @pytest.mark.parametrize("dim", [0, -1, 1.5, "4"])
    def test_a_bad_dim_is_refused(self, dim: Any) -> None:
        config = _replaceSpace(_severalConfig(), "z_signal", LatentSpaceConfig(dim, ["signal"]))
        with pytest.raises(ValueError, match=r"model.latent.spaces.z_signal.dim: expected"):
            validateModelConfig(config)

    def test_a_missing_dim_is_refused(self) -> None:
        config = _replaceSpace(_severalConfig(), "z_signal", LatentSpaceConfig(fed_by=["signal"]))
        with pytest.raises(ValueError, match=r"model.latent.spaces.z_signal.dim: required"):
            validateModelConfig(config)

    def test_a_missing_fed_by_is_refused(self) -> None:
        config = _replaceSpace(_severalConfig(), "z_signal", LatentSpaceConfig(dim=16))
        with pytest.raises(ValueError, match=r"model.latent.spaces.z_signal.fed_by: required"):
            validateModelConfig(config)

    def test_an_unknown_encoder_in_fed_by_is_refused_and_lists_the_known_ones(self) -> None:
        config = _replaceSpace(_severalConfig(), "z_signal", LatentSpaceConfig(16, ["audio"]))
        with pytest.raises(ValueError, match=r"z_signal.fed_by: unknown encoder 'audio'.*image"):
            validateModelConfig(config)

    def test_a_space_fed_by_two_encoders_needs_a_fusion(self) -> None:
        space = LatentSpaceConfig(dim=16, fed_by=["signal", "image"])
        config = _replaceSpace(_severalConfig(), "z_signal", space)
        with pytest.raises(ValueError, match=r"z_signal.fusion: required, since 2 encoders"):
            validateModelConfig(config)

    def test_a_fusion_on_a_space_fed_by_one_encoder_is_refused(self) -> None:
        space = LatentSpaceConfig(dim=16, fed_by=["signal"], fusion=FusionConfig("poe"))
        config = _replaceSpace(_severalConfig(), "z_signal", space)
        with pytest.raises(ValueError, match=r"z_signal.fusion: .*fed by one encoder"):
            validateModelConfig(config)

    def test_a_fusion_without_a_strategy_is_refused(self) -> None:
        space = LatentSpaceConfig(dim=16, fed_by=["signal", "image"], fusion=FusionConfig())
        config = _replaceSpace(_severalConfig(), "z_signal", space)
        with pytest.raises(ValueError, match=r"z_signal.fusion.strategy: required"):
            validateModelConfig(config)

    def test_an_encoder_that_feeds_nothing_is_refused(self) -> None:
        config = _severalConfig(encoders={"signal": _encoder(), "image": _encoder(IMAGE_ENCODER)})
        config = _replaceSpace(config, "z_image", LatentSpaceConfig(dim=16, fed_by=["signal"]))
        with pytest.raises(ValueError, match=r"model.encoders.image: encoder 'image' feeds no"):
            validateModelConfig(config)

    def test_two_identity_edges_from_one_encoder_are_refused(self) -> None:
        latent = LatentConfig(
            spaces={
                "z_a": LatentSpaceConfig(dim=16, fed_by=["signal"]),
                "z_b": LatentSpaceConfig(dim=16, fed_by=["signal"]),
            },
            decoders_consume={
                "image_out": DecoderConsumptionConfig(spaces=["z_a", "z_b"], assembler="concat")
            },
        )
        config = _severalConfig(latent=latent, encoders={"signal": _encoder()})
        with pytest.raises(ValueError, match=r"z_b.fed_by: encoder 'signal' feeds 2.*identical"):
            validateModelConfig(config)

    def test_an_explicit_identity_head_counts_as_an_identity_edge(self) -> None:
        latent = LatentConfig(
            spaces={
                "z_a": LatentSpaceConfig(dim=16, fed_by=["signal"]),
                "z_b": LatentSpaceConfig(dim=16, fed_by={"signal": {"head": "identity"}}),
            },
            decoders_consume={
                "image_out": DecoderConsumptionConfig(spaces=["z_a", "z_b"], assembler="concat")
            },
        )
        config = _severalConfig(latent=latent, encoders={"signal": _encoder()})
        with pytest.raises(ValueError, match=r"identity edges"):
            validateModelConfig(config)

    def test_one_identity_edge_and_one_head_is_accepted(self) -> None:
        validateModelConfig(_sharedPrivateConfig())

    def test_an_explicit_latent_dim_must_match_an_identity_edge(self) -> None:
        config = _severalConfig(encoders={"signal": _encoder(latent_dim=7), "image": _encoder()})
        with pytest.raises(ValueError, match=r"model.encoders.signal.kwargs.latent_dim: 7 differs"):
            validateModelConfig(config)

    def test_an_explicit_latent_dim_may_differ_through_a_head(self) -> None:
        config = _sharedPrivateConfig()
        config.modalities["signal"].encoder.kwargs["latent_dim"] = 32
        validateModelConfig(config)
        assert config.latent is not None
        config.latent.spaces["z_signal_private"].dim = 5
        validateModelConfig(config)


class TestConsumption:
    @staticmethod
    def _withConsumption(entry: DecoderConsumptionConfig | None) -> ModelConfig:
        config = _severalConfig(decoders={"image_out": _decoder(), "other": _decoder()})
        assert config.latent is not None
        config.latent.decoders_consume = {
            "image_out": DecoderConsumptionConfig(
                spaces=["z_signal", "z_image"], assembler="concat"
            ),
        }
        if entry is not None:
            config.latent.decoders_consume["other"] = entry
        return config

    def test_an_unknown_decoder_is_refused(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        config.latent.decoders_consume["ghost"] = DecoderConsumptionConfig(spaces=["z_signal"])
        with pytest.raises(ValueError, match=r"decoders_consume.ghost: unknown decoder 'ghost'"):
            validateModelConfig(config)

    @pytest.mark.parametrize("spaces", [[], "???"])
    def test_a_decoder_that_names_no_space_is_refused(self, spaces: Any) -> None:
        config = self._withConsumption(DecoderConsumptionConfig(spaces=spaces))
        with pytest.raises(ValueError, match=r"decoders_consume.other.spaces: required"):
            validateModelConfig(config)

    def test_an_unknown_space_is_refused_and_lists_the_known_ones(self) -> None:
        config = self._withConsumption(DecoderConsumptionConfig(spaces=["z_audio"]))
        with pytest.raises(
            ValueError, match=r"other.spaces\[0\]: unknown latent space 'z_audio'.*z_image"
        ):
            validateModelConfig(config)

    def test_a_space_listed_twice_is_refused(self) -> None:
        config = self._withConsumption(
            DecoderConsumptionConfig(spaces=["z_signal", "z_signal"], assembler="sum")
        )
        with pytest.raises(ValueError, match=r"other.spaces\[0\]: 'z_signal' is listed twice"):
            validateModelConfig(config)

    def test_several_spaces_need_an_assembler(self) -> None:
        config = self._withConsumption(DecoderConsumptionConfig(spaces=["z_signal", "z_image"]))
        with pytest.raises(ValueError, match=r"decoders_consume.other.assembler: required"):
            validateModelConfig(config)

    def test_an_assembler_on_one_space_is_refused(self) -> None:
        config = self._withConsumption(
            DecoderConsumptionConfig(spaces=["z_signal"], assembler="concat")
        )
        with pytest.raises(ValueError, match=r"decoders_consume.other: .*nothing to assemble"):
            validateModelConfig(config)

    def test_assembler_kwargs_on_one_space_are_refused(self) -> None:
        config = self._withConsumption(
            DecoderConsumptionConfig(spaces=["z_signal"], kwargs={"scale": 2.0})
        )
        with pytest.raises(ValueError, match=r"decoders_consume.other: .*nothing to assemble"):
            validateModelConfig(config)

    def test_a_decoder_that_consumes_nothing_is_refused(self) -> None:
        config = self._withConsumption(None)
        with pytest.raises(ValueError, match=r"model.decoders.other: decoder 'other' consumes no"):
            validateModelConfig(config)

    def test_a_space_that_no_decoder_consumes_is_refused(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        config.latent.decoders_consume["image_out"] = DecoderConsumptionConfig(spaces=["z_signal"])
        with pytest.raises(ValueError, match=r"spaces.z_image: no decoder consumes"):
            validateModelConfig(config)


class TestResolveSingle:
    def test_widths_are_filled_in_for_every_encoder_and_decoder(self) -> None:
        resolved = resolveModelConfig(_singleConfig())
        assert resolved.encoder_kwargs == {"signal": {"latent_dim": 16}}
        assert resolved.decoder_kwargs == {"signal": {"latent_dim": 16}}

    def test_an_explicit_latent_dim_is_never_overwritten(self) -> None:
        config = _singleConfig()
        config.modalities["signal"].encoder.kwargs["latent_dim"] = 8
        assert resolveModelConfig(config).encoder_kwargs["signal"]["latent_dim"] == 8

    def test_the_graph_is_the_single_latent_graph(self) -> None:
        config = _singleConfig(
            modalities={},
            encoders={"signal": _encoder(), "image_in": _encoder(IMAGE_ENCODER)},
            decoders={"image_out": _decoder()},
            single_latent=SingleLatentConfig(dim=8, name="z", fusion=FusionConfig("poe")),
        )
        graph = resolveModelConfig(config).routing_graph
        assert list(graph.latent_specs) == ["z"]
        assert graph.encoder_to_latents == {"signal": ["z"], "image_in": ["z"]}
        assert graph.latent_to_decoders == {"z": ["image_out"]}
        assert graph.encoder_heads == {}

    def test_fusion_regularizer_and_residual_are_mapped_per_latent_space(self) -> None:
        config = _singleConfig(
            modalities={},
            encoders={"signal": _encoder(), "image_in": _encoder(IMAGE_ENCODER)},
            decoders={"image_out": _decoder()},
            single_latent=SingleLatentConfig(
                dim=8,
                name="z",
                fusion=FusionConfig("poe", kwargs={}, residual=True),
                regularizer=RegularizerConfig("free_bits_kl", {"free_bits": 0.5}),
            ),
        )
        resolved = resolveModelConfig(config)
        assert resolved.fusion_strategies == {"z": "poe"}
        assert resolved.fusion_residual == {"z": True}
        assert resolved.regularizer_strategies == {"z": "free_bits_kl"}
        assert resolved.regularizer_kwargs == {"z": {"free_bits": 0.5}}

    def test_no_fusion_means_no_fusion_entries(self) -> None:
        resolved = resolveModelConfig(_singleConfig())
        assert resolved.fusion_strategies == {}
        assert resolved.fusion_kwargs == {}
        assert resolved.fusion_residual == {}

    @pytest.mark.parametrize(
        ("config", "path"),
        [
            (
                _singleConfig(modalities={"signal": ModalityConfig(_encoder("nope"), _decoder())}),
                "model.modalities.signal.encoder.name",
            ),
            (
                _singleConfig(modalities={"signal": ModalityConfig(_encoder(), _decoder("nope"))}),
                "model.modalities.signal.decoder.name",
            ),
            (
                _singleConfig(single_latent=SingleLatentConfig(dim=4, fusion=FusionConfig("nope"))),
                "model.single_latent.fusion.strategy",
            ),
            (
                _singleConfig(
                    single_latent=SingleLatentConfig(dim=4, regularizer=RegularizerConfig("nope"))
                ),
                "model.single_latent.regularizer.strategy",
            ),
        ],
    )
    def test_an_unregistered_name_raises_key_error_with_the_field_path(
        self, config: ModelConfig, path: str
    ) -> None:
        with pytest.raises(KeyError, match=f"{path}: .*nope"):
            resolveModelConfig(config)


class TestResolveSeveral:
    def test_every_encoder_takes_the_dim_of_the_space_it_feeds(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        config.latent.spaces["z_image"].dim = 24
        config.latent.decoders_consume["image_out"].assembler = "concat"
        resolved = resolveModelConfig(config)
        assert resolved.encoder_kwargs["signal"]["latent_dim"] == 16
        assert resolved.encoder_kwargs["image"]["latent_dim"] == 24

    def test_a_decoder_takes_the_width_concat_produces(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        config.latent.spaces["z_image"].dim = 24
        assert resolveModelConfig(config).decoder_kwargs["image_out"]["latent_dim"] == 40

    def test_a_decoder_takes_the_common_dim_for_sum_and_average(self) -> None:
        for assembler in ("sum", "average"):
            config = _severalConfig()
            assert config.latent is not None
            config.latent.decoders_consume["image_out"].assembler = assembler
            assert resolveModelConfig(config).decoder_kwargs["image_out"]["latent_dim"] == 16

    def test_a_decoder_of_one_space_takes_that_dim(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        config.latent.decoders_consume["image_out"] = DecoderConsumptionConfig(spaces=["z_signal"])
        config.latent.spaces["z_image"].fed_by = ["image"]
        config.decoders["image_aux"] = _decoder()
        config.latent.decoders_consume["image_aux"] = DecoderConsumptionConfig(spaces=["z_image"])
        resolved = resolveModelConfig(config)
        assert resolved.decoder_kwargs["image_out"]["latent_dim"] == 16
        assert resolved.decoder_kwargs["image_aux"]["latent_dim"] == 16

    def test_an_explicit_decoder_latent_dim_that_matches_is_kept(self) -> None:
        config = _severalConfig(decoders={"image_out": _decoder(latent_dim=32)})
        assert resolveModelConfig(config).decoder_kwargs["image_out"]["latent_dim"] == 32

    def test_an_explicit_decoder_latent_dim_that_differs_is_refused(self) -> None:
        config = _severalConfig(decoders={"image_out": _decoder(latent_dim=31)})
        with pytest.raises(ValueError, match=r"model.decoders.image_out.kwargs.latent_dim: 31"):
            resolveModelConfig(config)

    def test_an_encoder_behind_a_head_needs_an_explicit_latent_dim(self) -> None:
        latent = LatentConfig(
            spaces={
                "z_a": LatentSpaceConfig(dim=8, fed_by={"signal": {"head": "linear"}}),
                "z_b": LatentSpaceConfig(dim=8, fed_by=["image"]),
            },
            decoders_consume={
                "image_out": DecoderConsumptionConfig(spaces=["z_a", "z_b"], assembler="concat")
            },
        )
        with pytest.raises(ValueError, match=r"model.encoders.signal.kwargs.latent_dim: required"):
            resolveModelConfig(_severalConfig(latent=latent))
        explicit = _severalConfig(
            latent=latent, encoders={"signal": _encoder(latent_dim=12), "image": _encoder()}
        )
        assert resolveModelConfig(explicit).encoder_kwargs["signal"]["latent_dim"] == 12

    def test_the_graph_has_one_head_spec_per_mapping_edge(self) -> None:
        resolved = resolveModelConfig(_sharedPrivateConfig())
        graph = resolved.routing_graph
        assert graph.encoder_to_latents == {
            "signal": ["z_shared", "z_signal_private"],
            "image": ["z_shared", "z_image_private"],
        }
        assert graph.encoder_heads == {
            "signal": {"z_signal_private": HeadSpec(name="linear")},
            "image": {"z_image_private": HeadSpec(name="linear")},
        }
        assert graph.latent_to_decoders == {
            "z_shared": ["signal", "image"],
            "z_signal_private": ["signal"],
            "z_image_private": ["image"],
        }
        assert graph.decoder_assemblers == {"signal": "concat", "image": "concat"}

    def test_a_null_mapping_edge_builds_an_explicit_identity_head_spec(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        config.latent.spaces["z_signal"].fed_by = {"signal": None}
        graph = resolveModelConfig(config).routing_graph
        assert graph.encoder_heads == {"signal": {"z_signal": HeadSpec(name="identity")}}

    def test_a_list_edge_builds_no_head_spec(self) -> None:
        assert resolveModelConfig(_severalConfig()).routing_graph.encoder_heads == {}

    def test_head_kwargs_reach_the_head_spec(self) -> None:
        config = _sharedPrivateConfig()
        assert config.latent is not None
        config.latent.spaces["z_signal_private"].fed_by = {
            "signal": {"head": "linear", "kwargs": {"bias": False}}
        }
        graph = resolveModelConfig(config).routing_graph
        assert graph.encoder_heads["signal"]["z_signal_private"].kwargs == {"bias": False}

    def test_fusion_and_regularizer_are_mapped_per_latent_space(self) -> None:
        config = _sharedPrivateConfig()
        assert config.latent is not None
        config.latent.spaces["z_shared"].fusion = FusionConfig("poe", residual=True)
        config.latent.spaces["z_image_private"].regularizer = RegularizerConfig(
            "free_bits_kl", {"free_bits": 0.25}
        )
        resolved = resolveModelConfig(config)
        assert resolved.fusion_strategies == {"z_shared": "poe"}
        assert resolved.fusion_residual == {"z_shared": True}
        assert resolved.regularizer_strategies == {
            "z_shared": "kl_standard_normal",
            "z_signal_private": "kl_standard_normal",
            "z_image_private": "free_bits_kl",
        }
        assert resolved.regularizer_kwargs["z_image_private"] == {"free_bits": 0.25}

    def test_the_routing_graph_rules_are_reported_under_the_latent_section(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        config.latent.spaces["z_image"].dim = 24
        config.latent.decoders_consume["image_out"].assembler = "sum"
        with pytest.raises(ValueError, match=r"^model.latent: .*requires matching dimensionality"):
            resolveModelConfig(config)

    def test_assembler_kwargs_are_resolved_per_decoder(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        assert resolveModelConfig(config).assembler_kwargs == {"image_out": {}}

    def test_assembler_kwargs_reach_the_probe_and_the_resolved_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Widening(AbstractAssembler):
            def __init__(self, input_dims: Sequence[int], extra: int) -> None:
                super().__init__(input_dims)
                self.extra = extra

            @property
            def output_dim(self) -> int:
                return sum(self.input_dims) + self.extra

            def forward(
                self, latents: list[torch.Tensor], present: Sequence[int] | None = None
            ) -> torch.Tensor:
                return torch.cat([*latents, latents[0][:, : self.extra]], dim=-1)

        monkeypatch.setitem(assembler_registry._ASSEMBLER_REGISTRY, "widening_test", _Widening)
        config = _severalConfig()
        assert config.latent is not None
        config.latent.decoders_consume["image_out"] = DecoderConsumptionConfig(
            spaces=["z_signal", "z_image"], assembler="widening_test", kwargs={"extra": 3}
        )
        resolved = resolveModelConfig(config)
        assert resolved.decoder_kwargs["image_out"]["latent_dim"] == 35
        assert resolved.assembler_kwargs == {"image_out": {"extra": 3}}

    def test_an_assembler_that_cannot_be_built_is_reported_with_its_path(self) -> None:
        config = _severalConfig()
        assert config.latent is not None
        config.latent.decoders_consume["image_out"].kwargs = {"nonsense": 1}
        with pytest.raises(
            ValueError, match=r"model.latent.decoders_consume.image_out: assembler 'concat'"
        ):
            resolveModelConfig(config)

    @pytest.mark.parametrize(
        ("mutate", "path"),
        [
            (lambda c: c.encoders["signal"].__setattr__("name", "nope"), "model.encoders.signal"),
            (lambda c: c.decoders["image_out"].__setattr__("name", "nope"), "model.decoders.image"),
            (
                lambda c: setattr(
                    c.latent.spaces["z_signal"].regularizer,
                    "strategy",
                    "nope",
                ),
                "model.latent.spaces.z_signal.regularizer.strategy",
            ),
            (
                lambda c: setattr(
                    c.latent.decoders_consume["image_out"],
                    "assembler",
                    "nope",
                ),
                "model.latent.decoders_consume.image_out.assembler",
            ),
        ],
    )
    def test_an_unregistered_name_raises_key_error_with_the_field_path(
        self, mutate: Any, path: str
    ) -> None:
        config = _severalConfig()
        mutate(config)
        with pytest.raises(KeyError, match=f"{path}.*nope"):
            resolveModelConfig(config)

    def test_an_unregistered_head_and_fusion_raise_key_error_with_the_field_path(self) -> None:
        config = _sharedPrivateConfig()
        assert config.latent is not None
        config.latent.spaces["z_signal_private"].fed_by = {"signal": {"head": "nope"}}
        with pytest.raises(KeyError, match=r"z_signal_private.fed_by.signal.head: .*nope"):
            resolveModelConfig(config)

        config = _sharedPrivateConfig()
        assert config.latent is not None
        config.latent.spaces["z_shared"].fusion = FusionConfig("nope")
        with pytest.raises(KeyError, match=r"z_shared.fusion.strategy: .*nope"):
            resolveModelConfig(config)

    def test_a_custom_prefix_starts_the_paths_of_resolution_errors(self) -> None:
        config = _severalConfig(decoders={"image_out": _decoder(latent_dim=31)})
        with pytest.raises(ValueError, match=r"^run.model.decoders.image_out.kwargs.latent_dim"):
            resolveModelConfig(config, prefix="run.model")
