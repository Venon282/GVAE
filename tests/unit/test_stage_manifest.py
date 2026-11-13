"""Unit tests for `training/stage_manifest.py` (roadmap P6-1, ADR 0051).

The manifest is a record of plain values and the digest a fingerprint of a config, so both are
tested without a model: the shape of `toDict`, the round trip, what `fromDict` tolerates and what it
refuses, and the properties the digest promises (stable, independent of key order, sensitive to a
value, honest about a value it cannot represent). `buildStageManifest` on a real model is tested
in `tests/integration/test_stage_manifest.py`.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from global_vae.training.stage_manifest import StageManifest, computeConfigDigest


@dataclass
class _Inner:
    kernel: int = 5
    channels: list[int] = field(default_factory=lambda: [8, 16])


@dataclass
class _Config:
    name: str = "model"
    width: int = 16
    inner: _Inner = field(default_factory=_Inner)


class TestStageManifest:
    def test_the_defaults_record_nothing(self) -> None:
        manifest = StageManifest()
        assert manifest.stage is None
        assert manifest.modalities == ()
        assert manifest.latent_spaces == {}
        assert manifest.config_digest is None
        assert manifest.module_digests == {}

    def test_to_dict_is_made_of_plain_values(self) -> None:
        manifest = StageManifest(
            stage="solo_signal",
            modalities=("signal",),
            latent_spaces={"z_fused": 16},
            config_digest="ab" * 32,
            module_digests={"encoders.signal": "cd" * 32},
        )
        data = manifest.toDict()
        assert data == {
            "stage": "solo_signal",
            "modalities": ["signal"],
            "latent_spaces": {"z_fused": 16},
            "config_digest": "ab" * 32,
            "module_digests": {"encoders.signal": "cd" * 32},
        }
        assert isinstance(data["modalities"], list)
        assert json.loads(json.dumps(data)) == data

    def test_to_dict_does_not_share_the_manifest_dict(self) -> None:
        manifest = StageManifest(latent_spaces={"z": 4}, module_digests={"encoders.a": "d"})
        manifest.toDict()["latent_spaces"]["z"] = 99
        manifest.toDict()["module_digests"]["encoders.a"] = "other"
        assert manifest.latent_spaces == {"z": 4}
        assert manifest.module_digests == {"encoders.a": "d"}

    def test_the_round_trip_keeps_the_module_digests(self) -> None:
        manifest = StageManifest(
            "solo_signal",
            ("signal",),
            {"z_fused": 16},
            "a" * 64,
            {"encoders.signal": "b" * 64, "decoders.signal": "c" * 64},
        )
        assert StageManifest.fromDict(manifest.toDict()) == manifest
        assert StageManifest.fromDict(json.loads(json.dumps(manifest.toDict()))) == manifest

    def test_a_manifest_written_before_module_digests_existed_reads_with_none(self) -> None:
        data = {
            "stage": "solo_signal",
            "modalities": ["signal"],
            "latent_spaces": {"z_fused": 16},
            "config_digest": "a" * 64,
        }
        assert StageManifest.fromDict(data).module_digests == {}

    def test_the_round_trip_gives_the_same_manifest(self) -> None:
        manifest = StageManifest(
            "pair", ("signal", "image"), {"z_signal": 8, "z_image": 8}, "f" * 64
        )
        assert StageManifest.fromDict(manifest.toDict()) == manifest

    def test_the_round_trip_survives_json(self) -> None:
        manifest = StageManifest("solo_image", ("image",), {"z_fused": 16}, "0" * 64)
        assert StageManifest.fromDict(json.loads(json.dumps(manifest.toDict()))) == manifest

    def test_an_absent_key_keeps_its_default(self) -> None:
        assert StageManifest.fromDict({"stage": "only_a_label"}) == StageManifest(
            stage="only_a_label"
        )
        assert StageManifest.fromDict({}) == StageManifest()

    def test_a_key_this_version_does_not_know_is_ignored(self) -> None:
        data = {**StageManifest(stage="s").toDict(), "added_by_a_later_version": [1, 2, 3]}
        assert StageManifest.fromDict(data) == StageManifest(stage="s")

    @pytest.mark.parametrize("stage", ["", "   ", "\t"])
    def test_an_empty_stage_label_is_refused(self, stage: str) -> None:
        with pytest.raises(ValueError, match="stage label"):
            StageManifest(stage=stage)

    def test_a_stage_label_that_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(ValueError, match="stage label"):
            StageManifest.fromDict({"stage": 3})

    @pytest.mark.parametrize(
        ("data", "field_name"),
        [
            ({"modalities": "signal"}, "modalities"),
            ({"modalities": ["signal", 3]}, "modalities"),
            ({"latent_spaces": [["z", 4]]}, "latent_spaces"),
            ({"latent_spaces": {"z": "wide"}}, "latent_spaces"),
            ({"latent_spaces": {"z": True}}, "latent_spaces"),
            ({"latent_spaces": {3: 4}}, "latent_spaces"),
            ({"config_digest": 12}, "config_digest"),
            ({"module_digests": ["encoders.a"]}, "module_digests"),
            ({"module_digests": {"encoders.a": 3}}, "module_digests"),
            ({"module_digests": {3: "d"}}, "module_digests"),
        ],
    )
    def test_a_field_of_the_wrong_type_is_refused_and_named(
        self, data: dict[str, Any], field_name: str
    ) -> None:
        with pytest.raises(ValueError, match=field_name):
            StageManifest.fromDict(data)

    def test_the_manifest_cannot_be_changed_in_place(self) -> None:
        manifest = StageManifest(stage="s")
        with pytest.raises(AttributeError):
            manifest.stage = "other"  # type: ignore[misc]


class TestComputeConfigDigest:
    def test_it_is_the_sha256_of_the_canonical_json(self) -> None:
        expected = hashlib.sha256(b'{"a":1,"b":[1,2]}').hexdigest()
        assert computeConfigDigest({"b": [1, 2], "a": 1}) == expected

    def test_it_is_64_lowercase_hexadecimal_characters(self) -> None:
        digest = computeConfigDigest({"width": 16})
        assert len(digest) == 64
        assert digest == digest.lower()
        assert int(digest, 16) >= 0

    def test_it_is_the_same_on_every_call(self) -> None:
        assert computeConfigDigest(_Config()) == computeConfigDigest(_Config())

    def test_the_order_of_the_keys_does_not_matter(self) -> None:
        first = {"encoder": {"kernel": 5, "channels": [8, 16]}, "latent": {"dim": 16}}
        second = {"latent": {"dim": 16}, "encoder": {"channels": [8, 16], "kernel": 5}}
        assert computeConfigDigest(first) == computeConfigDigest(second)

    def test_a_dataclass_has_the_digest_of_its_dict(self) -> None:
        config = _Config()
        assert computeConfigDigest(config) == computeConfigDigest(
            {"name": "model", "width": 16, "inner": {"kernel": 5, "channels": [8, 16]}}
        )

    def test_a_changed_value_changes_the_digest(self) -> None:
        base = computeConfigDigest(_Config())
        assert computeConfigDigest(_Config(width=8)) != base
        assert computeConfigDigest(_Config(inner=_Inner(kernel=3))) != base
        assert computeConfigDigest(_Config(inner=_Inner(channels=[8, 32]))) != base

    def test_the_order_of_a_list_matters(self) -> None:
        assert computeConfigDigest({"channels": [8, 16]}) != computeConfigDigest(
            {"channels": [16, 8]}
        )

    def test_a_value_that_json_cannot_represent_is_an_error_that_names_it(self) -> None:
        with pytest.raises(TypeError, match="object"):
            computeConfigDigest({"activation": object()})

    def test_a_set_is_refused_because_its_order_is_not_stable(self) -> None:
        with pytest.raises(TypeError, match="set"):
            computeConfigDigest({"names": {"a", "b"}})

    def test_a_non_ascii_value_is_digested_as_escaped_ascii(self) -> None:
        assert (
            computeConfigDigest({"name": "café"})
            == hashlib.sha256(
                json.dumps({"name": "café"}, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
