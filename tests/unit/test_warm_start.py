"""Pieces of the warm start that need no trained model (roadmap P6-2, ADR 0052).

The behavior on real solo checkpoints is in `tests/integration/test_warm_start.py`.
"""

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    TrainingConfig,
    WarmStartEntryConfig,
    buildStageManifestFromConfig,
    buildWarmStartEntries,
    computeModuleDigests,
)
from global_vae.training.checkpoint import (
    CheckpointWeights,
    loadCheckpoint,
    readCheckpointWeights,
    saveCheckpoint,
)
from global_vae.training.stage_manifest import StageManifest, buildStageManifest
from global_vae.training.warm_start import WarmStartEntry


class _NotAllowedBySafeMode:
    """An object `torch.load(weights_only=True)` refuses to rebuild."""


def _experiment(*overrides: str) -> Any:
    return loadExperimentConfig(
        "experiment/pair_solo_signal",
        overrides=["data.loader_factory=unused:unused", "data.train_path=unused", *overrides],
    )


class TestWarmStartEntry:
    def test_the_defaults_are_the_strict_ones(self) -> None:
        entry = WarmStartEntry("solo.pt", ["signal"])
        assert entry.modules == ("signal",)
        assert entry.latent_space is None
        assert entry.check_manifest is True
        assert entry.min_active_units == 1

    def test_modules_are_stored_as_a_tuple(self) -> None:
        assert WarmStartEntry("solo.pt", ["encoders.a", "decoders.a"]).modules == (
            "encoders.a",
            "decoders.a",
        )

    @pytest.mark.parametrize("checkpoint", ["", "   "])
    def test_an_empty_checkpoint_is_refused(self, checkpoint: str) -> None:
        with pytest.raises(ValueError, match="checkpoint must be a non-empty path"):
            WarmStartEntry(checkpoint, ("signal",))

    @pytest.mark.parametrize("modules", [(), ("",), ("signal", "  ")])
    def test_no_module_or_an_empty_name_is_refused(self, modules: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="modules must be a non-empty list"):
            WarmStartEntry("solo.pt", modules)

    def test_an_empty_latent_space_is_refused(self) -> None:
        with pytest.raises(ValueError, match="latent_space must be a non-empty name"):
            WarmStartEntry("solo.pt", ("signal",), latent_space=" ")

    @pytest.mark.parametrize("bar", [-1, 1.5, True, "1"])
    def test_a_bar_that_is_not_a_count_is_refused(self, bar: Any) -> None:
        with pytest.raises(ValueError, match="min_active_units"):
            WarmStartEntry("solo.pt", ("signal",), min_active_units=bar)

    def test_zero_switches_the_guard_off_and_is_accepted(self) -> None:
        assert WarmStartEntry("solo.pt", ("signal",), min_active_units=0).min_active_units == 0

    def test_the_entry_cannot_be_changed_in_place(self) -> None:
        entry = WarmStartEntry("solo.pt", ("signal",))
        with pytest.raises(AttributeError):
            entry.check_manifest = False  # type: ignore[misc]


class TestReadCheckpointWeights:
    def _model(self) -> torch.nn.Module:
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(3, 2))

    def test_it_returns_the_weights_and_the_manifest(self, tmp_path: Path) -> None:
        model = self._model()
        manifest = StageManifest("solo", ("signal",), {"z": 2}, "d" * 64, {"encoders.a": "e" * 64})
        path = tmp_path / "solo.pt"
        saveCheckpoint(path, model, stage_manifest=manifest)

        weights = readCheckpointWeights(path)

        assert isinstance(weights, CheckpointWeights)
        assert weights.manifest == manifest
        assert weights.state_dict.keys() == model.state_dict().keys()
        assert all(torch.equal(weights.state_dict[k], v) for k, v in model.state_dict().items())
        assert weights.global_vae_version is not None

    def test_a_file_with_no_manifest_gives_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.pt"
        saveCheckpoint(path, self._model())
        assert readCheckpointWeights(path).manifest is None

    def test_nothing_else_of_the_file_is_returned(self, tmp_path: Path) -> None:
        model = self._model()
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        path = tmp_path / "full.pt"
        saveCheckpoint(path, model, optimizer=optimizer, global_step=40, history=[{"loss": 1.0}])

        weights = readCheckpointWeights(path)

        assert {field for field in vars(weights)} == {
            "state_dict",
            "manifest",
            "global_vae_version",
        }

    def test_a_missing_file_is_a_file_not_found_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nope.pt"):
            readCheckpointWeights(tmp_path / "nope.pt")

    def test_the_safe_mode_refuses_a_file_that_needs_pickle(self, tmp_path: Path) -> None:
        path = tmp_path / "unsafe.pt"
        torch.save({"model_state_dict": {}, "payload": _NotAllowedBySafeMode()}, path)
        with pytest.raises(pickle.UnpicklingError, match="upgradeCheckpoint"):
            readCheckpointWeights(path)

    def test_a_file_with_no_weights_is_refused_by_name(self, tmp_path: Path) -> None:
        path = tmp_path / "other.pt"
        torch.save({"something": 1}, path)
        with pytest.raises(ValueError, match=r"other\.pt.*no model_state_dict"):
            readCheckpointWeights(path)

    def test_a_malformed_manifest_is_refused_by_name(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save(
            {"model_state_dict": {}, "stage_manifest": {"module_digests": ["not", "a", "map"]}},
            path,
        )
        with pytest.raises(ValueError, match=r"bad\.pt"):
            readCheckpointWeights(path)

    def test_load_checkpoint_still_returns_the_module_digests(self, tmp_path: Path) -> None:
        model = self._model()
        manifest = StageManifest(stage="s", module_digests={"decoders.a": "f" * 64})
        path = tmp_path / "solo.pt"
        saveCheckpoint(path, model, stage_manifest=manifest)
        metadata = loadCheckpoint(path, self._model(), restore_rng_state=False)
        assert metadata.module_digests == {"decoders.a": "f" * 64}

    def test_load_checkpoint_gives_no_digests_for_a_file_without_a_manifest(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "bare.pt"
        saveCheckpoint(path, self._model())
        assert loadCheckpoint(path, self._model(), restore_rng_state=False).module_digests == {}


class TestComputeModuleDigests:
    def test_there_is_one_digest_per_encoder_and_decoder(self) -> None:
        digests = computeModuleDigests(_experiment().model)
        assert list(digests) == ["encoders.signal", "decoders.signal"]
        assert all(len(digest) == 64 for digest in digests.values())
        assert len(set(digests.values())) == 2

    def test_the_pair_model_gives_the_digests_of_the_solo_models(self) -> None:
        pair = computeModuleDigests(_experiment("model=signal_image_pair").model)
        signal = computeModuleDigests(_experiment().model)
        image = computeModuleDigests(
            loadExperimentConfig(
                "experiment/pair_solo_image",
                overrides=["data.loader_factory=unused:unused", "data.train_path=unused"],
            ).model
        )
        assert list(pair) == [
            "encoders.signal",
            "encoders.image",
            "decoders.signal",
            "decoders.image",
        ]
        assert pair["encoders.signal"] == signal["encoders.signal"]
        assert pair["decoders.signal"] == signal["decoders.signal"]
        assert pair["encoders.image"] == image["encoders.image"]
        assert pair["decoders.image"] == image["decoders.image"]

    @pytest.mark.parametrize(
        "override",
        [
            "model.encoders.signal.kwargs.kernel_sizes=3",
            "+model.encoders.signal.kwargs.global_pool=max",
        ],
    )
    def test_a_changed_kwarg_changes_the_digest_of_that_module_only(self, override: str) -> None:
        base = computeModuleDigests(_experiment("model=signal_image_pair").model)
        changed = computeModuleDigests(_experiment("model=signal_image_pair", override).model)
        assert changed["encoders.signal"] != base["encoders.signal"]
        assert {k: v for k, v in changed.items() if k != "encoders.signal"} == {
            k: v for k, v in base.items() if k != "encoders.signal"
        }

    def test_the_width_of_the_space_a_module_touches_is_part_of_its_digest(self) -> None:
        base = computeModuleDigests(_experiment("model=signal_image_pair").model)
        narrow = computeModuleDigests(
            _experiment("model=signal_image_pair", "model.latent.spaces.z_signal.dim=8").model
        )
        assert narrow["encoders.signal"] != base["encoders.signal"]
        assert narrow["decoders.signal"] != base["decoders.signal"]
        assert narrow["encoders.image"] == base["encoders.image"]

    def test_the_digest_does_not_depend_on_the_seed_or_the_training_config(self) -> None:
        one = computeModuleDigests(_experiment().model)
        other = computeModuleDigests(_experiment("seed=5", "training.beta=0.5").model)
        assert one == other


class TestTheStageManifestOfARun:
    def test_it_records_the_module_digests_of_the_model_config(self) -> None:
        experiment = _experiment()
        model = buildModelFromConfig(experiment.model)
        manifest = buildStageManifestFromConfig(model, experiment.training, experiment.model)
        assert manifest is not None
        assert manifest.module_digests == computeModuleDigests(experiment.model)

    def test_a_plain_mapping_cannot_be_resolved_so_it_records_none(self) -> None:
        experiment = _experiment()
        model = buildModelFromConfig(experiment.model)
        manifest = buildStageManifestFromConfig(
            model, experiment.training, {"name": "plain", "latent_dim": 4}
        )
        assert manifest is not None
        assert manifest.config_digest is not None
        assert manifest.module_digests == {}

    def test_build_stage_manifest_takes_the_digests_as_given(self) -> None:
        experiment = _experiment()
        model = buildModelFromConfig(experiment.model)
        manifest = buildStageManifest(model, "s", module_digests={"encoders.signal": "x"})
        assert manifest.module_digests == {"encoders.signal": "x"}
        assert buildStageManifest(model, "s").module_digests == {}


class TestBuildWarmStartEntries:
    def test_the_default_is_no_entry(self) -> None:
        assert TrainingConfig().warm_start == []
        assert buildWarmStartEntries(TrainingConfig()) == []

    def test_the_fields_are_carried_over_in_order(self) -> None:
        config = TrainingConfig(
            warm_start=[
                WarmStartEntryConfig(checkpoint="a.pt", modules=["signal"]),
                WarmStartEntryConfig(
                    checkpoint="b.pt",
                    modules=["encoders.image"],
                    latent_space="z_image",
                    check_manifest=False,
                    min_active_units=3,
                ),
            ]
        )
        assert buildWarmStartEntries(config) == [
            WarmStartEntry("a.pt", ("signal",)),
            WarmStartEntry("b.pt", ("encoders.image",), "z_image", False, 3),
        ]

    def test_the_place_of_a_bad_entry_is_named(self) -> None:
        config = TrainingConfig(
            warm_start=[
                WarmStartEntryConfig(checkpoint="a.pt", modules=["signal"]),
                WarmStartEntryConfig(checkpoint="b.pt", modules=["signal"], min_active_units=-2),
            ]
        )
        with pytest.raises(ValueError, match=r"training\.warm_start\[1\]: min_active_units"):
            buildWarmStartEntries(config)

    def test_hydra_builds_the_entries_from_the_command_line(self) -> None:
        experiment = _experiment(
            "training.warm_start=[{checkpoint: a.pt, modules: [signal]},"
            "{checkpoint: b.pt, modules: [image], min_active_units: 2}]"
        )
        assert [entry.checkpoint for entry in experiment.training.warm_start] == ["a.pt", "b.pt"]
        assert experiment.training.warm_start[1].min_active_units == 2
        assert experiment.training.warm_start[0].check_manifest is True
