"""Integration tests for the stage manifest and the solo recipes (roadmap P6-1, ADR 0051).

What is pinned here, with the real models of the two solo experiments and not with dummies:

- the manifest a solo experiment records (modalities, latent space and width, digest of the model
  config) and what changes it;
- that every checkpoint a run writes carries it, whichever way it is written (a direct call, the
  `checkpoint` callback, the `best_checkpoint` callback, a resumed run), in a form the safe loader
  (`weights_only=True`, ADR 0034) reads;
- that a run with no `training.stage`, and a file written by 2.0.0, are unchanged: no new key, and
  the metadata keeps its "nothing recorded" defaults;
- that the two solo recipes of the shipped synthetic pair run end to end through `scripts/train.py`
  (the `slow` class, which runs subprocesses like the other script tests).

The unit tests of the record itself and of the digest are in `tests/unit/test_stage_manifest.py`.
"""

import dataclasses
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import buildTrainerFromConfig, resumeTrainerFromConfig
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import CheckpointMetadata, loadCheckpoint, saveCheckpoint
from global_vae.training.stage_manifest import (
    StageManifest,
    buildStageManifest,
    computeConfigDigest,
)
from global_vae.training.trainer import Trainer
from global_vae.utils.seed import setGlobalSeed

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_EXAMPLE_PATH = _REPO_ROOT / "examples" / "04_paired_loader.py"

SIGNAL_EXPERIMENT = "experiment/pair_solo_signal"
IMAGE_EXPERIMENT = "experiment/pair_solo_image"

# (experiment file, its stage label, the batch key of its modality)
SOLO_STAGES = [
    (SIGNAL_EXPERIMENT, "solo_signal", "signal"),
    (IMAGE_EXPERIMENT, "solo_image", "image"),
]

# The keys a checkpoint of a trainer with no loss term had in 2.0.0, before `callbacks_state`,
# which a trainer with checkpoint callbacks (every solo experiment has two) has always added.
KEYS_OF_2_0_0 = {
    "global_vae_version",
    "model_state_dict",
    "optimizer_state_dict",
    "global_step",
    "start_epoch",
    "history",
    "config",
    "rng_state",
}


def _experiment(name: str, tmp_path: Path, *overrides: str) -> ExperimentConfig:
    """Compose a solo experiment with its run folder in `tmp_path` and no experiment logger."""
    return loadExperimentConfig(
        name,
        overrides=[
            "data.loader_factory=unused:unused",
            "data.train_path=unused",
            f"output_dir={tmp_path / 'run'}",
            "training.device=cpu",
            "training.loggers=[]",
            "training.callbacks.checkpoint.every_n_epochs=1",
            *overrides,
        ],
    )


def _trainer(experiment: ExperimentConfig, seed: int = 0) -> Trainer:
    setGlobalSeed(seed)
    model = buildModelFromConfig(experiment.model)
    return buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)


def _batches(modality: str) -> list[dict[str, torch.Tensor]]:
    """Two small batches of the modality, positive so that any transform of the data is defined."""
    torch.manual_seed(0)
    shape = (4, 256) if modality == "signal" else (4, 64, 64)
    return [{modality: torch.rand(*shape) + 0.1} for _ in range(2)]


def _readFile(path: Path) -> dict[str, Any]:
    """Read a checkpoint in the safe mode, as `loadCheckpoint` does by default (ADR 0034)."""
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint


class TestTheSoloExperimentsRecordTheirStage:
    @pytest.mark.parametrize(("name", "label", "modality"), SOLO_STAGES)
    def test_the_experiment_sets_its_label(self, name: str, label: str, modality: str) -> None:
        experiment = loadExperimentConfig(
            name, overrides=["data.loader_factory=unused:unused", "data.train_path=unused"]
        )
        assert experiment.training.stage == label

    @pytest.mark.parametrize(("name", "label", "modality"), SOLO_STAGES)
    def test_the_manifest_names_the_modality_the_latent_space_and_the_digest(
        self, name: str, label: str, modality: str, tmp_path: Path
    ) -> None:
        experiment = _experiment(name, tmp_path)
        manifest = _trainer(experiment).stage_manifest
        assert manifest is not None
        assert experiment.model.single_latent is not None
        assert manifest.stage == label
        assert manifest.modalities == (modality,)
        assert manifest.latent_spaces == {"z_fused": experiment.model.single_latent.dim}
        assert manifest.config_digest == computeConfigDigest(experiment.model)

    def test_the_two_solo_stages_have_different_digests(self, tmp_path: Path) -> None:
        signal = _trainer(_experiment(SIGNAL_EXPERIMENT, tmp_path)).stage_manifest
        image = _trainer(_experiment(IMAGE_EXPERIMENT, tmp_path)).stage_manifest
        assert signal is not None and image is not None
        assert signal.config_digest != image.config_digest
        assert signal.modalities != image.modalities

    def test_the_digest_covers_the_model_and_not_the_rest_of_the_experiment(
        self, tmp_path: Path
    ) -> None:
        base = _trainer(_experiment(SIGNAL_EXPERIMENT, tmp_path)).stage_manifest
        other_run = _trainer(
            _experiment(
                SIGNAL_EXPERIMENT,
                tmp_path / "other",
                "seed=7",
                "training.optimizer.kwargs.lr=0.01",
                "training.num_epochs=3",
            ),
            seed=7,
        ).stage_manifest
        assert base is not None and other_run is not None
        assert other_run.config_digest == base.config_digest

    def test_a_changed_latent_width_changes_the_digest_and_the_width(self, tmp_path: Path) -> None:
        base = _trainer(_experiment(SIGNAL_EXPERIMENT, tmp_path)).stage_manifest
        narrow = _trainer(
            _experiment(SIGNAL_EXPERIMENT, tmp_path, "model.single_latent.dim=8")
        ).stage_manifest
        assert base is not None and narrow is not None
        assert narrow.config_digest != base.config_digest
        assert (base.latent_spaces, narrow.latent_spaces) == ({"z_fused": 16}, {"z_fused": 8})

    def test_build_stage_manifest_reads_the_model(self, tmp_path: Path) -> None:
        experiment = _experiment(IMAGE_EXPERIMENT, tmp_path)
        model = buildModelFromConfig(experiment.model)
        manifest = buildStageManifest(model, stage="x", model_config=experiment.model)
        assert manifest == StageManifest(
            stage="x",
            modalities=("image",),
            latent_spaces={"z_fused": 16},
            config_digest=computeConfigDigest(experiment.model),
        )

    def test_without_a_model_config_the_digest_is_left_out(self, tmp_path: Path) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path)
        model = buildModelFromConfig(experiment.model)
        trainer = buildTrainerFromConfig(model, experiment.training)
        assert trainer.stage_manifest is not None
        assert trainer.stage_manifest.stage == "solo_signal"
        assert trainer.stage_manifest.config_digest is None

    @pytest.mark.parametrize("label", ["", "   "])
    def test_an_empty_label_is_refused_and_the_field_is_named(
        self, label: str, tmp_path: Path
    ) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path)
        model = buildModelFromConfig(experiment.model)
        training = dataclasses.replace(experiment.training, stage=label)
        with pytest.raises(ValueError, match=r"training\.stage"):
            buildTrainerFromConfig(model, training)

    def test_the_label_can_be_set_from_the_command_line(self, tmp_path: Path) -> None:
        experiment = _experiment("experiment/signal_vae", tmp_path, "training.stage=my_stage")
        assert experiment.training.stage == "my_stage"

    def test_an_experiment_that_sets_no_stage_records_none(self, tmp_path: Path) -> None:
        experiment = _experiment("experiment/signal_vae", tmp_path)
        assert experiment.training.stage is None
        assert _trainer(experiment).stage_manifest is None


class TestTheCheckpointCarriesTheManifest:
    def test_a_saved_file_holds_the_manifest_as_plain_values(self, tmp_path: Path) -> None:
        trainer = _trainer(_experiment(SIGNAL_EXPERIMENT, tmp_path))
        path = tmp_path / "solo.pt"
        trainer.saveCheckpoint(path)
        assert trainer.stage_manifest is not None
        assert _readFile(path)["stage_manifest"] == trainer.stage_manifest.toDict()

    def test_load_checkpoint_returns_the_manifest_in_the_metadata(self, tmp_path: Path) -> None:
        trainer = _trainer(_experiment(IMAGE_EXPERIMENT, tmp_path))
        path = tmp_path / "solo.pt"
        trainer.saveCheckpoint(path)
        metadata = loadCheckpoint(
            path, buildModelFromConfig(_experiment(IMAGE_EXPERIMENT, tmp_path).model)
        )
        assert trainer.stage_manifest is not None
        assert metadata.stage == "solo_image"
        assert metadata.modalities == ["image"]
        assert metadata.latent_spaces == {"z_fused": 16}
        assert metadata.config_digest == trainer.stage_manifest.config_digest

    def test_the_weights_load_as_before(self, tmp_path: Path) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path)
        trainer = _trainer(experiment, seed=1)
        path = tmp_path / "solo.pt"
        trainer.saveCheckpoint(path)
        fresh = buildModelFromConfig(experiment.model)
        loadCheckpoint(path, fresh)
        for name, value in trainer.model.state_dict().items():
            assert torch.equal(value, fresh.state_dict()[name]), name

    def test_a_run_with_no_stage_writes_exactly_the_keys_of_2_0_0(self, tmp_path: Path) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path, "training.stage=null")
        trainer = _trainer(experiment)
        path = tmp_path / "plain.pt"
        trainer.saveCheckpoint(path)
        assert set(_readFile(path)) == KEYS_OF_2_0_0 | {"callbacks_state"}

    def test_a_file_with_no_manifest_gives_the_defaults(self, tmp_path: Path) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path, "training.stage=null")
        path = tmp_path / "plain.pt"
        _trainer(experiment).saveCheckpoint(path)
        metadata = loadCheckpoint(path, buildModelFromConfig(experiment.model))
        assert (metadata.stage, metadata.modalities) == (None, [])
        assert (metadata.latent_spaces, metadata.config_digest) == ({}, None)

    def test_a_file_written_by_2_0_0_still_loads(self, tmp_path: Path) -> None:
        """A checkpoint built by hand with the keys of 2.0.0, in the safe mode."""
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path)
        source = buildModelFromConfig(experiment.model)
        path = tmp_path / "v2_0_0.pt"
        torch.save(
            {
                "global_vae_version": "2.0.0",
                "model_state_dict": source.state_dict(),
                "global_step": 12,
                "start_epoch": 3,
                "history": [{"loss/total": 1.5}],
                "config": {"seed": 0},
            },
            path,
        )
        target = buildModelFromConfig(experiment.model)
        metadata = loadCheckpoint(path, target)
        assert isinstance(metadata, CheckpointMetadata)
        assert (metadata.global_step, metadata.start_epoch) == (12, 3)
        assert metadata.global_vae_version == "2.0.0"
        assert metadata.config == {"seed": 0}
        assert (metadata.stage, metadata.modalities) == (None, [])
        assert (metadata.latent_spaces, metadata.config_digest) == ({}, None)
        for name, value in source.state_dict().items():
            assert torch.equal(value, target.state_dict()[name]), name

    def test_save_checkpoint_without_a_manifest_adds_no_key(self, tmp_path: Path) -> None:
        model = buildModelFromConfig(_experiment(SIGNAL_EXPERIMENT, tmp_path).model)
        path = tmp_path / "plain.pt"
        saveCheckpoint(path, model)
        assert "stage_manifest" not in _readFile(path)

    def test_save_checkpoint_writes_a_manifest_it_is_given(self, tmp_path: Path) -> None:
        model = buildModelFromConfig(_experiment(SIGNAL_EXPERIMENT, tmp_path).model)
        manifest = StageManifest(stage="by_hand", modalities=("signal",), latent_spaces={"z": 3})
        path = tmp_path / "by_hand.pt"
        saveCheckpoint(path, model, stage_manifest=manifest)
        assert _readFile(path)["stage_manifest"] == manifest.toDict()

    def test_a_malformed_manifest_is_an_error_that_names_the_file(self, tmp_path: Path) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path)
        model = buildModelFromConfig(experiment.model)
        path = tmp_path / "bad.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "stage_manifest": {"modalities": "signal"}},
            path,
        )
        with pytest.raises(ValueError, match=r"bad\.pt.*malformed stage_manifest.*modalities"):
            loadCheckpoint(path, model)

    def test_a_manifest_that_is_not_a_mapping_is_an_error_that_names_the_file(
        self, tmp_path: Path
    ) -> None:
        model = buildModelFromConfig(_experiment(SIGNAL_EXPERIMENT, tmp_path).model)
        path = tmp_path / "bad.pt"
        torch.save({"model_state_dict": model.state_dict(), "stage_manifest": "solo"}, path)
        with pytest.raises(ValueError, match=r"bad\.pt.*malformed stage_manifest"):
            loadCheckpoint(path, model)

    def test_a_manifest_with_a_key_of_a_later_version_still_loads(self, tmp_path: Path) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path)
        model = buildModelFromConfig(experiment.model)
        path = tmp_path / "later.pt"
        block = {**StageManifest(stage="later", modalities=("signal",)).toDict(), "extra": [1]}
        torch.save({"model_state_dict": model.state_dict(), "stage_manifest": block}, path)
        assert loadCheckpoint(path, model).stage == "later"


class TestEveryWriterOfACheckpointWritesIt:
    def test_the_checkpoint_callbacks_and_a_resumed_run_write_the_manifest(
        self, tmp_path: Path
    ) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path, "training.num_epochs=2")
        trainer = _trainer(experiment)
        trainer.fit(_batches("signal"), num_epochs=1, val_dataloader=_batches("signal"))
        assert trainer.stage_manifest is not None
        expected = trainer.stage_manifest.toDict()

        checkpoints = tmp_path / "run" / "checkpoints"
        periodic = checkpoints / "checkpoint_epoch_0000.pt"
        best = checkpoints / "best.pt"
        for path in (periodic, best):
            assert path.is_file(), path
            assert _readFile(path)["stage_manifest"] == expected, path

        resumed = _trainer(experiment, seed=5)
        assert (
            resumeTrainerFromConfig(
                resumed, dataclasses.replace(experiment.training, resume_from=str(periodic))
            )
            == 1
        )
        resumed.fit(_batches("signal"), num_epochs=1, val_dataloader=_batches("signal"))
        second = checkpoints / "checkpoint_epoch_0001.pt"
        assert _readFile(second)["stage_manifest"] == expected

    def test_a_trainer_built_by_hand_writes_the_manifest_it_was_given(self, tmp_path: Path) -> None:
        experiment = _experiment(SIGNAL_EXPERIMENT, tmp_path)
        model: GlobalVae = buildModelFromConfig(experiment.model)
        manifest = StageManifest(
            stage="by_hand", modalities=("signal",), latent_spaces={"z_fused": 16}
        )
        trainer = Trainer(model, device="cpu", stage_manifest=manifest)
        path = tmp_path / "by_hand.pt"
        trainer.saveCheckpoint(path)
        assert _readFile(path)["stage_manifest"] == manifest.toDict()

    def test_a_trainer_with_no_manifest_has_none(self, tmp_path: Path) -> None:
        model = buildModelFromConfig(_experiment(SIGNAL_EXPERIMENT, tmp_path).model)
        assert Trainer(model, device="cpu").stage_manifest is None


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic paired dataset (signal and image files, 40 pairs) written once per module."""
    spec = importlib.util.spec_from_file_location("global_vae_example_04_stages", _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    example: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    root = tmp_path_factory.mktemp("paired_dataset")
    example.writeSyntheticPairedDataset(root, num_pairs=40, seed=0)
    return root


# Every test below runs `scripts/train.py` in a subprocess, on the synthetic pair of
# `examples/04_paired_loader.py`.
@pytest.mark.slow
class TestTheSoloRecipesRunThroughTheTrainScript:
    def _run(self, experiment: str, dataset: Path, output: Path) -> str:
        python_path = [str(_REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
        completed = subprocess.run(
            [
                sys.executable,
                str(_TRAIN_SCRIPT),
                "--config-name",
                experiment,
                "data.loader_factory=examples.04_paired_loader:buildPairedDataloaders",
                f"data.train_path={dataset}",
                "data.batch_size=20",
                f"output_dir={output}",
                "training.num_epochs=1",
                "training.loggers=[]",
                "training.device=cpu",
                "training.beta_schedules.z_fused.kwargs.warmup_steps=2",
                f"hydra.run.dir={output / 'hydra'}",
            ],
            cwd=_REPO_ROOT,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(python_path)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        output_text = completed.stdout + completed.stderr
        assert completed.returncode == 0, output_text
        assert "Training complete" in output_text
        return output_text

    @pytest.mark.parametrize(("name", "label", "modality"), SOLO_STAGES)
    def test_a_solo_run_writes_the_manifest(
        self, name: str, label: str, modality: str, dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / label
        self._run(name, dataset, output)
        experiment = _experiment(name, tmp_path)
        best = _readFile(output / "checkpoints" / "best.pt")
        assert best["stage_manifest"] == {
            "stage": label,
            "modalities": [modality],
            "latent_spaces": {"z_fused": 16},
            "config_digest": computeConfigDigest(experiment.model),
        }
        metadata = loadCheckpoint(
            output / "checkpoints" / "best.pt", buildModelFromConfig(experiment.model)
        )
        assert (metadata.stage, metadata.modalities) == (label, [modality])

    def test_the_two_stages_are_told_apart_by_their_checkpoints(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        manifests = {}
        for name, label, _ in SOLO_STAGES:
            self._run(name, dataset, tmp_path / label)
            manifests[label] = _readFile(tmp_path / label / "checkpoints" / "best.pt")[
                "stage_manifest"
            ]
        assert manifests["solo_signal"]["modalities"] != manifests["solo_image"]["modalities"]
        assert manifests["solo_signal"]["config_digest"] != manifests["solo_image"]["config_digest"]
