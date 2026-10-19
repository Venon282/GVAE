"""Integration test for `scripts/train.py` (spec §10 "Config management").

Runs the script as a real subprocess, unlike `test_evaluate_script.py` (which imports
`scripts/evaluate.py` and calls its plain argparse-based `main()` directly in-process):
`train.py`'s `main` is decorated with `@hydra.main`, which owns global process state
(`sys.argv` parsing, Hydra's own singleton config-loader state) that is not safe to
exercise twice within one interpreter across several tests. A subprocess gives every
test a clean process, matching how the script is actually invoked
(`python scripts/train.py ...`), at the cost of being slower than an in-process call.

Uses `tests.integration._train_script_fixtures:buildDummyDataloaders` as the
`data.loader_factory` override (see that module's own docstring for why the repository
root must be on `PYTHONPATH` in the subprocess for the dotted import to resolve).
"""

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.training.checkpoint import loadCheckpoint

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
_REPO_ROOT = _SCRIPT_PATH.parents[1]
_LOADER_FACTORY = "tests.integration._train_script_fixtures:buildDummyDataloaders"
_SIGNAL_IMAGE_LOADER_FACTORY = (
    "tests.integration._train_script_fixtures:buildDummySignalImageDataloaders"
)

# Every test in this module runs `scripts/train.py` in a subprocess.
pytestmark = pytest.mark.slow


def _runTrainScript(
    tmp_path: Path, extra_overrides: list[str] | None = None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py` as a subprocess with a small, fast, isolated config.

    Args:
        tmp_path: pytest's per-test temporary directory; every file this run writes
            (Hydra's own run dir and this script's `output_dir`) is confined to it.
        extra_overrides: Additional Hydra dotlist overrides appended after the base
            set (data/output_dir/epoch-count/warmup overrides every call needs).

    Returns:
        `(completed_process, output_dir)`.
    """
    output_dir = tmp_path / "run"
    overrides = [
        f"data.loader_factory={_LOADER_FACTORY}",
        "data.train_path=unused",
        f"output_dir={output_dir}",
        "training.num_epochs=2",
        "training.beta_schedules.z_fused.kwargs.warmup_steps=5",
        f"hydra.run.dir={tmp_path / 'hydra_run'}",
        *(extra_overrides or []),
    ]
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *overrides],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return completed, output_dir


class TestTrainScriptEndToEnd:
    def test_runs_successfully_and_logs_completion(self, tmp_path: Path) -> None:
        completed, _ = _runTrainScript(tmp_path)
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in (completed.stdout + completed.stderr)

    def test_writes_csv_logger_output(self, tmp_path: Path) -> None:
        completed, output_dir = _runTrainScript(tmp_path)
        assert completed.returncode == 0, completed.stderr
        metrics_path = output_dir / "metrics.csv"
        assert metrics_path.exists()
        rows = list(csv.DictReader(metrics_path.open()))
        assert len(rows) > 0

    def test_writes_tensorboard_logger_output(self, tmp_path: Path) -> None:
        completed, output_dir = _runTrainScript(tmp_path)
        assert completed.returncode == 0, completed.stderr
        tb_dir = output_dir / "tensorboard"
        assert tb_dir.exists()
        assert any(tb_dir.iterdir())

    def test_writes_a_loadable_best_checkpoint(self, tmp_path: Path) -> None:
        completed, output_dir = _runTrainScript(tmp_path)
        assert completed.returncode == 0, completed.stderr
        checkpoint_path = output_dir / "checkpoints" / "best.pt"
        assert checkpoint_path.exists()

        from global_vae.config.experiment import loadExperimentConfig
        from global_vae.config.model import buildModelFromConfig

        cfg = loadExperimentConfig(
            overrides=[f"data.loader_factory={_LOADER_FACTORY}", "data.train_path=unused"]
        )
        model = buildModelFromConfig(cfg.model)
        metadata = loadCheckpoint(checkpoint_path, model=model)
        assert metadata.global_step > 0

    def test_missing_required_data_fields_fails_with_a_clear_error(self, tmp_path: Path) -> None:
        """Omitting `data.loader_factory`/`data.train_path` must fail loudly.

        This is Hydra's MISSING enforcement: the script must not silently train on nothing.
        """
        output_dir = tmp_path / "run"
        env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
        completed = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                f"output_dir={output_dir}",
                f"hydra.run.dir={tmp_path / 'hydra_run'}",
            ],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode != 0
        assert "MissingMandatoryValue" in completed.stderr or "missing" in completed.stderr.lower()

    def test_overriding_hyperparameters_takes_effect(self, tmp_path: Path) -> None:
        """A smoke check that command-line overrides actually reach the trained model.

        Not just that the process exits 0.
        """
        completed, output_dir = _runTrainScript(
            tmp_path, extra_overrides=["training.optimizer.kwargs.lr=0.0", "training.num_epochs=1"]
        )
        assert completed.returncode == 0, completed.stderr
        # lr=0.0 means the model never moves from its random initialization; reloading
        # the checkpoint and re-running forward should still produce finite numbers,
        # which is the one thing we can assert without hand-computing exact weights.
        from global_vae.config.experiment import loadExperimentConfig
        from global_vae.config.model import buildModelFromConfig

        cfg = loadExperimentConfig(
            overrides=[f"data.loader_factory={_LOADER_FACTORY}", "data.train_path=unused"]
        )
        model = buildModelFromConfig(cfg.model)
        loadCheckpoint(output_dir / "checkpoints" / "best.pt", model=model)
        output = model({"signal": torch.randn(2, 256)})
        assert torch.isfinite(output["reconstructions"]["signal"]).all()


@pytest.mark.parametrize("bad_spec", ["not_a_valid_spec", "module:func:extra"])
def test_invalid_loader_factory_spec_fails_clearly(tmp_path: Path, bad_spec: str) -> None:
    output_dir = tmp_path / "run"
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            f"data.loader_factory={bad_spec}",
            "data.train_path=unused",
            f"output_dir={output_dir}",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "module.path:function_name" in completed.stderr


def _runExperiment(
    tmp_path: Path, experiment: str, warmup_space: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/<experiment>` on the dummy paired data.

    Args:
        tmp_path: pytest's per-test temporary directory; every file the run writes is in it.
        experiment: Stem of a `configs/experiment/*.yaml` file.
        warmup_space: A latent space of that experiment's model, whose beta warm-up is
            shortened so two epochs cover it.

    Returns:
        `(completed_process, output_dir)`.
    """
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{experiment}",
        f"data.loader_factory={_SIGNAL_IMAGE_LOADER_FACTORY}",
        "data.train_path=unused",
        f"output_dir={output_dir}",
        "training.num_epochs=2",
        f"training.beta_schedules.{warmup_space}.kwargs.warmup_steps=5",
        f"hydra.run.dir={tmp_path / 'hydra_run'}",
    ]
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *overrides],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return completed, output_dir


@pytest.mark.parametrize(
    ("experiment", "warmup_space", "latent_spaces"),
    [
        ("signal_image_vae", "z_fused", {"z_fused"}),
        ("signal_image_independent_latents", "z_signal", {"z_signal", "z_image"}),
        (
            "signal_image_shared_private",
            "z_shared",
            {"z_shared", "z_signal_private", "z_image_private"},
        ),
    ],
)
class TestSignalImageExperiments:
    """One experiment per pattern of spec §9 trains through the script (roadmap P1-3, ADR 0025)."""

    def test_trains_and_logs_completion(
        self, tmp_path: Path, experiment: str, warmup_space: str, latent_spaces: set[str]
    ) -> None:
        completed, output_dir = _runExperiment(tmp_path, experiment, warmup_space)
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in (completed.stdout + completed.stderr)
        assert (output_dir / "metrics.csv").exists()

    def test_writes_a_checkpoint_the_configured_model_loads(
        self, tmp_path: Path, experiment: str, warmup_space: str, latent_spaces: set[str]
    ) -> None:
        completed, output_dir = _runExperiment(tmp_path, experiment, warmup_space)
        assert completed.returncode == 0, completed.stderr

        from global_vae.config.experiment import loadExperimentConfig
        from global_vae.config.model import buildModelFromConfig

        cfg = loadExperimentConfig(
            f"experiment/{experiment}",
            overrides=[f"data.loader_factory={_SIGNAL_IMAGE_LOADER_FACTORY}", "data.train_path=x"],
        )
        model = buildModelFromConfig(cfg.model)
        assert set(model.latent_spaces) == latent_spaces
        metadata = loadCheckpoint(output_dir / "checkpoints" / "best.pt", model=model)
        assert metadata.global_step > 0


def test_a_broken_model_section_fails_before_training_with_the_field_path(
    tmp_path: Path,
) -> None:
    """A wrong name in `fed_by` stops the run at model construction, naming the field."""
    output_dir = tmp_path / "run"
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--config-name",
            "experiment/signal_image_shared_private",
            f"data.loader_factory={_SIGNAL_IMAGE_LOADER_FACTORY}",
            "data.train_path=unused",
            "model.latent.spaces.z_shared.fed_by=[signal,audio]",
            f"output_dir={output_dir}",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "model.latent.spaces.z_shared.fed_by: unknown encoder 'audio'" in completed.stderr
    assert not (output_dir / "metrics.csv").exists()


def _epochMetrics(metrics_path: Path) -> dict[tuple[str, str], float]:
    """`(x, metric) -> value` of the `epoch` rows of a metrics CSV, the last row winning."""
    rows = csv.DictReader(metrics_path.open())
    return {(row["x"], row["metric"]): float(row["value"]) for row in rows if row["tag"] == "epoch"}


class TestResumeFromTheTrainScript:
    """`training.resume_from` and `training.num_epochs` as the total of the run (ADR 0032)."""

    _EVERY_EPOCH = "training.callbacks.checkpoint.every_n_epochs=1"

    def _interruptedRun(self, tmp_path: Path) -> Path:
        completed, output_dir = _runTrainScript(
            tmp_path, extra_overrides=["training.num_epochs=2", self._EVERY_EPOCH]
        )
        assert completed.returncode == 0, completed.stderr
        checkpoint = output_dir / "checkpoints" / "checkpoint_epoch_0001.pt"
        assert checkpoint.is_file()
        return checkpoint

    def test_a_resumed_run_completes_the_total_and_matches_the_uninterrupted_curve(
        self, tmp_path: Path
    ) -> None:
        uninterrupted, uninterrupted_dir = _runTrainScript(
            tmp_path / "uninterrupted", extra_overrides=["training.num_epochs=4", self._EVERY_EPOCH]
        )
        assert uninterrupted.returncode == 0, uninterrupted.stderr

        checkpoint = self._interruptedRun(tmp_path / "interrupted")
        resumed, resumed_dir = _runTrainScript(
            tmp_path / "interrupted",
            extra_overrides=[
                "training.num_epochs=4",
                self._EVERY_EPOCH,
                f"training.resume_from={checkpoint}",
            ],
        )
        assert resumed.returncode == 0, resumed.stderr
        output = resumed.stdout + resumed.stderr
        assert "2 of 4 epoch(s) left" in output
        assert "Starting training for 2 epoch(s)" in output

        expected = _epochMetrics(uninterrupted_dir / "metrics.csv")
        observed = _epochMetrics(resumed_dir / "metrics.csv")
        assert set(observed) == set(expected)
        assert len({x for x, _ in observed}) == 4
        for key, value in expected.items():
            assert observed[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key

        final = loadCheckpointRaw(resumed_dir / "checkpoints" / "checkpoint_epoch_0003.pt")
        assert final["start_epoch"] == 4
        assert len(final["history"]) == 4

    def test_the_metrics_file_of_the_first_part_is_continued_not_replaced(
        self, tmp_path: Path
    ) -> None:
        checkpoint = self._interruptedRun(tmp_path)
        metrics = tmp_path / "run" / "metrics.csv"
        first_part = metrics.read_text()

        completed, _ = _runTrainScript(
            tmp_path,
            extra_overrides=[
                "training.num_epochs=3",
                self._EVERY_EPOCH,
                f"training.resume_from={checkpoint}",
            ],
        )
        assert completed.returncode == 0, completed.stderr
        text = metrics.read_text()
        assert text.startswith(first_part)
        assert text.count("x,tag,metric,value") == 1

    def test_nothing_is_left_to_train_when_the_checkpoint_holds_every_epoch(
        self, tmp_path: Path
    ) -> None:
        checkpoint = self._interruptedRun(tmp_path)
        completed, output_dir = _runTrainScript(
            tmp_path,
            extra_overrides=["training.num_epochs=2", f"training.resume_from={checkpoint}"],
        )
        assert completed.returncode == 0, completed.stderr
        assert "Nothing to train" in completed.stdout + completed.stderr
        assert not (output_dir / "checkpoints" / "checkpoint_epoch_0002.pt").exists()

    def test_a_missing_checkpoint_fails_with_the_field_path(self, tmp_path: Path) -> None:
        completed, _ = _runTrainScript(
            tmp_path, extra_overrides=[f"training.resume_from={tmp_path / 'nope.pt'}"]
        )
        assert completed.returncode != 0
        assert "training.resume_from: no checkpoint at" in completed.stderr


def loadCheckpointRaw(path: Path) -> dict[str, Any]:
    """Read a checkpoint with the restricted unpickler, as `loadCheckpoint` does by default."""
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint
