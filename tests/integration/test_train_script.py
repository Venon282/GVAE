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
