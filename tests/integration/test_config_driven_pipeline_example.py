"""Integration test for `examples/02_config_driven_pipeline.py`'s CLI (spec §9, §10
"Config management").

Runs the script as a real subprocess, mirroring `test_train_script.py`'s own reasoning:
even though this script is not `@hydra.main`-decorated (it calls
`global_vae.config.experiment.loadExperimentConfig` directly, so several variants can
be composed and trained back to back within one process, unlike `scripts/train.py`),
it does a bare `from _synthetic_signal_data import ...` sibling import that only
resolves when `examples/` itself is on `sys.path`, which is exactly what happens when
the interpreter runs `python examples/02_config_driven_pipeline.py` directly (or, as
here, via `subprocess.run([sys.executable, str(script_path), ...])`) but not when the
module is loaded in-process via `importlib.util.spec_from_file_location` the way
`test_evaluate_script.py` loads `scripts/evaluate.py` (which has no such sibling
import to resolve, so that lighter-weight approach is safe there but not here).

Every run below passes `--num-epochs` and `--variants baseline` (a single variant) to
keep the subprocess fast: this test exercises the CLI's own config-selection wiring
(which flags reach which composed values), not the full two-variant, hundred-epoch
demonstration a real, unhurried run of this script performs.
"""

import subprocess
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "examples" / "02_config_driven_pipeline.py"


def _runExample(
    tmp_path: Path, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `examples/02_config_driven_pipeline.py` as a subprocess with a small, fast,
    isolated configuration.

    Args:
        tmp_path: pytest's per-test temporary directory; this run's `--output-root` is
            confined to it.
        extra_args: Additional CLI arguments appended after the base, fast-and-isolated
            set every call needs (`--variants baseline --num-epochs 1`).

    Returns:
        The completed subprocess.
    """
    args = [
        sys.executable,
        str(_SCRIPT_PATH),
        "--variants",
        "baseline",
        "--num-epochs",
        "1",
        "--output-root",
        str(tmp_path),
        *(extra_args or []),
    ]
    return subprocess.run(
        args, cwd=_SCRIPT_PATH.parents[1], capture_output=True, text=True, timeout=180
    )


class TestDefaultConfigSelection:
    def test_runs_successfully_with_the_default_resnet_config(self, tmp_path: Path) -> None:
        completed = _runExample(tmp_path)
        assert completed.returncode == 0, completed.stderr
        combined = completed.stdout + completed.stderr
        assert "experiment/signal_resnet_vae" in combined
        assert "model=global_vae_signal_resnet_single_latent" in combined

    def test_writes_the_expected_output_files(self, tmp_path: Path) -> None:
        completed = _runExample(tmp_path)
        assert completed.returncode == 0, completed.stderr
        variant_dir = tmp_path / "baseline"
        assert (variant_dir / "checkpoints" / "best.pt").exists()
        assert (variant_dir / "evaluation.json").exists()
        assert (variant_dir / "latent_space.png").exists()
        assert (variant_dir / "reconstructions.png").exists()
        assert (variant_dir / "loss_curves.png").exists()


class TestConfigSelectionFlags:
    def test_model_config_flag_switches_to_the_plain_conv_model(self, tmp_path: Path) -> None:
        completed = _runExample(tmp_path, ["--model-config", "signal_single_latent"])
        assert completed.returncode == 0, completed.stderr
        combined = completed.stdout + completed.stderr
        assert "model=global_vae_signal_single_latent" in combined

    def test_experiment_config_flag_switches_the_whole_experiment_file(
        self, tmp_path: Path
    ) -> None:
        completed = _runExample(tmp_path, ["--experiment-config", "experiment/signal_vae"])
        assert completed.returncode == 0, completed.stderr
        combined = completed.stdout + completed.stderr
        assert "experiment/signal_vae" in combined
        # signal_vae.yaml's own default model, not this script's own resnet default
        assert "model=global_vae_signal_single_latent" in combined

    def test_override_flag_reaches_the_composed_config(self, tmp_path: Path) -> None:
        completed = _runExample(
            tmp_path,
            [
                "--override",
                "training.optimizer.kwargs.lr=0.01",
                "--override",
                "model.single_latent.dim=32",
            ],
        )
        assert completed.returncode == 0, completed.stderr
        combined = completed.stdout + completed.stderr
        assert "lr=0.01" in combined

    def test_invalid_variant_name_fails_with_a_clear_argparse_error(self, tmp_path: Path) -> None:
        completed = _runExample(tmp_path, [])
        # sanity: the base call above must succeed; the actual assertion is the next call
        assert completed.returncode == 0, completed.stderr

        args = [
            sys.executable,
            str(_SCRIPT_PATH),
            "--variants",
            "does_not_exist",
            "--output-root",
            str(tmp_path),
        ]
        invalid = subprocess.run(
            args, cwd=_SCRIPT_PATH.parents[1], capture_output=True, text=True, timeout=60
        )
        assert invalid.returncode != 0
        assert "invalid choice" in invalid.stderr

    def test_missing_required_argument_is_fine_since_everything_has_a_default(
        self, tmp_path: Path
    ) -> None:
        """Unlike scripts/evaluate.py or scripts/visualize_latent.py, this script's CLI
        has no required flags at all: every argument has a sensible default (spec's own
        "use this config by default" requirement)."""
        completed = subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), "--help"],
            cwd=_SCRIPT_PATH.parents[1],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0
        assert "--model-config" in completed.stdout
        assert "--variants" in completed.stdout


class TestVariantSelection:
    def test_only_the_requested_variant_directory_is_written(self, tmp_path: Path) -> None:
        completed = _runExample(tmp_path)
        assert completed.returncode == 0, completed.stderr
        assert (tmp_path / "baseline").exists()
        assert not (tmp_path / "tuned").exists()

    def test_tuned_variant_can_be_selected_on_its_own(self, tmp_path: Path) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--variants",
                "tuned",
                "--num-epochs",
                "1",
                "--output-root",
                str(tmp_path),
            ],
            cwd=_SCRIPT_PATH.parents[1],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr
        assert (tmp_path / "tuned").exists()
        assert not (tmp_path / "baseline").exists()


# Deliberately not covered here: running both variants back to back at the script's
# own default --num-epochs. Every test above exercises "baseline" and "tuned"
# individually, which already covers the CLI's own config-selection wiring; running
# both together adds wall-clock time without adding further coverage.
