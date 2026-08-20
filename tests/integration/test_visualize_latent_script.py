"""Integration tests for `scripts/visualize_latent.py`.

Loaded via `importlib` since it lives outside the `global_vae` package, under
`scripts/`, matching how it is actually invoked (`python scripts/visualize_latent.py`),
and how `test_evaluate_script.py` already tests its own sibling script. Model/dataloader
factories live in `_script_fixtures.py` (see that file's docstring for why it is named
to avoid pytest's own test discovery); `buildLabeledDataloaderForScript` additionally
carries a "label" batch key for the `--label-key` tests below.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import pytest

from global_vae.training.checkpoint import saveCheckpoint
from global_vae.training.trainer import Trainer
from tests.integration._script_fixtures import (
    buildDataloaderForScript,
    buildModelForScript,
)

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "visualize_latent.py"

_MODEL_FACTORY = "tests.integration._script_fixtures:buildModelForScript"
_DATALOADER_FACTORY = "tests.integration._script_fixtures:buildDataloaderForScript"
_LABELED_DATALOADER_FACTORY = "tests.integration._script_fixtures:buildLabeledDataloaderForScript"


def _loadScriptModule() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "global_vae_visualize_latent_script", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _loadScriptModule()


@pytest.fixture
def checkpoint_with_history(tmp_path: Path) -> Path:
    """A checkpoint saved after a couple of real `Trainer.fit` epochs, so
    `metadata.history` is non-empty and the loss-curve plot has something to draw."""
    model = buildModelForScript()
    trainer = Trainer(model, device="cpu")
    trainer.fit(buildDataloaderForScript(), num_epochs=2)
    path = tmp_path / "model_with_history.pt"
    trainer.saveCheckpoint(path)
    return path


@pytest.fixture
def checkpoint_without_history(tmp_path: Path) -> Path:
    """A bare checkpoint, saved outside of any `Trainer`: `metadata.history == []`."""
    model = buildModelForScript()
    path = tmp_path / "model_bare.pt"
    saveCheckpoint(path, model=model)
    return path


class TestMain:
    def test_runs_end_to_end_and_saves_latent_and_kl_figures(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
            ]
        )
        assert (output_dir / "latent_z_fused.png").exists()
        assert (output_dir / "kl_z_fused.png").exists()

    def test_history_plot_is_saved_when_checkpoint_carries_one(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
            ]
        )
        assert (output_dir / "loss_curves.png").exists()

    def test_history_plot_is_skipped_when_checkpoint_has_none(
        self, script: ModuleType, checkpoint_without_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_without_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
            ]
        )
        assert not (output_dir / "loss_curves.png").exists()
        # the latent/KL figures still must have been produced regardless
        assert (output_dir / "latent_z_fused.png").exists()

    def test_skip_history_flag_skips_it_even_with_a_real_history(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--skip-history",
            ]
        )
        assert not (output_dir / "loss_curves.png").exists()

    def test_skip_kl_flag_skips_only_the_kl_figure(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--skip-kl",
            ]
        )
        assert (output_dir / "latent_z_fused.png").exists()
        assert not (output_dir / "kl_z_fused.png").exists()

    def test_default_output_dir_is_next_to_the_checkpoint(
        self, script: ModuleType, checkpoint_with_history: Path
    ) -> None:
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
            ]
        )
        expected_dir = checkpoint_with_history.parent / "visualizations"
        assert (expected_dir / "latent_z_fused.png").exists()

    def test_label_key_colors_the_scatter_plot_without_raising(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _LABELED_DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--label-key",
                "label",
            ]
        )
        assert (output_dir / "latent_z_fused.png").exists()

    def test_missing_label_key_raises_key_error(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(KeyError, match="does_not_exist"):
            script.main(
                [
                    "--checkpoint",
                    str(checkpoint_with_history),
                    "--model-factory",
                    _MODEL_FACTORY,
                    "--dataloader-factory",
                    _DATALOADER_FACTORY,
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(tmp_path / "viz"),
                    "--label-key",
                    "does_not_exist",
                ]
            )

    def test_use_samples_flag_does_not_raise(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--use-samples",
            ]
        )
        assert (output_dir / "latent_z_fused.png").exists()

    def test_latent_names_filter_restricts_which_spaces_are_plotted(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--latent-names",
                "z_fused",
            ]
        )
        assert (output_dir / "latent_z_fused.png").exists()

    def test_unknown_latent_name_raises_value_error(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="does_not_exist"):
            script.main(
                [
                    "--checkpoint",
                    str(checkpoint_with_history),
                    "--model-factory",
                    _MODEL_FACTORY,
                    "--dataloader-factory",
                    _DATALOADER_FACTORY,
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(tmp_path / "viz"),
                    "--latent-names",
                    "does_not_exist",
                ]
            )

    def test_max_samples_does_not_raise(
        self, script: ModuleType, checkpoint_with_history: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "viz"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_with_history),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--max-samples",
                "5",
            ]
        )
        assert (output_dir / "latent_z_fused.png").exists()

    def test_missing_checkpoint_raises(self, script: ModuleType, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            script.main(
                [
                    "--checkpoint",
                    str(tmp_path / "does_not_exist.pt"),
                    "--model-factory",
                    _MODEL_FACTORY,
                    "--dataloader-factory",
                    _DATALOADER_FACTORY,
                    "--device",
                    "cpu",
                ]
            )

    def test_missing_required_argument_raises_system_exit(self, script: ModuleType) -> None:
        with pytest.raises(SystemExit):
            script.main(["--checkpoint", "somewhere.pt"])
