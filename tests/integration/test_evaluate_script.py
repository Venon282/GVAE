"""Integration tests for `scripts/evaluate.py`.

Loaded via `importlib` since it lives outside the `global_vae` package, under
`scripts/`, matching how it is actually invoked (`python scripts/evaluate.py`).
Model/dataloader factories live in `_script_fixtures.py` (see that file's docstring
for why it is named to avoid pytest's own test discovery).
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import saveCheckpoint
from tests.integration._script_fixtures import buildModelForScript

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"

_MODEL_FACTORY = "tests.integration._script_fixtures:buildModelForScript"
_DATALOADER_FACTORY = "tests.integration._script_fixtures:buildDataloaderForScript"


def _loadScriptModule() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_evaluate_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _loadScriptModule()


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    model = buildModelForScript()
    path = tmp_path / "model.pt"
    saveCheckpoint(path, model=model)
    return path


class TestImportCallable:
    def test_imports_a_function_by_module_colon_name(self, script: ModuleType) -> None:
        func = script._importCallable(_MODEL_FACTORY)
        model = func()
        assert isinstance(model, GlobalVae)

    def test_missing_colon_raises_value_error(self, script: ModuleType) -> None:
        with pytest.raises(ValueError, match="module.path:function_name"):
            script._importCallable("no_colon_here")

    def test_unknown_module_raises(self, script: ModuleType) -> None:
        with pytest.raises(ModuleNotFoundError):
            script._importCallable("does.not.exist:func")

    def test_unknown_function_raises(self, script: ModuleType) -> None:
        with pytest.raises(AttributeError):
            script._importCallable("tests.integration._script_fixtures:does_not_exist")


class TestMain:
    def test_runs_end_to_end_and_prints_a_summary(
        self,
        script: ModuleType,
        checkpoint_path: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        output_dir = tmp_path / "results"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_path),
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
        captured = capsys.readouterr()
        assert "Evaluation over" in captured.out

        report_path = output_dir / "results.json"
        assert report_path.exists()
        with report_path.open() as f:
            report = json.load(f)
        assert report["num_samples"] > 0

        assert (output_dir / "reconstructions_signal.png").exists()
        assert (output_dir / "latent_z_fused.png").exists()
        assert (output_dir / "kl_z_fused.png").exists()

    def test_no_figures_flag_skips_figure_export(
        self, script: ModuleType, checkpoint_path: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "results_no_figures"
        script.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--no-figures",
            ]
        )
        assert (output_dir / "results.json").exists()
        assert not any(output_dir.glob("*.png"))

    def test_without_output_dir_only_prints(
        self, script: ModuleType, checkpoint_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--model-factory",
                _MODEL_FACTORY,
                "--dataloader-factory",
                _DATALOADER_FACTORY,
                "--device",
                "cpu",
            ]
        )
        captured = capsys.readouterr()
        assert "Evaluation over" in captured.out

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
