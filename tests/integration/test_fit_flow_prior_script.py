"""`scripts/fit_flow_prior.py`: fit and write a flow prior from a checkpoint (ADR 0061).

The model and the data come from `_latent_alignment_fixtures.py`. The loader of the unpaired
samples is a mixed stream (signal-only batches, pairs, and one image-only batch), which the script
reads as it is.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

from global_vae.losses.priors import loadFlowPrior
from global_vae.training.checkpoint import saveCheckpoint
from tests.integration._latent_alignment_fixtures import (
    buildAlignedPairModel,
    buildSignalOnlyBatches,
    buildTrainPairs,
)
from tests.unit._tiny_pair_model import LATENT, tinyBatches

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fit_flow_prior.py"
_THIS = "tests.integration.test_fit_flow_prior_script"
_FAST = ["--steps", "30", "--hidden", "8", "--layers", "1", "--device", "cpu"]


def buildMixedStream() -> list[dict[str, torch.Tensor]]:
    """Signal-only batches, pairs and one image-only batch."""
    image_only = [{"image": tinyBatches(1, seed=7)[0]["image"]}]
    return [*buildSignalOnlyBatches(), *buildTrainPairs(), *image_only]


def buildImageOnly() -> list[dict[str, torch.Tensor]]:
    return [{"image": batch["image"]} for batch in tinyBatches(2, seed=3)]


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_fit_flow_prior_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    saveCheckpoint(path, model=buildAlignedPairModel())
    return path


def _args(
    checkpoint: Path, output: Path, *extra: str, loader: str = "buildMixedStream"
) -> list[str]:
    return [
        "--checkpoint",
        str(checkpoint),
        "--model-factory",
        "tests.integration._latent_alignment_fixtures:buildAlignedPairModel",
        "--dataloader-factory",
        f"{_THIS}:{loader}",
        "--space",
        "z_signal",
        "--output",
        str(output),
        *_FAST,
        *extra,
    ]


class TestFit:
    def test_it_writes_a_prior_file_and_a_summary(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "priors" / "z_signal.pt"
        script.main(_args(checkpoint, output))
        prior = loadFlowPrior(output)
        assert prior.dim == LATENT and prior.projection is None
        summary = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        assert summary["space"] == "z_signal" and summary["source_width"] == LATENT
        # Two signal-only batches and the six pairs carry the signal; the image-only one is skipped.
        assert summary["num_points"] == (2 + 6) * 8 and summary["projected"] is False
        assert summary["reference_loss"] == pytest.approx(float(prior.reference_loss))

    def test_dim_gives_a_prior_over_another_width_with_a_projection(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "wide.pt"
        script.main(_args(checkpoint, output, "--dim", "5"))
        prior = loadFlowPrior(output)
        assert prior.dim == 5 and prior.source_dim == LATENT and prior.projection is not None
        assert json.loads(output.with_suffix(".json").read_text())["projected"] is True

    def test_max_samples_limits_the_points(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "few.pt"
        script.main(_args(checkpoint, output, "--max-samples", "20"))
        assert json.loads(output.with_suffix(".json").read_text())["num_points"] == 20

    def test_the_same_seed_writes_the_same_prior(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        first, second = tmp_path / "a.pt", tmp_path / "b.pt"
        script.main(_args(checkpoint, first, "--seed", "3"))
        script.main(_args(checkpoint, second, "--seed", "3"))
        for (name, a), (_, b) in zip(
            loadFlowPrior(first).state_dict().items(),
            loadFlowPrior(second).state_dict().items(),
            strict=True,
        ):
            assert torch.equal(a, b), name


class TestRefusals:
    def _refused(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str], arguments: list[str]
    ) -> str:
        with pytest.raises(SystemExit) as raised:
            script.main(arguments)
        assert raised.value.code == 2
        return capsys.readouterr().err

    def test_a_loader_without_the_modality_of_the_space_is_refused(
        self,
        script: ModuleType,
        checkpoint: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        error = self._refused(
            script, capsys, _args(checkpoint, tmp_path / "p.pt", loader="buildImageOnly")
        )
        assert "nothing to fit a prior to" in error
        assert not (tmp_path / "p.pt").exists()

    def test_a_space_the_model_does_not_have_is_refused(
        self,
        script: ModuleType,
        checkpoint: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        arguments = _args(checkpoint, tmp_path / "p.pt")
        arguments[arguments.index("z_signal")] = "z_audio"
        assert "z_audio" in self._refused(script, capsys, arguments)

    def test_a_setting_out_of_range_is_refused_by_the_fit(
        self,
        script: ModuleType,
        checkpoint: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        error = self._refused(script, capsys, _args(checkpoint, tmp_path / "p.pt", "--lr", "0"))
        assert "lr must be a finite number above 0" in error
