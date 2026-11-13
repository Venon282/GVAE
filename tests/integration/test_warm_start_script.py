"""The warm start through `scripts/train.py`: the three stages of the recipe, end to end (P6-2).

Each test runs the script in a subprocess, on the structured paired data of
`tests/integration/_warm_start_fixtures.py` (the solo encoders must not collapse, or the guard of
ADR 0052 refuses them, and a model trained on noise collapses):

- stage 1 and stage 2: the two solo experiments, written to `best.pt` with their manifests;
- stage 3: the pair model (`model=signal_image_pair`, `training=several_latents`) with
  `training.warm_start` listing the two files, run with a learning rate of 0, so that after a
  full epoch every parameter of the copied modules is still the solo one. That proves that the
  script applies the entries, applies them before the optimizer is built, and that the run starts
  its own count: the step is that of its four batches, and its manifest says `pair`.

A wrong file makes the script stop before the first step, with the message of the loader.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_LOADER = "tests.integration._warm_start_fixtures:buildStructuredPairDataloaders"
_BUFFER_MARKERS = ("running_mean", "running_var", "num_batches_tracked")
_BATCHES_PER_EPOCH = 4


def _run(
    config_name: str, overrides: list[str], output: Path, *, expect_success: bool = True
) -> str:
    python_path = [str(_REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    completed = subprocess.run(
        [
            sys.executable,
            str(_TRAIN_SCRIPT),
            "--config-name",
            config_name,
            f"data.loader_factory={_LOADER}",
            "data.train_path=unused",
            f"output_dir={output}",
            "training.loggers=[]",
            "training.device=cpu",
            f"hydra.run.dir={output / 'hydra'}",
            *overrides,
        ],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(python_path), "HYDRA_FULL_ERROR": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    text = completed.stdout + completed.stderr
    if expect_success:
        assert completed.returncode == 0, text
        assert "Training complete" in text
    else:
        assert completed.returncode != 0, text
    return text


def _readFile(path: Path) -> dict[str, Any]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint


def _warmStartArgument(entries: list[tuple[Path, str]]) -> str:
    items = ",".join(f'{{checkpoint: "{path}", modules: [{modules}]}}' for path, modules in entries)
    return f"training.warm_start=[{items}]"


@pytest.fixture(scope="module")
def solo_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Stages 1 and 2 through the script: the `best.pt` of each, written once for the module."""
    root = tmp_path_factory.mktemp("stages")
    best = {}
    for modality, extra in (("signal", ["training.reconstruction_reduction=sum"]), ("image", [])):
        output = root / modality
        _run(f"experiment/pair_solo_{modality}", ["training.num_epochs=15", *extra], output)
        best[modality] = output / "checkpoints" / "best.pt"
    return best


_PAIR_CONFIG = "experiment/pair_solo_signal"


def _pairOverrides(extra: list[str]) -> list[str]:
    return [
        "model=signal_image_pair",
        "training=several_latents",
        "training.stage=pair",
        "training.num_epochs=1",
        "training.optimizer.kwargs.lr=0.0",
        *extra,
    ]


@pytest.mark.slow
class TestTheThreeStagesThroughTheTrainScript:
    def test_the_pair_run_starts_from_the_two_solo_files(
        self, solo_runs: dict[str, Path], tmp_path: Path
    ) -> None:
        output = tmp_path / "pair"
        text = _run(
            _PAIR_CONFIG,
            _pairOverrides(
                [
                    _warmStartArgument(
                        [(solo_runs["signal"], "signal"), (solo_runs["image"], "image")]
                    )
                ]
            ),
            output,
        )
        assert "Warm start: copied encoders.signal, decoders.signal" in text
        assert "Warm start: copied encoders.image, decoders.image" in text

        pair = _readFile(output / "checkpoints" / "best.pt")
        # A new run: its own step count, and its own manifest.
        assert pair["global_step"] == _BATCHES_PER_EPOCH
        assert pair["stage_manifest"]["stage"] == "pair"
        assert pair["stage_manifest"]["modalities"] == ["signal", "image"]
        assert pair["stage_manifest"]["latent_spaces"] == {"z_signal": 16, "z_image": 16}

        # A learning rate of 0: every parameter the run trained from is the solo parameter.
        copied = 0
        for modality in ("signal", "image"):
            solo = _readFile(solo_runs[modality])["model_state_dict"]
            for key, value in solo.items():
                if any(marker in key for marker in _BUFFER_MARKERS):
                    continue
                assert torch.equal(pair["model_state_dict"][key], value), key
                copied += 1
        assert copied > 20

    def test_without_warm_start_the_same_run_does_not_start_from_the_solo_files(
        self, solo_runs: dict[str, Path], tmp_path: Path
    ) -> None:
        output = tmp_path / "pair"
        text = _run(_PAIR_CONFIG, _pairOverrides([]), output)
        assert "Warm start" not in text
        pair = _readFile(output / "checkpoints" / "best.pt")["model_state_dict"]
        solo = _readFile(solo_runs["signal"])["model_state_dict"]
        assert not torch.equal(
            pair["encoders.signal.to_mu.weight"], solo["encoders.signal.to_mu.weight"]
        )

    def test_the_wrong_file_stops_the_run_before_the_first_step(
        self, solo_runs: dict[str, Path], tmp_path: Path
    ) -> None:
        output = tmp_path / "pair"
        text = _run(
            _PAIR_CONFIG,
            _pairOverrides([_warmStartArgument([(solo_runs["image"], "signal")])]),
            output,
            expect_success=False,
        )
        assert "training.warm_start[0]" in text
        assert "solo model of ['image']" in text
        assert not (output / "checkpoints").exists()

    def test_a_resumed_pair_run_does_not_read_the_solo_files_again(
        self, solo_runs: dict[str, Path], tmp_path: Path
    ) -> None:
        first = tmp_path / "first"
        _run(
            _PAIR_CONFIG,
            _pairOverrides(
                [
                    _warmStartArgument(
                        [(solo_runs["signal"], "signal"), (solo_runs["image"], "image")]
                    )
                ]
            ),
            first,
        )
        # The solo files are gone: the run that resumes must not need them.
        warm_start = _warmStartArgument(
            [(tmp_path / "gone_signal.pt", "signal"), (tmp_path / "gone_image.pt", "image")]
        )
        text = _run(
            _PAIR_CONFIG,
            [
                *_pairOverrides([warm_start]),
                f"training.resume_from={first / 'checkpoints' / 'best.pt'}",
                "training.num_epochs=2",
            ],
            tmp_path / "second",
        )
        assert "training.warm_start is not applied" in text
