"""`scripts/evaluate.py --alignment-adapter`: spaces of different widths (roadmap P6-8, ADR 0060).

Loaded via `importlib` like `test_latent_alignment_script.py`. The model and the data come from
`_latent_alignment_fixtures.py`: `z_text` has width 5 and holds an embedding of `z_signal`, which
has width 3, and the checkpoint carries the loss-term state of a `latent_alignment` term whose
adapter on `z_text` undoes the embedding, so that through it the two views are one posterior.

What is pinned: that two widths are refused without the flag (with the way out in the message), the
report with it (retrieval, the adapter and the widths it names, the baseline that reads the raw
spaces), the head written out, and each message that refuses a setting.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import torch
from torch import nn

from global_vae.heads.linear import LinearHead
from global_vae.losses.terms.adapters import AdapterSpec
from global_vae.losses.terms.latent_alignment import LatentAlignmentTerm
from global_vae.training.checkpoint import saveCheckpoint
from tests.integration._latent_alignment_fixtures import buildEmbeddedModel, embeddingMatrix

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
_FIXTURES = "tests.integration._latent_alignment_fixtures"
_MODEL = f"{_FIXTURES}:buildEmbeddedModel"
_HELDOUT = f"{_FIXTURES}:buildTripleHeldout"
_TRAIN = f"{_FIXTURES}:buildTripleTrain"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_evaluate_script_p68b", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saveWithTerm(path: Path, term_name: str = "latent_alignment") -> Path:
    """Save the embedded model with a `latent_alignment` term whose adapter undoes the embedding."""
    model = buildEmbeddedModel()
    term = LatentAlignmentTerm(
        [["z_signal", "z_text"]],
        distance="wasserstein2_diag",
        stop_gradient="target",
        adapters={"z_text": 3},
    )
    term.validateModel(model)
    embedding = embeddingMatrix()
    with torch.no_grad():
        head = cast(LinearHead, term.adapters["z_text"])
        head.to_mu.weight.copy_(embedding)
        head.to_mu.bias.zero_()
        head.to_logvar.weight.copy_(embedding)
        head.to_logvar.bias.zero_()
    saveCheckpoint(path, model=model, loss_terms=nn.ModuleDict({term_name: term}))
    return path


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    return _saveWithTerm(tmp_path / "embedded.pt")


def _args(checkpoint: Path, *extra: str, output_dir: Path | None = None) -> list[str]:
    arguments = [
        "--checkpoint",
        str(checkpoint),
        "--model-factory",
        _MODEL,
        "--dataloader-factory",
        _HELDOUT,
        "--device",
        "cpu",
        "--no-figures",
    ]
    if output_dir is not None:
        arguments += ["--output-dir", str(output_dir)]
    return arguments + list(extra)


_PAIR = ("--latent-pair", "z_signal", "z_text")


def _report(directory: Path) -> dict[str, object]:
    loaded: dict[str, object] = json.loads(
        (directory / "latent_alignment.json").read_text(encoding="utf-8")
    )
    return loaded


def _refused(script: ModuleType, capsys: pytest.CaptureFixture[str], arguments: list[str]) -> str:
    with pytest.raises(SystemExit) as raised:
        script.main(arguments)
    assert raised.value.code == 2
    return capsys.readouterr().err


class TestReadThroughTheAdapter:
    def test_without_the_flag_two_widths_are_refused_and_the_way_out_is_named(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(script, capsys, _args(checkpoint, *_PAIR))
        assert "--latent-pair z_signal z_text" in error
        assert "adapters" in error and "D-16" in error

    def test_with_the_flag_the_two_views_are_one_posterior(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                checkpoint,
                *_PAIR,
                "--alignment-adapter",
                "z_text=3",
                "--train-dataloader-factory",
                _TRAIN,
                output_dir=out,
            )
        )
        report = _report(out)
        assert report["widths"] == {"z_signal": 3, "z_text": 5}
        adapters = report["adapters"]
        assert isinstance(adapters, dict)
        assert (adapters["z_text"]["in_dim"], adapters["z_text"]["out_dim"]) == (5, 3)
        retrieval = report["retrieval"]
        assert isinstance(retrieval, dict)
        for direction in ("z_signal->z_text", "z_text->z_signal"):
            assert retrieval[direction]["recall@1"] == pytest.approx(1.0)
        assert (out / "latent_alignment.md").exists()
        assert "adapter" in (out / "latent_alignment.md").read_text(encoding="utf-8").lower()

    def test_the_summary_names_the_adapter(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.main(_args(checkpoint, *_PAIR, "--alignment-adapter", "z_text=3"))
        printed = capsys.readouterr().out
        assert "z_text" in printed and "adapt" in printed.lower()

    def test_the_head_can_be_written_out(self, script: ModuleType, checkpoint: Path) -> None:
        script.main(_args(checkpoint, *_PAIR, "--alignment-adapter", "z_text=linear:3"))

    def test_without_training_pairs_the_report_still_runs_and_says_what_is_skipped(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(_args(checkpoint, *_PAIR, "--alignment-adapter", "z_text=3", output_dir=out))
        assert _report(out)["procrustes"] is None

    def test_a_term_under_another_name_is_read_with_alignment_term(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        saved = _saveWithTerm(tmp_path / "named.pt", term_name="align_images")
        script.main(
            _args(
                saved,
                *_PAIR,
                "--alignment-adapter",
                "z_text=3",
                "--alignment-term",
                "align_images",
            )
        )


class TestRefusals:
    def test_the_adapter_flag_needs_a_pair(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(script, capsys, _args(checkpoint, "--alignment-adapter", "z_text=3"))
        assert "--alignment-adapter needs --latent-pair" in error

    @pytest.mark.parametrize("text", ["z_text", "z_text=", "=3", "z_text=wide", "z_text=0"])
    def test_a_malformed_adapter_is_refused(
        self,
        script: ModuleType,
        checkpoint: Path,
        capsys: pytest.CaptureFixture[str],
        text: str,
    ) -> None:
        error = _refused(script, capsys, _args(checkpoint, *_PAIR, "--alignment-adapter", text))
        assert "--alignment-adapter" in error

    def test_a_floor_that_is_negative_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(
            script,
            capsys,
            _args(
                checkpoint,
                *_PAIR,
                "--alignment-adapter",
                "z_text=3",
                "--min-adapter-spread",
                "-1",
            ),
        )
        assert "min_adapter_spread" in error

    def test_a_space_given_twice_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(
            script,
            capsys,
            _args(
                checkpoint,
                *_PAIR,
                "--alignment-adapter",
                "z_text=3",
                "--alignment-adapter",
                "z_text=4",
            ),
        )
        assert "'z_text' is given twice" in error

    def test_a_width_that_the_checkpoint_does_not_hold_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(
            script, capsys, _args(checkpoint, *_PAIR, "--alignment-adapter", "z_text=4")
        )
        assert "does not fit" in error and "width 4" in error

    def test_a_space_the_model_does_not_have_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(
            script, capsys, _args(checkpoint, *_PAIR, "--alignment-adapter", "z_audio=3")
        )
        assert "z_audio" in error and "latent spaces" in error

    def test_a_space_that_no_pair_reads_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(
            script,
            capsys,
            _args(
                checkpoint,
                *_PAIR,
                "--alignment-adapter",
                "z_text=3",
                "--alignment-adapter",
                "z_image=3",
            ),
        )
        assert "z_image" in error

    def test_a_checkpoint_with_no_such_term_is_refused_with_the_terms_it_has(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(
            script,
            capsys,
            _args(
                checkpoint, *_PAIR, "--alignment-adapter", "z_text=3", "--alignment-term", "other"
            ),
        )
        assert "'other'" in error and "latent_alignment" in error

    def test_a_checkpoint_without_any_term_state_is_refused(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bare = tmp_path / "bare.pt"
        saveCheckpoint(bare, model=buildEmbeddedModel())
        error = _refused(script, capsys, _args(bare, *_PAIR, "--alignment-adapter", "z_text=3"))
        assert "holds no parameters" in error and "none" in error

    def test_an_unregistered_head_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        error = _refused(
            script, capsys, _args(checkpoint, *_PAIR, "--alignment-adapter", "z_text=nonesuch:3")
        )
        assert "nonesuch" in error


def test_the_spec_is_what_the_flag_writes() -> None:
    assert AdapterSpec("linear", 3).toText() == "3"
