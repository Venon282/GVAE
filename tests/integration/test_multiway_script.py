"""`scripts/evaluate.py --latent-set`: across three latent spaces or more (roadmap P6-8, ADR 0059).

Loaded via `importlib` like `test_latent_alignment_script.py`. The model and the data come from
`_latent_alignment_fixtures.py`: a three-modality model whose spaces are one cloud turned three
ways, so that the spaces as they are sit at chance and every post-hoc orthogonal map recovers the
partner.

What is pinned: the report written next to `results.json`, what the training pairs switch on, what
the anchor adds, how several sets are named, and each message that refuses a setting.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from global_vae.training.checkpoint import saveCheckpoint
from tests.integration._latent_alignment_fixtures import buildRotatedTripleModel

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
_FIXTURES = "tests.integration._latent_alignment_fixtures"
_MODEL = f"{_FIXTURES}:buildRotatedTripleModel"
_HELDOUT = f"{_FIXTURES}:buildTripleHeldout"
_TRAIN = f"{_FIXTURES}:buildTripleTrain"
_VIEWS = ("z_signal", "z_image", "z_text")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_evaluate_script_p68", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "triple.pt"
    saveCheckpoint(path, model=buildRotatedTripleModel())
    return path


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


def _read(directory: Path, stem: str = "multiway_alignment") -> dict[str, object]:
    loaded: dict[str, object] = json.loads((directory / f"{stem}.json").read_text(encoding="utf-8"))
    return loaded


class TestTheReport:
    def test_a_set_writes_its_report_next_to_results_json(
        self,
        script: ModuleType,
        checkpoint: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                checkpoint,
                "--latent-set",
                *_VIEWS,
                "--train-dataloader-factory",
                _TRAIN,
                "--latent-anchor",
                "z_signal",
                output_dir=out,
            )
        )
        assert (out / "results.json").exists()
        assert (out / "multiway_alignment.json").exists()
        assert (out / "multiway_alignment.md").exists()
        printed = capsys.readouterr().out
        assert (
            "Multi-way alignment: 3 views (z_signal, z_image, z_text, anchor z_signal)" in printed
        )
        assert "pairwise  maps=3" in printed and "anchor    maps=2" in printed

    def test_the_spaces_as_they_are_sit_at_chance_and_every_map_recovers_the_partner(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                checkpoint,
                "--latent-set",
                *_VIEWS,
                "--train-dataloader-factory",
                _TRAIN,
                "--latent-anchor",
                "z_signal",
                output_dir=out,
            )
        )
        report = _read(out)
        assert report["labels"] == list(_VIEWS)
        assert report["num_heldout"] == 96 and report["num_train"] == 128
        retrieval = report["retrieval"]
        assert isinstance(retrieval, dict)
        assert list(retrieval) == ["identity", "pairwise", "common", "anchor"]
        for pair, row in retrieval["identity"].items():
            assert row["recall@1"] < 0.1, pair
        for baseline in ("pairwise", "common", "anchor"):
            for pair, row in retrieval[baseline].items():
                assert row["recall@1"] == pytest.approx(1.0), (baseline, pair)
                assert row["chance_recall@1"] == pytest.approx(1 / 96)
        assert report["num_maps"] == {"identity": 0, "pairwise": 3, "common": 3, "anchor": 2}

    def test_without_an_anchor_the_anchor_baseline_is_left_out(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                checkpoint,
                "--latent-set",
                *_VIEWS,
                "--train-dataloader-factory",
                _TRAIN,
                output_dir=out,
            )
        )
        retrieval = _read(out)["retrieval"]
        assert isinstance(retrieval, dict)
        assert list(retrieval) == ["identity", "pairwise", "common"]

    def test_without_training_pairs_only_the_identity_is_read_and_the_report_says_so(
        self,
        script: ModuleType,
        checkpoint: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "out"
        script.main(_args(checkpoint, "--latent-set", *_VIEWS, output_dir=out))
        retrieval = _read(out)["retrieval"]
        assert isinstance(retrieval, dict) and list(retrieval) == ["identity"]
        assert "No training data" in capsys.readouterr().out
        assert "no training data" in (out / "multiway_alignment.md").read_text(encoding="utf-8")

    def test_the_retrieval_settings_are_read(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                checkpoint,
                "--latent-set",
                *_VIEWS,
                "--train-dataloader-factory",
                _TRAIN,
                "--retrieval-k",
                "2",
                "7",
                "--retrieval-similarity",
                "l2",
                output_dir=out,
            )
        )
        report = _read(out)
        assert report["ks"] == [2, 7] and report["similarity"] == "l2"

    def test_the_maps_can_be_centered(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                checkpoint,
                "--latent-set",
                *_VIEWS,
                "--train-dataloader-factory",
                _TRAIN,
                "--procrustes-center",
                output_dir=out,
            )
        )
        assert _read(out)["center"] is True

    def test_several_sets_are_named_by_their_views(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                checkpoint,
                "--latent-set",
                "z_signal",
                "z_image",
                "z_text",
                "--latent-set",
                "z_text",
                "z_image",
                "z_signal",
                output_dir=out,
            )
        )
        assert (out / "multiway_alignment_z_signal_z_image_z_text.json").exists()
        assert (out / "multiway_alignment_z_text_z_image_z_signal.md").exists()
        assert not (out / "multiway_alignment.json").exists()

    def test_a_pair_and_a_set_can_be_asked_together(
        self, script: ModuleType, checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--latent-set",
                *_VIEWS,
                "--train-dataloader-factory",
                _TRAIN,
                output_dir=out,
            )
        )
        assert (out / "latent_alignment.json").exists()
        assert (out / "multiway_alignment.json").exists()


class TestWhatIsRefused:
    def _refused(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str], arguments: list[str]
    ) -> str:
        with pytest.raises(SystemExit) as raised:
            script.main(arguments)
        assert raised.value.code == 2
        return capsys.readouterr().err

    def test_a_set_of_two_is_sent_to_latent_pair(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = self._refused(
            script, capsys, _args(checkpoint, "--latent-set", "z_signal", "z_image")
        )
        assert "at least three views" in err and "--latent-pair" in err

    def test_an_anchor_without_a_set_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = self._refused(script, capsys, _args(checkpoint, "--latent-anchor", "z_signal"))
        assert "--latent-anchor needs --latent-set" in err

    def test_an_anchor_that_is_not_a_view_of_the_set_is_refused_and_the_views_are_listed(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = self._refused(
            script,
            capsys,
            _args(checkpoint, "--latent-set", *_VIEWS, "--latent-anchor", "z_nope"),
        )
        assert "--latent-set z_signal z_image z_text" in err
        assert "anchor 'z_nope' is not one of the views" in err

    def test_a_space_the_model_lacks_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = self._refused(
            script, capsys, _args(checkpoint, "--latent-set", "z_signal", "z_image", "z_nope")
        )
        assert "--latent-set" in err and "z_nope" in err

    def test_the_training_flag_alone_is_refused(
        self, script: ModuleType, checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = self._refused(script, capsys, _args(checkpoint, "--train-dataloader-factory", _TRAIN))
        assert "needs --latent-pair or --latent-set" in err
