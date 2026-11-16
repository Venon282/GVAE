"""`scripts/evaluate.py --latent-pair`: across two separate latent spaces (roadmap P6-6, ADR 0056).

Loaded via `importlib` like `test_evaluate_script.py`. The models and the data come from
`_latent_alignment_fixtures.py`: a tiny pair model whose two encoders are the same function (so the
Procrustes map recovers the partner), the same model with both encoders collapsed, and a shared
latent space read from one modality at a time.

What is pinned: the flags that name a pair, the report written next to `results.json`, the cross
and gap rows of the metrics table of P3-6, the Procrustes baseline and the pair-integrity check
that the training pairs switch on, the collapse guard (with the solo stage's active units, and on a
deliberately collapsed model), and each message that refuses a setting.
"""

import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType

import pytest

from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import saveCheckpoint
from tests.integration._latent_alignment_fixtures import (
    buildAlignedPairModel,
    buildCollapsedPairModel,
)
from tests.integration._script_fixtures import buildTwoModalityModelForScript

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
_FIXTURES = "tests.integration._latent_alignment_fixtures"
_MODEL = f"{_FIXTURES}:buildAlignedPairModel"
_HELDOUT = f"{_FIXTURES}:buildHeldoutPairs"
_TRAIN = f"{_FIXTURES}:buildTrainPairs"
_SHUFFLED_TRAIN = f"{_FIXTURES}:buildShuffledTrainPairs"
_SIGNAL_ONLY = f"{_FIXTURES}:buildSignalOnlyBatches"
_SHARED_MODEL = "tests.integration._script_fixtures:buildTwoModalityModelForScript"
_SHARED_DATA = "tests.integration._script_fixtures:buildTwoModalityDataloaderForScript"


def _loadScriptModule() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_evaluate_script_p66", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _loadScriptModule()


def _checkpoint(tmp_path: Path, model: GlobalVae, name: str = "model.pt") -> Path:
    path = tmp_path / name
    saveCheckpoint(path, model=model)
    return path


@pytest.fixture
def aligned_checkpoint(tmp_path: Path) -> Path:
    return _checkpoint(tmp_path, buildAlignedPairModel())


@pytest.fixture
def collapsed_checkpoint(tmp_path: Path) -> Path:
    return _checkpoint(tmp_path, buildCollapsedPairModel(), "collapsed.pt")


def _args(
    checkpoint: Path,
    *extra: str,
    output_dir: Path | None = None,
    model: str = _MODEL,
    data: str = _HELDOUT,
) -> list[str]:
    arguments = [
        "--checkpoint",
        str(checkpoint),
        "--model-factory",
        model,
        "--dataloader-factory",
        data,
        "--device",
        "cpu",
        "--no-figures",
    ]
    if output_dir is not None:
        arguments += ["--output-dir", str(output_dir)]
    return arguments + list(extra)


def _report(directory: Path, stem: str = "latent_alignment") -> dict[str, object]:
    loaded: dict[str, object] = json.loads((directory / f"{stem}.json").read_text(encoding="utf-8"))
    return loaded


class TestTheReportNextToResults:
    def test_a_pair_writes_its_report_next_to_results_json(
        self,
        script: ModuleType,
        aligned_checkpoint: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--train-dataloader-factory",
                _TRAIN,
                output_dir=out,
            )
        )
        assert (out / "results.json").exists()
        assert (out / "latent_alignment.json").exists()
        assert (out / "latent_alignment.md").exists()
        printed = capsys.readouterr().out
        assert "Latent alignment: z_signal" in printed
        assert "retrieval z_signal->z_image" in printed
        assert "pair integrity: ok" in printed

    def test_the_json_report_has_every_part_of_the_roadmap_item(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--train-dataloader-factory",
                _TRAIN,
                output_dir=out,
            )
        )
        report = _report(out)
        assert report["labels"] == ["z_signal", "z_image"]
        assert report["num_heldout"] == 24 and report["num_train"] == 48
        retrieval = report["retrieval"]
        assert isinstance(retrieval, dict)
        assert set(retrieval) == {"z_signal->z_image", "z_image->z_signal"}
        assert retrieval["z_signal->z_image"]["recall@1"] == pytest.approx(1.0)
        assert retrieval["z_signal->z_image"]["chance_recall@1"] == pytest.approx(1 / 24)
        distances = report["distances"]
        assert isinstance(distances, dict)
        assert set(distances) == {
            "cosine",
            "info_nce",
            "mean_mse",
            "mixture_jsd",
            "symmetric_kl",
            "wasserstein2_diag",
        }
        assert set(report["cross_reconstruction"]) == {"z_signal->z_image", "z_image->z_signal"}  # type: ignore[call-overload]
        assert set(report["reconstruction_gap"]) == {"z_signal->z_image", "z_image->z_signal"}  # type: ignore[call-overload]
        assert set(report["diagnostics"]) == {"z_signal", "z_image"}  # type: ignore[call-overload]
        assert report["procrustes"] is not None
        assert report["pair_integrity"]["status"] == "ok"  # type: ignore[index]
        assert report["flags"] == []

    def test_the_cross_rows_and_their_gap_join_the_metrics_table(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(aligned_checkpoint, "--latent-pair", "z_signal", "z_image", output_dir=out)
        )
        table = (out / "metrics_table.md").read_text(encoding="utf-8")
        assert "| signal | image |" in table
        assert "| image | signal |" in table
        assert "| signal | image (cross minus self) |" in table
        assert "| image | signal (cross minus self) |" in table
        assert "| all | signal |" in table

    def test_without_an_output_directory_it_only_prints(
        self, script: ModuleType, aligned_checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.main(_args(aligned_checkpoint, "--latent-pair", "z_signal", "z_image"))
        out = capsys.readouterr().out
        assert "Latent alignment:" in out
        assert "| signal | image (cross minus self) |" in out

    def test_without_the_flag_nothing_of_it_is_written(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(_args(aligned_checkpoint, output_dir=out))
        assert (out / "results.json").exists()
        assert not (out / "latent_alignment.json").exists()
        assert not (out / "latent_alignment.md").exists()

    def test_several_pairs_each_have_their_own_files(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--latent-pair",
                "z_image",
                "z_signal",
                output_dir=out,
            )
        )
        assert (out / "latent_alignment_z_signal_z_image.json").exists()
        assert (out / "latent_alignment_z_image_z_signal.md").exists()
        assert not (out / "latent_alignment.json").exists()

    def test_figures_can_be_drawn_in_the_same_run(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        arguments = _args(
            aligned_checkpoint, "--latent-pair", "z_signal", "z_image", output_dir=out
        )
        arguments.remove("--no-figures")
        script.main(arguments)
        assert (out / "latent_alignment.json").exists()
        assert any(out.glob("*.png"))


class TestTheSettings:
    def test_the_distances_can_be_picked(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--latent-distances",
                "cosine",
                "mean_mse",
                output_dir=out,
            )
        )
        assert set(_report(out)["distances"]) == {"cosine", "mean_mse"}  # type: ignore[call-overload]

    def test_the_retrieval_k_and_the_similarity_are_those_of_the_paired_flags(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--retrieval-k",
                "1",
                "3",
                "--retrieval-similarity",
                "l2",
                output_dir=out,
            )
        )
        report = _report(out)
        assert report["ks"] == [1, 3]
        assert report["similarity"] == "l2"

    def test_max_samples_limits_the_heldout_pairs_and_max_train_samples_the_training_pairs(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--train-dataloader-factory",
                _TRAIN,
                "--max-samples",
                "12",
                "--max-train-samples",
                "30",
                output_dir=out,
            )
        )
        report = _report(out)
        assert report["num_heldout"] == 12 and report["num_train"] == 30

    def test_the_procrustes_fit_can_be_centered(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--train-dataloader-factory",
                _TRAIN,
                "--procrustes-center",
                output_dir=out,
            )
        )
        assert _report(out)["procrustes"]["center"] is True  # type: ignore[index]

    def test_the_integrity_shuffles_and_the_margin_are_passed_on(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--train-dataloader-factory",
                _TRAIN,
                "--integrity-shuffles",
                "3",
                "--integrity-margin",
                "0.1",
                output_dir=out,
            )
        )
        integrity = _report(out)["pair_integrity"]
        assert integrity["shuffles"] == 3  # type: ignore[index]
        assert integrity["margin"] == pytest.approx(0.1)  # type: ignore[index]


class TestTheBaselineAndTheIntegrityCheck:
    def test_without_training_pairs_they_are_skipped_and_the_report_says_so(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(aligned_checkpoint, "--latent-pair", "z_signal", "z_image", output_dir=out)
        )
        report = _report(out)
        assert report["procrustes"] is None
        assert report["pair_integrity"]["status"] == "not_checked"  # type: ignore[index]
        assert "No training pairs" in report["pair_integrity"]["message"]  # type: ignore[index]

    def test_the_training_pairs_fit_a_map_that_is_applied_to_the_heldout_pairs(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--train-dataloader-factory",
                _TRAIN,
                output_dir=out,
            )
        )
        procrustes = _report(out)["procrustes"]
        assert isinstance(procrustes, dict)
        assert procrustes["num_fit_pairs"] == 48
        assert procrustes["retrieval"]["z_signal->z_image"]["recall@1"] == pytest.approx(1.0)
        assert "z_signal->z_image" in procrustes["cross_reconstruction"]

    def test_pairs_that_do_not_match_are_flagged_and_logged(
        self,
        script: ModuleType,
        aligned_checkpoint: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        out = tmp_path / "out"
        with caplog.at_level(logging.WARNING):
            script.main(
                _args(
                    aligned_checkpoint,
                    "--latent-pair",
                    "z_signal",
                    "z_image",
                    "--train-dataloader-factory",
                    _SHUFFLED_TRAIN,
                    output_dir=out,
                )
            )
        report = _report(out)
        assert report["pair_integrity"]["status"] == "suspect"  # type: ignore[index]
        assert any(flag.startswith("Pair integrity:") for flag in report["flags"])  # type: ignore[attr-defined]
        assert any("Pair integrity" in record.getMessage() for record in caplog.records)

    def test_the_training_flag_needs_a_pair(
        self, script: ModuleType, aligned_checkpoint: Path
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            script.main(_args(aligned_checkpoint, "--train-dataloader-factory", _TRAIN))
        assert raised.value.code == 2


class TestTheCollapseGuard:
    def test_a_collapsed_model_is_flagged_by_the_script(
        self,
        script: ModuleType,
        collapsed_checkpoint: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "out"
        with caplog.at_level(logging.WARNING):
            script.main(
                _args(
                    collapsed_checkpoint,
                    "--latent-pair",
                    "z_signal",
                    "z_image",
                    "--train-dataloader-factory",
                    _TRAIN,
                    output_dir=out,
                    model=f"{_FIXTURES}:buildCollapsedPairModel",
                )
            )
        report = _report(out)
        assert all(check["flagged"] for check in report["collapse"].values())  # type: ignore[attr-defined]
        flags = report["flags"]
        assert any("Collapse: 'z_signal'" in flag for flag in flags)  # type: ignore[attr-defined]
        assert any("Collapse: 'z_image'" in flag for flag in flags)  # type: ignore[attr-defined]
        assert any("0 for mismatched pairs too" in flag for flag in flags)  # type: ignore[attr-defined]
        assert any("Collapse" in record.getMessage() for record in caplog.records)
        assert "FLAG: Collapse" in capsys.readouterr().out
        md = (out / "latent_alignment.md").read_text(encoding="utf-8")
        assert "Collapse" in md

    def test_a_space_with_fewer_units_than_its_solo_stage_is_flagged(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--solo-active-units",
                "z_signal=99",
                output_dir=out,
            )
        )
        report = _report(out)
        collapse = report["collapse"]
        assert collapse["z_signal"]["flagged"] is True  # type: ignore[index]
        assert collapse["z_signal"]["reference_active_units"] == 99  # type: ignore[index]
        assert collapse["z_image"]["flagged"] is False  # type: ignore[index]

    def test_the_floor_applies_to_a_view_without_a_solo_count(
        self, script: ModuleType, aligned_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                aligned_checkpoint,
                "--latent-pair",
                "z_signal",
                "z_image",
                "--min-active-units",
                "4",
                output_dir=out,
            )
        )
        collapse = _report(out)["collapse"]
        assert collapse["z_signal"]["required"] == 4  # type: ignore[index]
        assert collapse["z_signal"]["flagged"] is True  # type: ignore[index]

    def test_a_view_that_is_not_in_any_pair_is_refused(
        self, script: ModuleType, aligned_checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            script.main(
                _args(
                    aligned_checkpoint,
                    "--latent-pair",
                    "z_signal",
                    "z_image",
                    "--solo-active-units",
                    "z_audio=3",
                )
            )
        assert raised.value.code == 2
        assert "z_audio" in capsys.readouterr().err

    @pytest.mark.parametrize("entry", ["z_signal", "z_signal=many", "z_signal=-1", "=3"])
    def test_an_entry_that_is_not_a_view_and_a_count_is_refused(
        self,
        script: ModuleType,
        aligned_checkpoint: Path,
        entry: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            script.main(
                _args(
                    aligned_checkpoint,
                    "--latent-pair",
                    "z_signal",
                    "z_image",
                    "--solo-active-units",
                    entry,
                )
            )
        assert raised.value.code == 2
        assert "solo-active-units" in capsys.readouterr().err


class TestASharedSpace:
    @pytest.fixture
    def shared_checkpoint(self, tmp_path: Path) -> Path:
        return _checkpoint(tmp_path, buildTwoModalityModelForScript(), "shared.pt")

    def test_one_space_read_from_each_modality_is_a_pair(
        self, script: ModuleType, shared_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                shared_checkpoint,
                "--latent-pair",
                "z_fused@x",
                "z_fused@y",
                output_dir=out,
                model=_SHARED_MODEL,
                data=_SHARED_DATA,
            )
        )
        report = _report(out)
        assert report["labels"] == ["x", "y"]
        assert report["views"] == {
            "x": {"space": "z_fused", "inputs": ["x"]},
            "y": {"space": "z_fused", "inputs": ["y"]},
        }
        assert set(report["retrieval"]) == {"x->y", "y->x"}  # type: ignore[call-overload]

    def test_the_cells_that_paired_already_measured_are_kept(
        self, script: ModuleType, shared_checkpoint: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        script.main(
            _args(
                shared_checkpoint,
                "--paired",
                "--latent-pair",
                "z_fused@x",
                "z_fused@y",
                output_dir=out,
                model=_SHARED_MODEL,
                data=_SHARED_DATA,
            )
        )
        table = (out / "metrics_table.md").read_text(encoding="utf-8")
        assert "| x | y |" in table
        assert "| y | x |" in table
        assert "(cross minus self)" in table
        assert (out / "paired_results.json").exists()
        assert (out / "latent_alignment.json").exists()


class TestRefusals:
    @pytest.mark.parametrize(
        ("pair", "message"),
        [
            (("z_signal", "z_nowhere"), "z_nowhere"),
            (("z_signal", "z_signal"), "same"),
            (("z_signal", "z_image@signal"), "not fed"),
            (("z_signal@", "z_image"), "empty modality"),
        ],
    )
    def test_a_pair_that_is_not_valid_for_the_model_is_refused_with_the_reason(
        self,
        script: ModuleType,
        aligned_checkpoint: Path,
        pair: tuple[str, str],
        message: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            script.main(_args(aligned_checkpoint, "--latent-pair", *pair))
        assert raised.value.code == 2
        err = capsys.readouterr().err
        assert "--latent-pair" in err
        assert message in err

    def test_batches_without_a_modality_of_the_pair_are_refused(
        self, script: ModuleType, aligned_checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            script.main(
                _args(
                    aligned_checkpoint,
                    "--latent-pair",
                    "z_signal",
                    "z_image",
                    data=_SIGNAL_ONLY,
                )
            )
        assert raised.value.code == 2
        assert "image" in capsys.readouterr().err

    def test_a_distance_that_is_not_registered_is_refused_by_the_parser(
        self, script: ModuleType, aligned_checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            script.main(
                _args(
                    aligned_checkpoint,
                    "--latent-pair",
                    "z_signal",
                    "z_image",
                    "--latent-distances",
                    "no_such_distance",
                )
            )
        assert raised.value.code == 2
        assert "no_such_distance" in capsys.readouterr().err

    def test_an_invalid_retrieval_k_is_refused_for_a_pair_too(
        self, script: ModuleType, aligned_checkpoint: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            script.main(
                _args(
                    aligned_checkpoint,
                    "--latent-pair",
                    "z_signal",
                    "z_image",
                    "--retrieval-k",
                    "0",
                )
            )
        assert raised.value.code == 2
        assert "--retrieval-k" in capsys.readouterr().err


class TestReportName:
    def test_one_pair_has_the_plain_name_and_several_have_their_labels(
        self, script: ModuleType
    ) -> None:
        from global_vae.evaluation.latent_alignment import evaluateLatentAlignment
        from tests.integration._latent_alignment_fixtures import buildHeldoutPairs

        report = evaluateLatentAlignment(
            buildAlignedPairModel(), "z_signal", "z_image", buildHeldoutPairs()
        )
        assert script._alignmentReportName(report, several=False) == "latent_alignment"
        assert (
            script._alignmentReportName(report, several=True) == "latent_alignment_z_signal_z_image"
        )
        report.labels = ["z fused@x", "y/b"]
        assert script._alignmentReportName(report, several=True) == "latent_alignment_z_fused_x_y_b"
