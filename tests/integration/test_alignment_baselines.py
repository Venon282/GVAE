"""The four alignment baselines in one command (`examples/05_alignment_baselines.py`, roadmap P6-6).

The example is loaded in-process through `importlib` (its file name starts with a digit) and run on
its synthetic linked pair at small epochs. Nothing here measures anything about real data (roadmap
I-15): what is pinned is that the harness runs and reports what the roadmap asks for.

- **the four baselines in one command**: the solo spaces as they are (0), the Procrustes map (1),
  stage 3 once per distance (2) and the joint model with a shared latent space (3), each over
  repeated random splits and seeds with a mean and a spread, never one run;
- **the pair-integrity check** counted per repeat, and the chance level next to every recall;
- **the collapse guard on a deliberately collapsed run**: a stage 3 under an enormous KL weight
  loses the units its solo stages had, and the guard flags it while the solo baseline is clean;
- **a solo stage that collapsed** is refused by the warm start (ADR 0052), not scored.

These tests train a few small models each and are marked `slow` (skip with `-m "not slow"`).
"""

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest
import torch

from global_vae.evaluation.alignment_baselines import (
    BASELINE_JOINT,
    BASELINE_PROCRUSTES,
    BASELINE_SOLO,
    BASELINE_STAGE3,
    BaselineComparison,
    BaselineSettings,
    runAlignmentBaselines,
)

pytestmark = pytest.mark.slow

_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "05_alignment_baselines.py"
_REPEATS = 2
_NUM_PAIRS = 72


def _loadExample() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_example_05", _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _loadExample()


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("alignment_baselines")


@pytest.fixture(scope="module")
def comparison(example: ModuleType, output_dir: Path) -> BaselineComparison:
    """One command: every baseline, two random splits and seeds, one distance."""
    result: BaselineComparison = example.run(
        [
            "--pairs",
            str(_NUM_PAIRS),
            "--repeats",
            str(_REPEATS),
            "--solo-epochs",
            "40",
            "--pair-epochs",
            "8",
            "--joint-epochs",
            "8",
            "--distances",
            "cosine",
            "--output-dir",
            str(output_dir),
        ]
    )
    return result


class TestTheSyntheticPair:
    def test_the_rows_of_a_sample_describe_the_same_two_numbers(self, example: ModuleType) -> None:
        generator = torch.Generator().manual_seed(0)
        signals, images = example.makeLinkedRows(12, generator)
        assert signals.shape == (12, 256)
        assert images.shape == (12, 64, 64)
        assert torch.isfinite(signals).all() and torch.isfinite(images).all()

    def test_one_seed_is_one_pool(self, example: ModuleType) -> None:
        one, other = example.makeLinkedPool(10, seed=4), example.makeLinkedPool(10, seed=4)
        different = example.makeLinkedPool(10, seed=5)
        assert torch.equal(one.paired["signal"], other.paired["signal"])
        assert not torch.equal(one.paired["signal"], different.paired["signal"])
        assert one.num_pairs == 10 and one.unpaired == {}

    def test_unpaired_samples_are_extra_rows_of_each_modality(self, example: ModuleType) -> None:
        pool = example.makeLinkedPool(10, num_unpaired=7, seed=1)
        assert pool.unpaired["signal"].shape == (7, 256)
        assert pool.unpaired["image"].shape == (7, 64, 64)
        assert pool.num_pairs == 10

    def test_the_defaults_are_the_reference_of_the_roadmap(self, example: ModuleType) -> None:
        args = example.buildArgumentParser().parse_args([])
        assert (args.pairs, args.repeats) == (72, 5)
        assert args.distances is None and not args.no_joint

    def test_a_distance_that_is_not_registered_is_refused_by_the_parser(
        self, example: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            example.buildArgumentParser().parse_args(["--distances", "no_such_distance"])
        assert raised.value.code == 2
        assert "no_such_distance" in capsys.readouterr().err


class TestTheFourBaselinesInOneCommand:
    def test_each_baseline_of_the_roadmap_has_a_row_in_its_order(
        self, comparison: BaselineComparison
    ) -> None:
        assert [row.name for row in comparison.rows] == [
            BASELINE_SOLO,
            BASELINE_PROCRUSTES,
            f"{BASELINE_STAGE3} cosine",
            BASELINE_JOINT,
        ]
        assert comparison.labels == ["z_signal", "z_image"]

    def test_the_numbers_are_over_repeated_splits_never_one_run(
        self, comparison: BaselineComparison
    ) -> None:
        assert comparison.settings["repeats"] == _REPEATS
        for row in comparison.rows:
            assert len(row.runs) == _REPEATS, row.name
            recall = row.spread[f"recall@{comparison.headline_k}"]
            assert recall.count == _REPEATS
            assert recall.minimum <= recall.mean <= recall.maximum
            assert recall.std >= 0.0
        # Two seeds are two splits and two sets of models, so the numbers differ between them.
        solo = comparison.row(BASELINE_SOLO)
        assert solo.runs[0] != solo.runs[1]

    def test_the_split_is_what_the_settings_say(self, comparison: BaselineComparison) -> None:
        assert comparison.num_pairs == _NUM_PAIRS
        assert comparison.num_heldout == 18
        assert comparison.num_train == _NUM_PAIRS - 18

    def test_every_recall_is_read_against_its_chance_level(
        self, comparison: BaselineComparison
    ) -> None:
        k = comparison.headline_k
        expected = min(k, comparison.num_heldout) / comparison.num_heldout
        for row in comparison.rows:
            for run in row.runs:
                assert run[f"chance_recall@{k}"] == pytest.approx(expected), row.name
                assert 0.0 <= run[f"recall@{k}"] <= 1.0, row.name
                assert run["chance_mean_rank"] == pytest.approx((comparison.num_heldout + 1) / 2)
                assert 1.0 <= run["mean_rank"] <= comparison.num_heldout, row.name

    def test_the_headline_k_is_selective_for_the_heldout_size(
        self, comparison: BaselineComparison
    ) -> None:
        assert comparison.headline_k == 1

    def test_the_pairs_are_checked_in_every_repeat_and_these_ones_match(
        self, comparison: BaselineComparison
    ) -> None:
        assert comparison.integrity == {"ok": _REPEATS, "suspect": 0, "not_checked": 0}

    def test_cross_reconstruction_is_read_next_to_the_self_reconstruction(
        self, comparison: BaselineComparison
    ) -> None:
        for row in comparison.rows:
            for run in row.runs:
                assert math.isfinite(run["cross_r2"]) and math.isfinite(run["self_r2"]), row.name
                assert "gap_r2" in run, row.name
        solo = comparison.row(BASELINE_SOLO).runs[0]
        assert "cross_mse/z_signal->z_image/image" in solo
        assert "self_mse/z_image/image" in solo
        assert "gap_mse/z_signal->z_image/image" in solo

    def test_a_model_has_its_distances_and_its_guard_and_the_baseline_map_has_neither(
        self, comparison: BaselineComparison
    ) -> None:
        procrustes = comparison.row(BASELINE_PROCRUSTES).runs[0]
        assert not any(key.startswith("distance/") for key in procrustes)
        assert "active_units" not in procrustes
        for name in (BASELINE_SOLO, f"{BASELINE_STAGE3} cosine", BASELINE_JOINT):
            run = comparison.row(name).runs[0]
            assert "distance/cosine/ratio" in run, name
            assert run["active_units"] >= 0.0, name
            assert run["collapsed_views"] in (0.0, 1.0, 2.0), name

    def test_the_joint_model_is_read_from_one_shared_space_one_modality_at_a_time(
        self, comparison: BaselineComparison
    ) -> None:
        joint = comparison.row(BASELINE_JOINT).runs[0]
        assert "recall@1/signal->image" in joint
        assert "recall@1/image->signal" in joint

    def test_the_stage_3_model_has_two_separate_spaces(
        self, comparison: BaselineComparison
    ) -> None:
        stage3 = comparison.row(f"{BASELINE_STAGE3} cosine").runs[0]
        assert "recall@1/z_signal->z_image" in stage3
        assert "recall@1/z_image->z_signal" in stage3

    def test_the_table_and_the_numbers_are_written(
        self, comparison: BaselineComparison, output_dir: Path
    ) -> None:
        text = (output_dir / "comparison.md").read_text(encoding="utf-8")
        for name in (BASELINE_SOLO, BASELINE_PROCRUSTES, BASELINE_JOINT):
            assert f"| {name} |" in text
        assert "+/-" in text and "(chance)" in text
        loaded = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
        assert [row["name"] for row in loaded["rows"]] == [row.name for row in comparison.rows]
        assert len(loaded["rows"][0]["runs"]) == _REPEATS
        assert loaded["integrity"] == comparison.integrity

    def test_the_command_prints_the_table(
        self, example: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        example.run(
            [
                "--pairs",
                "40",
                "--repeats",
                "1",
                "--solo-epochs",
                "40",
                "--pair-epochs",
                "2",
                "--distances",
                "cosine",
                "--no-joint",
                "--output-dir",
                str(tmp_path),
            ]
        )
        out = capsys.readouterr().out
        assert "Alignment baselines" in out
        assert BASELINE_JOINT not in out
        assert (tmp_path / "comparison.md").exists()


@pytest.fixture(scope="module")
def collapsed(example: ModuleType) -> BaselineComparison:
    # An enormous KL weight in stage 3 pulls every posterior to the prior: both spaces lose
    # the units their solo stages had, and the distance between them falls for the wrong reason.
    settings = BaselineSettings(
        repeats=1,
        solo_epochs=40,
        pair_epochs=8,
        beta=300.0,
        distances=("mean_mse",),
        include_joint=False,
    )
    return runAlignmentBaselines(example.makeLinkedPool(_NUM_PAIRS, seed=0), settings=settings)


class TestTheCollapseGuardOnARealRun:
    """The done-criterion of P6-6: the guard fires on a deliberately collapsed run."""

    def test_the_collapsed_stage_3_is_flagged_in_both_spaces(
        self, collapsed: BaselineComparison
    ) -> None:
        run = collapsed.row(f"{BASELINE_STAGE3} mean_mse").runs[0]
        assert run["collapsed_views"] == 2.0
        assert run["num_flags"] >= 2.0

    def test_it_has_fewer_active_units_than_the_solo_stages_it_started_from(
        self, collapsed: BaselineComparison
    ) -> None:
        stage3 = collapsed.row(f"{BASELINE_STAGE3} mean_mse").runs[0]
        solo = collapsed.row(BASELINE_SOLO).runs[0]
        assert stage3["active_units"] < solo["active_units"]

    def test_the_solo_spaces_it_started_from_are_not_flagged(
        self, collapsed: BaselineComparison
    ) -> None:
        solo = collapsed.row(BASELINE_SOLO).runs[0]
        assert solo["collapsed_views"] == 0.0
        assert solo["num_flags"] == 0.0

    def test_the_table_counts_the_flagged_repeats(self, collapsed: BaselineComparison) -> None:
        lines = collapsed.toMarkdown().splitlines()
        stage3 = next(line for line in lines if line.startswith(f"| {BASELINE_STAGE3} mean_mse |"))
        assert stage3.rstrip().endswith("| 1/1 |")
        solo = next(line for line in lines if line.startswith(f"| {BASELINE_SOLO} |"))
        assert solo.rstrip().endswith("| 0/1 |")


class TestASoloStageThatCollapsed:
    def test_the_warm_start_refuses_it_instead_of_scoring_it(self, example: ModuleType) -> None:
        settings = BaselineSettings(
            repeats=1, solo_epochs=1, pair_epochs=1, distances=("cosine",), include_joint=False
        )
        with pytest.raises(ValueError, match="looks collapsed"):
            runAlignmentBaselines(example.makeLinkedPool(_NUM_PAIRS, seed=0), settings=settings)
