"""The harness of the four alignment baselines (`evaluation/alignment_baselines.py`, roadmap P6-6).

What is pinned here is everything that does not train a model (the run itself is in
`tests/integration/test_alignment_baselines.py`):

- the pool and its random split: pairs stay together, the held-out pairs are never in the training
  rows, the unpaired rows go to the solo stages only, one seed gives one split;
- the settings that are refused, and the epochs of the joint model (equal training by default);
- the mean and the spread over the repeats, and `nan` left out of them;
- the flat numbers of a report, for the model and for the Procrustes baseline;
- the table: mean +/- standard deviation per baseline, a dash where a baseline has no number, the
  integrity count, the chance level next to every recall, no em dash and no unicode arrow;
- a distance that is not registered is refused before anything trains.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from global_vae.evaluation.alignment_baselines import (
    BASELINE_JOINT,
    BASELINE_PROCRUSTES,
    BASELINE_SOLO,
    BASELINE_STAGE3,
    BaselineComparison,
    BaselineRecipe,
    BaselineRow,
    BaselineSettings,
    PairPool,
    flattenAlignmentReport,
    runAlignmentBaselines,
    splitPool,
    summarizeValues,
)
from global_vae.evaluation.latent_alignment import LatentAlignmentReport, evaluateLatentAlignment
from tests.unit._tiny_pair_model import buildTinyModel, tinyBatches


def _identifiedPool(count: int = 20, extra: int = 0) -> PairPool:
    """A pool whose row `i` is the number `i` (and `10 i` in the image), to follow rows by value."""
    index = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    unpaired = {}
    if extra:
        unpaired = {"signal": 1000.0 + torch.arange(extra, dtype=torch.float32).unsqueeze(1)}
    return PairPool(paired={"signal": index, "image": index * 10.0}, unpaired=unpaired)


def _values(batches: list[dict[str, torch.Tensor]], modality: str) -> list[float]:
    return [float(v) for batch in batches for v in batch[modality].flatten()]


class TestPairPool:
    def test_it_counts_its_pairs(self) -> None:
        assert _identifiedPool(20).num_pairs == 20

    def test_a_pool_needs_a_paired_modality(self) -> None:
        with pytest.raises(ValueError, match="at least one paired modality"):
            PairPool(paired={})

    def test_every_modality_has_one_row_per_pair(self) -> None:
        with pytest.raises(ValueError, match="same number of rows"):
            PairPool(paired={"signal": torch.zeros(5, 2), "image": torch.zeros(4, 2)})

    def test_unpaired_rows_belong_to_a_paired_modality(self) -> None:
        with pytest.raises(ValueError, match="not paired modalities"):
            PairPool(paired={"signal": torch.zeros(5, 2)}, unpaired={"audio": torch.zeros(3, 2)})


class TestSplitPool:
    def test_the_sizes_follow_the_fraction(self) -> None:
        split = splitPool(_identifiedPool(72), heldout_fraction=0.25, seed=0, batch_size=16)
        assert len(_values(split.heldout, "signal")) == 18
        assert len(_values(split.train, "signal")) == 54

    def test_the_heldout_pairs_are_never_in_the_training_rows(self) -> None:
        split = splitPool(_identifiedPool(40), 0.25, seed=3, batch_size=8)
        held, train = set(_values(split.heldout, "signal")), set(_values(split.train, "signal"))
        assert held.isdisjoint(train)
        assert held | train == {float(i) for i in range(40)}

    def test_the_two_modalities_of_a_pair_stay_together(self) -> None:
        split = splitPool(_identifiedPool(40), 0.25, seed=3, batch_size=8)
        for batches in (split.train, split.heldout):
            assert _values(batches, "image") == [10.0 * v for v in _values(batches, "signal")]

    def test_one_seed_is_one_split_and_another_seed_is_another(self) -> None:
        pool = _identifiedPool(40)
        one, again = splitPool(pool, 0.25, 7, 8), splitPool(pool, 0.25, 7, 8)
        other = splitPool(pool, 0.25, 8, 8)
        assert _values(one.heldout, "signal") == _values(again.heldout, "signal")
        assert _values(one.heldout, "signal") != _values(other.heldout, "signal")

    def test_at_least_two_pairs_are_held_out(self) -> None:
        split = splitPool(_identifiedPool(10), 0.01, seed=0, batch_size=4)
        assert len(_values(split.heldout, "signal")) == 2

    def test_a_batch_of_one_row_is_dropped(self) -> None:
        split = splitPool(_identifiedPool(20), 0.25, seed=0, batch_size=4)
        assert all(batch["signal"].shape[0] >= 2 for batch in split.train + split.heldout)
        assert len(_values(split.heldout, "signal")) == 4
        assert len(_values(split.train, "signal")) == 15

    def test_the_solo_stages_get_the_training_rows_and_then_the_unpaired_ones(self) -> None:
        split = splitPool(_identifiedPool(20, extra=6), 0.25, seed=0, batch_size=8)
        train_values = set(_values(split.train, "signal"))
        solo_signal = _values(split.solo["signal"], "signal")
        assert train_values <= set(solo_signal)
        assert {v for v in solo_signal if v >= 1000.0} == {1000.0 + i for i in range(6)}
        assert set(_values(split.solo["image"], "image")) == set(_values(split.train, "image"))

    def test_the_unpaired_rows_never_reach_the_pairs_or_the_heldout_set(self) -> None:
        split = splitPool(_identifiedPool(20, extra=6), 0.25, seed=0, batch_size=8)
        for batches in (split.train, split.heldout):
            assert all(v < 1000.0 for v in _values(batches, "signal"))

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_a_fraction_outside_zero_one_raises(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="heldout_fraction"):
            splitPool(_identifiedPool(), fraction, 0, 4)

    def test_a_batch_size_below_two_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            splitPool(_identifiedPool(), 0.25, 0, 1)

    def test_a_pool_that_leaves_fewer_than_two_for_training_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 are needed"):
            splitPool(_identifiedPool(3), 0.25, 0, 4)


class TestSettings:
    def test_the_defaults_are_those_of_the_roadmap_reference(self) -> None:
        settings = BaselineSettings()
        assert settings.repeats == 5
        assert settings.distances is None
        assert settings.include_joint

    def test_the_joint_model_trains_as_long_as_the_staged_route_unless_told_otherwise(self) -> None:
        assert BaselineSettings(solo_epochs=7, pair_epochs=5).total_joint_epochs == 12
        assert (
            BaselineSettings(solo_epochs=7, pair_epochs=5, joint_epochs=3).total_joint_epochs == 3
        )

    @pytest.mark.parametrize("name", ["repeats", "solo_epochs", "pair_epochs", "ramp_steps"])
    def test_a_count_below_one_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match=name):
            BaselineSettings(**{name: 0})  # type: ignore[arg-type]

    def test_joint_epochs_below_one_raise(self) -> None:
        with pytest.raises(ValueError, match="joint_epochs"):
            BaselineSettings(joint_epochs=0)

    @pytest.mark.parametrize("name", ["beta", "alignment_weight", "cross_weight"])
    @pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
    def test_a_weight_that_is_negative_or_not_finite_raises(self, name: str, value: float) -> None:
        with pytest.raises(ValueError, match=name):
            BaselineSettings(**{name: value})  # type: ignore[arg-type]

    def test_the_default_recipe_is_the_shipped_signal_plus_image_pair(self) -> None:
        recipe = BaselineRecipe()
        assert recipe.modalities == ("signal", "image")
        assert recipe.pair_spaces == ("z_signal", "z_image")
        assert "model=signal_image_pair" in recipe.pair_overrides
        assert "training=several_latents" in recipe.pair_overrides
        assert recipe.joint_space == "z_fused"
        assert "model.single_latent.dim=16" in recipe.joint_overrides


class TestSummarizeValues:
    def test_mean_spread_and_range(self) -> None:
        spread = summarizeValues([1.0, 2.0, 3.0, 4.0])
        assert spread.mean == 2.5
        assert spread.std == pytest.approx(1.2909944)
        assert (spread.minimum, spread.maximum, spread.count) == (1.0, 4.0, 4)

    def test_one_repeat_has_no_spread(self) -> None:
        spread = summarizeValues([0.5])
        assert (spread.mean, spread.std, spread.count) == (0.5, 0.0, 1)

    def test_nan_is_left_out_and_not_counted(self) -> None:
        spread = summarizeValues([1.0, math.nan, 3.0])
        assert (spread.mean, spread.count) == (2.0, 2)

    def test_nothing_but_nan_is_nan_with_a_count_of_zero(self) -> None:
        spread = summarizeValues([math.nan, math.nan])
        assert spread.count == 0
        assert math.isnan(spread.mean) and math.isnan(spread.std)

    def test_an_empty_list_is_the_same(self) -> None:
        assert summarizeValues([]).count == 0


def _alignedReport(train: bool = True) -> LatentAlignmentReport:
    model = buildTinyModel(0)
    model.encoders["image"].load_state_dict(model.encoders["signal"].state_dict())

    def linked(count: int, seed: int) -> list[dict[str, torch.Tensor]]:
        return [
            {"signal": b["signal"], "image": b["signal"].clone()} for b in tinyBatches(count, seed)
        ]

    return evaluateLatentAlignment(
        model,
        "z_signal",
        "z_image",
        linked(3, 1),
        train=linked(4, 2) if train else None,
        distances=["cosine", "mean_mse"],
    )


class TestFlattenAlignmentReport:
    def test_the_numbers_a_comparison_averages(self) -> None:
        flat = flattenAlignmentReport(_alignedReport())
        for key in ("recall@1", "chance_recall@1", "recall@5", "mean_rank", "chance_mean_rank"):
            assert key in flat, key
        assert flat["recall@1"] == pytest.approx(1.0)
        assert flat["chance_recall@1"] == pytest.approx(1 / 24)
        assert flat["mean_rank"] == pytest.approx(1.0)
        assert flat["chance_mean_rank"] == pytest.approx(12.5)
        assert "recall@1/z_signal->z_image" in flat
        assert "mean_rank/z_image->z_signal" in flat
        assert "cross_mse/z_signal->z_image/image" in flat
        assert "self_mse/z_image/image" in flat
        assert "gap_mse/z_signal->z_image/image" in flat
        for key in ("cross_r2", "self_r2", "gap_r2", "active_units", "collapsed_views"):
            assert key in flat, key
        assert flat["collapsed_views"] == 0.0
        assert flat["num_flags"] == 0.0
        assert set(k for k in flat if k.startswith("distance/")) == {
            "distance/cosine/ratio",
            "distance/mean_mse/ratio",
        }

    def test_every_value_is_a_float(self) -> None:
        assert all(isinstance(v, float) for v in flattenAlignmentReport(_alignedReport()).values())

    def test_the_procrustes_baseline_has_its_own_retrieval_and_no_model_guard(self) -> None:
        report = _alignedReport()
        flat = flattenAlignmentReport(report, procrustes=True)
        assert report.procrustes is not None
        assert flat["recall@1/z_signal->z_image"] == pytest.approx(
            report.procrustes.retrieval["z_signal->z_image"]["recall@1"]
        )
        assert "cross_r2" in flat and "self_r2" in flat
        for key in ("active_units", "collapsed_views", "num_flags"):
            assert key not in flat, key
        assert not any(key.startswith("distance/") for key in flat)

    def test_a_report_without_training_pairs_has_no_baseline_to_flatten(self) -> None:
        with pytest.raises(ValueError, match="no Procrustes baseline"):
            flattenAlignmentReport(_alignedReport(train=False), procrustes=True)


def _row(name: str, runs: list[dict[str, float]]) -> BaselineRow:
    row = BaselineRow(name, f"The row {name}.", runs=runs)
    row.finish()
    return row


class TestBaselineRow:
    def test_finish_computes_the_spread_of_each_number_over_the_runs(self) -> None:
        row = _row("a", [{"x": 1.0, "y": 5.0}, {"x": 3.0}])
        assert row.spread["x"].mean == 2.0 and row.spread["x"].count == 2
        assert row.spread["y"].mean == 5.0 and row.spread["y"].count == 1

    def test_no_run_no_spread(self) -> None:
        assert _row("a", []).spread == {}


def _comparison(joint: bool = True) -> BaselineComparison:
    runs = [
        {
            "recall@5": 0.1,
            "chance_recall@5": 0.25,
            "mean_rank": 9.0,
            "chance_mean_rank": 9.5,
            "cross_r2": -0.1,
            "self_r2": 0.5,
            "active_units": 6.0,
            "num_flags": 0.0,
        },
        {
            "recall@5": 0.3,
            "chance_recall@5": 0.25,
            "mean_rank": 8.0,
            "chance_mean_rank": 9.5,
            "cross_r2": 0.1,
            "self_r2": 0.5,
            "active_units": 4.0,
            "num_flags": 2.0,
        },
    ]
    rows = [
        _row(BASELINE_SOLO, runs),
        _row(BASELINE_PROCRUSTES, [{k: v for k, v in r.items() if k != "num_flags"} for r in runs]),
        _row(f"{BASELINE_STAGE3} cosine", runs),
    ]
    if joint:
        rows.append(_row(BASELINE_JOINT, runs))
    return BaselineComparison(
        labels=["z_signal", "z_image"],
        settings={"repeats": 2, "seed": 0},
        headline_k=5,
        rows=rows,
        integrity={"ok": 2, "suspect": 0, "not_checked": 0},
        num_pairs=72,
        num_train=54,
        num_heldout=18,
    )


class TestBaselineComparison:
    def test_a_row_is_found_by_its_name(self) -> None:
        assert _comparison().row(BASELINE_SOLO).name == BASELINE_SOLO

    def test_an_unknown_row_lists_the_real_ones(self) -> None:
        with pytest.raises(KeyError, match="1 procrustes"):
            _comparison().row("nothing")

    def test_the_table_has_a_row_per_baseline_with_mean_and_spread_and_the_chance_level(
        self,
    ) -> None:
        text = _comparison().toMarkdown()
        assert text.startswith("# Alignment baselines: z_signal and z_image")
        assert "72 pairs, 54 for training and 18 held out in each of 2 random splits" in text
        assert "recall@5 (chance)" in text
        for name in (
            BASELINE_SOLO,
            BASELINE_PROCRUSTES,
            f"{BASELINE_STAGE3} cosine",
            BASELINE_JOINT,
        ):
            assert f"| {name} |" in text
        assert "0.200 +/- 0.141 (0.250 +/- 0.000)" in text
        assert "{'ok': 2, 'suspect': 0, 'not_checked': 0}" in text
        assert "I-15" in text

    def test_a_baseline_without_a_model_has_a_dash_where_it_has_no_guard(self) -> None:
        lines = _comparison().toMarkdown().splitlines()
        procrustes = next(line for line in lines if line.startswith(f"| {BASELINE_PROCRUSTES} |"))
        assert procrustes.rstrip().endswith("| - |")
        solo = next(line for line in lines if line.startswith(f"| {BASELINE_SOLO} |"))
        assert solo.rstrip().endswith("| 1/2 |")

    def test_every_row_of_the_table_has_the_same_number_of_cells(self) -> None:
        table = [line for line in _comparison().toMarkdown().splitlines() if line.startswith("|")]
        assert len({line.count("|") for line in table}) == 1

    def test_the_text_has_no_em_dash_and_no_unicode_arrow(self) -> None:
        text = _comparison().toMarkdown()
        assert chr(0x2014) not in text
        assert not any(chr(arrow) in text for arrow in (0x2192, 0x2190, 0x21D2))

    def test_the_summary_is_the_table(self) -> None:
        assert _comparison().summary() == _comparison().toMarkdown()

    def test_the_files_are_written_with_their_folders(self, tmp_path: Path) -> None:
        comparison = _comparison()
        comparison.save(tmp_path / "a" / "comparison.json")
        comparison.saveMarkdown(tmp_path / "b" / "comparison.md")
        loaded = json.loads((tmp_path / "a" / "comparison.json").read_text(encoding="utf-8"))
        assert loaded["labels"] == ["z_signal", "z_image"]
        assert loaded["rows"][0]["runs"][0]["recall@5"] == 0.1
        assert loaded["rows"][0]["spread"]["recall@5"]["count"] == 2
        assert (tmp_path / "b" / "comparison.md").read_text(
            encoding="utf-8"
        ) == comparison.toMarkdown()


class TestRunRefusals:
    def test_an_unknown_distance_is_refused_before_anything_trains(self, tmp_path: Path) -> None:
        pool = _identifiedPool(20)
        settings = BaselineSettings(repeats=1, distances=("cosine", "not_a_distance"))
        with pytest.raises(KeyError, match="not_a_distance"):
            runAlignmentBaselines(pool, settings=settings, work_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
