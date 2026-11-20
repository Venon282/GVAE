"""Evaluation across more than two latent spaces (`evaluation/multiway_alignment.py`, P6-8).

What is pinned, in the order of the module:

- the labels of the views, and the encode-only pass over `K` of them (the rows stay the same
  sample, the widths must agree, the sample ids are compared);
- the generalized Procrustes frame: it recovers views that are rotations of one another, equals the
  two-view Procrustes for two views, removes an offset when centered, never does worse than the
  maps to the first view it starts from, and refuses what it cannot fit;
- the anchored frame: `K - 1` maps, the anchor's own is the identity, and a pair of other views
  goes through the anchor;
- the report: the four baselines side by side, the number of maps each one fitted, chance, what is
  left out without training data or an anchor, and its JSON and Markdown forms.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.evaluation.latent_alignment import LatentView, fitOrthogonalProcrustes
from global_vae.evaluation.multiway_alignment import (
    BASELINES,
    CommonFrame,
    MultiwayReport,
    ViewMeans,
    collectViewMeans,
    evaluateMultiwayAlignment,
    fitAnchoredFrame,
    fitCommonFrame,
    viewLabels,
)
from tests.integration._latent_alignment_fixtures import (
    TRIPLE,
    TRIPLE_BATCH,
    buildRotatedTripleModel,
    buildTripleBatches,
    buildTripleModel,
)
from tests.unit._tiny_pair_model import LATENT

NAMES = TRIPLE
VIEWS = [LatentView(f"z_{name}", (name,)) for name in NAMES]
SAMPLES = 40


def _orthogonal(size: int, seed: int, reflection: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix: torch.Tensor = torch.linalg.qr(
        torch.randn(size, size, generator=generator, dtype=torch.float64)
    ).Q
    if reflection:
        matrix[:, 0] = -matrix[:, 0]
    return matrix


def _spaces(
    count: int = 3, samples: int = SAMPLES, width: int = 4, offset: float = 0.0
) -> dict[str, torch.Tensor]:
    """`count` views that are rotations of one base cloud (plus an offset of their own)."""
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(samples, width, generator=generator, dtype=torch.float64)
    spaces: dict[str, torch.Tensor] = {}
    for index, name in enumerate(NAMES[:count]):
        shift = offset * torch.arange(width, dtype=torch.float64) * (index + 1)
        spaces[name] = base @ _orthogonal(width, index + 1, reflection=index == 1) + shift
    return spaces


def _batches(count: int = 4, seed: int = 0, same: bool = True) -> list[dict[str, torch.Tensor]]:
    return buildTripleBatches(count, seed, same)


def _withIds(batch: dict[str, torch.Tensor], **ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {**batch, **{f"sample_id.{name}": value for name, value in ids.items()}}


class TestViewLabels:
    def test_views_of_different_spaces_are_named_by_their_space(self) -> None:
        assert viewLabels(VIEWS) == ("z_signal", "z_image", "z_text")

    def test_views_of_one_space_are_named_by_their_inputs(self) -> None:
        views = [LatentView("z_fused", ("signal",)), LatentView("z_fused", ("image",)), VIEWS[2]]
        assert viewLabels(views) == ("z_fused@signal", "z_fused@image", "z_text")

    def test_the_same_view_twice_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"views \['z_signal@signal'\] are listed more than"):
            viewLabels([VIEWS[0], VIEWS[0]])


class TestCollectViewMeans:
    def test_each_view_is_its_own_encoder_alone_and_row_i_is_sample_i(self) -> None:
        model = buildRotatedTripleModel()
        batches = _batches(2)
        got = collectViewMeans(model, batches, VIEWS)
        assert isinstance(got, ViewMeans)
        assert got.labels == ("z_signal", "z_image", "z_text")
        assert got.num_samples == 2 * TRIPLE_BATCH and got.width == LATENT
        for name, view in zip(NAMES, VIEWS, strict=True):
            expected = torch.cat(
                [
                    model.encode({name: b[name]}, use_mean=True)["latent_params"][view.space][0]
                    for b in batches
                ]
            )
            assert torch.allclose(got.means[view.space], expected.detach())

    def test_max_samples_trims_the_last_batch(self) -> None:
        got = collectViewMeans(buildTripleModel(), _batches(4), VIEWS, max_samples=TRIPLE_BATCH + 5)
        assert got.num_samples == TRIPLE_BATCH + 5

    def test_the_model_is_put_in_evaluation_mode(self) -> None:
        model = buildTripleModel()
        model.train()
        collectViewMeans(model, _batches(1), VIEWS)
        assert not model.training

    def test_fewer_than_two_views_are_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two views, got 1"):
            collectViewMeans(buildTripleModel(), _batches(1), VIEWS[:1])

    def test_a_number_of_samples_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_samples must be at least 1"):
            collectViewMeans(buildTripleModel(), _batches(1), VIEWS, max_samples=0)

    def test_an_empty_dataloader_is_refused(self) -> None:
        with pytest.raises(ValueError, match="gave no batch"):
            collectViewMeans(buildTripleModel(), [], VIEWS)

    def test_a_batch_without_a_modality_is_refused_by_name(self) -> None:
        batch = {n: b for n, b in _batches(1)[0].items() if n != "text"}
        with pytest.raises(ValueError, match=r"lacks \['text'\]"):
            collectViewMeans(buildTripleModel(), [batch], VIEWS)

    def test_views_of_different_widths_are_refused_with_the_decision_that_says_why(self) -> None:
        model = buildTripleModel({"signal": 3, "image": 3, "text": 4})
        with pytest.raises(ValueError, match=r"different widths.*roadmap D-16.*adapter"):
            collectViewMeans(model, _batches(1), VIEWS)

    def test_sample_ids_that_agree_are_counted(self) -> None:
        ids = torch.arange(TRIPLE_BATCH)
        batches = [_withIds(b, signal=ids, image=ids, text=ids) for b in _batches(2)]
        assert (
            collectViewMeans(buildTripleModel(), batches, VIEWS).sample_ids_checked
            == 2 * TRIPLE_BATCH
        )

    def test_batches_without_ids_are_trusted_and_say_so(self) -> None:
        assert collectViewMeans(buildTripleModel(), _batches(1), VIEWS).sample_ids_checked == 0

    def test_a_modality_whose_rows_are_in_another_order_stops_the_pass(self) -> None:
        ids = torch.arange(TRIPLE_BATCH)
        batch = _withIds(_batches(1)[0], signal=ids, image=ids, text=ids.flip(0))
        with pytest.raises(ValueError, match=r"Evaluating several latent spaces.*not the same"):
            collectViewMeans(buildTripleModel(), [batch], VIEWS)


class TestFitCommonFrame:
    def test_views_that_are_rotations_of_one_cloud_agree_exactly(self) -> None:
        spaces = _spaces()
        frame = fitCommonFrame(spaces)
        assert isinstance(frame, CommonFrame)
        assert frame.relative_residual == pytest.approx(0.0, abs=1e-20)
        for a in NAMES:
            for b in NAMES:
                assert torch.allclose(frame.between(a, b, spaces[a]), spaces[b], atol=1e-9)

    def test_every_map_is_orthogonal_and_there_is_one_per_view(self) -> None:
        frame = fitCommonFrame(_spaces())
        assert frame.num_maps == 3
        for rotation in frame.rotations.values():
            assert torch.allclose(rotation @ rotation.T, torch.eye(4, dtype=torch.float64))

    def test_two_views_are_the_orthogonal_procrustes_of_the_pair(self) -> None:
        spaces = _spaces(2)
        frame = fitCommonFrame(spaces)
        pair = fitOrthogonalProcrustes(spaces["signal"], spaces["image"])
        assert torch.allclose(
            frame.between("signal", "image", spaces["signal"]), pair.toSecond(spaces["signal"])
        )
        assert frame.num_iterations == 0 and frame.converged

    def test_paths_through_the_frame_agree(self) -> None:
        spaces = _spaces()
        noisy = {n: m + 0.05 * torch.randn_like(m) for n, m in spaces.items()}
        frame = fitCommonFrame(noisy)
        direct = frame.between("signal", "text", noisy["signal"])
        in_two = frame.between("image", "text", frame.between("signal", "image", noisy["signal"]))
        assert torch.allclose(direct, in_two, atol=1e-9)

    def test_centering_removes_an_offset_that_the_plain_fit_cannot(self) -> None:
        spaces = _spaces(offset=2.0)
        plain = fitCommonFrame(spaces)
        centered = fitCommonFrame(spaces, center=True)
        assert plain.relative_residual > 1e-3
        assert centered.relative_residual == pytest.approx(0.0, abs=1e-20)
        assert torch.allclose(
            centered.between("signal", "text", spaces["signal"]), spaces["text"], atol=1e-9
        )

    def test_the_iteration_does_not_do_worse_than_the_maps_it_starts_from(self) -> None:
        spaces = {n: m + 0.3 * torch.randn_like(m) for n, m in _spaces().items()}
        start = fitCommonFrame(spaces, max_iterations=0)
        finished = fitCommonFrame(spaces)
        assert finished.relative_residual <= start.relative_residual + 1e-12
        assert finished.num_iterations > 0 and finished.converged
        assert start.num_iterations == 0 and start.converged

    def test_more_rounds_never_raise_the_residual(self) -> None:
        spaces = {n: m + 0.5 * torch.randn_like(m) for n, m in _spaces(3).items()}
        residuals = [
            fitCommonFrame(spaces, max_iterations=n, tolerance=0.0).relative_residual
            for n in range(0, 6)
        ]
        assert all(
            later <= earlier + 1e-12
            for earlier, later in zip(residuals, residuals[1:], strict=False)
        )

    def test_it_is_not_converged_when_the_rounds_run_out_first(self) -> None:
        spaces = {n: m + 0.5 * torch.randn_like(m) for n, m in _spaces(3).items()}
        frame = fitCommonFrame(spaces, max_iterations=1, tolerance=0.0)
        assert frame.num_iterations == 1 and not frame.converged

    def test_the_fit_is_deterministic_and_leaves_its_inputs_alone(self) -> None:
        spaces = {n: m + 0.2 * torch.randn_like(m) for n, m in _spaces().items()}
        before = {n: m.clone() for n, m in spaces.items()}
        first, second = fitCommonFrame(spaces), fitCommonFrame(spaces)
        for name in NAMES:
            assert torch.equal(first.rotations[name], second.rotations[name])
            assert torch.equal(spaces[name], before[name])

    def test_independent_clouds_leave_most_of_their_size(self) -> None:
        generator = torch.Generator().manual_seed(5)
        clouds = {n: torch.randn(200, 4, generator=generator, dtype=torch.float64) for n in NAMES}
        assert fitCommonFrame(clouds).relative_residual > 0.5

    def test_one_view_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two views, got 1"):
            fitCommonFrame({"a": torch.zeros(3, 2)})

    def test_matrices_of_different_shapes_are_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"'b' gives \(3, 3\) and 'a' gives \(3, 2\)"):
            fitCommonFrame({"a": torch.zeros(3, 2), "b": torch.zeros(3, 3)})

    def test_a_value_that_is_not_finite_is_refused_by_name(self) -> None:
        bad = torch.ones(3, 2)
        bad[0, 0] = float("nan")
        with pytest.raises(ValueError, match="means of 'b' must be finite"):
            fitCommonFrame({"a": torch.ones(3, 2), "b": bad})

    def test_no_sample_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            fitCommonFrame({"a": torch.zeros(0, 2), "b": torch.zeros(0, 2)})

    @pytest.mark.parametrize("kwargs", [{"max_iterations": -1}, {"tolerance": -1e-3}])
    def test_a_negative_setting_is_refused(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="must be at least 0"):
            fitCommonFrame(_spaces(), **kwargs)  # type: ignore[arg-type]


class TestFitAnchoredFrame:
    def test_the_anchors_own_map_is_the_identity_and_the_others_are_fitted_to_it(self) -> None:
        spaces = _spaces()
        frame = fitAnchoredFrame(spaces, "image")
        assert torch.equal(frame.rotations["image"], torch.eye(4, dtype=torch.float64))
        assert frame.num_iterations == 0 and frame.converged
        pair = fitOrthogonalProcrustes(spaces["text"], spaces["image"])
        assert torch.allclose(frame.rotations["text"], pair.rotation)

    def test_two_other_views_go_through_the_anchor(self) -> None:
        spaces = _spaces()
        frame = fitAnchoredFrame(spaces, "signal")
        through = frame.between("image", "text", spaces["image"])
        via_anchor = frame.between(
            "signal", "text", frame.between("image", "signal", spaces["image"])
        )
        assert torch.allclose(through, via_anchor, atol=1e-9)
        assert torch.allclose(through, spaces["text"], atol=1e-9)

    def test_an_anchored_fit_is_no_better_than_the_common_one(self) -> None:
        spaces = {n: m + 0.3 * torch.randn_like(m) for n, m in _spaces().items()}
        anchored = fitAnchoredFrame(spaces, "signal").relative_residual
        assert fitCommonFrame(spaces).relative_residual <= anchored + 1e-12

    def test_an_anchor_that_is_not_a_view_is_refused_and_the_views_are_listed(self) -> None:
        with pytest.raises(ValueError, match=r"anchor 'nope' is not one of the views \['signal'"):
            fitAnchoredFrame(_spaces(), "nope")


class TestEvaluateMultiwayAlignment:
    def _report(self, **kwargs: Any) -> MultiwayReport:
        options: dict[str, Any] = {
            "train": _batches(4, seed=1),
            "anchor": "z_signal",
            "ks": (1, 5),
        }
        options.update(kwargs)
        return evaluateMultiwayAlignment(
            buildRotatedTripleModel(), VIEWS, _batches(3, seed=2), **options
        )

    def test_rotated_spaces_are_misaligned_as_they_are_and_recovered_by_every_map(self) -> None:
        report = self._report()
        assert list(report.retrieval) == list(BASELINES)
        assert report.meanRetrieval("identity")["recall@1"] < 0.1
        for baseline in ("pairwise", "common", "anchor"):
            assert report.meanRetrieval(baseline)["recall@1"] == pytest.approx(1.0, abs=1e-9)

    def test_every_ordered_pair_is_read(self) -> None:
        report = self._report()
        pairs = {f"{a}->{b}" for a in report.labels for b in report.labels if a != b}
        assert len(pairs) == 6
        for baseline in BASELINES:
            assert set(report.retrieval[baseline]) == pairs

    def test_the_bill_of_maps(self) -> None:
        assert self._report().num_maps == {"identity": 0, "pairwise": 3, "common": 3, "anchor": 2}

    def test_four_views_need_six_pairwise_maps_but_four_in_a_frame_and_three_to_an_anchor(
        self,
    ) -> None:
        spaces = _spaces(3, samples=60)
        four = {**spaces, "fourth": spaces["signal"] @ _orthogonal(4, 99)}
        assert fitCommonFrame(four).num_maps == 4
        assert fitAnchoredFrame(four, "signal").num_maps == 3

    def test_the_chance_level_is_reported_next_to_the_recalls(self) -> None:
        report = self._report()
        mean = report.meanRetrieval("identity")
        assert mean["chance_recall@1"] == pytest.approx(1 / report.num_heldout)
        assert mean["chance_recall@5"] == pytest.approx(5 / report.num_heldout)

    def test_the_maps_are_fitted_on_the_training_data_and_not_on_the_held_out_data(self) -> None:
        model = buildRotatedTripleModel()
        held = _batches(3, seed=2, same=False)  # strangers: no map can match them
        report = evaluateMultiwayAlignment(model, VIEWS, held, train=_batches(4, seed=1))
        assert report.meanRetrieval("pairwise")["recall@1"] < 0.5
        assert report.num_train == 4 * TRIPLE_BATCH and report.num_heldout == 3 * TRIPLE_BATCH

    def test_without_training_data_only_the_identity_is_read_and_the_report_says_so(self) -> None:
        report = evaluateMultiwayAlignment(
            buildRotatedTripleModel(), VIEWS, _batches(3), anchor="z_signal"
        )
        assert list(report.retrieval) == ["identity"] and report.num_train == 0
        assert report.frame_fit == {}
        assert "No training data" in report.summary()
        assert "no training data" in report.toMarkdown()

    def test_without_an_anchor_there_is_no_anchor_baseline(self) -> None:
        report = self._report(anchor=None)
        assert list(report.retrieval) == ["identity", "pairwise", "common"]
        assert "anchor" not in report.num_maps

    def test_the_fit_of_each_frame_is_reported_on_both_sets(self) -> None:
        fit = self._report().frame_fit
        assert set(fit) == {"common", "anchor"}
        assert fit["common"]["relative_residual_train"] == pytest.approx(0.0, abs=1e-12)
        assert fit["common"]["relative_residual_heldout"] == pytest.approx(0.0, abs=1e-12)
        assert fit["anchor"]["num_iterations"] == 0

    def test_the_anchor_must_be_one_of_the_views(self) -> None:
        with pytest.raises(ValueError, match=r"anchor 'z_nope' is not one of the views"):
            self._report(anchor="z_nope")

    def test_a_space_the_model_lacks_lists_the_real_ones(self) -> None:
        with pytest.raises(KeyError, match="z_signal"):
            evaluateMultiwayAlignment(
                buildRotatedTripleModel(), ["z_signal", "z_nope"], _batches(1), train=_batches(1)
            )

    def test_one_held_out_sample_is_not_enough_for_a_retrieval(self) -> None:
        with pytest.raises(ValueError, match="at least 2 held-out samples"):
            evaluateMultiwayAlignment(buildRotatedTripleModel(), VIEWS, _batches(1), max_samples=1)

    def test_a_view_given_as_text_is_resolved(self) -> None:
        report = evaluateMultiwayAlignment(
            buildRotatedTripleModel(), ["z_signal", "z_image@image", "z_text"], _batches(2)
        )
        assert report.labels == ["z_signal", "z_image", "z_text"]
        assert report.views["z_image"] == {"space": "z_image", "inputs": ["image"]}

    def test_the_sample_ids_that_were_compared_are_counted_over_both_sets(self) -> None:
        ids = torch.arange(TRIPLE_BATCH)

        def marked(count: int, seed: int) -> list[dict[str, torch.Tensor]]:
            return [_withIds(b, signal=ids, image=ids, text=ids) for b in _batches(count, seed)]

        report = evaluateMultiwayAlignment(
            buildRotatedTripleModel(), VIEWS, marked(2, 2), train=marked(3, 1), anchor="z_signal"
        )
        assert report.sample_ids_checked == 5 * TRIPLE_BATCH
        assert "agreed across the modalities" in report.toMarkdown()

    def test_without_ids_the_markdown_says_the_rows_were_taken_on_trust(self) -> None:
        assert "taken on trust" in self._report().toMarkdown()

    def test_the_summary_has_one_line_per_baseline(self) -> None:
        summary = self._report().summary().splitlines()
        assert summary[0].startswith("Multi-way alignment: 3 views (z_signal, z_image, z_text, ")
        assert [line.split()[0] for line in summary[1:]] == list(BASELINES)
        assert "maps=2" in summary[-1]

    def test_the_markdown_has_the_two_tables_and_the_fit_lines(self) -> None:
        text = self._report().toMarkdown()
        assert "# Multi-way alignment: 3 latent views" in text
        assert "| baseline | maps | recall@1 | recall@5 | mean rank | mrr |" in text
        assert "| chance | 0 |" in text
        assert "| z_image->z_text |" in text
        assert "`common` fit:" in text and "converged" in text
        assert text.endswith("\n")

    def test_the_report_is_saved_as_json_and_markdown(self, tmp_path: Path) -> None:
        report = self._report()
        report.save(tmp_path / "deep" / "multiway.json")
        report.saveMarkdown(tmp_path / "deep" / "multiway.md")
        loaded = json.loads((tmp_path / "deep" / "multiway.json").read_text(encoding="utf-8"))
        assert loaded["labels"] == report.labels and loaded["num_maps"]["anchor"] == 2
        assert (tmp_path / "deep" / "multiway.md").read_text(
            encoding="utf-8"
        ) == report.toMarkdown()

    def test_a_baseline_the_report_lacks_is_a_key_error_that_lists_the_ones_it_has(self) -> None:
        with pytest.raises(KeyError, match=r"No baseline 'anchor'.*\['identity'\]"):
            evaluateMultiwayAlignment(buildRotatedTripleModel(), VIEWS, _batches(2)).meanRetrieval(
                "anchor"
            )

    def test_two_views_work_too(self) -> None:
        report = evaluateMultiwayAlignment(
            buildRotatedTripleModel(), VIEWS[:2], _batches(2), train=_batches(2, seed=1)
        )
        assert report.meanRetrieval("common")["recall@1"] == pytest.approx(1.0, abs=1e-9)
        assert report.num_maps["pairwise"] == 1 and report.num_maps["common"] == 2
