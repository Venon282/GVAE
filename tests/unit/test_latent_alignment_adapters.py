"""The pair evaluation of spaces of different widths (`evaluation/latent_alignment.py`, P6-8).

What is pinned: the orthogonal map and the integrity check between spaces of different widths (a
partial isometry), the two collectors that keep the posteriors at their own widths and skip what a
difference of widths makes impossible, `adaptPosteriors`, and `evaluateLatentAlignment` with
`adapters`: what it reads adapted and what it reads as the model has it, the notes and fields of
the report, the collapse guard on an adapter that shrinks its output, and each refusal.

The model of most tests has `z_text` of width 5 holding an embedding of `z_signal`, which has width
3 (`mu_text = mu_signal E` with `E` of orthonormal rows), and an adapter whose weight is `E`: read
through it, the two views are the same posterior, so retrieval is exact and the distance is zero.
"""

import json
from typing import cast

import pytest
import torch

from global_vae.evaluation.latent_alignment import (
    LatentView,
    PairPosteriors,
    adaptPosteriors,
    checkPairIntegrity,
    collectPairPosteriors,
    collectPairStatistics,
    evaluateLatentAlignment,
    fitOrthogonalProcrustes,
)
from global_vae.heads.linear import LinearHead
from global_vae.losses.terms.adapters import AdapterSpec, buildAdapter
from global_vae.models.global_vae import GlobalVae
from tests.integration._latent_alignment_fixtures import (
    EMBEDDED_WIDTHS,
    buildEmbeddedModel,
    buildTripleBatches,
    buildTripleModel,
    embeddingMatrix,
)

SIGNAL = LatentView("z_signal", ("signal",))
TEXT = LatentView("z_text", ("text",))
IMAGE = LatentView("z_image", ("image",))
WIDTHS = EMBEDDED_WIDTHS


def _embeddedModel() -> tuple[GlobalVae, LinearHead]:
    """`z_text` holds `mu_signal E` (and the log-variance likewise); the adapter undoes it."""
    model = buildEmbeddedModel()
    embedding = embeddingMatrix()
    adapter = cast(LinearHead, buildAdapter(model, "z_text", AdapterSpec("linear", 3), "t"))
    with torch.no_grad():
        adapter.to_mu.weight.copy_(embedding)
        adapter.to_mu.bias.zero_()
        adapter.to_logvar.weight.copy_(embedding)
        adapter.to_logvar.bias.zero_()
    return model, adapter


def _heldout() -> list[dict[str, torch.Tensor]]:
    return buildTripleBatches(3, seed=2)


def _train() -> list[dict[str, torch.Tensor]]:
    return buildTripleBatches(4, seed=1)


class TestProcrustesBetweenDifferentWidths:
    def _pair(
        self, narrow: int = 3, wide: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(3)
        q = torch.linalg.qr(torch.randn(wide, narrow, generator=generator, dtype=torch.float64)).Q
        rotation = q.T  # (narrow, wide), orthonormal rows
        first = torch.randn(60, narrow, generator=generator, dtype=torch.float64)
        return first, first @ rotation, rotation

    def test_an_embedding_of_a_narrow_space_in_a_wide_one_is_recovered(self) -> None:
        first, second, rotation = self._pair()
        fitted = fitOrthogonalProcrustes(first, second)
        assert fitted.rotation.shape == (3, 5)
        assert torch.allclose(fitted.rotation, rotation, atol=1e-8)
        assert torch.allclose(fitted.toSecond(first), second, atol=1e-8)

    def test_the_rows_of_the_map_are_orthonormal_and_the_way_back_projects(self) -> None:
        first, second, _ = self._pair()
        fitted = fitOrthogonalProcrustes(first, second)
        assert torch.allclose(fitted.rotation @ fitted.rotation.T, torch.eye(3).double(), atol=1e-8)
        assert torch.allclose(fitted.toFirst(second), first, atol=1e-8)

    def test_the_wide_space_first_gives_orthonormal_columns(self) -> None:
        first, second, rotation = self._pair()
        fitted = fitOrthogonalProcrustes(second, first)
        assert fitted.rotation.shape == (5, 3)
        assert torch.allclose(fitted.rotation, rotation.T, atol=1e-8)
        assert torch.allclose(fitted.rotation.T @ fitted.rotation, torch.eye(3).double(), atol=1e-8)

    def test_centering_removes_an_offset_between_spaces_of_different_widths(self) -> None:
        first, second, _ = self._pair()
        shifted = second + torch.arange(5, dtype=torch.float64)
        fitted = fitOrthogonalProcrustes(first, shifted, center=True)
        assert torch.allclose(fitted.toSecond(first), shifted, atol=1e-8)

    def test_the_integrity_check_runs_between_spaces_of_different_widths(self) -> None:
        first, second, _ = self._pair()
        ok = checkPairIntegrity(first.float(), second.float(), ks=(1, 5), shuffles=5)
        assert ok.status == "ok" and ok.recall == pytest.approx(1.0)
        wrong = checkPairIntegrity(
            first.float(), second.float().flip(0), ks=(1, 5), shuffles=5, margin=0.05
        )
        assert wrong.status == "suspect"


class TestCollectorsWithDifferentWidths:
    def test_by_default_two_widths_are_refused_on_the_first_batch_with_the_way_out(self) -> None:
        with pytest.raises(ValueError, match=r"width 3.*width 5.*D-16.*adapters"):
            collectPairPosteriors(buildTripleModel(WIDTHS), _heldout(), SIGNAL, TEXT)
        with pytest.raises(ValueError, match=r"width 3.*width 5.*D-16.*adapters"):
            collectPairStatistics(buildTripleModel(WIDTHS), _heldout(), SIGNAL, TEXT)

    def test_the_posteriors_can_be_kept_at_their_own_widths(self) -> None:
        got = collectPairPosteriors(
            buildTripleModel(WIDTHS), _heldout(), SIGNAL, TEXT, require_equal_widths=False
        )
        assert got.first_mu.shape == (96, 3) and got.second_mu.shape == (96, 5)
        assert got.second_logvar.shape == (96, 5)

    def test_without_equal_widths_the_cross_reconstruction_is_skipped_and_the_self_is_not(
        self,
    ) -> None:
        model = buildTripleModel(WIDTHS)
        mapping = fitOrthogonalProcrustes(
            *(lambda p: (p.first_mu, p.second_mu))(
                collectPairPosteriors(model, _train(), SIGNAL, TEXT, require_equal_widths=False)
            )
        )
        stats = collectPairStatistics(
            model, _heldout(), SIGNAL, TEXT, mapping=mapping, require_equal_widths=False
        )
        reconstruction = stats.reconstruction
        assert reconstruction.cross_reconstruction == {}
        assert set(reconstruction.self_reconstruction) == {"z_signal", "z_text"}
        assert set(reconstruction.mapped_cross_reconstruction) == {
            "z_signal->z_text",
            "z_text->z_signal",
        }

    def test_equal_widths_still_compute_the_cross_reconstruction_when_not_required(self) -> None:
        model = buildTripleModel({"signal": 3, "image": 3, "text": 3})
        stats = collectPairStatistics(model, _heldout(), SIGNAL, IMAGE, require_equal_widths=False)
        assert set(stats.reconstruction.cross_reconstruction) == {
            "z_signal->z_image",
            "z_image->z_signal",
        }


class TestAdaptPosteriors:
    def test_the_adapted_view_changes_width_and_the_other_is_returned_as_it_is(self) -> None:
        model, adapter = _embeddedModel()
        raw = collectPairPosteriors(model, _heldout(), SIGNAL, TEXT, require_equal_widths=False)
        adapted = adaptPosteriors(raw, SIGNAL, TEXT, {"z_text": adapter})
        assert adapted.first_mu is raw.first_mu
        assert adapted.second_mu.shape == (96, 3)
        assert torch.allclose(adapted.second_mu, raw.first_mu, atol=1e-5)
        assert torch.allclose(adapted.second_logvar, raw.first_logvar, atol=1e-5)

    def test_the_ids_checked_are_carried_over(self) -> None:
        zeros = torch.zeros(4, 3)
        raw = PairPosteriors(zeros, zeros, zeros, zeros, sample_ids_checked=4)
        assert adaptPosteriors(raw, SIGNAL, IMAGE, {}).sample_ids_checked == 4

    def test_no_gradient_is_kept(self) -> None:
        model, adapter = _embeddedModel()
        raw = collectPairPosteriors(model, _heldout(), SIGNAL, TEXT, require_equal_widths=False)
        assert not adaptPosteriors(raw, SIGNAL, TEXT, {"z_text": adapter}).second_mu.requires_grad


class TestEvaluateWithAdapters:
    def _report(self, **kwargs: object):  # type: ignore[no-untyped-def]
        model, adapter = _embeddedModel()
        options: dict[str, object] = {
            "train": _train(),
            "adapters": {"z_text": adapter},
            "ks": (1, 5),
            "integrity_shuffles": 5,
        }
        options.update(kwargs)
        return evaluateLatentAlignment(model, "z_signal", "z_text", _heldout(), **options)  # type: ignore[arg-type]

    def test_without_adapters_two_widths_are_refused(self) -> None:
        model, _ = _embeddedModel()
        with pytest.raises(ValueError, match=r"D-16.*adapters"):
            evaluateLatentAlignment(model, "z_signal", "z_text", _heldout())

    def test_read_through_the_adapter_the_two_views_are_the_same_posterior(self) -> None:
        report = self._report()
        for metrics in report.retrieval.values():
            assert metrics["recall@1"] == pytest.approx(1.0)
        for name, result in report.distances.items():
            assert result.separates, name
        # `info_nce` is contrastive: it is not zero on identical posteriors, by construction.
        for name in ("mean_mse", "wasserstein2_diag", "symmetric_kl", "cosine"):
            assert report.distances[name].paired == pytest.approx(0.0, abs=1e-4)

    def test_the_diagnostics_are_read_as_the_model_has_them_and_adapted(self) -> None:
        report = self._report()
        assert report.diagnostics["z_signal"].num_dimensions == 3
        assert report.diagnostics["z_text"].num_dimensions == 5
        assert report.diagnostics["z_text (adapted)"].num_dimensions == 3
        assert set(report.collapse) == {"z_signal", "z_text", "z_text (adapted)"}

    def test_the_procrustes_baseline_and_the_integrity_check_read_the_raw_spaces(self) -> None:
        report = self._report()
        assert report.procrustes is not None
        assert report.procrustes.retrieval["z_signal->z_text"]["recall@1"] == pytest.approx(1.0)
        assert report.pair_integrity.status == "ok"
        assert report.pair_integrity.recall == pytest.approx(1.0)

    def test_the_cross_reconstruction_is_skipped_and_the_mapped_one_is_there(self) -> None:
        report = self._report()
        assert report.cross_reconstruction == {} and report.reconstruction_gap == {}
        assert report.procrustes is not None
        assert set(report.procrustes.reconstruction_gap) == set(
            report.procrustes.cross_reconstruction
        )
        assert set(report.procrustes.cross_reconstruction) == {
            "z_signal->z_text",
            "z_text->z_signal",
        }

    def test_the_report_says_what_was_adapted_and_what_was_not_computed(self) -> None:
        report = self._report()
        assert report.widths == {"z_signal": 3, "z_text": 5}
        assert report.adapters["z_text"]["head"] == "LinearHead"
        assert (report.adapters["z_text"]["in_dim"], report.adapters["z_text"]["out_dim"]) == (5, 3)
        assert report.adapters["z_text"]["parameters"] == 2 * (5 * 3 + 3)
        text = " ".join(report.notes)
        assert "read through the adapter of 'z_text' (LinearHead, 5 -> 3)" in text
        assert "widths 3 and 5" in text and "partial isometry" in text

    def test_the_notes_are_in_the_summary_and_the_markdown(self) -> None:
        report = self._report()
        assert "  note: 'z_text' is read through the adapter" in report.summary()
        assert "- 'z_text' is read through the adapter" in report.toMarkdown()
        assert "z_text (adapted)" in report.toMarkdown()

    def test_the_report_is_json_serializable_with_its_new_fields(self) -> None:
        loaded = json.loads(json.dumps(self._report().toDict()))
        assert loaded["adapters"]["z_text"]["out_dim"] == 3 and loaded["widths"]["z_text"] == 5

    def test_without_training_data_the_adapted_numbers_are_still_read(self) -> None:
        report = self._report(train=None)
        assert report.procrustes is None
        assert report.pair_integrity.status == "not_checked"
        assert report.retrieval["z_signal->z_text"]["recall@1"] == pytest.approx(1.0)

    def test_an_adapter_that_shrinks_its_output_is_flagged_by_the_collapse_guard(self) -> None:
        model, adapter = _embeddedModel()
        with torch.no_grad():
            adapter.to_mu.weight.zero_()
        report = evaluateLatentAlignment(
            model, "z_signal", "z_text", _heldout(), adapters={"z_text": adapter}
        )
        assert report.collapse["z_text (adapted)"].flagged
        assert any("Collapse: 'z_text (adapted)'" in flag for flag in report.flags)
        assert report.retrieval["z_signal->z_text"]["recall@1"] < 0.5

    def test_the_spread_ratio_compares_per_dimension_so_that_two_widths_compare(self) -> None:
        # The embedding keeps the total spread: 5 dimensions before the adapter, 3 after it.
        ratio = self._report().adapters["z_text"]["spread_ratio"]
        assert ratio == pytest.approx(5 / 3, rel=1e-3)

    def test_an_adapter_that_keeps_its_spread_raises_no_adapter_flag(self) -> None:
        assert not [flag for flag in self._report().flags if flag.startswith("Adapter:")]

    def test_an_adapter_that_shrinks_its_output_is_flagged_by_its_spread(self) -> None:
        model, adapter = _embeddedModel()
        with torch.no_grad():
            adapter.to_mu.weight.mul_(0.1)
        report = evaluateLatentAlignment(
            model, "z_signal", "z_text", _heldout(), adapters={"z_text": adapter}
        )
        assert report.adapters["z_text"]["spread_ratio"] == pytest.approx(5 / 3 * 0.01, rel=1e-3)
        flagged = [flag for flag in report.flags if flag.startswith("Adapter:")]
        assert len(flagged) == 1 and "'z_text'" in flagged[0] and "retrieval" in flagged[0]
        # The retrieval does not change with the scale: the flag is about the distance.
        assert report.retrieval["z_signal->z_text"]["recall@1"] == pytest.approx(1.0)

    def test_the_floor_can_be_lowered_to_silence_the_flag(self) -> None:
        model, adapter = _embeddedModel()
        with torch.no_grad():
            adapter.to_mu.weight.mul_(0.1)
        report = evaluateLatentAlignment(
            model,
            "z_signal",
            "z_text",
            _heldout(),
            adapters={"z_text": adapter},
            min_adapter_spread=0.0,
        )
        assert not [flag for flag in report.flags if flag.startswith("Adapter:")]

    @pytest.mark.parametrize("floor", [-0.1, float("nan"), float("inf")])
    def test_a_floor_that_is_not_a_finite_non_negative_number_is_refused(
        self, floor: float
    ) -> None:
        model, adapter = _embeddedModel()
        with pytest.raises(ValueError, match="min_adapter_spread"):
            evaluateLatentAlignment(
                model,
                "z_signal",
                "z_text",
                _heldout(),
                adapters={"z_text": adapter},
                min_adapter_spread=floor,
            )

    def test_adapters_that_leave_the_widths_unequal_are_refused_by_name(self) -> None:
        model, _ = _embeddedModel()
        wrong = buildAdapter(model, "z_text", AdapterSpec("linear", 4), "t")
        with pytest.raises(ValueError, match=r"width 3.*width 4.*once the adapters have read"):
            evaluateLatentAlignment(
                model, "z_signal", "z_text", _heldout(), adapters={"z_text": wrong}
            )

    def test_an_adapter_of_a_space_that_is_not_in_the_pair_is_refused(self) -> None:
        model, adapter = _embeddedModel()
        with pytest.raises(ValueError, match=r"names \['z_image'\].*'z_signal' and 'z_text'"):
            evaluateLatentAlignment(
                model, "z_signal", "z_text", _heldout(), adapters={"z_image": adapter}
            )

    def test_equal_widths_with_an_adapter_read_the_cross_reconstruction_as_before(self) -> None:
        model = buildTripleModel({"signal": 3, "image": 3, "text": 3})
        adapter = buildAdapter(model, "z_image", AdapterSpec("linear", 3), "t")
        report = evaluateLatentAlignment(
            model, "z_signal", "z_image", _heldout(), adapters={"z_image": adapter}
        )
        assert set(report.cross_reconstruction) == {"z_signal->z_image", "z_image->z_signal"}
        assert len(report.notes) == 2 and "was not computed" not in " ".join(report.notes)

    def test_a_report_without_adapters_has_no_notes_and_empty_adapters(self) -> None:
        model = buildTripleModel({"signal": 3, "image": 3, "text": 3})
        report = evaluateLatentAlignment(model, "z_signal", "z_image", _heldout())
        assert report.adapters == {} and report.notes == []
        assert report.widths == {"z_signal": 3, "z_image": 3}
