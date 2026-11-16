"""Evaluation across two separate latent spaces (`evaluation/latent_alignment.py`, roadmap P6-6).

What is pinned, in the order of the module:

- the views: the text of a view, its resolution against a model (a separate space read from its own
  modality, a shared space read from one modality alone), and the labels of a pair;
- the encode-only pass (ADR 0046) and the single pass that also scores self and cross
  reconstruction;
- the post-hoc orthogonal Procrustes baseline: it recovers a known orthogonal map in closed form,
  centered or not, and one fit serves both directions;
- the pair-integrity check (roadmap I-8): true pairs pass, shuffled pairs and a collapsed space do
  not, and what it refuses;
- the distance table: every registered distance on paired and on mismatched samples, so that a
  distance that is 0 for strangers too (two collapsed spaces) is marked as proving nothing, and the
  global random generator is left as it was;
- the collapse guard, which fires on a deliberately collapsed run;
- the report: its structure, its JSON and Markdown forms, and the shapes that feed the metrics
  table of P3-6.
"""

import json
import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from global_vae.evaluation import latent_alignment
from global_vae.evaluation.latent_alignment import (
    DEFAULT_MIN_ACTIVE_UNITS,
    INTEGRITY_NOT_CHECKED,
    INTEGRITY_OK,
    INTEGRITY_SUSPECT,
    LatentAlignmentReport,
    LatentView,
    PairPosteriors,
    checkCollapse,
    checkPairIntegrity,
    chooseSelectiveK,
    collectPairPosteriors,
    collectPairStatistics,
    computeDistanceTable,
    computeReconstructionGap,
    evaluateLatentAlignment,
    fitOrthogonalProcrustes,
    pairLabels,
    parseLatentView,
    resolveLatentView,
)
from global_vae.evaluation.latent_diagnostics import LatentDiagnostics
from global_vae.models.global_vae import GlobalVae
from tests.integration._script_fixtures import (
    buildTwoModalityDataloaderForScript,
    buildTwoModalityModelForScript,
)
from tests.unit._tiny_pair_model import LATENT, buildTinyModel, tinyBatches

SIGNAL = LatentView("z_signal", ("signal",))
IMAGE = LatentView("z_image", ("image",))


def _alignedModel() -> GlobalVae:
    """A pair model whose two encoders are the same function: row `i` gives the same code twice."""
    model = buildTinyModel(0)
    model.encoders["image"].load_state_dict(model.encoders["signal"].state_dict())
    return model


def _collapse(model: GlobalVae, modalities: tuple[str, ...] = ("signal", "image")) -> GlobalVae:
    """Zero the heads of the encoders: the posterior is the same `N(0, 1)` for every sample."""
    for modality in modalities:
        for name in ("to_mu", "to_logvar"):
            head = cast(nn.Linear, model.encoders[modality].get_submodule(name))
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
    return model


def _linked(count: int = 3, seed: int = 1) -> list[dict[str, torch.Tensor]]:
    """Batches whose image is the signal: the same sample twice, for an aligned model."""
    return [{"signal": b["signal"], "image": b["signal"].clone()} for b in tinyBatches(count, seed)]


def _orthogonal(size: int, seed: int, reflection: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix: torch.Tensor = torch.linalg.qr(
        torch.randn(size, size, generator=generator, dtype=torch.float64)
    ).Q
    if reflection:
        matrix[:, 0] = -matrix[:, 0]
    return matrix


def _posteriors(count: int = 12, noise: float = 0.0) -> PairPosteriors:
    """Posteriors of two views over `count` samples: the second is the first, plus noise."""
    generator = torch.Generator().manual_seed(3)
    mu = torch.randn(count, 4, generator=generator)
    logvar = torch.randn(count, 4, generator=generator) * 0.1
    second_mu = mu + noise * torch.randn(count, 4, generator=generator)
    second_logvar = logvar + noise * 0.1 * torch.randn(count, 4, generator=generator)
    return PairPosteriors(mu, logvar, second_mu, second_logvar)


def _diagnostic(active: int, dims: int = 4) -> LatentDiagnostics:
    return LatentDiagnostics(num_dimensions=dims, active_threshold=0.01, active_units=active)


class TestParseLatentView:
    def test_a_space_name_alone_has_no_inputs(self) -> None:
        assert parseLatentView("z_signal") == ("z_signal", None)

    def test_a_space_read_from_one_modality(self) -> None:
        assert parseLatentView("z_fused@signal") == ("z_fused", ("signal",))

    def test_a_space_read_from_several_modalities(self) -> None:
        assert parseLatentView("z_fused@signal+image") == ("z_fused", ("signal", "image"))

    def test_spaces_around_the_parts_are_dropped(self) -> None:
        assert parseLatentView(" z_fused @ signal + image ") == ("z_fused", ("signal", "image"))

    @pytest.mark.parametrize("text", ["", "   ", "@signal"])
    def test_a_missing_space_name_raises(self, text: str) -> None:
        with pytest.raises(ValueError, match="needs a space name"):
            parseLatentView(text)

    @pytest.mark.parametrize("text", ["z@", "z@signal+", "z@+image"])
    def test_an_empty_modality_raises(self, text: str) -> None:
        with pytest.raises(ValueError, match="empty modality"):
            parseLatentView(text)

    def test_a_view_writes_itself_in_the_form_it_is_read_from(self) -> None:
        view = LatentView("z_fused", ("signal", "image"))
        assert view.toText() == "z_fused@signal+image"
        assert parseLatentView(view.toText()) == ("z_fused", ("signal", "image"))


class TestResolveLatentView:
    def test_a_separate_space_is_read_from_the_modalities_of_its_encoders(self) -> None:
        assert resolveLatentView(buildTinyModel(), "z_signal") == SIGNAL

    def test_a_view_is_accepted_as_it_is(self) -> None:
        assert resolveLatentView(buildTinyModel(), IMAGE) == IMAGE

    def test_a_shared_space_needs_the_modality_to_be_read_from_one(self) -> None:
        model = buildTwoModalityModelForScript()
        both = resolveLatentView(model, "z_fused")
        assert set(both.inputs) == {"x", "y"}
        assert resolveLatentView(model, "z_fused@x") == LatentView("z_fused", ("x",))

    def test_inputs_given_beside_a_text_without_modality_are_used(self) -> None:
        model = buildTwoModalityModelForScript()
        assert resolveLatentView(model, "z_fused", inputs=["y"]) == LatentView("z_fused", ("y",))

    def test_an_unknown_space_lists_the_real_ones(self) -> None:
        with pytest.raises(KeyError, match="z_signal"):
            resolveLatentView(buildTinyModel(), "z_nowhere")

    def test_a_modality_that_does_not_feed_the_space_raises(self) -> None:
        with pytest.raises(ValueError, match="is not fed by the modality"):
            resolveLatentView(buildTinyModel(), "z_signal@image")

    def test_a_repeated_modality_raises(self) -> None:
        with pytest.raises(ValueError, match="repeats a modality"):
            resolveLatentView(buildTwoModalityModelForScript(), "z_fused@x+x")

    def test_no_modality_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one input modality"):
            resolveLatentView(buildTinyModel(), "z_signal", inputs=[])


class TestPairLabels:
    def test_two_spaces_are_named_by_their_spaces(self) -> None:
        assert pairLabels(SIGNAL, IMAGE) == ("z_signal", "z_image")

    def test_one_space_read_from_two_inputs_is_named_by_the_inputs(self) -> None:
        first, second = LatentView("z_fused", ("x",)), LatentView("z_fused", ("y",))
        assert pairLabels(first, second) == ("x", "y")

    def test_the_same_view_twice_raises(self) -> None:
        with pytest.raises(ValueError, match="the same"):
            pairLabels(SIGNAL, SIGNAL)


class TestCollectPairPosteriors:
    def test_each_view_is_its_own_encoder_alone_and_row_i_is_sample_i(self) -> None:
        model = buildTinyModel()
        batches = tinyBatches(2)
        got = collectPairPosteriors(model, batches, SIGNAL, IMAGE)
        assert got.num_samples == 16
        model.eval()
        with torch.no_grad():
            expected = model.encode({"signal": batches[0]["signal"]}, use_mean=True)
        mu, logvar = expected["latent_params"]["z_signal"]
        assert torch.allclose(got.first_mu[:8], mu)
        assert torch.allclose(got.first_logvar[:8], logvar)
        assert got.first_mu.shape == got.second_mu.shape == (16, LATENT)

    def test_max_samples_trims_the_last_batch(self) -> None:
        got = collectPairPosteriors(buildTinyModel(), tinyBatches(3), SIGNAL, IMAGE, max_samples=11)
        assert got.num_samples == 11

    def test_the_model_is_put_in_evaluation_mode(self) -> None:
        model = buildTinyModel()
        model.train()
        collectPairPosteriors(model, tinyBatches(1), SIGNAL, IMAGE)
        assert not model.training

    def test_a_shared_space_is_read_from_one_modality_at_a_time(self) -> None:
        model = buildTwoModalityModelForScript()
        first, second = LatentView("z_fused", ("x",)), LatentView("z_fused", ("y",))
        got = collectPairPosteriors(model, buildTwoModalityDataloaderForScript(), first, second)
        assert not torch.allclose(got.first_mu, got.second_mu)

    def test_a_batch_without_a_modality_raises(self) -> None:
        batches = [{"signal": torch.randn(8, 8)}]
        with pytest.raises(ValueError, match="lacks \\['image'\\]"):
            collectPairPosteriors(buildTinyModel(), batches, SIGNAL, IMAGE)

    def test_an_empty_dataloader_raises(self) -> None:
        with pytest.raises(ValueError, match="no batch"):
            collectPairPosteriors(buildTinyModel(), [], SIGNAL, IMAGE)

    @pytest.mark.parametrize("room", [0, -1])
    def test_a_number_of_samples_below_one_raises(self, room: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            collectPairPosteriors(buildTinyModel(), tinyBatches(1), SIGNAL, IMAGE, max_samples=room)

    def test_two_widths_are_refused_with_the_decision_that_says_why(self) -> None:
        posteriors = PairPosteriors(
            torch.zeros(4, 3), torch.zeros(4, 3), torch.zeros(4, 5), torch.zeros(4, 5)
        )
        with pytest.raises(ValueError, match="D-16"):
            latent_alignment._checkWidths(posteriors, SIGNAL, IMAGE)


class TestFitOrthogonalProcrustes:
    def test_it_recovers_a_known_orthogonal_map(self) -> None:
        rotation = _orthogonal(5, seed=0)
        first = torch.randn(40, 5, dtype=torch.float64)
        fitted = fitOrthogonalProcrustes(first, first @ rotation)
        assert torch.allclose(fitted.rotation, rotation, atol=1e-8)
        assert torch.allclose(fitted.rotation @ fitted.rotation.T, torch.eye(5).double(), atol=1e-8)
        assert fitted.num_pairs == 40
        assert not fitted.center

    def test_a_reflection_is_recovered_too(self) -> None:
        rotation = _orthogonal(4, seed=1, reflection=True)
        first = torch.randn(30, 4, dtype=torch.float64)
        fitted = fitOrthogonalProcrustes(first, first @ rotation)
        assert torch.allclose(fitted.toSecond(first), first @ rotation, atol=1e-8)

    def test_one_fit_serves_both_directions(self) -> None:
        rotation = _orthogonal(4, seed=2)
        first = torch.randn(30, 4, dtype=torch.float64)
        second = first @ rotation
        fitted = fitOrthogonalProcrustes(first, second)
        assert torch.allclose(fitted.toSecond(first), second, atol=1e-8)
        assert torch.allclose(fitted.toFirst(second), first, atol=1e-8)
        assert torch.allclose(fitted.toFirst(fitted.toSecond(first)), first, atol=1e-8)

    def test_a_centered_fit_also_removes_a_constant_offset(self) -> None:
        rotation = _orthogonal(4, seed=3)
        first = torch.randn(50, 4, dtype=torch.float64) + 3.0
        second = (first - first.mean(0)) @ rotation + 7.0
        fitted = fitOrthogonalProcrustes(first, second, center=True)
        assert fitted.center
        assert torch.allclose(fitted.first_mean, first.mean(0))
        assert torch.allclose(fitted.toSecond(first), second, atol=1e-8)
        assert torch.allclose(fitted.toFirst(second), first, atol=1e-8)

    def test_an_uncentered_fit_cannot_remove_an_offset(self) -> None:
        first = torch.randn(50, 4, dtype=torch.float64)
        second = first @ _orthogonal(4, seed=3) + 7.0
        fitted = fitOrthogonalProcrustes(first, second)
        assert not torch.allclose(fitted.toSecond(first), second, atol=1e-3)

    def test_the_fit_is_closed_form_and_repeatable(self) -> None:
        first, second = torch.randn(20, 3), torch.randn(20, 3)
        one, other = fitOrthogonalProcrustes(first, second), fitOrthogonalProcrustes(first, second)
        assert torch.equal(one.rotation, other.rotation)

    def test_the_inputs_are_not_changed(self) -> None:
        first, second = torch.randn(10, 3), torch.randn(10, 3)
        before = first.clone(), second.clone()
        fitOrthogonalProcrustes(first, second, center=True)
        assert torch.equal(first, before[0]) and torch.equal(second, before[1])

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (torch.zeros(4), torch.zeros(4)),
            (torch.zeros(4, 3), torch.zeros(5, 3)),
            (torch.zeros(4, 3), torch.zeros(4, 2)),
        ],
    )
    def test_inputs_that_are_not_two_matrices_of_one_shape_raise(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="one shape"):
            fitOrthogonalProcrustes(first, second)

    def test_no_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one pair"):
            fitOrthogonalProcrustes(torch.zeros(0, 3), torch.zeros(0, 3))

    def test_a_value_that_is_not_finite_raises(self) -> None:
        bad = torch.zeros(4, 3)
        bad[1, 1] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            fitOrthogonalProcrustes(bad, torch.zeros(4, 3))


class TestPairIntegrity:
    def _pairs(self, count: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(5)
        first = torch.randn(count, 4, generator=generator, dtype=torch.float64)
        return first, first @ _orthogonal(4, seed=5)

    def test_true_pairs_pass(self) -> None:
        first, second = self._pairs()
        result = checkPairIntegrity(first, second)
        assert result.status == INTEGRITY_OK
        assert result.recall == pytest.approx(1.0)
        assert result.recall > result.chance_recall
        assert result.num_pairs == 48
        assert result.shuffles == latent_alignment.DEFAULT_INTEGRITY_SHUFFLES
        assert "recovers the partner" in result.message

    def test_pairs_whose_rows_do_not_match_are_suspect(self) -> None:
        first, second = self._pairs()
        permutation = torch.randperm(48, generator=torch.Generator().manual_seed(9))
        result = checkPairIntegrity(first, second[permutation])
        assert result.status == INTEGRITY_SUSPECT
        assert "may not be the same sample" in result.message
        assert "roadmap I-8" in result.message

    def test_pairs_of_independent_noise_are_suspect(self) -> None:
        generator = torch.Generator().manual_seed(6)
        result = checkPairIntegrity(torch.randn(48, 4, generator=generator), torch.randn(48, 4))
        assert result.status == INTEGRITY_SUSPECT

    def test_the_shuffled_control_is_reported(self) -> None:
        first, second = self._pairs()
        result = checkPairIntegrity(first, second, shuffles=7)
        assert result.shuffles == 7
        assert 0.0 <= result.shuffled_recall_mean <= result.shuffled_recall_max <= 1.0
        assert result.recall > result.shuffled_recall_max

    def test_without_the_control_nothing_is_said_about_it(self) -> None:
        first, second = self._pairs()
        result = checkPairIntegrity(first, second, shuffles=0)
        assert result.status == INTEGRITY_OK
        assert math.isnan(result.shuffled_recall_max)
        assert "shuffled" not in result.message

    def test_the_k_that_is_read_is_still_selective(self) -> None:
        first, second = self._pairs(48)
        result = checkPairIntegrity(first, second, ks=[1, 5, 10, 40])
        assert result.k == 10
        assert result.chance_recall == pytest.approx(10 / 48)

    def test_a_k_can_be_asked_for(self) -> None:
        first, second = self._pairs()
        result = checkPairIntegrity(first, second, k=3)
        assert result.k == 3
        assert result.chance_recall == pytest.approx(3 / 48)

    def test_the_same_seed_gives_the_same_check(self) -> None:
        first, second = self._pairs()
        one = checkPairIntegrity(first, second, seed=4)
        other = checkPairIntegrity(first, second, seed=4)
        assert one == other

    def test_a_centered_check_passes_through_an_offset(self) -> None:
        first, second = self._pairs()
        assert checkPairIntegrity(first, second + 5.0, center=True).status == INTEGRITY_OK

    @pytest.mark.parametrize("count", [0, 1])
    def test_fewer_than_two_pairs_are_not_checked(self, count: int) -> None:
        result = checkPairIntegrity(torch.zeros(count, 4), torch.zeros(count, 4))
        assert result.status == INTEGRITY_NOT_CHECKED
        assert result.k is None
        assert result.num_pairs == count

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"shuffles": -1}, "shuffles"),
            ({"margin": -0.1}, "margin"),
            ({"margin": float("nan")}, "margin"),
            ({"k": 0}, "integrity_k"),
            ({"k": True}, "integrity_k"),
            ({"similarity": "dot"}, "similarity"),
        ],
    )
    def test_an_invalid_setting_raises(self, kwargs: dict[str, object], match: str) -> None:
        first, second = self._pairs(10)
        with pytest.raises(ValueError, match=match):
            checkPairIntegrity(first, second, **kwargs)  # type: ignore[arg-type]

    def test_two_sets_of_different_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="one shape"):
            checkPairIntegrity(torch.zeros(5, 3), torch.zeros(6, 3))

    @pytest.mark.parametrize(
        ("ks", "pairs", "expected"),
        [
            ([1, 5, 10], 72, 10),
            ([1, 5, 10], 20, 5),
            ([1, 5, 10], 6, 1),
            ([5, 10], 6, 5),
            ([1, 5, 10], 1000, 10),
        ],
    )
    def test_the_selective_k_is_the_largest_whose_chance_is_at_most_a_quarter(
        self, ks: list[int], pairs: int, expected: int
    ) -> None:
        assert chooseSelectiveK(ks, pairs) == expected


class TestDistanceTable:
    def test_every_registered_distance_is_evaluated_by_default(self) -> None:
        from global_vae.losses.distances.registry import listRegisteredLatentDistances

        values, skipped = computeDistanceTable(_posteriors(noise=0.5))
        assert sorted(values) == sorted(listRegisteredLatentDistances())
        assert skipped == {}

    def test_equal_posteriors_are_closer_than_strangers_for_every_distance(self) -> None:
        values, _ = computeDistanceTable(_posteriors())
        for name, result in values.items():
            assert result.separates, name
            assert result.paired < result.mismatched, name
            assert result.ratio == pytest.approx(result.paired / result.mismatched)
        for name in ("mean_mse", "symmetric_kl", "wasserstein2_diag", "cosine"):
            assert values[name].paired == pytest.approx(0.0, abs=1e-5), name

    def test_a_distance_that_is_zero_for_strangers_too_proves_nothing(self) -> None:
        constant = PairPosteriors(*(torch.zeros(12, 4) for _ in range(4)))
        values, _ = computeDistanceTable(
            constant, ["mean_mse", "symmetric_kl", "wasserstein2_diag"]
        )
        for name, result in values.items():
            assert not result.separates, name
            assert math.isnan(result.ratio), name
            assert result.mismatched == pytest.approx(0.0)

    def test_the_names_pick_the_distances(self) -> None:
        values, _ = computeDistanceTable(_posteriors(), ["cosine"])
        assert list(values) == ["cosine"]

    def test_the_global_random_generator_is_left_as_it_was(self) -> None:
        torch.manual_seed(123)
        before = torch.get_rng_state()
        computeDistanceTable(_posteriors(noise=0.5))
        assert torch.equal(torch.get_rng_state(), before)

    def test_a_stochastic_distance_is_repeatable_for_a_seed(self) -> None:
        posteriors = _posteriors(noise=0.5)
        one, _ = computeDistanceTable(posteriors, ["mixture_jsd"], seed=3)
        other, _ = computeDistanceTable(posteriors, ["mixture_jsd"], seed=3)
        assert one["mixture_jsd"].separates
        assert one["mixture_jsd"].paired == other["mixture_jsd"].paired
        assert one["mixture_jsd"].mismatched == other["mixture_jsd"].mismatched

    def test_the_value_does_not_depend_on_how_samples_are_chunked_for_a_mean_distance(self) -> None:
        posteriors = _posteriors(count=10, noise=0.5)
        whole, _ = computeDistanceTable(posteriors, ["mean_mse"], chunk_size=10)
        chunked, _ = computeDistanceTable(posteriors, ["mean_mse"], chunk_size=3)
        assert whole["mean_mse"].paired == pytest.approx(chunked["mean_mse"].paired)
        assert whole["mean_mse"].mismatched == pytest.approx(chunked["mean_mse"].mismatched)

    def test_a_trailing_chunk_too_small_for_the_distance_is_merged_into_the_one_before(
        self,
    ) -> None:
        assert latent_alignment._chunkBounds(10, 4, minimum=2) == [(0, 4), (4, 8), (8, 10)]
        assert latent_alignment._chunkBounds(9, 4, minimum=2) == [(0, 4), (4, 9)]
        assert latent_alignment._chunkBounds(9, 4, minimum=1) == [(0, 4), (4, 8), (8, 9)]
        assert latent_alignment._chunkBounds(3, 4, minimum=2) == [(0, 3)]

    def test_a_distance_that_needs_more_samples_than_there_are_is_skipped_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from global_vae.losses.distances.registry import getLatentDistanceClass

        real = getLatentDistanceClass("mean_mse")

        class NeedsFour(real):  # type: ignore[valid-type, misc]
            min_batch_size = 4

        monkeypatch.setattr(latent_alignment, "getLatentDistanceClass", lambda name: NeedsFour)
        values, skipped = computeDistanceTable(_posteriors(count=3), ["demanding"])
        assert values == {}
        assert "at least 4 samples" in skipped["demanding"]

    def test_a_distance_that_cannot_be_built_without_arguments_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from global_vae.losses.distances.registry import getLatentDistanceClass

        real = getLatentDistanceClass("mean_mse")

        class NeedsAnArgument(real):  # type: ignore[valid-type, misc]
            def __init__(self, required: int) -> None:
                super().__init__()

        monkeypatch.setattr(
            latent_alignment, "getLatentDistanceClass", lambda name: NeedsAnArgument
        )
        values, skipped = computeDistanceTable(_posteriors(), ["picky"])
        assert values == {}
        assert "without arguments" in skipped["picky"]

    def test_an_unknown_distance_raises_and_lists_the_real_ones(self) -> None:
        with pytest.raises(KeyError, match="cosine"):
            computeDistanceTable(_posteriors(), ["no_such_distance"])

    @pytest.mark.parametrize("chunk_size", [0, -1, True])
    def test_a_chunk_size_that_is_not_a_positive_integer_raises(self, chunk_size: int) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            computeDistanceTable(_posteriors(), chunk_size=chunk_size)

    def test_fewer_than_two_samples_raise(self) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            computeDistanceTable(_posteriors(count=1))


class TestReconstructionGap:
    SELF = {
        "z_signal": {"signal": {"mse": 1.0, "r2": 0.9}},
        "z_image": {"image": {"mse": 2.0, "r2": 0.8}},
    }
    CROSS = {
        "z_signal->z_image": {"image": {"mse": 5.0, "r2": 0.3}},
        "z_image->z_signal": {"signal": {"mse": 1.5, "r2": 0.7}},
    }

    def test_the_gap_is_cross_minus_self_of_the_target_view(self) -> None:
        gap = computeReconstructionGap(self.SELF, self.CROSS)
        assert gap["z_signal->z_image"]["image"] == {"mse": 3.0, "r2": pytest.approx(-0.5)}
        assert gap["z_image->z_signal"]["signal"] == {
            "mse": pytest.approx(0.5),
            "r2": pytest.approx(-0.2),
        }

    def test_a_cell_without_a_self_value_has_no_gap(self) -> None:
        gap = computeReconstructionGap({"z_signal": self.SELF["z_signal"]}, self.CROSS)
        assert list(gap) == ["z_image->z_signal"]

    def test_a_metric_without_a_self_value_is_left_out(self) -> None:
        cross = {"z_signal->z_image": {"image": {"mse": 5.0, "pearson_r": 0.4}}}
        gap = computeReconstructionGap(self.SELF, cross)
        assert gap["z_signal->z_image"]["image"] == {"mse": 3.0}

    def test_nothing_in_nothing_out(self) -> None:
        assert computeReconstructionGap({}, {}) == {}


class TestCheckCollapse:
    def test_a_view_with_fewer_units_than_its_solo_stage_is_flagged(self) -> None:
        checks = checkCollapse(
            {"a": _diagnostic(2), "b": _diagnostic(4)}, reference_active_units={"a": 4, "b": 4}
        )
        assert checks["a"].flagged and not checks["b"].flagged
        assert checks["a"].required == 4
        assert checks["a"].reference_active_units == 4
        assert checks["a"].active_units == 2

    def test_a_view_that_kept_its_units_or_gained_some_is_not_flagged(self) -> None:
        checks = checkCollapse({"a": _diagnostic(3)}, reference_active_units={"a": 3})
        assert not checks["a"].flagged
        checks = checkCollapse({"a": _diagnostic(4)}, reference_active_units={"a": 3})
        assert not checks["a"].flagged

    def test_a_view_without_a_reference_falls_back_to_the_floor(self) -> None:
        checks = checkCollapse({"a": _diagnostic(0), "b": _diagnostic(1)})
        assert checks["a"].flagged and not checks["b"].flagged
        assert checks["a"].required == DEFAULT_MIN_ACTIVE_UNITS
        assert checks["a"].reference_active_units is None

    def test_the_floor_can_be_raised_or_dropped(self) -> None:
        assert checkCollapse({"a": _diagnostic(1)}, min_active_units=2)["a"].flagged
        assert not checkCollapse({"a": _diagnostic(0)}, min_active_units=0)["a"].flagged

    def test_a_reference_beats_the_floor(self) -> None:
        checks = checkCollapse({"a": _diagnostic(1)}, {"a": 0}, min_active_units=3)
        assert not checks["a"].flagged

    def test_a_reference_for_a_view_that_is_not_in_the_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="not views of this pair"):
            checkCollapse({"a": _diagnostic(1)}, {"z": 2})

    @pytest.mark.parametrize("count", [-1, True])
    def test_an_invalid_solo_count_raises(self, count: int) -> None:
        with pytest.raises(ValueError, match="solo active units"):
            checkCollapse({"a": _diagnostic(1)}, {"a": count})

    @pytest.mark.parametrize("floor", [-1, True])
    def test_an_invalid_floor_raises(self, floor: int) -> None:
        with pytest.raises(ValueError, match="min_active_units"):
            checkCollapse({"a": _diagnostic(1)}, min_active_units=floor)


class TestCollectPairStatistics:
    def test_self_and_cross_reconstruction_are_scored_for_the_modalities_of_each_view(self) -> None:
        stats = collectPairStatistics(buildTinyModel(), tinyBatches(2), SIGNAL, IMAGE)
        rec = stats.reconstruction
        assert set(rec.self_reconstruction) == {"z_signal", "z_image"}
        assert set(rec.self_reconstruction["z_signal"]) == {"signal"}
        assert set(rec.cross_reconstruction) == {"z_signal->z_image", "z_image->z_signal"}
        assert set(rec.cross_reconstruction["z_signal->z_image"]) == {"image"}
        assert set(rec.cross_reconstruction["z_image->z_signal"]) == {"signal"}
        assert {"mse", "r2", "pearson_r"} <= set(rec.self_reconstruction["z_signal"]["signal"])
        assert rec.mapped_cross_reconstruction == {}

    def test_the_cross_reconstruction_decodes_the_code_of_one_view_through_the_other_space(
        self,
    ) -> None:
        model = buildTinyModel()
        batches = tinyBatches(1)
        stats = collectPairStatistics(model, batches, SIGNAL, IMAGE)
        model.eval()
        with torch.no_grad():
            code = model.encode({"signal": batches[0]["signal"]}, use_mean=True)["latent_params"][
                "z_signal"
            ][0]
            decoded = model.decode({"z_image": code})["image"]
        expected = torch.mean((decoded - batches[0]["image"]) ** 2).item()
        got = stats.reconstruction.cross_reconstruction["z_signal->z_image"]["image"]["mse"]
        assert got == pytest.approx(expected, rel=1e-5)

    def test_with_a_map_the_carried_code_is_scored_as_well(self) -> None:
        model = _alignedModel()
        mapping = fitOrthogonalProcrustes(torch.randn(10, LATENT), torch.randn(10, LATENT))
        stats = collectPairStatistics(model, _linked(), SIGNAL, IMAGE, mapping=mapping)
        mapped = stats.reconstruction.mapped_cross_reconstruction
        assert set(mapped) == {"z_signal->z_image", "z_image->z_signal"}
        plain = stats.reconstruction.cross_reconstruction["z_signal->z_image"]["image"]["mse"]
        assert mapped["z_signal->z_image"]["image"]["mse"] != pytest.approx(plain)

    def test_the_streaming_metrics_do_not_depend_on_the_batch_size(self) -> None:
        model = buildTinyModel()
        rows = torch.cat([b["signal"] for b in tinyBatches(3)])
        images = torch.cat([b["image"] for b in tinyBatches(3)])
        whole = [{"signal": rows, "image": images}]
        pieces = [{"signal": rows[i : i + 6], "image": images[i : i + 6]} for i in range(0, 24, 6)]
        a = collectPairStatistics(model, whole, SIGNAL, IMAGE).reconstruction
        b = collectPairStatistics(model, pieces, SIGNAL, IMAGE).reconstruction
        for kind in ("self_reconstruction", "cross_reconstruction"):
            for key, per_modality in getattr(a, kind).items():
                for modality, metrics in per_modality.items():
                    for name, value in metrics.items():
                        assert getattr(b, kind)[key][modality][name] == pytest.approx(
                            value, rel=1e-4, abs=1e-6
                        ), (kind, key, modality, name)

    def test_a_shared_space_has_the_same_space_read_from_each_modality(self) -> None:
        model = buildTwoModalityModelForScript()
        first, second = LatentView("z_fused", ("x",)), LatentView("z_fused", ("y",))
        stats = collectPairStatistics(model, buildTwoModalityDataloaderForScript(), first, second)
        rec = stats.reconstruction
        assert set(rec.cross_reconstruction) == {"x->y", "y->x"}
        assert set(rec.cross_reconstruction["x->y"]) == {"y"}
        assert set(rec.self_reconstruction) == {"x", "y"}

    def test_the_same_view_twice_raises(self) -> None:
        with pytest.raises(ValueError, match="the same"):
            collectPairStatistics(buildTinyModel(), tinyBatches(1), SIGNAL, SIGNAL)

    def test_an_empty_dataloader_raises(self) -> None:
        with pytest.raises(ValueError, match="no batch"):
            collectPairStatistics(buildTinyModel(), [], SIGNAL, IMAGE)

    def test_max_samples_trims_and_must_be_positive(self) -> None:
        stats = collectPairStatistics(
            buildTinyModel(), tinyBatches(3), SIGNAL, IMAGE, max_samples=9
        )
        assert stats.posteriors.num_samples == 9
        with pytest.raises(ValueError, match="at least 1"):
            collectPairStatistics(buildTinyModel(), tinyBatches(1), SIGNAL, IMAGE, max_samples=0)


class TestEvaluateLatentAlignment:
    def test_the_report_of_a_pair_with_training_pairs(self) -> None:
        report = evaluateLatentAlignment(
            _alignedModel(), "z_signal", "z_image", _linked(3, 1), train=_linked(6, 2)
        )
        assert isinstance(report, LatentAlignmentReport)
        assert report.labels == ["z_signal", "z_image"]
        assert report.views == {
            "z_signal": {"space": "z_signal", "inputs": ["signal"]},
            "z_image": {"space": "z_image", "inputs": ["image"]},
        }
        assert report.num_heldout == 24
        assert report.num_train == 48
        assert set(report.retrieval) == {"z_signal->z_image", "z_image->z_signal"}
        for metrics in report.retrieval.values():
            for k in report.ks:
                assert f"recall@{k}" in metrics and f"chance_recall@{k}" in metrics
            assert "mean_rank" in metrics
            assert metrics["recall@1"] == pytest.approx(1.0)
            assert metrics["chance_recall@1"] == pytest.approx(1 / 24)
        assert report.procrustes is not None
        assert report.procrustes.num_fit_pairs == 48
        assert report.pair_integrity.status == INTEGRITY_OK
        assert report.flags == []
        assert all(not check.flagged for check in report.collapse.values())

    def test_every_registered_distance_is_reported_against_mismatched_pairs(self) -> None:
        report = evaluateLatentAlignment(_alignedModel(), "z_signal", "z_image", _linked())
        assert sorted(report.distances) == [
            "cosine",
            "info_nce",
            "mean_mse",
            "mixture_jsd",
            "symmetric_kl",
            "wasserstein2_diag",
        ]
        assert all(result.separates for result in report.distances.values())

    def test_the_distances_can_be_picked(self) -> None:
        report = evaluateLatentAlignment(
            _alignedModel(), "z_signal", "z_image", _linked(), distances=["cosine", "mean_mse"]
        )
        assert sorted(report.distances) == ["cosine", "mean_mse"]

    def test_without_training_pairs_the_baseline_and_the_check_are_left_out_and_say_so(
        self,
    ) -> None:
        report = evaluateLatentAlignment(_alignedModel(), "z_signal", "z_image", _linked())
        assert report.procrustes is None
        assert report.num_train == 0
        assert report.pair_integrity.status == INTEGRITY_NOT_CHECKED
        assert "No training pairs" in report.pair_integrity.message
        assert report.flags == []

    def test_a_random_pair_of_spaces_is_at_chance_and_its_pairs_are_suspect(self) -> None:
        report = evaluateLatentAlignment(
            buildTinyModel(), "z_signal", "z_image", tinyBatches(4, 1), train=tinyBatches(6, 2)
        )
        assert report.pair_integrity.status == INTEGRITY_SUSPECT
        assert any(flag.startswith("Pair integrity:") for flag in report.flags)
        recall = report.retrieval["z_signal->z_image"]["recall@1"]
        assert recall <= 0.25

    def test_the_gap_is_cross_minus_self(self) -> None:
        report = evaluateLatentAlignment(buildTinyModel(), "z_signal", "z_image", tinyBatches(2))
        cross = report.cross_reconstruction["z_signal->z_image"]["image"]["mse"]
        own = report.self_reconstruction["z_image"]["image"]["mse"]
        gap = report.reconstruction_gap["z_signal->z_image"]["image"]["mse"]
        assert gap == pytest.approx(cross - own)

    def test_max_samples_limits_the_heldout_pairs_and_max_train_samples_the_training_pairs(
        self,
    ) -> None:
        report = evaluateLatentAlignment(
            _alignedModel(),
            "z_signal",
            "z_image",
            _linked(3),
            train=_linked(6, 2),
            max_samples=10,
            max_train_samples=30,
        )
        assert report.num_heldout == 10
        assert report.num_train == 30

    def test_the_heldout_data_is_read_once_and_the_training_data_is_read_once(self) -> None:
        reads = {"heldout": 0, "train": 0}

        def counted(name: str, batches: list[dict[str, torch.Tensor]]):  # type: ignore[no-untyped-def]
            for batch in batches:
                reads[name] += 1
                yield batch

        evaluateLatentAlignment(
            _alignedModel(),
            "z_signal",
            "z_image",
            counted("heldout", _linked(3)),
            train=counted("train", _linked(2, 2)),
        )
        assert reads == {"heldout": 3, "train": 2}

    def test_the_pair_can_be_given_as_views_or_as_text(self) -> None:
        model = _alignedModel()
        by_text = evaluateLatentAlignment(model, "z_signal@signal", "z_image@image", _linked())
        by_view = evaluateLatentAlignment(model, SIGNAL, IMAGE, _linked())
        assert by_text.retrieval == by_view.retrieval

    def test_a_shared_space_is_compared_with_itself_read_from_two_modalities(self) -> None:
        report = evaluateLatentAlignment(
            buildTwoModalityModelForScript(),
            "z_fused@x",
            "z_fused@y",
            buildTwoModalityDataloaderForScript(),
        )
        assert report.labels == ["x", "y"]
        assert set(report.retrieval) == {"x->y", "y->x"}
        assert report.views["x"] == {"space": "z_fused", "inputs": ["x"]}

    def test_the_same_seed_gives_the_same_report(self) -> None:
        one = evaluateLatentAlignment(
            buildTinyModel(), "z_signal", "z_image", tinyBatches(3, 1), train=tinyBatches(3, 2)
        )
        other = evaluateLatentAlignment(
            buildTinyModel(), "z_signal", "z_image", tinyBatches(3, 1), train=tinyBatches(3, 2)
        )
        assert one.toDict() == other.toDict()

    def test_the_model_is_left_in_evaluation_mode_and_unchanged(self) -> None:
        model = _alignedModel()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        model.train()
        evaluateLatentAlignment(model, "z_signal", "z_image", _linked(), train=_linked(2, 2))
        assert not model.training
        assert all(torch.equal(before[name], t) for name, t in model.state_dict().items())

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"first": "z_nowhere"}, KeyError, "z_nowhere"),
            ({"second": "z_signal"}, ValueError, "different widths|the same|not fed"),
            ({"distances": ["no_such"]}, KeyError, "no_such"),
            ({"ks": [0]}, ValueError, "positive integer"),
            ({"similarity": "dot"}, ValueError, "similarity"),
            ({"chunk_size": 0}, ValueError, "chunk_size"),
            ({"active_unit_threshold": -1.0}, ValueError, "threshold"),
            ({"integrity_margin": -1.0, "train": _linked(2)}, ValueError, "margin"),
        ],
    )
    def test_invalid_arguments_raise(
        self, kwargs: dict[str, object], error: type[Exception], match: str
    ) -> None:
        arguments: dict[str, object] = {"first": "z_signal", "second": "z_image"}
        arguments.update(kwargs)
        with pytest.raises(error, match=match):
            evaluateLatentAlignment(
                _alignedModel(),
                arguments.pop("first"),  # type: ignore[arg-type]
                arguments.pop("second"),  # type: ignore[arg-type]
                _linked(),
                **arguments,  # type: ignore[arg-type]
            )

    def test_a_batch_without_a_modality_of_the_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="carry the modalities of one sample"):
            evaluateLatentAlignment(
                _alignedModel(), "z_signal", "z_image", [{"signal": torch.randn(8, 8)}]
            )

    def test_fewer_than_two_heldout_pairs_raise(self) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            evaluateLatentAlignment(
                _alignedModel(), "z_signal", "z_image", _linked(), max_samples=1
            )


class TestTheCollapseGuardFires:
    """The done-criterion of P6-6: the guard fires on a deliberately collapsed run."""

    def test_two_collapsed_spaces_are_flagged_and_their_distances_prove_nothing(self) -> None:
        model = _collapse(_alignedModel())
        report = evaluateLatentAlignment(
            model, "z_signal", "z_image", _linked(), train=_linked(4, 2)
        )
        assert all(diagnostic.active_units == 0 for diagnostic in report.diagnostics.values())
        assert all(check.flagged for check in report.collapse.values())
        assert any(flag.startswith("Collapse: 'z_signal'") for flag in report.flags)
        assert any(flag.startswith("Collapse: 'z_image'") for flag in report.flags)
        assert not report.distances["mean_mse"].separates
        assert math.isnan(report.distances["mean_mse"].ratio)
        assert report.distances["mean_mse"].paired == pytest.approx(0.0)
        assert any("0 for mismatched pairs too" in flag for flag in report.flags)

    def test_a_collapsed_pair_cannot_pass_the_integrity_check_and_retrieves_at_chance(self) -> None:
        model = _collapse(_alignedModel())
        report = evaluateLatentAlignment(
            model, "z_signal", "z_image", _linked(), train=_linked(4, 2)
        )
        assert report.pair_integrity.status == INTEGRITY_SUSPECT
        for metrics in report.retrieval.values():
            for k in report.ks:
                assert metrics[f"recall@{k}"] == pytest.approx(metrics[f"chance_recall@{k}"])

    def test_one_collapsed_space_is_named(self) -> None:
        model = _collapse(_alignedModel(), ("image",))
        report = evaluateLatentAlignment(model, "z_signal", "z_image", _linked())
        assert not report.collapse["z_signal"].flagged
        assert report.collapse["z_image"].flagged
        assert [flag for flag in report.flags if flag.startswith("Collapse")] == [
            flag for flag in report.flags if "'z_image'" in flag and flag.startswith("Collapse")
        ]

    def test_a_space_with_fewer_units_than_its_solo_stage_is_flagged_even_above_the_floor(
        self,
    ) -> None:
        model = _alignedModel()
        baseline = evaluateLatentAlignment(model, "z_signal", "z_image", _linked())
        active = baseline.diagnostics["z_signal"].active_units
        assert active >= 1
        report = evaluateLatentAlignment(
            model,
            "z_signal",
            "z_image",
            _linked(),
            reference_active_units={"z_signal": active + 1},
        )
        assert report.collapse["z_signal"].flagged
        assert report.collapse["z_signal"].reference_active_units == active + 1
        assert not report.collapse["z_image"].flagged
        assert any(f"its solo stage had {active + 1}" in flag for flag in report.flags)

    def test_the_same_run_against_its_own_solo_count_is_not_flagged(self) -> None:
        model = _alignedModel()
        first = evaluateLatentAlignment(model, "z_signal", "z_image", _linked())
        solo = {label: d.active_units for label, d in first.diagnostics.items()}
        again = evaluateLatentAlignment(
            model, "z_signal", "z_image", _linked(), reference_active_units=solo
        )
        assert again.flags == []


class TestReportForms:
    def _report(self) -> LatentAlignmentReport:
        return evaluateLatentAlignment(
            _alignedModel(),
            "z_signal",
            "z_image",
            _linked(),
            train=_linked(4, 2),
            reference_active_units={"z_signal": LATENT + 1},
        )

    def test_the_summary_names_the_pair_the_flags_and_the_numbers_to_read(self) -> None:
        text = self._report().summary()
        assert text.startswith("Latent alignment: z_signal")
        assert "FLAG: Collapse: 'z_signal'" in text
        assert "retrieval z_signal->z_image: recall@1=" in text
        assert "(chance " in text
        assert "distance cosine: paired=" in text
        assert "gap to self" in text
        assert "active units [z_signal]" in text
        assert "procrustes baseline (fitted on 32 pairs)" in text
        assert "pair integrity: ok" in text

    def test_the_summary_says_when_nothing_was_flagged(self) -> None:
        report = evaluateLatentAlignment(_alignedModel(), "z_signal", "z_image", _linked())
        assert "no flag raised" in report.summary()

    def test_the_json_form_is_plain_and_keeps_every_section(self, tmp_path: Path) -> None:
        report = self._report()
        path = tmp_path / "nested" / "latent_alignment.json"
        report.save(path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "labels",
            "views",
            "retrieval",
            "alignment",
            "distances",
            "self_reconstruction",
            "cross_reconstruction",
            "reconstruction_gap",
            "diagnostics",
            "collapse",
            "procrustes",
            "pair_integrity",
            "flags",
        ):
            assert key in loaded, key
        assert loaded["distances"]["cosine"].keys() >= {
            "paired",
            "mismatched",
            "ratio",
            "separates",
        }
        assert loaded["collapse"]["z_signal"]["flagged"] is True
        assert loaded["pair_integrity"]["status"] == INTEGRITY_OK

    def test_the_markdown_form_has_a_section_for_each_thing_to_read(self, tmp_path: Path) -> None:
        report = self._report()
        text = report.toMarkdown()
        for needle in (
            "retrieval",
            "chance",
            "distance",
            "cross",
            "active units",
            "Procrustes",
            "integrity",
            "Collapse",
        ):
            assert needle.lower() in text.lower(), needle
        path = tmp_path / "report.md"
        report.saveMarkdown(path)
        assert path.read_text(encoding="utf-8") == text

    def test_the_markdown_form_without_training_pairs_says_the_baseline_was_not_run(self) -> None:
        report = evaluateLatentAlignment(_alignedModel(), "z_signal", "z_image", _linked())
        text = report.toMarkdown()
        assert "No training pairs were given" in text
        assert "Procrustes baseline (held-out pairs)" not in text

    def test_the_text_has_no_em_dash_and_no_unicode_arrow(self) -> None:
        text = self._report().summary() + self._report().toMarkdown()
        assert chr(0x2014) not in text
        assert not any(chr(arrow) in text for arrow in (0x2192, 0x2190, 0x21D2))

    def test_the_cross_reconstruction_is_keyed_by_the_inputs_of_its_source_view(self) -> None:
        report = evaluateLatentAlignment(buildTinyModel(), "z_signal", "z_image", tinyBatches(2))
        by_subset = report.crossReconstructionBySubset()
        assert set(by_subset) == {frozenset({"signal"}), frozenset({"image"})}
        assert set(by_subset[frozenset({"signal"})]) == {"image"}
        assert set(by_subset[frozenset({"image"})]) == {"signal"}
        gaps = report.gapBySubset()
        assert set(gaps) == set(by_subset)
        assert (
            gaps[frozenset({"signal"})]["image"]["mse"]
            == report.reconstruction_gap["z_signal->z_image"]["image"]["mse"]
        )
