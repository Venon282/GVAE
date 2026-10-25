"""Unit tests for the functions of `evaluation.paired_alignment` that take plain tensors.

The model-facing parts (`collectModalityPosteriorMeans`, `evaluatePairedAlignment`) are in
`tests/integration/test_paired_evaluation.py`. Here every number is checked against a brute-force
reference written with loops, which is the definition in the module docstring and not an
optimized variant of it.
"""

import math

import pytest
import torch

from global_vae.evaluation.paired_alignment import (
    DEFAULT_RETRIEVAL_KS,
    SIMILARITIES,
    computeAlignmentMetrics,
    computeRetrievalMetrics,
    validateRetrievalSettings,
)


def _embeddings(
    samples: int = 40, dim: int = 5, noise: float = 0.3, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two noisy views of the same latent points."""
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(samples, dim, generator=generator)
    first = latent + noise * torch.randn(samples, dim, generator=generator)
    second = latent + noise * torch.randn(samples, dim, generator=generator)
    return first, second


def _similarity(query: torch.Tensor, candidate: torch.Tensor, similarity: str) -> float:
    if similarity == "cosine":
        norms = query.norm() * candidate.norm()
        return 0.0 if norms == 0 else (query @ candidate / norms).item()
    return -((query - candidate).pow(2).sum()).item()


def _referenceRetrieval(
    queries: torch.Tensor, gallery: torch.Tensor, ks: tuple[int, ...], similarity: str
) -> dict[str, float]:
    """The definition: the partner's rank among ties is uniform over the positions they take."""
    queries, gallery = queries.double(), gallery.double()
    count = queries.shape[0]
    hits = {k: 0.0 for k in ks}
    ranks: list[float] = []
    reciprocals: list[float] = []
    for i in range(count):
        scores = [_similarity(queries[i], gallery[j], similarity) for j in range(count)]
        greater = sum(score > scores[i] + 1e-12 for score in scores)
        tied = sum(abs(score - scores[i]) <= 1e-12 for score in scores) - 1
        positions = range(greater + 1, greater + tied + 2)
        for k in ks:
            hits[k] += sum(position <= k for position in positions) / len(positions)
        ranks.append(sum(positions) / len(positions))
        reciprocals.append(sum(1 / position for position in positions) / len(positions))
    reference = {}
    for k in ks:
        reference[f"recall@{k}"] = hits[k] / count
        reference[f"chance_recall@{k}"] = min(k, count) / count
    ordered = sorted(ranks)
    middle = count // 2
    reference["mean_rank"] = sum(ranks) / count
    reference["median_rank"] = (
        ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    )
    reference["mrr"] = sum(reciprocals) / count
    return reference


class TestValidateRetrievalSettings:
    def test_returns_the_ks_sorted_and_without_duplicates(self) -> None:
        assert validateRetrievalSettings([10, 1, 5, 1], "cosine", 8) == (1, 5, 10)

    def test_the_defaults_are_valid(self) -> None:
        assert validateRetrievalSettings(DEFAULT_RETRIEVAL_KS, "cosine", 1024) == (1, 5, 10)
        for similarity in SIMILARITIES:
            validateRetrievalSettings((1,), similarity, 1)

    def test_rejects_an_empty_ks(self) -> None:
        with pytest.raises(ValueError, match="at least one value of k"):
            validateRetrievalSettings([], "cosine", 8)

    @pytest.mark.parametrize("bad", [0, -3, 1.5, True, "5"])
    def test_rejects_a_k_that_is_not_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            validateRetrievalSettings([1, bad], "cosine", 8)  # type: ignore[list-item]

    def test_rejects_an_unknown_similarity(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown similarity 'dot'.*cosine"):
            validateRetrievalSettings((1,), "dot", 8)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_rejects_a_chunk_size_that_is_not_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            validateRetrievalSettings((1,), "cosine", bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("similarity", SIMILARITIES)
class TestRetrievalAgainstTheDefinition:
    @pytest.mark.parametrize(("samples", "dim"), [(2, 1), (7, 3), (40, 5), (65, 2)])
    def test_matches_the_brute_force_reference(
        self, similarity: str, samples: int, dim: int
    ) -> None:
        first, second = _embeddings(samples, dim, noise=0.5)
        ks = (1, 3, 10)
        result = computeRetrievalMetrics(first, second, ks, similarity)
        expected = _referenceRetrieval(first, second, ks, similarity)
        assert result.keys() == expected.keys()
        for name, value in expected.items():
            assert result[name] == pytest.approx(value, abs=1e-9), name

    @pytest.mark.parametrize("chunk_size", [1, 3, 40, 1000])
    def test_the_chunk_size_changes_nothing(self, similarity: str, chunk_size: int) -> None:
        first, second = _embeddings(40, 5, noise=0.8)
        assert computeRetrievalMetrics(
            first, second, similarity=similarity, chunk_size=chunk_size
        ) == computeRetrievalMetrics(first, second, similarity=similarity, chunk_size=40)

    def test_matches_the_reference_when_many_candidates_tie(self, similarity: str) -> None:
        """Eight samples on three distinct points: ties everywhere, partner included."""
        points = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        queries = points[torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])]
        gallery = points[torch.tensor([0, 1, 1, 2, 2, 0, 0, 1])]
        ks = (1, 2, 4)
        result = computeRetrievalMetrics(queries, gallery, ks, similarity)
        expected = _referenceRetrieval(queries, gallery, ks, similarity)
        for name, value in expected.items():
            assert result[name] == pytest.approx(value, abs=1e-9), name


class TestRetrievalKnownCases:
    def test_a_perfect_embedding_retrieves_every_partner_first(self) -> None:
        points = torch.randn(30, 4, generator=torch.Generator().manual_seed(2))
        result = computeRetrievalMetrics(points, points.clone(), (1, 5))
        assert result["recall@1"] == 1.0
        assert result["recall@5"] == 1.0
        assert result["mean_rank"] == 1.0
        assert result["median_rank"] == 1.0
        assert result["mrr"] == pytest.approx(1.0)

    def test_a_hand_made_pair_with_one_tie(self) -> None:
        """Query 0 ties its partner with another candidate, query 1 ranks its partner second."""
        queries = torch.tensor([[0.0], [10.0]])
        gallery = torch.tensor([[1.0], [-1.0]])
        result = computeRetrievalMetrics(queries, gallery, (1, 2), "l2")
        assert result["recall@1"] == pytest.approx(0.25)  # (1/2 + 0) / 2
        assert result["recall@2"] == pytest.approx(1.0)
        assert result["mean_rank"] == pytest.approx(1.75)  # (1.5 + 2) / 2
        assert result["median_rank"] == pytest.approx(1.75)
        assert result["mrr"] == pytest.approx(0.625)  # ((1 + 1/2) / 2 + 1/2) / 2

    def test_a_worst_case_ranks_the_partner_last(self) -> None:
        """Each query has a gallery row at its own value, which is not its partner."""
        queries = torch.tensor([[0.0], [10.0]])
        gallery = torch.tensor([[10.0], [0.0]])
        result = computeRetrievalMetrics(queries, gallery, (1, 2), "l2")
        assert result["recall@1"] == 0.0
        assert result["recall@2"] == 1.0
        assert result["mean_rank"] == 2.0
        assert result["mrr"] == pytest.approx(0.5)

    def test_a_collapsed_latent_space_scores_chance_and_not_a_perfect_retrieval(self) -> None:
        """Every sample at the same point: all candidates tie with the partner."""
        count = 50
        constant = torch.ones(count, 4) * 0.7
        for similarity in SIMILARITIES:
            result = computeRetrievalMetrics(constant, constant.clone(), (1, 5, 10), similarity)
            for k in (1, 5, 10):
                assert result[f"recall@{k}"] == pytest.approx(k / count)
            assert result["mean_rank"] == pytest.approx((count + 1) / 2)
            assert result["median_rank"] == pytest.approx((count + 1) / 2)
            harmonic = sum(1 / position for position in range(1, count + 1))
            assert result["mrr"] == pytest.approx(harmonic / count)

    def test_zero_vectors_tie_under_cosine_and_do_not_divide_by_zero(self) -> None:
        zeros = torch.zeros(20, 3)
        result = computeRetrievalMetrics(zeros, zeros.clone(), (1,), "cosine")
        assert result["recall@1"] == pytest.approx(1 / 20)

    def test_ties_are_not_broken_by_the_order_of_the_rows(self) -> None:
        constant = torch.ones(10, 2)
        reversed_gallery = constant.flip(0)
        assert computeRetrievalMetrics(constant, reversed_gallery) == computeRetrievalMetrics(
            constant, constant
        )

    def test_a_k_beyond_the_number_of_samples_retrieves_everything(self) -> None:
        first, second = _embeddings(6, 3, noise=3.0)
        result = computeRetrievalMetrics(first, second, (50,))
        assert result["recall@50"] == 1.0
        assert result["chance_recall@50"] == 1.0

    def test_the_chance_level_is_k_over_the_number_of_samples(self) -> None:
        first, second = _embeddings(40)
        result = computeRetrievalMetrics(first, second, (1, 5, 10))
        assert [result[f"chance_recall@{k}"] for k in (1, 5, 10)] == [1 / 40, 5 / 40, 10 / 40]

    def test_recall_never_decreases_with_k_and_stays_between_zero_and_one(self) -> None:
        first, second = _embeddings(60, 4, noise=1.5)
        result = computeRetrievalMetrics(first, second, (1, 2, 5, 20, 60))
        recalls = [result[f"recall@{k}"] for k in (1, 2, 5, 20, 60)]
        assert recalls == sorted(recalls)
        assert all(0.0 <= value <= 1.0 for value in recalls)
        assert recalls[-1] == 1.0

    def test_the_ks_may_come_in_any_order(self) -> None:
        first, second = _embeddings(30)
        shuffled = computeRetrievalMetrics(first, second, (10, 1, 5, 1))
        assert shuffled == computeRetrievalMetrics(first, second, (1, 5, 10))
        assert list(shuffled)[:2] == ["recall@1", "chance_recall@1"]

    def test_the_rank_statistics_are_consistent(self) -> None:
        first, second = _embeddings(50, 3, noise=1.0)
        result = computeRetrievalMetrics(first, second, (1,))
        assert 1.0 <= result["median_rank"] <= 50.0
        assert 1.0 <= result["mean_rank"] <= 50.0
        assert 1.0 / 50 <= result["mrr"] <= 1.0
        assert result["mrr"] >= result["recall@1"] - 1e-12

    def test_noisy_copies_beat_chance_and_independent_points_do_not(self) -> None:
        first, second = _embeddings(300, 8, noise=0.3)
        copies = computeRetrievalMetrics(first, second, (1,))
        independent = computeRetrievalMetrics(
            first, torch.randn(300, 8, generator=torch.Generator().manual_seed(9)), (1,)
        )
        assert copies["recall@1"] > 0.5
        assert independent["recall@1"] < 0.05

    def test_cosine_ignores_the_length_of_each_vector_and_l2_does_not(self) -> None:
        first, second = _embeddings(30, 4, noise=0.6)
        scales = torch.rand(30, 1, generator=torch.Generator().manual_seed(4)) * 9 + 0.5
        plain = computeRetrievalMetrics(first, second, similarity="cosine")
        assert computeRetrievalMetrics(first * scales, second) == pytest.approx(plain)
        assert computeRetrievalMetrics(first, second * scales) == pytest.approx(plain)
        plain_l2 = computeRetrievalMetrics(first, second, similarity="l2")
        assert computeRetrievalMetrics(first * scales, second, similarity="l2") != pytest.approx(
            plain_l2
        )

    def test_float32_and_float16_inputs_are_accepted(self) -> None:
        first, second = _embeddings(20)
        assert computeRetrievalMetrics(first, second) == pytest.approx(
            computeRetrievalMetrics(first.double(), second.double()), abs=1e-9
        )
        half = computeRetrievalMetrics(first.half(), second.half())
        assert 0.0 <= half["recall@1"] <= 1.0

    def test_a_tensor_that_requires_grad_is_accepted(self) -> None:
        first, second = _embeddings(10)
        assert computeRetrievalMetrics(first.requires_grad_(), second)["recall@1"] >= 0.0


class TestRetrievalErrors:
    def test_rejects_inputs_that_are_not_matrices_of_one_shape(self) -> None:
        with pytest.raises(ValueError, match="matrices"):
            computeRetrievalMetrics(torch.zeros(5), torch.zeros(5))
        with pytest.raises(ValueError, match="matrices"):
            computeRetrievalMetrics(torch.zeros(5, 3), torch.zeros(6, 3))
        with pytest.raises(ValueError, match="matrices"):
            computeRetrievalMetrics(torch.zeros(5, 3), torch.zeros(5, 4))

    def test_needs_at_least_two_pairs(self) -> None:
        with pytest.raises(ValueError, match="At least 2 pairs"):
            computeRetrievalMetrics(torch.zeros(1, 3), torch.zeros(1, 3))

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_rejects_a_value_that_is_not_finite(self, bad: float) -> None:
        first, second = _embeddings(10)
        first[3, 1] = bad
        with pytest.raises(ValueError, match="finite"):
            computeRetrievalMetrics(first, second)

    def test_validates_the_settings_before_reading_the_data(self) -> None:
        first, second = _embeddings(10)
        with pytest.raises(ValueError, match="Unknown similarity"):
            computeRetrievalMetrics(first, second, similarity="manhattan")
        with pytest.raises(ValueError, match="at least one value of k"):
            computeRetrievalMetrics(first, second, ks=[])


def _referenceAlignment(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    first, second = first.double(), second.double()
    count = first.shape[0]
    paired_cos: list[float] = []
    paired_sq: list[float] = []
    mismatched_cos: list[float] = []
    mismatched_sq: list[float] = []
    for i in range(count):
        for j in range(count):
            cosine = _similarity(first[i], second[j], "cosine")
            squared = (first[i] - second[j]).pow(2).sum().item()
            (paired_cos if i == j else mismatched_cos).append(cosine)
            (paired_sq if i == j else mismatched_sq).append(squared)
    correlations = [
        torch.corrcoef(torch.stack([first[:, d], second[:, d]]))[0, 1].item()
        for d in range(first.shape[1])
    ]
    return {
        "cosine_paired": sum(paired_cos) / len(paired_cos),
        "cosine_mismatched": sum(mismatched_cos) / len(mismatched_cos),
        "squared_distance_paired": sum(paired_sq) / len(paired_sq),
        "squared_distance_mismatched": sum(mismatched_sq) / len(mismatched_sq),
        "dimension_correlation": sum(correlations) / len(correlations),
    }


class TestAlignmentAgainstTheDefinition:
    @pytest.mark.parametrize(("samples", "dim"), [(2, 1), (7, 3), (25, 6)])
    def test_matches_the_brute_force_reference(self, samples: int, dim: int) -> None:
        first, second = _embeddings(samples, dim, noise=0.8, seed=samples)
        result = computeAlignmentMetrics(first, second)
        for name, value in _referenceAlignment(first, second).items():
            assert result[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name

    def test_derived_numbers_are_consistent_with_the_ones_they_come_from(self) -> None:
        first, second = _embeddings(30, 4, noise=0.5)
        result = computeAlignmentMetrics(first, second)
        assert result["cosine_gap"] == pytest.approx(
            result["cosine_paired"] - result["cosine_mismatched"]
        )
        assert result["squared_distance_ratio"] == pytest.approx(
            result["squared_distance_paired"] / result["squared_distance_mismatched"]
        )
        assert result["dimensions_compared"] == 4.0

    def test_a_mean_that_is_not_centred_does_not_change_the_correlation(self) -> None:
        first, second = _embeddings(40, 4, noise=0.5)
        assert computeAlignmentMetrics(first.double() + 100.0, second.double() - 7.0)[
            "dimension_correlation"
        ] == pytest.approx(computeAlignmentMetrics(first, second)["dimension_correlation"])


class TestAlignmentKnownCases:
    def test_identical_views_are_perfectly_aligned(self) -> None:
        points = torch.randn(30, 4, generator=torch.Generator().manual_seed(3))
        result = computeAlignmentMetrics(points, points.clone())
        assert result["cosine_paired"] == pytest.approx(1.0)
        assert result["squared_distance_paired"] == pytest.approx(0.0, abs=1e-12)
        assert result["squared_distance_ratio"] == pytest.approx(0.0, abs=1e-12)
        assert result["dimension_correlation"] == pytest.approx(1.0)
        assert result["cosine_gap"] > 0.5

    def test_opposite_views_are_perfectly_anti_aligned(self) -> None:
        points = torch.randn(30, 4, generator=torch.Generator().manual_seed(3))
        result = computeAlignmentMetrics(points, -points)
        assert result["cosine_paired"] == pytest.approx(-1.0)
        assert result["dimension_correlation"] == pytest.approx(-1.0)
        assert result["squared_distance_ratio"] > 1.0

    def test_independent_views_look_like_mismatched_ones(self) -> None:
        generator = torch.Generator().manual_seed(5)
        first = torch.randn(600, 6, generator=generator)
        second = torch.randn(600, 6, generator=generator)
        result = computeAlignmentMetrics(first, second)
        assert abs(result["cosine_gap"]) < 0.1
        assert result["squared_distance_ratio"] == pytest.approx(1.0, abs=0.1)
        assert abs(result["dimension_correlation"]) < 0.15

    def test_a_noisier_pair_is_less_aligned(self) -> None:
        clean = computeAlignmentMetrics(*_embeddings(200, 5, noise=0.1))
        noisy = computeAlignmentMetrics(*_embeddings(200, 5, noise=2.0))
        assert clean["cosine_gap"] > noisy["cosine_gap"]
        assert clean["squared_distance_ratio"] < noisy["squared_distance_ratio"]
        assert clean["dimension_correlation"] > noisy["dimension_correlation"]

    def test_a_constant_dimension_is_left_out_of_the_correlation(self) -> None:
        first, second = _embeddings(40, 4)
        first[:, 2] = 1e6  # constant, and far from zero
        second[:, 2] = -3.0
        result = computeAlignmentMetrics(first, second)
        assert result["dimensions_compared"] == 3.0
        assert math.isfinite(result["dimension_correlation"])

    def test_with_no_varying_dimension_the_correlation_is_undefined(self) -> None:
        constant = torch.ones(10, 3)
        result = computeAlignmentMetrics(constant, constant.clone() * 2.0)
        assert result["dimensions_compared"] == 0.0
        assert math.isnan(result["dimension_correlation"])

    def test_if_every_sample_is_the_same_point_the_distance_ratio_is_undefined(self) -> None:
        constant = torch.ones(10, 3)
        result = computeAlignmentMetrics(constant, constant.clone())
        assert result["squared_distance_mismatched"] == pytest.approx(0.0, abs=1e-12)
        assert math.isnan(result["squared_distance_ratio"])

    def test_zero_vectors_have_a_cosine_of_zero(self) -> None:
        zeros = torch.zeros(10, 3)
        result = computeAlignmentMetrics(zeros, zeros.clone())
        assert result["cosine_paired"] == 0.0
        assert result["cosine_mismatched"] == 0.0


class TestAlignmentErrors:
    def test_rejects_inputs_that_are_not_matrices_of_one_shape(self) -> None:
        with pytest.raises(ValueError, match="matrices"):
            computeAlignmentMetrics(torch.zeros(5, 3), torch.zeros(5, 2))

    def test_needs_at_least_two_pairs(self) -> None:
        with pytest.raises(ValueError, match="At least 2 pairs"):
            computeAlignmentMetrics(torch.zeros(1, 3), torch.zeros(1, 3))

    def test_rejects_a_value_that_is_not_finite(self) -> None:
        first, second = _embeddings(10)
        second[0, 0] = math.nan
        with pytest.raises(ValueError, match="finite"):
            computeAlignmentMetrics(first, second)
