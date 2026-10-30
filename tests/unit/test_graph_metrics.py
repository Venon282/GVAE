"""Unit tests for the scores of a reconstructed graph (roadmap P3-8, ADR 0043)."""

import math

import pytest
import torch

from global_vae.utils.graph_metrics import computeAuc, computeEdgeScores


def _bruteForceAuc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """The definition: the chance that a positive outscores a negative, a tie counting a half."""
    positive, negative = scores[labels > 0.5], scores[labels <= 0.5]
    wins = (positive[:, None] > negative[None, :]).float().sum()
    ties = (positive[:, None] == negative[None, :]).float().sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def _graph(edges: list[tuple[int, int]], size: int) -> torch.Tensor:
    matrix = torch.zeros(size, size)
    for i, j in edges:
        matrix[i, j] = matrix[j, i] = 1.0
    return matrix


class TestAuc:
    def test_a_perfect_ranking_is_one_and_a_reversed_one_is_zero(self) -> None:
        scores, labels = torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0, 0, 1, 1])
        assert computeAuc(scores, labels) == 1.0
        assert computeAuc(-scores, labels) == 0.0

    def test_scores_that_are_all_equal_give_one_half(self) -> None:
        assert computeAuc(torch.full((6,), 0.3), torch.tensor([0, 1, 0, 1, 1, 0])) == 0.5

    def test_a_tie_between_a_positive_and_a_negative_counts_a_half(self) -> None:
        scores, labels = torch.tensor([0.5, 0.5, 0.9]), torch.tensor([0, 1, 1])
        assert computeAuc(scores, labels) == pytest.approx(0.75)

    def test_it_matches_the_definition_on_random_scores_with_ties(self) -> None:
        generator = torch.Generator().manual_seed(0)
        scores = torch.randint(0, 6, (200,), generator=generator).float()
        labels = (torch.rand(200, generator=generator) < 0.4).float()
        assert computeAuc(scores, labels) == pytest.approx(_bruteForceAuc(scores, labels))

    def test_it_reads_a_tensor_of_any_shape(self) -> None:
        scores, labels = torch.tensor([[0.1, 0.9], [0.2, 0.8]]), torch.tensor([[0, 1], [0, 1]])
        assert computeAuc(scores, labels) == 1.0

    @pytest.mark.parametrize("label", [0, 1])
    def test_it_is_not_defined_when_every_label_is_the_same(self, label: int) -> None:
        assert math.isnan(computeAuc(torch.rand(5), torch.full((5,), label)))


class TestEdgeScores:
    def test_a_perfect_reconstruction_scores_perfectly(self) -> None:
        adjacency = _graph([(0, 1), (1, 2), (2, 3)], 4)[None]
        scores = computeEdgeScores(adjacency.clone(), adjacency)
        assert scores["auc"] == 1.0
        assert scores["f1"] == 1.0
        assert scores["bce"] < 1e-4

    def test_the_pairs_are_the_upper_triangle_of_each_graph(self) -> None:
        scores = computeEdgeScores(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4))
        assert scores["num_pairs"] == 2 * 6

    def test_a_directed_graph_counts_every_ordered_pair_but_not_the_diagonal(self) -> None:
        scores = computeEdgeScores(torch.zeros(1, 4, 4), torch.zeros(1, 4, 4), directed=True)
        assert scores["num_pairs"] == 12

    def test_the_lower_triangle_and_the_diagonal_of_an_undirected_graph_are_ignored(self) -> None:
        adjacency = _graph([(0, 1), (1, 2)], 4)[None]
        probabilities = adjacency.clone()
        probabilities[0, 3, 0] = probabilities[0, 2, 1] = probabilities[0, 1, 1] = 0.99
        assert computeEdgeScores(probabilities, adjacency)["auc"] == 1.0

    def test_padded_nodes_are_not_pairs(self) -> None:
        """A graph of 3 nodes padded to 5: whatever is said about the padding does not count."""
        adjacency = _graph([(0, 1), (1, 2)], 5)[None]
        probabilities = adjacency.clone()
        probabilities[0, 3:, :] = 0.99
        probabilities[0, :, 3:] = 0.99
        scores = computeEdgeScores(probabilities, adjacency, torch.tensor([3]))
        assert scores["num_pairs"] == 3
        assert scores["auc"] == 1.0
        assert scores["f1"] == 1.0

    def test_without_the_number_of_nodes_the_padding_counts_as_pairs(self) -> None:
        adjacency = _graph([(0, 1), (1, 2)], 5)[None]
        assert computeEdgeScores(adjacency.clone(), adjacency)["num_pairs"] == 10

    def test_each_graph_has_its_own_number_of_nodes(self) -> None:
        scores = computeEdgeScores(torch.zeros(2, 6, 6), torch.zeros(2, 6, 6), torch.tensor([3, 6]))
        assert scores["num_pairs"] == 3 + 15

    def test_the_density_is_the_share_of_pairs_that_are_edges(self) -> None:
        adjacency = _graph([(0, 1), (1, 2), (2, 3)], 4)[None]  # 3 edges of 6 pairs
        assert computeEdgeScores(adjacency.clone(), adjacency)["density"] == 0.5

    def test_the_threshold_decides_the_f1_and_not_the_auc(self) -> None:
        adjacency = _graph([(0, 1), (1, 2)], 4)[None]
        probabilities = adjacency * 0.3 + 0.1  # edges at 0.4, other pairs at 0.1
        low = computeEdgeScores(probabilities, adjacency, threshold=0.3)
        high = computeEdgeScores(probabilities, adjacency, threshold=0.45)
        assert low["f1"] == 1.0
        assert high["f1"] == 0.0
        assert low["auc"] == high["auc"] == 1.0

    def test_a_confident_wrong_answer_has_a_large_but_finite_bce(self) -> None:
        adjacency = _graph([(0, 1)], 3)[None]
        scores = computeEdgeScores(1 - adjacency, adjacency)
        assert 5 < scores["bce"] < 20
        assert math.isfinite(scores["bce"])

    def test_graphs_with_no_edge_have_no_defined_auc_and_an_f1_of_zero(self) -> None:
        scores = computeEdgeScores(torch.full((1, 4, 4), 0.2), torch.zeros(1, 4, 4))
        assert math.isnan(scores["auc"])
        assert scores["f1"] == 0.0

    def test_two_shapes_that_differ_are_refused_with_both(self) -> None:
        with pytest.raises(ValueError, match=r"\(1, 4, 4\) and \(1, 3, 3\)"):
            computeEdgeScores(torch.zeros(1, 4, 4), torch.zeros(1, 3, 3))

    def test_a_tensor_without_a_batch_axis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="batch, N, N"):
            computeEdgeScores(torch.zeros(4, 4), torch.zeros(4, 4))

    def test_a_matrix_that_is_not_square_is_refused(self) -> None:
        with pytest.raises(ValueError, match="square"):
            computeEdgeScores(torch.zeros(1, 4, 5), torch.zeros(1, 4, 5))

    def test_a_number_of_nodes_of_the_wrong_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="num_nodes"):
            computeEdgeScores(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4), torch.tensor([4]))

    def test_graphs_of_less_than_two_nodes_have_nothing_to_score(self) -> None:
        with pytest.raises(ValueError, match="no pair"):
            computeEdgeScores(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4), torch.tensor([1, 0]))
