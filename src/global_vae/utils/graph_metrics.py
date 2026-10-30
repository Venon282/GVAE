"""Scores of a reconstructed graph over the pairs of nodes it really has (ADR 0043).

A batch of graphs is a dense adjacency tensor `(batch, N, N)`, padded with isolated nodes up to the
largest graph. The metrics of `evaluate` read it as an image, so they count the padding (a block of
zeros that is trivially right) and the diagonal, and they score an edge as a pixel.
`computeEdgeScores` scores what a graph model is for, whether an edge is there or not, over the
pairs of real nodes only.
"""

import torch
import torch.nn.functional as functional


def computeAuc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the area under the ROC curve, with tied scores counted as one half.

    Args:
        scores: A float tensor of any shape, higher meaning more likely to be positive.
        labels: A tensor of the same number of values, `1` for a positive and `0` for a negative.

    Returns:
        The probability that a random positive scores above a random negative, or `nan` when
        `labels` has no positive or no negative (the area is not defined then).
    """
    flat_scores = scores.flatten().double()
    positive = labels.flatten() > 0.5
    num_positive = int(positive.sum())
    num_negative = positive.numel() - num_positive
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    values, inverse, counts = torch.unique(flat_scores, return_inverse=True, return_counts=True)
    last_rank = counts.cumsum(0).double()
    average_rank = last_rank - (counts.double() - 1) / 2
    ranks = average_rank[inverse]
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative)


def computeEdgeScores(
    probabilities: torch.Tensor,
    adjacency: torch.Tensor,
    num_nodes: torch.Tensor | None = None,
    directed: bool = False,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Score the edges of reconstructed graphs against the graphs, over pairs of real nodes.

    A pair counts when both its nodes exist (index below the graph's `num_nodes`) and they are two
    different nodes. For an undirected graph (default) each pair counts once, from the upper
    triangle; with `directed` each ordered pair counts.

    Args:
        probabilities: Edge probabilities in `[0, 1]`, `(batch, N, N)`, such as the output of
            `graph_cnn_decoder_v1`.
        adjacency: The graphs, `(batch, N, N)`, `1` for an edge and `0` for none.
        num_nodes: The number of nodes of each graph, `(batch,)`. `None` means every graph has all
            `N` nodes, with no padding.
        directed: Whether the graph is directed.
        threshold: Probability above which an edge is predicted, for `f1`.

    Returns:
        A dict with `bce` (the mean negative log-likelihood of a pair), `auc` (`nan` if the pairs
        hold only edges or only non-edges), `f1` (of the edges, at `threshold`), `density` (the
        share of the pairs that are edges) and `num_pairs`.

    Raises:
        ValueError: If the two tensors do not have one shape `(batch, N, N)`, or `num_nodes` is not
            `(batch,)`.
    """
    if probabilities.shape != adjacency.shape or probabilities.dim() != 3:
        raise ValueError(
            "probabilities and adjacency must have one shape (batch, N, N), got "
            f"{tuple(probabilities.shape)} and {tuple(adjacency.shape)}."
        )
    batch, size = adjacency.shape[0], adjacency.shape[1]
    if adjacency.shape[2] != size:
        raise ValueError(f"A graph is a square matrix, got shape {tuple(adjacency.shape)}.")
    if num_nodes is None:
        num_nodes = torch.full((batch,), size, device=adjacency.device)
    if num_nodes.shape != (batch,):
        raise ValueError(f"num_nodes must have shape ({batch},), got {tuple(num_nodes.shape)}.")
    index = torch.arange(size, device=adjacency.device)
    exists = index[None, :] < num_nodes[:, None]
    pair = exists[:, :, None] & exists[:, None, :]
    distinct = index[:, None] != index[None, :]
    if not directed:
        distinct = index[:, None] < index[None, :]
    pair = pair & distinct[None]
    predicted = probabilities[pair].clamp(1e-6, 1 - 1e-6)
    truth = adjacency[pair].to(predicted.dtype)
    if truth.numel() == 0:
        raise ValueError("There is no pair of nodes to score: every graph has fewer than 2 nodes.")
    bce = float(functional.binary_cross_entropy(predicted, truth))
    guess = (predicted > threshold).to(truth.dtype)
    true_positive = float((guess * truth).sum())
    denominator = float(guess.sum() + truth.sum())
    f1 = 2 * true_positive / denominator if denominator > 0 else 0.0
    return {
        "bce": bce,
        "auc": computeAuc(predicted, truth),
        "f1": f1,
        "density": float(truth.mean()),
        "num_pairs": float(truth.numel()),
    }
