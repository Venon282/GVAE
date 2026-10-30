"""Synthetic graphs of several families and sizes, for the graph tests.

Each graph is drawn from one of four families, in a canonical node order, so that a model that
reconstructs the adjacency matrix has something to learn: a stochastic block model of two blocks
(the first block's nodes come first), a ring lattice, a random recursive tree (a node is attached to
an earlier one) and an Erdos-Renyi graph. It is a stand-in for a dataset: no real graph dataset is
named anywhere in this framework (`data/NOTE.md`).
"""

import random

import torch

N_MAX = 16
N_MIN = 8
FAMILIES = ("block", "ring", "tree", "random")


def _blockGraph(n: int, rng: random.Random) -> list[list[int]]:
    first = max(2, round(n * rng.uniform(0.3, 0.7)))
    p_in, p_out = rng.uniform(0.6, 0.9), rng.uniform(0.03, 0.2)
    adjacency = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < first) == (j < first)
            if rng.random() < (p_in if same else p_out):
                adjacency[i][j] = adjacency[j][i] = 1
    return adjacency


def _ringGraph(n: int, rng: random.Random) -> list[list[int]]:
    reach = rng.randint(1, 3)
    adjacency = [[0] * n for _ in range(n)]
    for i in range(n):
        for step in range(1, reach + 1):
            j = (i + step) % n
            adjacency[i][j] = adjacency[j][i] = 1
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.02:
                adjacency[i][j] = adjacency[j][i] = 1 - adjacency[i][j]
    return adjacency


def _treeGraph(n: int, rng: random.Random) -> list[list[int]]:
    adjacency = [[0] * n for _ in range(n)]
    for i in range(1, n):
        j = rng.randrange(i)
        adjacency[i][j] = adjacency[j][i] = 1
    return adjacency


def _randomGraph(n: int, rng: random.Random) -> list[list[int]]:
    p = rng.uniform(0.1, 0.5)
    adjacency = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adjacency[i][j] = adjacency[j][i] = 1
    return adjacency


_BUILDERS = (_blockGraph, _ringGraph, _treeGraph, _randomGraph)


def shuffleNodes(adjacency: torch.Tensor, num_nodes: torch.Tensor, seed: int) -> torch.Tensor:
    """Put the real nodes of every graph in a random order, leaving the padding where it is."""
    generator = torch.Generator().manual_seed(seed)
    shuffled = adjacency.clone()
    for index in range(adjacency.shape[0]):
        n = int(num_nodes[index])
        order = torch.randperm(n, generator=generator)
        shuffled[index, :n, :n] = adjacency[index, :n, :n][order][:, order]
    return shuffled


def makeGraphs(
    count: int, seed: int, variable: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build `count` graphs as padded dense adjacency matrices.

    Args:
        count: Number of graphs.
        seed: Seed of the generator, so a call is reproducible.
        variable: Whether the number of nodes varies, from `N_MIN` to `N_MAX`. `False` gives
            graphs of `N_MAX` nodes, so there is no padding.

    Returns:
        `(adjacency, num_nodes, family)`: the matrices `(count, N_MAX, N_MAX)`, with zeros in the
        rows and columns of the nodes a graph does not have (a padded node is an isolated one), the
        number of nodes of each graph, and the index of its family in `FAMILIES`.
    """
    rng = random.Random(seed)
    adjacency = torch.zeros(count, N_MAX, N_MAX)
    num_nodes = torch.zeros(count, dtype=torch.long)
    family = torch.zeros(count, dtype=torch.long)
    for index in range(count):
        n = rng.randint(N_MIN, N_MAX) if variable else N_MAX
        kind = rng.randrange(len(_BUILDERS))
        adjacency[index, :n, :n] = torch.tensor(_BUILDERS[kind](n, rng), dtype=torch.float)
        num_nodes[index] = n
        family[index] = kind
    return adjacency, num_nodes, family
