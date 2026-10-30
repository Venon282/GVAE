"""Graph decoder: the 2D CNN decoder with edge probabilities as output (roadmap P3-8, ADR 0043)."""

from typing import Any

import torch

from global_vae.decoders.registry import registerDecoder
from global_vae.decoders.two_d_cnn_decoder import TwoDCnnDecoder


@registerDecoder("graph_cnn_decoder_v1")
class GraphCnnDecoder(TwoDCnnDecoder):
    """`TwoDCnnDecoder` that returns the probability of an edge between every pair of nodes.

    A graph of at most `N` nodes is a dense adjacency matrix `(batch, N, N)`, which is an image to
    the 2D convolutions (ADR 0043), so this class reuses `TwoDCnnDecoder` whole: every argument of
    that class is accepted and passed on, and the `output_shape` must be square. It changes only
    the end of `forward`. The scores of the convolutions are averaged with their transpose, so the
    graph is undirected (`symmetric`), squashed by a sigmoid into probabilities, and the diagonal is
    set to exactly `0` (`zero_diagonal`), so a node has no edge to itself.

    The sigmoid is the reason for the class. The loss names of a training config compare a
    reconstruction with the batch tensor of the same name, and `bce` refuses values outside `[0,
    1]`, which the raw output of `TwoDCnnDecoder` is not. With probabilities,
    `reconstruction_loss: bce` with the `sum` reduction is the Bernoulli likelihood of the
    edges, the best loss measured in ADR 0043, and it runs from `scripts/train.py`. The quality
    of the reconstruction is that of the parent class, within the noise of the measurements.

    With `out_channels` above `1`, the output is `(batch, out_channels, N, N)`, one adjacency
    matrix per type of edge, and every channel is symmetrized and zeroed on its diagonal.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int] | list[int],
        symmetric: bool = True,
        zero_diagonal: bool = True,
        modality_name: str = "graph",
        **kwargs: Any,
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the latent vector it receives.
            output_shape: `(N, N)`, the padded number of nodes on both axes.
            symmetric: Whether the graph is undirected, so that the edge from `i` to `j` and the
                one from `j` to `i` are the same probability. `False` keeps a directed graph.
            zero_diagonal: Whether the probability of an edge from a node to itself is exactly
                `0`. `False` allows self-loops.
            modality_name: Name of the modality this decoder reconstructs.
            **kwargs: Every other argument of `TwoDCnnDecoder` (`out_channels`, `hidden_channels`,
                `seed_shape`, `kernel_sizes`, `paddings`, ...), unchanged.

        Raises:
            ValueError: If `output_shape` is not square, or as `TwoDCnnDecoder` for a
                configuration that does not produce exactly `output_shape`.
        """
        shape = tuple(output_shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"A graph is a square adjacency matrix, so output_shape must be (N, N), "
                f"got {shape}."
            )
        super().__init__(
            latent_dim=latent_dim,
            output_shape=output_shape,
            modality_name=modality_name,
            **kwargs,
        )
        self._symmetric = symmetric
        self._zero_diagonal = zero_diagonal

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of graphs from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Edge probabilities, `(batch, N, N)` for one channel or `(batch, out_channels, N, N)`,
            within `[0, 1]`, symmetric and with a zero diagonal unless turned off.
        """
        scores = super().forward(z)
        if self._symmetric:
            scores = (scores + scores.transpose(-1, -2)) / 2
        probabilities = torch.sigmoid(scores)
        if self._zero_diagonal:
            size = probabilities.shape[-1]
            off_diagonal = 1.0 - torch.eye(size, dtype=probabilities.dtype, device=z.device)
            probabilities = probabilities * off_diagonal
        return probabilities
