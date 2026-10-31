"""Point cloud decoder (spec §6 and §10, roadmap P3-8, ADR 0044)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.mlp_blocks import buildMlp

DEFAULT_HIDDEN_DIMS = (256, 256)
CODE_NORMALIZATIONS = (None, "layer")


@registerDecoder("point_cloud_decoder_v1")
class PointCloudDecoder(AbstractDecoder):
    """Decoder that writes a fixed number of points from one latent vector, folding a template.

    Every output point has its own learned code (`num_points` vectors of `code_dim`, trained with
    the rest). The latent vector is copied next to each code, and one MLP shared by all points maps
    each pair to the position of that point. So the latent vector bends a template (the codes) into
    the cloud, like the folding of a grid in FoldingNet, with no grid and no assumption about
    the neighbours of a point. Nothing ties the point at one index to the point at the next, so the
    output has no sequence structure the way a 1D convolutional decoder does, and `num_points` can
    be any positive integer.

    The output is a set in no particular order, so it is compared with the target by a loss that
    does not depend on order, such as the Chamfer distance of `utils/point_cloud_metrics.py`.
    """

    def __init__(
        self,
        latent_dim: int,
        num_points: int,
        out_dim: int = 3,
        code_dim: int = 32,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        activation: str = "relu",
        normalization: str | None = None,
        modality_name: str = "point_cloud",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the latent vector it receives.
            num_points: Number of points to write. Must match the encoder's batches and the
                number of rows of the data the caller's `loader_factory` yields, and it is
                data-dependent, so the config layer never fills it in.
            out_dim: Number of values per point: `3` for coordinates, more if a point carries
                normals or colors.
            code_dim: Width of the learned code of each point.
            hidden_dims: Widths of the MLP shared by the points, from the input outwards. Empty goes
                straight from the latent vector and the code to the output layer.
            activation: Activation inside the MLP, one of
                `global_vae.utils.activations.listSupportedActivations()`.
            normalization: `None` (default) or `"layer"`. `"batch"` is not offered: it would mix the
                points of different clouds.
            modality_name: Name of the modality this decoder reconstructs.

        Raises:
            ValueError: If `latent_dim`, `num_points`, `out_dim` or `code_dim` is below `1`,
                `normalization` is not a known choice, or an argument of the MLP is out of range
                (the message names it).
        """
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("num_points", num_points),
            ("out_dim", out_dim),
            ("code_dim", code_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}.")
        if normalization not in CODE_NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization {normalization!r} for a point cloud. Expected one of "
                f"{list(CODE_NORMALIZATIONS)}; 'batch' would mix the points of different clouds."
            )
        self._latent_dim = latent_dim
        self._num_points = num_points
        self._modality_name = modality_name
        self.codes = nn.Parameter(torch.randn(num_points, code_dim))
        self.body, width = buildMlp(latent_dim + code_dim, hidden_dims, activation, normalization)
        self.to_output = nn.Linear(width, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of point clouds from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Points, shape `(batch, num_points, out_dim)`, unbounded.

        Raises:
            ValueError: If `z` is not of shape `(batch, latent_dim)`.
        """
        if z.dim() != 2 or z.shape[-1] != self._latent_dim:
            raise ValueError(f"z must be (batch, {self._latent_dim}), got {tuple(z.shape)}.")
        batch = z.shape[0]
        per_point = torch.cat(
            [
                z.unsqueeze(1).expand(batch, self._num_points, self._latent_dim),
                self.codes.unsqueeze(0).expand(batch, -1, -1),
            ],
            dim=-1,
        )
        points: torch.Tensor = self.to_output(self.body(per_point))
        return points

    @property
    def modality_name(self) -> str:
        """Name of the modality this decoder reconstructs."""
        return self._modality_name
