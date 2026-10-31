"""Point cloud encoder (spec §6 and §10, roadmap P3-8, ADR 0044)."""

from collections.abc import Sequence

import torch

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.mlp_blocks import buildMlp
from global_vae.utils.point_cloud_blocks import POOLING_CHOICES, poolPoints, splitPoints
from global_vae.utils.transformer_blocks import buildPoolingHead

DEFAULT_POINT_DIMS = (64, 128, 256)
POINT_NORMALIZATIONS = (None, "layer")


@registerEncoder("point_cloud_encoder_v1")
class PointCloudEncoder(AbstractEncoder):
    """PointNet-style encoder of a set of points, blind to their order and to padding.

    The same small MLP reads every point on its own (`point_dims`), and the features of the points
    are pooled with a max, a mean or both (`pooling`), which are symmetric functions: the same cloud
    in any order gives the same `(mu, logvar)`. A row of `NaN` is a point that does not exist, left
    out of the pooling, so a cloud padded with rows of `NaN` to the length of the batch also gives
    the same `(mu, logvar)`, and a cloud with no point at all stays finite. The pooled vector goes
    through an optional MLP and two linear layers.

    The cost is linear in the number of points, where attention is quadratic. Nothing in it sees
    the neighbours of a point, so it reads the distribution of the points and not local structure.
    """

    def __init__(
        self,
        latent_dim: int,
        in_dim: int = 3,
        point_dims: Sequence[int] = DEFAULT_POINT_DIMS,
        head_hidden_dims: Sequence[int] = (256,),
        activation: str = "relu",
        normalization: str | None = None,
        pooling: str = "max_mean",
        modality_name: str = "point_cloud",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            in_dim: Number of values per point: `3` for coordinates, more if a point carries
                normals or colors, which are read together with the position.
            point_dims: Widths of the per-point MLP, from the input outwards. At least one.
            head_hidden_dims: Widths of an optional MLP between the pooled vector and the `mu` and
                `logvar` layers. Empty goes straight to them.
            activation: Activation inside the per-point MLP, one of
                `global_vae.utils.activations.listSupportedActivations()`.
            normalization: `None` (default) or `"layer"` inside the per-point MLP. `"batch"` is not
                offered: it would mix the points of different clouds.
            pooling: `"max_mean"` (default), `"max"` or `"mean"`.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If `latent_dim` or `in_dim` is below `1`, `point_dims` is empty, `pooling`
                or `normalization` is not a known choice, or another argument is out of range (the
                message names it).
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if in_dim < 1:
            raise ValueError(f"in_dim must be at least 1, got {in_dim}.")
        if len(point_dims) == 0:
            raise ValueError("point_dims needs at least one layer, got an empty sequence.")
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"Unknown pooling '{pooling}'. Expected one of {POOLING_CHOICES}.")
        if normalization not in POINT_NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization {normalization!r} for a point cloud. Expected one of "
                f"{list(POINT_NORMALIZATIONS)}; 'batch' would mix the points of different clouds."
            )
        self._latent_dim = latent_dim
        self._in_dim = in_dim
        self._pooling = pooling
        self._modality_name = modality_name
        self.point_mlp, width = buildMlp(in_dim, point_dims, activation, normalization)
        pooled_width = width * 2 if pooling == "max_mean" else width
        self.head, self.to_mu, self.to_logvar = buildPoolingHead(
            pooled_width, head_hidden_dims, latent_dim
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of point clouds.

        Args:
            x: A float tensor `(batch, points, in_dim)`, with a row of `NaN` where a cloud has no
                point. `points` is at least `1` and may differ from batch to batch.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `x` is not a float tensor of shape `(batch, points, in_dim)`.
        """
        filled, valid = splitPoints(x, self._in_dim)
        pooled = self.head(poolPoints(self.point_mlp(filled), valid, self._pooling))
        mu: torch.Tensor = self.to_mu(pooled)
        logvar: torch.Tensor = self.to_logvar(pooled)
        return mu, logvar

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the `(mu, logvar)` output."""
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        """Name of the modality this encoder handles."""
        return self._modality_name

    @property
    def minimal_input_length(self) -> int:
        """Smallest cloud this encoder accepts: `1` point.

        Nothing is strided or pooled over a window, so no number of points collapses a feature map.

        Returns:
            `1`.
        """
        return 1
