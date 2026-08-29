"""Generic fixed-size resampling transform (spec §6.2).

One implementation covering 1D, 2D, 3D (and, by the same mechanism, any
number of spatial dimensions `torch.nn.functional.interpolate` itself
supports): which axes are "spatial" and what size to resample them to are
caller-supplied parameters (`num_spatial_dims`, `size`), never a hardcoded
assumption baked into a "1D version" versus a "2D version" of this class.
This is deliberate: spec §6.2 requires that nothing in this subpackage be
written specifically for one dimensionality or one dataset (e.g. resampling
a SAXS curve to a common *q*-grid, spec §6), so there is exactly one
`ResampleTransform`, reused for a 1D signal, a 2D image, a 3D volume, or a
batch of any of these, purely through configuration.

Unlike `LogTransform`/`StandardizeTransform`, resampling is **not** a
lossless bijection in general: shrinking a signal and then growing it back
discards information (a smaller intermediate representation cannot carry
more detail than it has room for), so `inverse` is a best-effort
reconstruction at the original size, not an exact inverse. This is a
mathematical property of resampling itself, not a shortcut taken here.
`source_size` (the size to resample back to) must be given explicitly for
`inverse` to be usable at all: a single `ResampleTransform` instance has no
way to know, only from `y`, what the original size of any particular sample
was (samples can vary in size before resampling), so guessing one would
silently be wrong for some inputs. If every sample in a dataset does share a
before-resampling size (or the caller only ever wants to view results at one
canonical size), passing that size as `source_size` makes `inverse` usable;
otherwise, tracking each sample's own original size is a data-pipeline
concern outside this transform's scope (`data/NOTE.md`).
"""

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.data.transforms.base import AbstractTransform
from global_vae.data.transforms.registry import registerTransform
from global_vae.utils.stage_config import ShapeLike, resolveSpatialShape

_ALIGN_CORNERS_MODES = frozenset({"linear", "bilinear", "bicubic", "trilinear"})
_DEFAULT_MODE_BY_NDIM = {1: "linear", 2: "bilinear", 3: "trilinear"}


@registerTransform("resample")
class ResampleTransform(AbstractTransform):
    """Resamples the last `num_spatial_dims` axes of a tensor to a fixed size.

    Any leading axes (batch, channel, or anything else the caller's data
    happens to carry) are left untouched: they are flattened into an
    interpolation "batch" dimension internally and restored afterward, so
    this transform works whether it is applied to one unbatched sample or to
    an already-batched tensor, with no shape convention beyond "the last
    `num_spatial_dims` axes are the ones to resample".

    Attributes:
        target_size: Size (or per-axis sizes) `apply` resamples to.
        source_size: Size (or per-axis sizes) `inverse` resamples back
            to. `None` means `inverse` is not usable (see the module
            docstring).
        num_spatial_dims: How many trailing axes are spatial (`1` for
            a plain series, `2` for an image, `3` for a volume, ...).
        mode: `torch.nn.functional.interpolate` interpolation mode.
        align_corners: Forwarded to `interpolate` for modes that
            accept it; ignored otherwise.
    """

    def __init__(
        self,
        target_size: ShapeLike,
        source_size: ShapeLike | None = None,
        num_spatial_dims: int = 1,
        mode: str | None = None,
        align_corners: bool = False,
    ) -> None:
        """Initialize the transform.

        Args:
            target_size: A single `int`, applied to every one of the
                `num_spatial_dims` spatial axes, or an explicit
                `tuple[int, ...]` of length `num_spatial_dims` (see
                `utils.stage_config.resolveSpatialShape`, reused here
                so this class follows the exact same
                shared-or-per-axis convention as every conv building
                block elsewhere in this codebase).
            source_size: As `target_size`, the size `inverse` resamples
                back to. `None` (default) means `inverse` raises
                instead of guessing (see the module docstring).
            num_spatial_dims: Number of trailing axes to resample.
                Must be `1`, `2`, or `3` (the dimensionalities
                `torch.nn.functional.interpolate` supports for a
                spatially-aware mode).
            mode: `torch.nn.functional.interpolate`'s `mode` argument.
                `None` (default) resolves to the natural choice for
                `num_spatial_dims` (`"linear"`, `"bilinear"`, or
                `"trilinear"`); pass `"nearest"` (or any other mode
                `interpolate` accepts for this many spatial dims)
                explicitly to override.
            align_corners: Forwarded to `interpolate`; only meaningful
                for modes in `{"linear", "bilinear", "bicubic",
                "trilinear"}`, silently ignored for others (matching
                `interpolate`'s own contract).

        Raises:
            ValueError: If `num_spatial_dims` is not `1`, `2`, or `3`.
        """
        if num_spatial_dims not in (1, 2, 3):
            raise ValueError(f"num_spatial_dims must be 1, 2, or 3, got {num_spatial_dims}.")
        self.target_size = target_size
        self.source_size = source_size
        self.num_spatial_dims = num_spatial_dims
        self.mode = mode if mode is not None else _DEFAULT_MODE_BY_NDIM[num_spatial_dims]
        self.align_corners = align_corners

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Resample the last `num_spatial_dims` axes of `x` to `target_size`.

        Args:
            x: Tensor with at least `num_spatial_dims` dimensions.

        Returns:
            A tensor with the same leading axes as `x` and its
            trailing `num_spatial_dims` axes resized to `target_size`.

        Raises:
            ValueError: If `x` has fewer than `num_spatial_dims`
                dimensions.
        """
        return self._resampleTo(x, self.target_size)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Resample the last `num_spatial_dims` axes of `y` back to `source_size`.

        Args:
            y: Tensor with at least `num_spatial_dims` dimensions,
                typically the output of `apply`.

        Returns:
            A best-effort reconstruction at `source_size` (see the
            module docstring: not an exact inverse in general).

        Raises:
            ValueError: If `source_size` was not given at construction,
                or if `y` has fewer than `num_spatial_dims` dimensions.
        """
        if self.source_size is None:
            raise ValueError(
                "ResampleTransform.inverse() requires source_size to have been given at "
                "construction time; this instance was built without one. Resampling is "
                "lossy, so there is no way to recover an original size from y alone."
            )
        return self._resampleTo(y, self.source_size)

    def _resampleTo(self, x: torch.Tensor, size: ShapeLike) -> torch.Tensor:
        """Shared resampling body for `apply`/`inverse`.

        Args:
            x: Tensor with at least `num_spatial_dims` dimensions.
            size: Target size for the trailing `num_spatial_dims` axes.

        Returns:
            The resampled tensor, same leading shape as `x`, dtype
            preserved.

        Raises:
            ValueError: If `x` has fewer than `num_spatial_dims`
                dimensions.
        """
        if x.dim() < self.num_spatial_dims:
            raise ValueError(
                f"ResampleTransform(num_spatial_dims={self.num_spatial_dims}) requires an "
                f"input with at least {self.num_spatial_dims} dimension(s), got a tensor "
                f"of shape {tuple(x.shape)}."
            )
        resolved_size = resolveSpatialShape(size, self.num_spatial_dims, "target_size/source_size")

        leading_shape = x.shape[: x.dim() - self.num_spatial_dims]
        spatial_shape = x.shape[x.dim() - self.num_spatial_dims :]
        flattened = x.reshape(-1, 1, *spatial_shape).to(torch.float32)

        if self.mode in _ALIGN_CORNERS_MODES:
            resampled = F.interpolate(
                flattened, size=resolved_size, mode=self.mode, align_corners=self.align_corners
            )
        else:
            resampled = F.interpolate(flattened, size=resolved_size, mode=self.mode)

        return resampled.reshape(*leading_shape, *resolved_size).to(x.dtype)
