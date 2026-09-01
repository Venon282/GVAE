"""Generic resampling / regridding transform (spec §6.2).

One implementation covering two genuinely different needs, both expressed
through the same class rather than as separate, dimensionality- or
domain-specific tools:

1. **"Just change the point count"** (`interpolation="torch"`, the default).
   Treats the input as sampled on an implicit, evenly-spaced index grid and
   resamples it to a new evenly-spaced grid of a different size, via
   `torch.nn.functional.interpolate`. Fast, has no extra dependency beyond
   torch, and works for 1D/2D/3D data through `num_spatial_dims`. This is
   exactly what the original version of this class did.

2. **"Resample onto specific x-positions"** (`interpolation="scipy"`).
   The case (1) cannot express: if two samples were measured at genuinely
   different positions along an axis (e.g. two curves recorded on different
   x-grids, of different length or range), resampling both to the *same
   number of points* via (1) does **not** put them on the same grid: index
   `n` of one curve and index `n` of the other still do not correspond to
   the same physical position, since (1) has no notion of position at all,
   only of point count. This mode takes explicit `source_coords`/
   `target_coords` (1D position arrays) and a choice of interpolation
   method from `scipy.interpolate` (`scipy_kind`: linear, nearest, and
   several polynomial/spline flavors: `"cubic_spline"`, `"pchip"`,
   `"akima"`, or any `scipy.interpolate.interp1d` `kind`), so that
   resampling genuinely means "evaluate this curve's fitted interpolant at
   these positions", not "reinterpret this many points as evenly spaced".
   `scipy` is a soft dependency of this mode only (imported lazily, only
   when `interpolation="scipy"` is actually used); the default
   `interpolation="torch"` mode needs nothing beyond torch.

Coordinates that are the **same for every sample** (e.g. a shared,
non-uniformly-spaced instrument grid) can be given once at construction
(`source_coords`/`target_coords`). Coordinates that **differ per sample**
(the case that originally motivated this mode: two curves genuinely
recorded at different positions) cannot be expressed as a single batched
tensor operation, since each sample would need its own position array; for
that case, `apply`/`inverse` accept `source_coords`/`target_coords` as
optional **per-call** overrides, meant to be used once per (unbatched)
sample from the caller's own per-sample loading code, before batching
samples of differing native grids into one tensor.

Only `num_spatial_dims=1` supports coordinate-aware resampling today:
scattered/rectilinear coordinate grids in 2D/3D are a materially larger
feature (which axes' coordinates, separable vs. fully scattered, ...) that
is not implemented; `interpolation="scipy"` combined with
`num_spatial_dims != 1` raises `NotImplementedError` rather than silently
resampling incorrectly. The evenly-spaced `interpolation="torch"` mode is
unaffected and still works at any `num_spatial_dims`.

As with every transform in this subpackage, none of this is specific to
any one dataset: `source_coords`/`target_coords` are plain position arrays
the caller supplies, with no assumption about what they represent (a
scattering vector, a wavelength, a timestamp, ...).
"""

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.data.transforms.base import AbstractTransform
from global_vae.data.transforms.registry import registerTransform
from global_vae.utils.stage_config import ShapeLike, resolveSpatialShape

_ALIGN_CORNERS_MODES = frozenset({"linear", "bilinear", "bicubic", "trilinear"})
_DEFAULT_MODE_BY_NDIM = {1: "linear", 2: "bilinear", 3: "trilinear"}
_SPLINE_KINDS = frozenset({"cubic_spline", "pchip", "akima"})
_INTERP1D_KINDS = frozenset(
    {"linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"}
)

CoordsLike = torch.Tensor | Sequence[float]


def _toCoordsArray(coords: CoordsLike, name: str) -> np.ndarray:
    """Convert a caller-supplied coordinate array to a validated 1D `np.ndarray`.

    Args:
        coords: A 1D `torch.Tensor` or a plain sequence of floats.
        name: Parameter name, used only for the error message.

    Returns:
        A 1D `float64` NumPy array.

    Raises:
        ValueError: If `coords` is not 1-dimensional.
    """
    array = (
        coords.detach().cpu().numpy().astype(np.float64)
        if isinstance(coords, torch.Tensor)
        else np.asarray(coords, dtype=np.float64)
    )
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {array.shape}.")
    return array


@registerTransform("resample")
class ResampleTransform(AbstractTransform):
    """Resamples the last `num_spatial_dims` axes of a tensor to a fixed size.

    Any leading axes (batch, channel, or anything else the caller's data
    happens to carry) are left untouched: they are flattened into an
    interpolation "batch" dimension internally and restored afterward, so
    this transform works whether it is applied to one unbatched sample or to
    an already-batched tensor sharing one coordinate grid.

    Attributes:
        target_size: Size (or per-axis sizes) `apply` resamples to, when no
            explicit `target_coords` is given.
        source_size: Size (or per-axis sizes) `inverse` resamples back to
            (`interpolation="torch"` only).
        num_spatial_dims: How many trailing axes are spatial.
        mode: `torch.nn.functional.interpolate` interpolation mode
            (`interpolation="torch"` only).
        align_corners: Forwarded to `interpolate` for modes that accept it
            (`interpolation="torch"` only).
        interpolation: `"torch"` (evenly-spaced grids, the default) or
            `"scipy"` (explicit coordinates and/or a richer choice of
            interpolation method; see the module docstring).
        source_coords: Shared source positions used by every `apply`/
            `inverse` call that does not pass its own (`interpolation=
            "scipy"` only). `None` defaults to an implicit evenly-spaced
            grid `0, 1, ..., length - 1`.
        target_coords: As `source_coords`, for the positions `apply`
            resamples onto. `None` defaults to `target_size` evenly-spaced
            points spanning the resolved source range.
        scipy_kind: Interpolation method (`interpolation="scipy"` only):
            any `scipy.interpolate.interp1d` `kind` (`"linear"`,
            `"nearest"`, `"nearest-up"`, `"zero"`, `"slinear"`,
            `"quadratic"`, `"cubic"`, `"previous"`, `"next"`), or one of
            `"cubic_spline"` (`scipy.interpolate.CubicSpline`), `"pchip"`
            (`scipy.interpolate.PchipInterpolator`, monotonicity-preserving),
            `"akima"` (`scipy.interpolate.Akima1DInterpolator`, robust to
            outliers).
        extrapolate: Whether resampling onto positions outside the source
            range is allowed (`interpolation="scipy"` only). `False`
            (default) raises a clear error instead of silently
            extrapolating or returning `nan`.
    """

    def __init__(
        self,
        target_size: ShapeLike | None = None,
        source_size: ShapeLike | None = None,
        num_spatial_dims: int = 1,
        mode: str | None = None,
        align_corners: bool = False,
        interpolation: str = "torch",
        source_coords: CoordsLike | None = None,
        target_coords: CoordsLike | None = None,
        scipy_kind: str = "linear",
        extrapolate: bool = False,
    ) -> None:
        """Initialize the transform.

        Args:
            target_size: A single `int`, applied to every one of the
                `num_spatial_dims` spatial axes, or an explicit
                `tuple[int, ...]` of length `num_spatial_dims` (see
                `utils.stage_config.resolveSpatialShape`). Only meaningful
                when `target_coords` is not given (`target_coords`, when
                given, determines the output size directly). Required
                unless `target_coords` is given.
            source_size: As `target_size`, the size `inverse` resamples
                back to. `interpolation="torch"` only; `interpolation=
                "scipy"` uses `source_coords` for the same purpose instead.
            num_spatial_dims: Number of trailing axes to resample. Must be
                `1`, `2`, or `3`; `interpolation="scipy"` additionally
                requires exactly `1` (see the module docstring).
            mode: `torch.nn.functional.interpolate`'s `mode` argument
                (`interpolation="torch"` only). `None` (default) resolves
                to the natural choice for `num_spatial_dims` (`"linear"`,
                `"bilinear"`, or `"trilinear"`).
            align_corners: Forwarded to `interpolate` (`interpolation=
                "torch"` only); only meaningful for modes in `{"linear",
                "bilinear", "bicubic", "trilinear"}`, silently ignored for
                others (matching `interpolate`'s own contract).
            interpolation: `"torch"` (default) or `"scipy"`; see the module
                docstring for the difference.
            source_coords: Shared source positions (`interpolation=
                "scipy"` only). Overridable per-call via `apply`'s own
                `source_coords` argument for samples whose positions differ
                from one another.
            target_coords: Shared target positions (`interpolation=
                "scipy"` only), overridable the same way.
            scipy_kind: See the class docstring (`interpolation="scipy"`
                only).
            extrapolate: See the class docstring (`interpolation="scipy"`
                only).

        Raises:
            ValueError: If `num_spatial_dims` is not `1`, `2`, or `3`; if
                `interpolation` is not `"torch"` or `"scipy"`; if
                `source_coords`/`target_coords` are given with
                `interpolation="torch"`; if `scipy_kind` is not a
                recognized method; or if neither `target_size` nor
                `target_coords` is given.
            NotImplementedError: If `interpolation="scipy"` is combined
                with `num_spatial_dims != 1`.
        """
        if num_spatial_dims not in (1, 2, 3):
            raise ValueError(f"num_spatial_dims must be 1, 2, or 3, got {num_spatial_dims}.")
        if interpolation not in ("torch", "scipy"):
            raise ValueError(f"interpolation must be 'torch' or 'scipy', got '{interpolation}'.")
        if interpolation == "torch" and (source_coords is not None or target_coords is not None):
            raise ValueError(
                "source_coords/target_coords require interpolation='scipy'; with the default "
                "interpolation='torch', resampling only ever knows point counts, not positions."
            )
        if interpolation == "scipy" and num_spatial_dims != 1:
            raise NotImplementedError(
                "interpolation='scipy' (coordinate-aware resampling) is only implemented for "
                "num_spatial_dims=1. interpolation='torch' still works at any num_spatial_dims "
                "if evenly-spaced resampling (no explicit coordinates) is enough."
            )
        if interpolation == "scipy" and scipy_kind not in (_INTERP1D_KINDS | _SPLINE_KINDS):
            available = ", ".join(sorted(_INTERP1D_KINDS | _SPLINE_KINDS))
            raise ValueError(f"Unknown scipy_kind '{scipy_kind}'. Available: {available}.")
        if target_size is None and target_coords is None:
            raise ValueError("At least one of target_size or target_coords must be given.")

        self.target_size = target_size
        self.source_size = source_size
        self.num_spatial_dims = num_spatial_dims
        self.mode = mode if mode is not None else _DEFAULT_MODE_BY_NDIM[num_spatial_dims]
        self.align_corners = align_corners
        self.interpolation = interpolation
        self.source_coords = (
            _toCoordsArray(source_coords, "source_coords") if source_coords is not None else None
        )
        self.target_coords = (
            _toCoordsArray(target_coords, "target_coords") if target_coords is not None else None
        )
        self.scipy_kind = scipy_kind
        self.extrapolate = extrapolate

    def apply(
        self,
        x: torch.Tensor,
        source_coords: CoordsLike | None = None,
        target_coords: CoordsLike | None = None,
    ) -> torch.Tensor:
        """Resample the last `num_spatial_dims` axes of `x` toward `target_size`/`target_coords`.

        Args:
            x: Tensor with at least `num_spatial_dims` dimensions.
            source_coords: Per-call override of `self.source_coords`
                (`interpolation="scipy"` only): pass this when `x` is a
                single sample whose own positions differ from every other
                sample's (the case a shared, construction-time
                `source_coords` cannot express).
            target_coords: Per-call override of `self.target_coords`, same
                convention.

        Returns:
            A tensor with the same leading axes as `x` and its trailing
            `num_spatial_dims` axes resampled.

        Raises:
            ValueError: If `x` has fewer than `num_spatial_dims`
                dimensions; if a coordinate-aware call is missing
                coordinates it needs to resolve a target grid; or if the
                resolved target range falls outside the source range and
                `extrapolate=False`.
        """
        if self.interpolation == "torch":
            return self._resampleUniform(x, self._resolveUniformSize(self.target_size))
        return self._resampleWithCoords(
            x,
            self._resolveCoords(self.source_coords, source_coords, "source_coords"),
            self._resolveCoords(self.target_coords, target_coords, "target_coords"),
        )

    def inverse(
        self,
        y: torch.Tensor,
        source_coords: CoordsLike | None = None,
        target_coords: CoordsLike | None = None,
    ) -> torch.Tensor:
        """Resample the last `num_spatial_dims` axes of `y` back toward the source grid.

        Args:
            y: Tensor with at least `num_spatial_dims` dimensions,
                typically the output of `apply`.
            source_coords: Per-call override, as in `apply` (used here as
                the *destination* grid, i.e. `apply`'s own `source_coords`
                argument name, so the two calls stay symmetric).
            target_coords: Per-call override of the grid `y` is currently
                on (`apply`'s own `target_coords`).

        Returns:
            A best-effort reconstruction toward the source grid (see the
            module docstring: not an exact inverse in general).

        Raises:
            ValueError: If neither `source_size` nor `source_coords` (at
                construction or per-call) was ever given, so there is
                nothing to invert back to; or the same range/shape errors
                as `apply`.
        """
        if self.interpolation == "torch":
            if self.source_size is None:
                raise ValueError(
                    "ResampleTransform.inverse() requires source_size to have been given at "
                    "construction time; this instance was built without one. Resampling is "
                    "lossy, so there is no way to recover an original size from y alone."
                )
            return self._resampleUniform(
                y, resolveSpatialShape(self.source_size, self.num_spatial_dims, "source_size")
            )

        resolved_source = self._resolveCoords(
            self.source_coords, source_coords, "source_coords", required=True
        )
        resolved_target = self._resolveCoords(self.target_coords, target_coords, "target_coords")
        # Inverting means resampling *from* the target grid *back to* the source grid.
        return self._resampleWithCoords(y, resolved_target, resolved_source)

    def _resolveUniformSize(self, size: ShapeLike | None) -> tuple[int, ...]:
        """Resolve an evenly-spaced target/source size for the torch backend.

        Args:
            size: As given to `__init__`.

        Returns:
            A resolved per-axis size tuple.

        Raises:
            ValueError: If `size` is `None`.
        """
        if size is None:
            raise ValueError(
                "ResampleTransform(interpolation='torch') requires target_size (source_size "
                "for inverse); target_coords/source_coords only apply to interpolation='scipy'."
            )
        return resolveSpatialShape(size, self.num_spatial_dims, "target_size/source_size")

    def _resolveCoords(
        self,
        constructed: np.ndarray | None,
        override: CoordsLike | None,
        name: str,
        required: bool = False,
    ) -> np.ndarray | None:
        """Resolve a per-call coordinate override against the construction-time default.

        Args:
            constructed: `self.source_coords`/`self.target_coords`.
            override: The per-call value passed to `apply`/`inverse`, if
                any.
            name: Parameter name, used only for the error message.
            required: If `True`, raise instead of returning `None` when
                neither `constructed` nor `override` is given (used for
                `source_coords` in `inverse`, which cannot fall back to an
                implicit uniform grid the way `apply` can).

        Returns:
            The resolved coordinate array, or `None` (meaning "implicit
            evenly-spaced grid", resolved later once the axis length is
            known) if not required.

        Raises:
            ValueError: If `required` and neither value is available.
        """
        resolved = _toCoordsArray(override, name) if override is not None else constructed
        if resolved is None and required:
            raise ValueError(
                f"ResampleTransform.inverse() requires {name} (either at construction time or "
                f"passed to this call) to invert an interpolation='scipy' resampling."
            )
        return resolved

    def _resampleUniform(self, x: torch.Tensor, resolved_size: tuple[int, ...]) -> torch.Tensor:
        """Evenly-spaced resampling via `torch.nn.functional.interpolate`.

        Args:
            x: Tensor with at least `num_spatial_dims` dimensions.
            resolved_size: Target size for the trailing `num_spatial_dims`
                axes.

        Returns:
            The resampled tensor, dtype preserved.

        Raises:
            ValueError: If `x` has fewer than `num_spatial_dims`
                dimensions.
        """
        leading_shape, spatial_shape = self._splitShape(x)
        flattened = x.reshape(-1, 1, *spatial_shape).to(torch.float32)

        if self.mode in _ALIGN_CORNERS_MODES:
            resampled = F.interpolate(
                flattened, size=resolved_size, mode=self.mode, align_corners=self.align_corners
            )
        else:
            resampled = F.interpolate(flattened, size=resolved_size, mode=self.mode)

        return resampled.reshape(*leading_shape, *resolved_size).to(x.dtype)

    def _resampleWithCoords(
        self,
        x: torch.Tensor,
        source_coords: np.ndarray | None,
        target_coords: np.ndarray | None,
    ) -> torch.Tensor:
        """Coordinate-aware resampling via `scipy.interpolate` (`num_spatial_dims=1` only).

        Args:
            x: Tensor with at least 1 dimension.
            source_coords: Source positions, or `None` for an implicit
                evenly-spaced grid `0, ..., length - 1`.
            target_coords: Target positions, or `None` to default to
                `target_size` evenly-spaced points spanning the resolved
                source range.

        Returns:
            The resampled tensor, dtype preserved.

        Raises:
            ValueError: If `x` has fewer than 1 dimension; if the resolved
                target range falls outside the resolved source range and
                `self.extrapolate` is `False`.
            ImportError: If `scipy` is not installed.
        """
        try:
            from scipy import interpolate as scipy_interpolate
        except ImportError as error:
            raise ImportError(
                "ResampleTransform(interpolation='scipy') requires the 'scipy' package, which "
                "is not installed. Install it with `pip install scipy` or "
                '`pip install -e ".[interpolation]"`.'
            ) from error

        leading_shape, spatial_shape = self._splitShape(x)
        length = spatial_shape[0]
        resolved_source = (
            source_coords if source_coords is not None else np.arange(length, dtype=np.float64)
        )
        if resolved_source.shape[0] != length:
            raise ValueError(
                f"source_coords has {resolved_source.shape[0]} position(s) but the input's "
                f"resampled axis has length {length}."
            )

        if target_coords is not None:
            resolved_target = target_coords
        else:
            # Guaranteed non-None here: __init__ requires target_size or target_coords, and
            # this branch is only reached when no target_coords (constructed or per-call) was
            # resolved, so target_size must be the one that was actually given.
            assert self.target_size is not None
            target_length = resolveSpatialShape(self.target_size, 1, "target_size")[0]
            resolved_target = np.linspace(
                resolved_source.min(), resolved_source.max(), target_length, dtype=np.float64
            )

        if not self.extrapolate:
            source_min, source_max = resolved_source.min(), resolved_source.max()
            target_min, target_max = resolved_target.min(), resolved_target.max()
            if target_min < source_min - 1e-9 or target_max > source_max + 1e-9:
                raise ValueError(
                    f"Target positions [{target_min}, {target_max}] fall outside the source "
                    f"positions' range [{source_min}, {source_max}]. Pass extrapolate=True to "
                    f"allow this, or choose target positions within range."
                )

        values_np = x.detach().cpu().numpy().astype(np.float64).reshape(-1, length)
        interpolator = self._buildScipyInterpolator(scipy_interpolate, resolved_source, values_np)
        resampled_np: np.ndarray = interpolator(resolved_target)

        resampled = torch.from_numpy(resampled_np).to(x.dtype)
        return resampled.reshape(*leading_shape, resolved_target.shape[0])

    def _buildScipyInterpolator(
        self, scipy_interpolate: ModuleType, source_coords: np.ndarray, values: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Build the scipy interpolator selected by `self.scipy_kind`.

        Args:
            scipy_interpolate: The `scipy.interpolate` module (imported
                lazily by the caller).
            source_coords: Source positions, shape `(length,)`.
            values: Values to interpolate, shape `(rows, length)`.

        Returns:
            A callable `scipy` interpolator: `interpolator(target_coords)
            -> ndarray` of shape `(rows, len(target_coords))`.
        """
        # scipy has no type stubs (pyproject.toml ignores it for mypy), so every constructor
        # call below returns `Any`; `cast` documents the true, checked-at-runtime contract
        # (every one of these classes/functions is callable as `interpolator(coords) -> ndarray`)
        # instead of letting `Any` silently propagate out of this method.
        if self.scipy_kind == "cubic_spline":
            return cast(
                Callable[[np.ndarray], np.ndarray],
                scipy_interpolate.CubicSpline(
                    source_coords, values, axis=-1, extrapolate=self.extrapolate
                ),
            )
        if self.scipy_kind == "pchip":
            return cast(
                Callable[[np.ndarray], np.ndarray],
                scipy_interpolate.PchipInterpolator(
                    source_coords, values, axis=-1, extrapolate=self.extrapolate
                ),
            )
        if self.scipy_kind == "akima":
            return cast(
                Callable[[np.ndarray], np.ndarray],
                scipy_interpolate.Akima1DInterpolator(
                    source_coords, values, axis=-1, extrapolate=self.extrapolate
                ),
            )
        return cast(
            Callable[[np.ndarray], np.ndarray],
            scipy_interpolate.interp1d(
                source_coords,
                values,
                kind=self.scipy_kind,
                axis=-1,
                bounds_error=not self.extrapolate,
                fill_value="extrapolate" if self.extrapolate else np.nan,
            ),
        )

    def _splitShape(self, x: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Split `x`'s shape into leading axes and the trailing spatial axes.

        Args:
            x: Tensor with at least `num_spatial_dims` dimensions.

        Returns:
            `(leading_shape, spatial_shape)`.

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
        leading_shape = x.shape[: x.dim() - self.num_spatial_dims]
        spatial_shape = x.shape[x.dim() - self.num_spatial_dims :]
        return tuple(leading_shape), tuple(spatial_shape)
