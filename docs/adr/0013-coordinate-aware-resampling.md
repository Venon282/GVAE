# 0013 — Coordinate-aware resampling (`ResampleTransform` refinement)

**Status:** accepted
**Date:** 2026-08-29

## Context

ADR 0012 introduced `ResampleTransform` as one of the three built-in generic data
transforms (spec §6.2). Its first version only ever knew how many points a tensor
had, resampling from `source_size` points to `target_size` points via
`torch.nn.functional.interpolate`, which implicitly treats both the input and the
output as evenly spaced.

That is a real gap for a common case this transform is explicitly meant to serve
(resampling to a common grid, spec §6.1's own decision A2): if two samples were
measured at genuinely different positions (a different number of points, a different
range, or both), resampling both to the same *point count* does not put them on the
same *grid*. Point `n` of one resampled curve and point `n` of the other still do not
correspond to the same physical position, because the first version of this transform
had no notion of position at all, only of count. Point-count-only resampling can
silently misalign exactly the data it was meant to align.

Separately, `torch.nn.functional.interpolate`'s available modes (linear/bilinear/
trilinear/nearest/area/bicubic) are a narrow slice of the interpolation methods a
real, noisy, irregularly-sampled dataset might need (e.g. monotonicity-preserving or
outlier-robust splines), all of which `scipy.interpolate` already provides.

Both gaps were raised together, directly against the first version of this class, and
both stem from the same missing capability: no way to describe *where* a sample's
points actually are.

## Decision

`ResampleTransform` gains a second backend, `interpolation="scipy"`, alongside the
original `interpolation="torch"` (kept as the default, unchanged in behavior):

- **Explicit coordinates.** `source_coords`/`target_coords` (1D position arrays)
  describe where the input's points are and where the output's points should be.
  Given once at construction, they apply to every call (the common case: a shared,
  possibly non-uniformly-spaced instrument grid). For the case that actually
  motivated this ADR — samples whose own positions genuinely differ from one
  another — `apply`/`inverse` also accept `source_coords`/`target_coords` as
  **per-call** overrides, so a caller's own per-sample loading code (a `Dataset
  .__getitem__`-style loop, necessarily per-sample anyway since ragged, differently-
  gridded samples cannot be one dense batched tensor before this step) can resample
  each sample onto one shared target grid with its own true positions taken into
  account. `AbstractTransform.apply(self, x) -> Tensor` remains the interface every
  other transform, `ComposeTransform`, and the config-wiring/visualization callers
  rely on; `ResampleTransform.apply`/`inverse` only *add* optional keyword parameters
  on top of it, so every existing caller (`transform.apply(x)`, `transform(x)`,
  `ComposeTransform`) is unaffected.
- **A richer choice of interpolation method**, `scipy_kind`: any
  `scipy.interpolate.interp1d` `kind` (`"linear"`, `"nearest"`, `"nearest-up"`,
  `"zero"`, `"slinear"`, `"quadratic"`, `"cubic"`, `"previous"`, `"next"`), plus three
  dedicated spline classes: `"cubic_spline"` (`CubicSpline`), `"pchip"`
  (`PchipInterpolator`, monotonicity-/shape-preserving, never overshoots between
  points), and `"akima"` (`Akima1DInterpolator`, robust to outliers). `scipy` is a
  **soft dependency of this mode only** (imported lazily inside the method, exactly
  the pattern already used for `tensorboard`/`scikit-learn`/`umap-learn`): the default
  `interpolation="torch"` path, and every other transform in this subpackage, need
  nothing beyond torch. `pyproject.toml` gained an `interpolation` extra
  (`scipy>=1.10`), also listed under `dev` so the test suite exercises it for real.
- **No silent extrapolation.** `extrapolate: bool = False` (default) raises a clear
  error if the resolved target range falls outside the resolved source range, instead
  of silently extrapolating or returning `nan` (which is what the underlying `scipy`
  classes do by default when asked to evaluate outside their fitted range); pass
  `extrapolate=True` to allow it explicitly.
- **Scope stays honest.** `interpolation="scipy"` combined with `num_spatial_dims !=
  1` raises `NotImplementedError`: a fully general N-D scattered/rectilinear
  coordinate grid is a materially larger feature (which axes have their own
  coordinates, separable vs. fully scattered positions, ...) that this ADR does not
  attempt. The evenly-spaced `interpolation="torch"` path is unaffected and still
  works at any `num_spatial_dims`.
- As with every transform in this subpackage (spec §6.2's hard requirement),
  `source_coords`/`target_coords` are plain position arrays with no assumption about
  what they represent; nothing added here is specific to any dataset or domain.

## Consequences

- Two curves recorded on genuinely different grids can now be resampled onto the same
  common positions correctly, not just to the same point count.
  `tests/integration/test_transforms.py::TestResampleTransformCoordinateAware
  ::test_per_sample_source_coords_align_two_different_grids` verifies this directly:
  the same underlying function, sampled at two different sets of positions and
  resampled onto one shared grid, agrees closely once aligned.
- `examples/01_signal_vae_pipeline.py` (new, see its own module docstring) exercises
  this end to end: synthetic curves are generated on deliberately irregular,
  per-sample grids, resampled onto one common grid (the intersection of every curve's
  own range, so no extrapolation is ever needed), and only then fed through the rest
  of the existing pipeline (log/standardize, model, training, evaluation,
  visualization). Building this example surfaced a related, practical lesson, also
  documented in that script's own comments: an unconstrained natural cubic spline
  (`scipy_kind="cubic_spline"`) fitted through many irregularly-spaced, noisy points
  can overshoot far outside the data's own range between points (classic spline
  ringing) — `"pchip"`, which never overshoots the local data range, is the safer
  default for this kind of data, and is what the example actually uses.
- `tests/integration/test_transforms.py` gained
  `TestResampleTransformCoordinateAware` (shared and per-call coordinates, every
  `scipy_kind`, the extrapolation error path and its explicit opt-in, batched
  shared-grid resampling, the `num_spatial_dims != 1` `NotImplementedError`, the
  missing-`scipy`-package `ImportError` path, and the pre-existing
  `interpolation="torch"` tests, unchanged, still passing).
- `docs/global-vae-project-specification.md` §6.2 and §9's config example updated to
  mention coordinate-aware resampling; `pyproject.toml` gained the `interpolation`
  extra.
- This ADR does not revise ADR 0012's other two transforms (`log`, `standardize`) or
  its overall design (registry pattern, `ComposeTransform`, the framework/data-pipeline
  boundary): only `ResampleTransform` itself changed, and only additively — every
  `interpolation="torch"` call from before this ADR behaves identically.
