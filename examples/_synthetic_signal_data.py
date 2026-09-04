"""Synthetic signal data, shared by every `examples/*.py` script.

Not part of the `global_vae` package: this is example-only scaffolding, analogous to
`tests/integration/_script_fixtures.py`'s dummy fixtures (see that module's own
docstring for why such helper modules are named without a `test_`/numeric prefix, and
`docs/adr/0012-generic-data-transforms.md`'s reasoning for why `data/transforms/`
itself never contains anything like this: this module is deliberately outside the
framework, exactly where dataset-specific code belongs, spec §6.2).

Deliberately on an **irregular, per-sample grid**: every curve is measured at its own,
randomly-perturbed positions, of its own length, so both example scripts actually need
`ResampleTransform`'s coordinate-aware mode (`interpolation="scipy"`) rather than a
plain, count-only resample (see `docs/adr/0013-coordinate-aware-resampling.md`).

`buildSyntheticSignalDataloaders` is the `"module.path:function_name"` `loader_factory`
`02_config_driven_pipeline.py` points `data.loader_factory` at (spec §9): it turns a
`global_vae.config.data.DataConfig` into a `DataloaderBundle` using nothing but the
functions above it in this file, plus `config.data.buildTransformPipeline` to apply
whatever `transforms` the composed YAML config actually specifies (spec §6.2), instead
of hand-building a pipeline in Python the way `01_signal_vae_pipeline.py` does.
"""

import numpy as np
import torch

from global_vae.config.data import DataConfig, DataloaderBundle, buildTransformPipeline
from global_vae.data.transforms.compose import ComposeTransform
from global_vae.data.transforms.resample import ResampleTransform

NUM_TRAIN, NUM_VAL, NUM_TEST = 240, 30, 30
DEFAULT_COMMON_GRID_LENGTH = 128


def generateSyntheticCurve(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic curve: a single positive-valued peak plus noise, on its
    own randomly-perturbed grid.

    This mimics a common real situation (e.g. two measurements taken over slightly
    different ranges, at a different resolution, or with a different number of
    points) without being tied to any one instrument or dataset: nothing here is
    specific to a domain, only the *shape* of the problem (irregular per-sample
    positions) is what matters for this example.

    Args:
        rng: A seeded NumPy random generator.

    Returns:
        `(positions, values)`, both shape `(n_points,)`, `n_points` itself random
        per curve. `positions` is sorted and strictly increasing.
    """
    n_points = rng.integers(50, 90)
    grid_start = rng.uniform(0.05, 0.15)
    grid_end = rng.uniform(0.85, 0.95)
    positions = np.sort(rng.uniform(grid_start, grid_end, n_points))

    center = rng.uniform(0.35, 0.65)
    width = rng.uniform(0.05, 0.12)
    amplitude = rng.uniform(2.0, 6.0)
    baseline = 1.0
    peak = baseline + amplitude * np.exp(-0.5 * ((positions - center) / width) ** 2)

    noise = rng.normal(0.0, 0.02, size=positions.shape) * peak
    values = np.clip(peak + noise, 1e-3, None)  # keep strictly positive (needed by LogTransform)
    return positions, values


def buildSyntheticDataset(
    num_curves: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate `num_curves` synthetic curves, each on its own grid.

    Args:
        num_curves: How many curves to generate.
        rng: A seeded NumPy random generator.

    Returns:
        A list of `(positions, values)` pairs.
    """
    return [generateSyntheticCurve(rng) for _ in range(num_curves)]


def computeCommonGrid(curves: list[tuple[np.ndarray, np.ndarray]], length: int) -> torch.Tensor:
    """Compute a common target grid as the *intersection* of every curve's own range.

    Args:
        curves: `(positions, values)` pairs covering every split (train/val/test):
            the intersection must hold across all of them, or some split's curves
            would need extrapolation later.
        length: Number of points in the common grid.

    Returns:
        A 1D tensor of `length` evenly-spaced positions, entirely inside every
        curve's own measured range.
    """
    common_min = max(positions.min() for positions, _ in curves)
    common_max = min(positions.max() for positions, _ in curves)
    # dtype=torch.float64, not torch.linspace's default float32: this grid is only
    # ever consumed as ResampleTransform's target_coords, which is stored (and
    # compared against source_coords) at float64 precision internally anyway
    # (resample.py's _toCoordsArray always upcasts). Building it at float32 first
    # would round every endpoint by up to ~1e-7, which can push it fractionally
    # outside the tightest curve's own range and trip the extrapolate=False check
    # for no real reason.
    return torch.linspace(float(common_min), float(common_max), length, dtype=torch.float64)


def buildResampleTransform(common_grid: torch.Tensor) -> ResampleTransform:
    """Build the coordinate-aware resample transform every curve is passed through.

    `scipy_kind="pchip"` (shape-preserving Hermite interpolation) is used instead of
    a plain natural cubic spline on purpose: an unconstrained cubic spline fitted
    through many irregularly-spaced, noisy points can overshoot wildly between
    points (classic spline ringing), which would violate this data's known
    positivity and break the log-transform step right after. "pchip" never
    overshoots the data's own local min/max, which matters more here than raw
    smoothness (`docs/adr/0013-coordinate-aware-resampling.md`).

    Args:
        common_grid: The shared target grid (see `computeCommonGrid`).

    Returns:
        A `ResampleTransform(interpolation="scipy", scipy_kind="pchip")` instance.
    """
    return ResampleTransform(target_coords=common_grid, interpolation="scipy", scipy_kind="pchip")


def resampleOntoCommonGrid(
    curves: list[tuple[np.ndarray, np.ndarray]], resample_transform: ResampleTransform
) -> torch.Tensor:
    """Resample every curve (its own grid, its own length) onto the shared common
    grid `resample_transform.target_coords`, one sample at a time.

    Per-sample, not batched: each curve has different `source_coords`, exactly the
    case `ResampleTransform`'s per-call `source_coords` override exists for (see its
    own module docstring). This has to happen before batching, since a batch is a
    single dense tensor and these curves do not start out the same length.

    Args:
        curves: `(positions, values)` pairs, as returned by `buildSyntheticDataset`.
        resample_transform: A `ResampleTransform(interpolation="scipy", ...)`
            instance with `target_coords` already set to the common grid.

    Returns:
        A single tensor, shape `(len(curves), common_grid_length)`.

    Note:
        `source_coords` is passed at `positions`' own native float64 precision, not
        downcast to float32 (`values`, the y-data, is downcast; only x-positions are
        precision-sensitive here). Two of a curve's ~50-90 positions can legitimately
        land extremely close together under `rng.uniform`, and float32 has just
        ~7 significant digits: rounding two already-close, genuinely distinct float64
        positions to float32 can collapse them to the exact same value, which breaks
        `np.sort`'s strictly-increasing guarantee and makes `scipy`'s spline
        interpolators (`pchip` included) raise `ValueError: x must be strictly
        increasing sequence.`. This is a probability-of-collision issue, not a
        one-off: it is rare enough at a few hundred curves to go unnoticed, but with
        enough curves (tens of thousands) it reliably happens for at least one of
        them.
    """
    resampled = [
        resample_transform.apply(
            torch.from_numpy(values).float(), source_coords=torch.from_numpy(positions)
        )
        for positions, values in curves
    ]
    return torch.stack(resampled)


def _buildSyntheticSignalArtifacts(
    config: DataConfig,
) -> tuple[DataloaderBundle, ComposeTransform, torch.Tensor]:
    """Shared implementation behind `buildSyntheticSignalDataloaders`.

    Returns the preprocessing pipeline and the common grid alongside the
    `DataloaderBundle` itself (not just the bundle `loader_factory`'s own contract
    requires), since the example script that drives this factory through
    `global_vae.config.data.buildDataloadersFromConfig` also needs the pipeline's
    `.inverse` and the grid's x-positions afterward, for evaluation/visualization
    (`pipeline.inverse` as `plotReconstructionGrid`'s own `inverse_transform`, exactly
    as `01_signal_vae_pipeline.py` uses its own hand-built pipeline). Calling this
    function directly (as the example script does) is a plain Python function call,
    not the `loader_factory` dotted-path indirection `buildDataloadersFromConfig`
    itself provides; both paths are deterministic in `config.seed`, so they agree.

    Args:
        config: A `DataConfig`, typically produced by
            `global_vae.config.experiment.loadExperimentConfig`.

    Returns:
        `(bundle, pipeline, common_grid)`.
    """
    rng = np.random.default_rng(config.seed)
    train_curves = buildSyntheticDataset(NUM_TRAIN, rng)
    val_curves = buildSyntheticDataset(NUM_VAL, rng)
    test_curves = buildSyntheticDataset(NUM_TEST, rng)
    all_curves = train_curves + val_curves + test_curves

    length = config.sequence_length or DEFAULT_COMMON_GRID_LENGTH
    common_grid = computeCommonGrid(all_curves, length)
    resample_transform = buildResampleTransform(common_grid)

    train_values = resampleOntoCommonGrid(train_curves, resample_transform)
    val_values = resampleOntoCommonGrid(val_curves, resample_transform)
    test_values = resampleOntoCommonGrid(test_curves, resample_transform)

    # Driven by config.transforms (spec §6.2), not hand-built: whatever the composed
    # YAML says (log, standardize, ...), in whatever order, is what gets applied here.
    pipeline = buildTransformPipeline(config)
    train_preprocessed = pipeline.apply(train_values)
    val_preprocessed = pipeline.apply(val_values)
    test_preprocessed = pipeline.apply(test_values)

    def toBatches(values: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [
            {"signal": values[start : start + config.batch_size]}
            for start in range(0, values.shape[0], config.batch_size)
        ]

    bundle = DataloaderBundle(
        train=toBatches(train_preprocessed),
        val=toBatches(val_preprocessed),
        test=toBatches(test_preprocessed),
    )
    return bundle, pipeline, common_grid


def buildSyntheticSignalDataloaders(config: DataConfig) -> DataloaderBundle:
    """`loader_factory` for `02_config_driven_pipeline.py` (spec §9): `data.loader_factory`
    points at this function's dotted path, and
    `global_vae.config.data.buildDataloadersFromConfig` resolves and calls it with the
    composed `DataConfig`.

    `config.train_path`/`config.val_path`/`config.test_path` are unused: this factory
    generates its data in-memory (see the module docstring), the same way every other
    part of this framework's own test suite does; a real `loader_factory` would read
    `config.train_path` et al. instead. `config.seed` seeds the generation, so this
    factory (and the example script's own direct call to
    `_buildSyntheticSignalArtifacts` for evaluation/visualization afterward) produce
    the exact same data whenever the same config is composed again.

    Args:
        config: A `DataConfig`, typically produced by
            `global_vae.config.experiment.loadExperimentConfig`.

    Returns:
        `train`/`val`/`test` batches, each `signal` values already preprocessed
        through `config.transforms` (`buildTransformPipeline`).
    """
    bundle, _pipeline, _common_grid = _buildSyntheticSignalArtifacts(config)
    return bundle
