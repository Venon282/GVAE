# 0012 — Generic, invertible data transforms (`data/transforms/`)

**Status:** accepted
**Date:** 2026-08-29

## Context

Spec §8's repository layout has reserved `data/transforms/` from the start, but
`data/NOTE.md` deferred it indefinitely, bundled together with `datamodule.py`
under one blanket "the data pipeline stays entirely the user's own
responsibility" decision. That framing conflated two things of a genuinely
different nature:

1. **Dataset loading, matching/pairing samples across modalities, and
   train/val/test splitting.** These are inherently dataset- and
   user-specific (which files, which naming convention, which split ratio),
   with no reusable structure to extract into this framework. `datamodule.py`
   correctly stays out of scope for this reason.
2. **Preprocessing steps such as a log transform, a normalization, or
   resampling to a common grid.** These are generic tensor operations,
   independent of file format or dataset identity. Spec §6 already said as
   much explicitly: "SAXS-specific preprocessing (e.g. log-scale intensity)
   belongs in `transforms/`, not in the encoder", and other signal sources
   "should slot into the same... family later, with only preprocessing
   differing, not the architecture" — exactly the kind of reuse this
   codebase already gives encoders and decoders themselves.

Leaving (2) unimplemented alongside (1) had two concrete costs, not just a
conceptual one: `DataConfig.transforms` (`config/data.py`) was a plain
`list[str]` nothing ever resolved, purely decorative; and
`visualization.reconstruction_plot`'s own `inverse_transform` hook had
nothing to plug into besides a function the caller had to hand-write every
time, with no framework-provided building block even for the common cases.
Separately, spec §10's testing checklist item **C11** explicitly calls for
"tests unitaires... pour les transforms (notamment l'invertibilité)", a
regularizer-registry test, a beta-schedule test, a datamodule test, a raw
trainer smoke test, and a real-module (`OneDCnnEncoder`/`OneDCnnDecoder` via
`GlobalVae.createSingleLatent`, no fusion) integration test for spec §6.1
milestone 1 specifically — none of which could be satisfied for transforms
without transform code existing in the first place, and two of which
(trainer smoke test, real-module milestone test) were simply missing
regardless of this decision.

## Decision

### Split the scope decision, not the code that was already correct

`datamodule.py` remains unbuilt and unplanned, now for a stated permanent
reason (§6.2 of the spec) rather than an open question awaiting one.
`DataConfig.loader_factory` remains this framework's one integration point
into a caller's own data pipeline, unchanged.

### `data/transforms/`: `AbstractTransform` + registry, mirroring every other pluggable strategy

- `base.py`: `AbstractTransform`, a plain `ABC` (not an `nn.Module`,
  mirroring `AbstractBetaSchedule`'s own reasoning: a transform is a
  deterministic function of a tensor, not a computation with learnable
  parameters that needs to participate in the autograd graph). Two abstract
  methods, `apply`/`inverse`, plus `__call__` aliasing `apply`.
- `registry.py`: `registerTransform`/`getTransformClass`/
  `listRegisteredTransforms`, the exact self-registration pattern already
  used for encoders, decoders, fusion, assemblers, regularizers, and beta
  schedules.
- Three built-in strategies:
  - `log` (`LogTransform`): `log(x + eps)` / `exp(y) - eps`. Exact up to
    floating-point error. Raises at `apply` time if `x + eps` is not
    strictly positive everywhere, rather than silently returning
    `-inf`/`nan`.
  - `standardize` (`StandardizeTransform`): `(x - mean) / std` /
    `y * std + mean`. Exact. `mean`/`std` are supplied explicitly (a
    `float` or a broadcastable `torch.Tensor`, e.g. one value per channel)
    rather than computed from whatever tensor is passed in: computing
    statistics from data is itself a data-pipeline decision (which split,
    how many samples) this transform has no business making, and doing so
    would silently couple it to one dataset. This mirrors the project's
    general "the user supplies the value, never a silently guessed one"
    convention (spec §10, §12).
  - `resample` (`ResampleTransform`): resamples the trailing
    `num_spatial_dims` axes of a tensor to a fixed size via
    `torch.nn.functional.interpolate`, with `mode` defaulting to the
    natural choice per `num_spatial_dims` (`"linear"`/`"bilinear"`/
    `"trilinear"`). Unlike the two transforms above, this one is not a
    lossless bijection: shrinking then growing a signal back discards
    information. `inverse` is therefore a best-effort reconstruction at an
    explicitly supplied `source_size`; omitting `source_size` makes
    `inverse` raise a clear error instead of silently guessing an original
    size no single instance can actually know (samples can vary in size
    before resampling).
- `compose.py`: `ComposeTransform`, chaining several transforms, applying in
  order and inverting in the exact reverse order. Deliberately **not**
  registered: it is a combinator over already-resolved `AbstractTransform`
  instances (typically produced by `buildTransformPipeline` below), not a
  named strategy a caller selects by string, the same reasoning that keeps
  `Trainer.callbacks` a plain list rather than a registry-selected entry
  (`docs/adr/0005-training-loop.md`).

### Hard genericity requirement, enforced by design and by test

Every transform above operates on a tensor of *any* shape or
dimensionality. Shape-dependent behavior is expressed only through explicit
constructor parameters (`ResampleTransform.num_spatial_dims`), never by
branching internally on `x.dim()`, and nothing in this subpackage is named
after, or contains logic specific to, one dataset (no "SAXS" anywhere here)
or one fixed dimensionality (one `ResampleTransform`, not a
`SignalResample`/`ImageResample` pair). `tests/integration/test_transforms.py`
exercises every transform at three different dimensionalities (a 1D vector,
a 2D image-shaped tensor, a 3D volume-shaped tensor) using the exact same
class and constructor arguments each time, so this is a checked property,
not only a documented intention.

### `config/data.py`: `DataConfig.transforms` becomes real

`TransformConfig` (name + kwargs, mirroring `FusionConfig`/
`LoggerEntryConfig`) replaces the old `list[str]`; `DataConfig.transforms:
list[TransformConfig]`. `buildTransformPipeline(config) -> ComposeTransform`
resolves every entry through the `data.transforms` registry, in order.
Nothing in this framework calls it automatically — a caller's own
`loader_factory` may call it while loading data, or ignore
`config.transforms` and preprocess however it likes — preserving the exact
framework/data-pipeline boundary this project already committed to. The
returned pipeline's `.inverse` is a plain `Callable[[Tensor], Tensor]`,
directly usable as `visualization.reconstruction_plot`'s
`inverse_transform` parameter or `evaluation.visual_export`'s
`inverse_transforms` dict, without the caller hand-writing one.

`configs/data/signal.yaml` updated to the new schema, shipping a
`log` + `standardize` example. The `standardize` step's `mean`/`std` are
left as placeholder identity values (`0.0`/`1.0`) with a comment directing
the user to compute real statistics from their own training split, matching
`StandardizeTransform`'s own "never guess a statistic" design.

### Testing (C11)

- `tests/integration/test_transforms.py`: the registry pattern (mirroring
  every other registry's own test file), value correctness and
  invertibility for `log`/`standardize` (exact, floating-point tolerance),
  `resample` (approximate, documented tolerance, plus the shape round-trip
  and the no-`source_size`-raises path), `ComposeTransform` (ordering,
  reverse-ordering on inverse, an empty pipeline as the identity), and
  `config.data.buildTransformPipeline` (including an end-to-end check
  against the real, shipped `configs/data/signal.yaml`).
- `tests/integration/test_trainer_smoke.py`: a small, self-contained smoke
  test independent of `test_trainer.py`'s larger suite — a handful of
  optimizer steps on dummy data, checking that the loss decreases over that
  short run and that no parameter's gradient is `None` after a step, at
  more than one point in the run (not only the first).
- `tests/integration/test_signal_vae_milestone.py`: the real-module gap
  `test_en_l1_dn_default.py` left open. Builds spec §6.1 milestone 1 exactly
  — `OneDCnnEncoder` + `OneDCnnDecoder` via `GlobalVae.createSingleLatent`,
  no fusion strategy, single modality — and covers model assembly (the real
  classes are used, no fusion module is built), forward-pass shapes,
  gradient flow into every parameter, and a short `Trainer.fit` run where
  the loss decreases, i.e. "trained end to end" as spec §6.1 itself puts it.
- The regularizer-registry and beta-schedule tests C11 also asks for
  already existed (`test_regularizers.py`, `test_beta_schedules.py`, both
  predating this ADR); nothing new was needed for those. No test exists for
  `datamodule.py`, since (per the scope decision above) there is no such
  module and none is planned.

## Consequences

- `DataConfig.transforms` is a real, working mechanism end to end, verified
  against the shipped `configs/data/signal.yaml`
  (`TestBuildTransformPipelineFromConfig::test_real_signal_yaml_config_wires_a_working_pipeline`),
  not merely schema-valid.
- Adding a further generic transform (a clip/winsorize step, a per-sample
  z-score, ...) is purely additive: one new file under `data/transforms/`,
  registered in its `__init__.py`, zero changes to `GlobalVae`, `Trainer`,
  `config/data.py`, or any existing transform.
- `datamodule.py` remains unbuilt, now for a clearly stated permanent
  reason instead of an open question; future conversations should not
  treat its absence as a gap to fill (see spec §12's updated guidance).
- Spec §11's previously-bundled "data pipeline scope" open question is
  resolved and removed from the open-questions list; spec §6 gains a new
  §6.2 documenting the transform pattern and its genericity requirement,
  §8's repository tree annotates `data/transforms/` and `datamodule.py`
  accordingly, §9 gains a `data.transforms` config example, §10's Testing
  bullet is expanded with the C11 checklist items (including the two that
  were missing independent of this transform work: the trainer smoke test
  and the real-module milestone integration test), and §12 gains guidance
  matching the new rule.
- `ruff check`, `ruff format --check`, and `mypy --strict` pass clean on
  every new/changed file; the full test suite (51 tests across the three
  new files) passes.
