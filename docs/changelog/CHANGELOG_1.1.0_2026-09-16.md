# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-09-16

Documentation catch-up for the last pre-0.1.0 iteration
(`examples/02_config_driven_pipeline.py`, `plotLossCurves`/`plotStepCurves`'s
`twin_metrics`, and the regularization/checkpoint-monitor choices baked into
`examples/01_signal_vae_pipeline.py`), which had already landed in code and in the
`[0.1.0]` entry's high-level description, but not with the level of detail below.
Also fixes real issues found while writing this entry and, separately, while a user
ran `01_signal_vae_pipeline.py` at a much larger scale than its shipped defaults (see
"Fixed").

### Added

- `encoders/OneDCnnResidualEncoder.py` (`1d_cnn_resnet_encoder_v1`) and
  `decoders/OneDCnnResidualDecoder.py` (`1d_cnn_resnet_decoder_v1`): ResNet-style 1D
  encoder/decoder built from `utils/conv_blocks.py`'s new `Residual1DBlock`/
  `Residual1DUpBlock` (spec §7, "scaling toward larger backbones"). Each stage's
  residual block has its own configurable depth (`block_depths`, per stage or
  shared, e.g. `(3, 4)` for a first stage with 3 internal layers before its
  shortcut and a deeper second stage with 4), and both classes are otherwise as
  permissive as `OneDCnnEncoder`/`OneDCnnDecoder` (per-stage kernel/stride/
  padding/dilation/pooling/activation/normalization, both decoder upsample
  modes, exact-output-length verification instead of resizing). A shortcut's
  hyperparameters are verified at construction time to reach the exact same
  output length as its main path for every input length (not just one example
  length), via two new pure helpers in `utils/conv_math.py`
  (`computeConv1dLengthOffset`/`computeConvTranspose1dLengthOffset`); a mismatch
  raises `ValueError` naming both offsets and how to fix them, matching this
  codebase's existing "verify, don't silently resize" convention. Deliberately
  does not attempt U-Net-style encoder-to-decoder skip connections: see
  `docs/adr/0014-residual-1d-encoder-decoder.md` for why that would give the
  decoder a deterministic, unregularized path around `z` (posterior collapse via
  skip paths, and loss of generation from the prior), and what the
  framework-consistent alternative would be instead.
- `utils/conv_math.py` gained `computeConv1dLengthOffset`, `isLengthPreservingConv1d`,
  `computeConvTranspose1dLengthOffset`, and `computeUpsampleStackOutputLength` (the
  last relocated, unchanged, from `OneDCnnDecoder.py`'s private
  `_computeLengthFromResolved`, now shared by both decoders; see "Changed" below).
- `tests/integration/test_conv_blocks.py`,
  `tests/integration/test_residual_signal_encoder.py`, and
  `tests/integration/test_residual_signal_decoder.py` covering the above: shapes
  across depth/stride/projection combinations, the flexible-depth case itself,
  gradient flow, both decoder upsample modes, the decoder's last-transition
  unconstrained-output guarantee (and that it does not also suppress an earlier
  internal layer's normalization/activation in a deep last-stage block), and every
  documented error path (non-positive depth, an even `kernel_size` with a
  multi-layer block, and an offset-mismatched shortcut).
- `docs/adr/0014-residual-1d-encoder-decoder.md` documenting the above, including
  the U-Net-skip-connections analysis and why it was rejected for this framework.

- `examples/02_config_driven_pipeline.py`: the same spec §6.1 milestone 1 pipeline as
  `01_signal_vae_pipeline.py`, assembled entirely from the `configs/` YAML files (spec
  §9, §10 "Config management") instead of hand-written Python kwargs, with a
  synthetic, in-memory `loader_factory`
  (`examples/_synthetic_signal_data.buildSyntheticSignalDataloaders`, see below)
  standing in for a real dataset. Demonstrates something the first example cannot show
  on its own: versioned, comparable experiment runs. Two named variants of the
  composed config (`EXPERIMENT_VARIANTS`) are composed and trained back to back, each
  expressed as nothing but a short list of Hydra dotlist overrides on top of the same
  base config: `"baseline"` (no overrides beyond wiring in the synthetic loader
  factory: `kl_standard_normal`, the stock beta warm-up) and `"tuned"` (switches the
  regularizer to `free_bits_kl`, slows the beta warm-up, and monitors
  `val/loss/reconstruction` for best-checkpoint selection, the same choices documented
  for `01_signal_vae_pipeline.py` below, now expressed as config overrides instead of
  Python kwargs, and model-architecture-agnostic so they apply identically whichever
  model config is selected). Each variant gets its own `output_dir`, so its
  checkpoint, config snapshot, CSV/TensorBoard logs, and figures never overwrite the
  other's (a plain, filesystem-level form of run versioning any config-driven
  workflow gets close to for free) and its own evaluation report; `main()` prints
  every selected variant's test-set reconstruction metrics side by side, so the value
  of driving hyperparameters from config is visible in the numbers, not only
  asserted.
  By default, composes `configs/experiment/signal_resnet_vae.yaml` (spec §7's residual
  encoder/decoder, see `docs/adr/0014-residual-1d-encoder-decoder.md`), added
  alongside `configs/model/signal_resnet_single_latent.yaml` in this same unreleased
  version (see the residual-encoder/decoder entry above): both are new, so this
  script's own default changing to use them is not itself a later "Changed" against a
  previously-released behavior. Which config groups (and which raw Hydra dotlist
  overrides) get composed is now a real, argparse-driven CLI (`_buildArgumentParser`),
  not a hardcoded Python constant, since choosing which configs to use is itself part
  of what this script demonstrates: `--experiment-config` (which
  `configs/experiment/*.yaml` to start from), `--model-config`/`--data-config`/
  `--training-config` (override one config group at a time on top of it, e.g.
  `--model-config signal_single_latent` to compare against the plain conv encoder/
  decoder instead), `--variants` (run a subset of `EXPERIMENT_VARIANTS` instead of
  every one, for a quick check), `--num-epochs`/`--seed`/`--output-root`, and
  `--override KEY=VALUE` (repeatable: arbitrary further raw Hydra dotlist overrides,
  the same mechanism `scripts/train.py` exposes directly on its own command line, for
  anything the named flags do not cover). `main(argv: list[str] | None = None)`
  mirrors `scripts/evaluate.py`/`scripts/visualize_latent.py`'s own plain-argparse
  convention (not `@hydra.main`, since this script composes and trains several
  variants back to back within one process, which an `@hydra.main`-decorated
  entry point cannot do safely, see `docs/adr/0011-hydra-config-layer.md`).
- `tests/integration/test_config_driven_pipeline_example.py`: covers the CLI above as
  a real subprocess (mirroring `test_train_script.py`'s own reasoning: this script's
  `from _synthetic_signal_data import ...` sibling import only resolves when
  `examples/` itself is on `sys.path`, which an in-process `importlib`-loaded module
  does not get for free) — the default resnet config, `--model-config`/
  `--experiment-config` switching to the plain conv model, `--override` reaching the
  composed config, `--variants` actually restricting which variant directories get
  written, and the invalid-variant-name argparse error path.
- `configs/model/signal_resnet_single_latent.yaml` and
  `configs/experiment/signal_resnet_vae.yaml`: the residual encoder/decoder's own
  model and experiment config, mirroring `signal_single_latent.yaml`/`signal_vae.yaml`
  exactly (same `signal -> z -> signal` shape, no fusion strategy, `output_length: 256`
  matching `configs/data/signal.yaml`'s `sequence_length`), with a per-stage
  `block_depths` example (`[2, 2, 3, 3, 2]` encoder, `[2, 3, 3, 2, 2]` decoder)
  demonstrating the residual classes' own flexible-depth feature. Selectable either as
  its own named experiment file or as a `model=signal_resnet_single_latent` override
  on top of `signal_vae.yaml`, both documented and both covered by
  `tests/integration/test_config.py`'s new `TestSignalResnetVaeExperiment` (composition,
  the real registered classes actually being built, forward-pass shapes,
  `block_depths` actually reaching the constructed modules, gradient flow, and a full
  `Trainer.fit` run end to end) and by a real `scripts/train.py` subprocess smoke run.
- `examples/_synthetic_signal_data.py`: the synthetic-curve generation, common-grid
  computation, and coordinate-aware-resampling helpers factored out of
  `01_signal_vae_pipeline.py` (unchanged behavior) so `02_config_driven_pipeline.py`
  can reuse the exact same data through its own `loader_factory`
  (`buildSyntheticSignalDataloaders`) without duplicating any of it. Not itself an
  example: the leading underscore marks it as shared scaffolding, mirroring
  `tests/integration/_script_fixtures.py`'s own naming convention and the reason
  behind it (pytest's test discovery, and here a reader browsing `examples/`, should
  not mistake it for a third, independently runnable example).
- `visualization.loss_curves.plotLossCurves` and `plotStepCurves` gained an optional
  `twin_metrics` parameter (plus `twin_log_scale`/`twin_ylabel`): a second group of
  metric keys plotted on a secondary, independently-scaled y-axis (`ax.twinx()`)
  instead of the primary one. Fixes a real readability gap, not a cosmetic one: this
  framework's own reconstruction and regularization losses routinely live one to
  several orders of magnitude apart (especially with `free_bits_kl` or a light beta),
  and sharing one axis (even a log-scaled one) still visually crushes whichever group
  is smaller, since matplotlib autoscales to the larger group's range alone.
  `twin_metrics=None` (the default) draws everything on one axis exactly as before, so
  every existing call site (including every test in `test_loss_curves.py` predating
  this change) is unaffected. Both examples' own final loss-curve plot now uses this
  to put reconstruction/total loss on the left axis and regularization on the right.

- `ResampleTransform(interpolation="scipy")` now validates that the resolved
  `source_coords` are strictly increasing *before* calling into `scipy`, raising a
  clear `ValueError` that names the likely cause (unsorted/duplicate positions, or
  precision loss from a lower-precision dtype such as `float32` upstream, exactly
  the bug above) instead of letting `scipy`'s own generic "`x` must be strictly
  increasing sequence" surface several stack frames down with no context. Covered
  by three new tests in `tests/integration/test_transforms.py`
  (`TestResampleTransformCoordinateAware`): an unsorted case, a duplicate-value
  case, and a direct regression test for the `float32`-collision scenario above
  (verified numerically before being written: `0.30000001` and `0.30000002` are
  distinct `float64` values but round to the identical `float32` one).
- `examples/02_config_driven_pipeline.py` now logs, at run time, exactly which
  `configs/*.yaml` files it composes and how many Hydra overrides are layered on
  top for the current variant, plus a couple of values actually resolved from that
  composed config (regularizer strategy, best-checkpoint monitor, learning rate).
  Prompted by a user reading the script and not finding it obvious that YAML config
  files were involved at all, since everything in the code body is plain Python
  (`EXPERIMENT_VARIANTS`'s dotlist override strings, `loadExperimentConfig(...)`
  calls) with no literal `.yaml` path in sight. The composition was already real
  (`loadExperimentConfig()` defaults to `config_name="experiment/signal_vae"`,
  genuinely reading and merging `configs/experiment/signal_vae.yaml` and, through
  its own `defaults:` list, `configs/model/signal_single_latent.yaml`,
  `configs/data/signal.yaml`, and `configs/training/default.yaml`, from disk via
  Hydra); this only makes it visible without having to already know that. The
  module docstring gained a paragraph spelling out the same thing.

- Cross-modal reconstruction reporting (spec §5: "the model can be trained and
  queried with any subset of available modalities"), making a mechanism
  `GlobalVae.forward` already implements (no code change to `GlobalVae` or any
  encoder/decoder/fusion strategy) visible and systematic as a reporting tool
  instead of something a caller had to reconstruct by hand for every input subset
  it wanted to compare:
  - `visualization/reconstruction_plot.py` gained `resolveDefaultInputSubsets`
    (one singleton subset per encoder, plus the full set together, deliberately
    not the whole power set), `collectCrossModalReconstructions` (runs the model
    under several input-modality subsets and collects every resulting
    `(original, reconstruction)` pair, ground truth always taken from the full,
    unrestricted batch so a decoder is comparable against ground truth even when
    its own modality was withheld from a given subset), and
    `plotCrossModalReconstructionMatrix` (a grid of input-subset x decoder overlay
    plots for one chosen example, reusing the module's existing `_plotOnePair`
    helper).
  - `evaluation/cross_modal.py` (new file): `computeCrossModalReconstructionMetrics`
    (applies `evaluation.metrics`'s existing metric functions to every cell of a
    collected cross-modal matrix, so "how much does reconstructing 'image' from
    'signal' alone actually cost" has a number, not only a picture) and
    `exportCrossModalFigures` (saves the matrix figure to disk, mirroring
    `exportEvaluationFigures`'s own shape; a deliberate no-op, not an error, for
    any model with fewer than two encoders, since there is nothing cross-modal to
    report for a single-modality model).
  - `scripts/evaluate.py` now calls `exportCrossModalFigures` unconditionally
    alongside its existing `exportEvaluationFigures` call whenever `--output-dir`
    is given: free (a no-op) for spec §6.1 milestone 1's single-modality model,
    immediately useful once a second modality exists.
  - `tests/integration/test_cross_modal_reconstruction.py` and
    `tests/integration/test_cross_modal_evaluation.py`, plus a new
    `TestCrossModalFigures` class in `tests/integration/test_evaluate_script.py`
    (using two new, purely additive dummy fixtures in
    `tests/integration/_script_fixtures.py`,
    `buildTwoModalityModelForScript`/`buildTwoModalityDataloaderForScript`).
    Exercises a three-modality dummy model as well as the two-modality case, since
    nothing in this feature assumes exactly two.
  - `docs/adr/0016-cross-modal-reconstruction-reporting.md` documenting the above,
    including why the figure-export half lives in its own `evaluation/cross_modal.py`
    rather than folded into `evaluate.py`/`visual_export.py`.
- `encoders/TwoDCnnEncoder.py` (`2d_cnn_encoder_v1`) and `decoders/TwoDCnnDecoder.py`
  (`2d_cnn_decoder_v1`): plain (non-residual) 2D CNN encoder/decoder for spec §6's
  image modality, a direct generalization of `OneDCnnEncoder`/`OneDCnnDecoder`
  matching every flexibility feature of the 1D pair (per-stage kernel/stride/
  padding/dilation, per-stage pooling/activation/normalization, adaptive global
  pooling for size-agnostic encoding, the custom-`nn.Module`-stage escape hatch,
  and the decoder's exact-output-shape verification instead of resizing, on both
  axes independently, with `output_padding` auto-solved per axis for
  `conv_transpose`). `TwoDCnnEncoder` also adds a new `minimal_input_shape ->
  tuple[int, int]` property alongside the base-class-required
  `minimal_input_length` (a single `int`, unable to fully express a 2D minimum on
  its own; see the ADR below). See `docs/adr/0017-2d-cnn-encoder-decoder.md`.
- `utils/stage_config.py` gained `broadcastPerStageShape` and
  `broadcastPerStageOptionalShape` (the latter tolerating a per-stage/shared
  `None`, needed for `pool_kernel_sizes`/`pool_strides`), finally implementing what
  the module's own docstring had already named and described but never shipped:
  the 2D generalization of `broadcastPerStage` that resolves the ambiguity a bare
  `Sequence`-based broadcast would have once a per-stage value can itself be a
  multi-component shape (`list` = one entry per stage, `tuple` = one shape shared
  by every stage). **Behavior note:** this makes `tuple` and `list` mean different
  things for shape-like arguments of the new 2D classes, unlike the 1D classes
  where they were interchangeable; see the ADR for the reasoning and the exact
  error a mis-sized tuple now raises instead of being silently misread.
- `utils/conv_math.py` gained the 2D counterpart of every 1D shape-arithmetic
  function `TwoDCnnDecoder` needs (`computeConv2dOutputShape`,
  `computeConvTranspose2dOutputShape`, `computeUpsampleThenConv2dOutputShape`,
  `solveConvTranspose2dOutputPadding`, `solveMinimumInputShapeForConv2d`,
  `computeUpsampleStack2dOutputShape`), each just its 1D counterpart applied once
  per axis (`Conv2d`/`ConvTranspose2d`'s own formulas are separable per axis, so no
  new math is derived here). `utils/builders.py` gained `build2DPoolLayer`/
  `build2DUpSampleStage`, mirroring `build1DPoolLayer`/`build1DUpSampleStage`.
- `tests/integration/test_image_encoder.py` and `tests/integration/test_image_decoder.py`
  covering the above: shapes (square, non-square, multi-channel, implicit-channel
  input), the `tuple`-vs-`list` behavior difference and its error path, per-stage
  pooling/activation/normalization, minimum-input-shape solving (including a
  non-square minimum from non-square hyperparameters), the decoder's exact-shape
  verification and its non-square, per-axis `output_padding` auto-solve, gradient
  flow, registration, and an encoder-decoder round trip (square RGB and
  rectangular grayscale) mirroring how `GlobalVae` wires a modality's encoder and
  decoder together.
- `docs/adr/0017-2d-cnn-encoder-decoder.md` documenting the above, including the
  `mypy --strict` reasoning behind splitting `broadcastPerStageShape` in two
  rather than making one `Optional`-typed function, and why `TwoDCnnEncoder`
  keeps `minimal_input_length` as a conservative `max(minimal_input_shape)`
  summary instead of widening `AbstractEncoder`'s own contract.

### Changed

- `examples/01_signal_vae_pipeline.py` regularizes its single latent space with
  `free_bits_kl` (`FREE_BITS = 1.0`) rather than the framework's own default,
  `kl_standard_normal`, and its `BestCheckpointCallback` monitors
  `"val/loss/reconstruction"` rather than that callback's own default,
  `"val/loss/total"`. Both choices were already part of the script when the `[0.1.0]`
  entry below was written; that entry only described the pipeline's stages, not these
  specific decisions or the reasoning behind them, which this entry now records:
  - An earlier version of this example used `kl_standard_normal` with a short beta
    warm-up, and reconstruction quality stayed poor (test-set R^2 around 0.3) no
    matter how much model capacity or data was added: the signature of posterior
    collapse, not underfitting. `beta` reaching full weight before the decoder has
    learned to rely on `z` lets the encoder cheaply satisfy the KL term by pushing
    every dimension's posterior toward the prior, after which extra capacity is never
    used either. `free_bits_kl` (spec §2.3) exists specifically for this: it gives
    every latent dimension a small KL budget it is never penalized for using, which
    alone raised this example's test-set R^2 from ~0.3 to ~0.97. Documented directly
    in the script's own module docstring ("On regularization") so the lesson stays
    visible to anyone reading it, not only recoverable from git history.
  - `free_bits_kl` holds the regularization term close to a constant per-dimension
    floor (`latent_dim * free_bits`), which dominates the *magnitude* of
    `"val/loss/total"` without being what actually distinguishes a better epoch from a
    worse one here; monitoring the dominated total would pick checkpoints essentially
    at random with respect to reconstruction quality. `BestCheckpointCallback` itself
    needed no code change for this: `monitor` was already a constructor parameter
    (`docs/adr/0007-best-checkpoint-callback.md`). This is a usage lesson specific to
    pairing it with `free_bits_kl`, not a bug fix.
  - The example's final loss-curve plot now uses `plotLossCurves`'s new
    `twin_metrics` (see "Added" above) instead of one shared axis.
- `examples/README.md` gained a section for `02_config_driven_pipeline.py`, and its
  `01_signal_vae_pipeline.py` section now mentions the `free_bits_kl` choice above.
  `README.md`'s own "Want to see it work" pointer now also mentions the second
  example.
- `decoders/OneDCnnDecoder.py`'s private `_computeLengthFromResolved` is now
  `utils.conv_math.computeUpsampleStackOutputLength` (imported back under its old
  private name at the one call site, so nothing else in this file changes),
  relocated so `OneDCnnResidualDecoder` can share the exact same per-transition
  length-chaining logic instead of duplicating it (see "Added" above and
  `docs/adr/0014-residual-1d-encoder-decoder.md`). Pure relocation, verified
  behavior-preserving by `test_signal_decoder.py`'s full existing suite passing
  unchanged. Also gained an explicit `upsample_modes_: tuple[str, ...]`
  annotation (a pre-existing `mypy --strict` gap this refactor's own new code
  would otherwise have repeated): mypy cannot always infer
  `broadcastPerStage`'s `TypeVar` through a `str | Sequence[str]`-typed
  argument without help; no behavior change.

### Fixed

- `DataConfig.transforms` and `DataConfig.sequence_length` were a single, flat
  value shared by the whole experiment (`transforms: list[TransformConfig]`,
  `sequence_length: int | None`), even though spec §6 explicitly anticipates more
  than one dataset sharing the 1D-signal encoder/decoder family "with only
  preprocessing differing, not the architecture": a second signal dataset had
  nowhere to configure its own steps, its own `log`/`standardize` statistics, or
  its own resampled length independently of the first. This was also inconsistent
  with `evaluation.visual_export.exportEvaluationFigures`'s and
  `visualization.reconstruction_plot.plotReconstructionGrid`'s own
  `inverse_transforms: dict[str, InverseTransform]` parameter, already keyed per
  modality, which the old flat pipeline had no way to produce correctly without
  bypassing the config layer entirely (as `examples/01_signal_vae_pipeline.py`
  already had to). Both fields are now per-modality mappings
  (`transforms: dict[str, list[TransformConfig]]`,
  `sequence_length: dict[str, int] | None`), and
  `buildTransformPipeline(config) -> dict[str, ComposeTransform]` now returns one
  pipeline per configured modality instead of one pipeline for the whole config; a
  modality absent from `config.transforms` is simply absent from the returned
  dict rather than getting an implicit identity pipeline. Breaking change to
  `DataConfig`'s shape (no real dataset depended on the old shape yet);
  `configs/data/signal.yaml`,
  `tests/integration/test_transforms.py::TestBuildTransformPipelineFromConfig`,
  `tests/integration/test_config.py`, `tests/integration/_train_script_fixtures.py`,
  and `examples/_synthetic_signal_data.py` updated accordingly. See
  `docs/adr/0015-per-modality-data-transforms.md`.
- `examples/02_config_driven_pipeline.py` read `DataloaderBundle.test` (via
  `_saveVariantFigures`'s `list(bundle.test)` and `runVariant`'s
  `evaluate(best_model, dataloaders.test, ...)`) as if it were never `None`. Its
  declared type is `Iterable[dict[str, torch.Tensor]] | None`
  (`global_vae/config/data.py`: a caller's `loader_factory` is not required to
  provide a test split), so both call sites failed `mypy --strict`: `list` and
  `evaluate` both require a non-`None` `Iterable` argument, even though this
  example's own `loader_factory` always populates `test` in practice. Fixed with the
  same `assert ... is not None` pattern this file already used for
  `cfg.training.checkpoint.best_path` (and `tests/integration/test_config.py`
  already used for `DataloaderBundle.val`), not by loosening either callee's
  signature: a genuinely optional test split is the correct type for the general
  `DataloaderBundle` contract, and this example's own factory is simply the one
  place that always happens to satisfy it.
- `examples/_synthetic_signal_data.py`'s `resampleOntoCommonGrid` cast each curve's
  positions to `float32` (`torch.from_numpy(positions).float()`) before handing them
  to `ResampleTransform` as `source_coords`. At the shipped dataset size
  (240/30/30 train/val/test curves) this went unnoticed; a user scaling the same
  script up to tens of thousands of curves hit `scipy`'s own
  `ValueError: x must be strictly increasing sequence.` from deep inside its
  `PchipInterpolator`. Root cause: two of a curve's
  ~50-90 positions, drawn from a continuous `rng.uniform`, are always distinct at
  `float64` precision, but can legitimately land close enough together that
  rounding both to `float32` (~7 significant digits) collapses them to the exact
  same value, which breaks the strictly-increasing input every `scipy` spline class
  (`pchip` included) requires. This is a probability-of-collision issue, not a
  one-off bug that only affects one input: rare enough at a few hundred curves to
  never trigger, reliably hit by at least one curve once there are tens of
  thousands. Fixed by passing `positions` at its own native `float64` precision
  instead (`values`, the curve's y-data, is still downcast to `float32`; only
  x-positions are precision-sensitive here). `computeCommonGrid`'s target grid is
  now also built at `float64` (`torch.linspace(..., dtype=torch.float64)` instead of
  that function's own `float32` default) for the same reason, even though it was
  not the one that actually crashed in this report: `ResampleTransform` always
  upcasts `target_coords` to `float64` internally anyway, so building it at
  `float32` first only ever loses precision for no benefit, and could in principle
  round a target endpoint fractionally outside the tightest curve's own range and
  trip the `extrapolate=False` check for no real reason.
