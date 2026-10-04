# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-10-04

### Added

- `training/callbacks/` subpackage: `TrainerCallback` (relocated from
  `training/callbacks.py`) and the checkpoint callbacks now self-register into a
  unified `training.callbacks` registry (`registerCallback`/`getCallbackClass`/
  `listRegisteredCallbacks`, mirroring every other pluggable strategy in this
  codebase), instead of `CheckpointCallback`/`BestCheckpointCallback` living
  ungoverned together in `training/checkpoint.py`. `TrainingConfig.callbacks:
  dict[str, dict[str, Any] | None]` (registry name -> constructor kwargs) replaces
  the old `checkpoint: CheckpointConfig` field: a run now opts into exactly the
  checkpoint/scheduling-style callbacks it wants, in any order, instead of always
  carrying one hardcoded field. Experiment loggers (`training/loggers/`,
  `TrainingConfig.loggers`) are unaffected: they keep their own, separate
  `registerLogger`/`getLoggerClass` registry and config field, unchanged by this
  work. See `docs/adr/0022-callback-registry.md`.
- `training/callbacks/reduce_lr_on_plateau.py` (`ReduceLrOnPlateau`, registered
  `reduce_lr_on_plateau`): the same algorithm and constructor arguments as
  `torch.optim.lr_scheduler.ReduceLROnPlateau` (`mode`, `factor`, `patience`, `threshold`,
  `threshold_mode`, `cooldown`, `min_lr`, `eps`), reimplemented as a `TrainerCallback` so it
  reads its monitored metric from `Trainer.fit`'s own epoch-metrics dict (`monitor`) and
  lowers `trainer.optimizer`'s learning rate on a plateau, with no separate manual `.step()`
  call needed.
- `training/callbacks/early_stopping.py` (`EarlyStopping`, registered `early_stopping`):
  plain `torch` ships no early-stopping class of its own, so this callback reuses the exact
  same plateau-detection algorithm `ReduceLrOnPlateau` does (`training/callbacks/_plateau.py`'s
  shared `PlateauTracker`) and stops training (`Trainer.should_stop = True`) instead of
  lowering the learning rate once the monitored metric plateaus.
- `Trainer.should_stop`: a new `bool` attribute (`False` by default, reset at the start of
  every `fit()` call) a callback's `onEpochEnd` can set to end the run after the current
  epoch's remaining callbacks have all run, rather than raising or returning a sentinel
  `fit()` would have to interpret. The one `Trainer` change `EarlyStopping` needed.
- `tests/integration/test_callback_registry.py`, `test_plateau_tracker.py`,
  `test_early_stopping.py`, `test_reduce_lr_on_plateau.py`, `test_checkpoint_callbacks.py`
  (new): registry mechanics and every built-in name (explicitly covering that experiment
  loggers are *not* registered here), the shared plateau algorithm's own value correctness,
  and each new callback's behavior both via direct calls and through a real `Trainer.fit()`
  run. `test_trainer.py` gained a `TestShouldStop` class.

- `fusion/moe.py` (`moe`), `fusion/concat_mlp.py` (`concat_mlp`) and
  `fusion/cross_attention.py` (`cross_attention`): the three remaining fusion strategies of
  spec §4, registered next to `poe` and selectable by name from `fusion_strategies` /
  `latent.fusion.strategy` with no change to `GlobalVae` or the config layer. `moe` fuses
  per-modality experts by closed-form moment matching of their mixture (fused variance grows
  when experts disagree). `concat_mlp` concatenates every modality's `(mu, logvar)` through an
  MLP and, per spec §5, handles a missing modality with an explicit scheme (zero imputation
  plus an optional per-modality presence mask) while still reporting
  `handlesMissingModalities=False`. `cross_attention` turns each active modality into one token
  fused by a Transformer encoder and mean-pooled, so a missing modality is just an omitted
  token. `latent_dim` (and `modality_dims` for `concat_mlp`) are passed through
  `fusion.kwargs`, not auto-filled. See `docs/adr/0020-additional-fusion-strategies.md`.
- `docs/how-to/choose-a-fusion-strategy.md` (new, added to the mkdocs nav): how to pick and configure a fusion strategy, with YAML and Python examples for each.
- `fusion/residual.py` (`ResidualFusion`) and the `residual` flag of spec §4/§9, implemented for
  every fusion strategy at once instead of some of them: `GlobalVae(..., fusion_residual={"z":
  True})`, `GlobalVae.createSingleLatent(..., fusion_residual=True)` and `FusionConfig.residual`
  (`residual: true` next to `strategy:` in YAML, as in spec §9). The fused posterior starts as the
  mean of the active experts and a learned gate (initialized at 0) opens the strategy's own
  correction: `fused = skip + gate * (strategy(params) - skip)`. `gate = 1` recovers the strategy
  exactly, and missing-modality support is inherited from it. Off by default, so existing models,
  configs and checkpoints are unchanged; enabling it changes the module tree, so checkpoints do not
  cross the flag. The spec names the flag without defining it, so the definition is a decision:
  see `docs/adr/0021-fusion-residual-connection.md`.
- `tests/integration/test_fusion_residual.py` (new): the wrapper's properties (identity at
  initialization, `gate = 1` recovers the strategy, interpolation, gradients, subsets, error
  paths) against all four strategies, plus the `GlobalVae` wiring and the YAML path.
- `tests/integration/test_yaml_shapes.py` (new): see the first entry under Fixed.
- `tests/integration/test_fusion.py` (new): the fusion registry had no unit test although
  spec §10 requires one, and `poe.py` had no direct test. Covers registration, lookup and the
  duplicate/unknown-name paths, `poe` and `moe` against hand-computed formulas, missing-modality
  behaviour, gradient flow for all four strategies, and each new strategy wired through
  `GlobalVae.createSingleLatent` with every modality present and with one missing.
- `encoders/TwoDCnnResidualEncoder.py` (`2d_cnn_resnet_encoder_v1`) and
  `decoders/TwoDCnnResidualDecoder.py` (`2d_cnn_resnet_decoder_v1`): the 2D residual
  ("ResNet-style") encoder/decoder pair for spec §6's image modality (spec §7), the 2D
  generalization of `OneDCnnResidualEncoder`/`OneDCnnResidualDecoder`, generalized the same
  way ADR 0017 generalized the plain pair (`int` / `tuple[int, int]` / `list` convention for
  every shape-like hyperparameter, per-stage `block_depths` and `shortcut_kernel_sizes`, exact
  per-axis output-shape verification instead of resizing). Built on `Residual2DBlock` /
  `Residual2DUpBlock` (`utils/conv_blocks.py`), whose shortcut is verified at construction time
  to reach the main path's shape for every input shape, independently on both axes, through the
  new `computeConv2dLengthOffset` / `computeConvTranspose2dLengthOffset`
  (`utils/conv_math.py`). See `docs/adr/0018-2d-residual-encoder-decoder.md`.
- `tests/integration/test_residual_image_encoder.py` and
  `tests/integration/test_residual_image_decoder.py` for the pair above.
- `tests/integration/test_conv_math.py` (new): unit tests for `utils/conv_math.py`, which had
  none. Every 1D and 2D formula is cross-checked against the shape a real `nn.Conv1d`/`Conv2d`/
  `ConvTranspose1d`/`ConvTranspose2d`/`MaxPool`/`Upsample` stack actually produces, over sweeps
  that include even kernels, strides and dilations above 1, and non-square configurations where
  the two axes disagree (so an axis mix-up cannot pass by symmetry). The length-offset helpers
  are checked against their defining property (equal offsets imply equal output shapes for every
  input length, the guarantee a residual shortcut relies on), and the solvers
  (`solveConvTranspose*OutputPadding`, `solveMinimumInput*`) against exact-minimum properties.
- `tests/integration/test_conv_blocks.py` gained `TestResidual2DBlock` and
  `TestResidual2DUpBlock` (it only covered the 1D blocks): shapes with non-square kernels and
  strides, flexible depths, identity vs projection shortcut, both upsample modes, gradient flow,
  the decoder-only output-suppression flags, every error path (including a shortcut mismatch on a
  *single* axis, which a symmetric test could not detect), and a sweep over odd and even input
  shapes on each axis independently for the "every input shape, not one example shape" guarantee.
- `examples/03_signal_image_to_image.py` and `tests/integration/test_signal_image_example.py`:
  the multimodal example, `(signal, image) -> image` where the output image is not the input
  image. Two encoders (1D and 2D residual) feed one latent space through PoE fusion and a single
  decoder produces the clean target, on synthetic data built so that neither input suffices
  alone (the 1D signal is blind to the vertical position, the degraded input image is noisy and
  partly erased). Reports, per input subset, what the decoder produces from each input alone and
  from both (ADR 0016), next to the "return the degraded input" baseline (typical test-set R^2:
  about 0.33 from the signal, 0.57 from the image, 0.76 from both), and draws a translation grid
  figure (the framework's cross-modal matrix plot handles 1D series only). The model is built
  from an explicit `RoutingGraph`, since `createSingleLatent` ties decoder names to encoder
  names. `Trainer` uses the whole batch as encoder input *and* as reconstruction target, and
  `GlobalVae.forward` raises `KeyError` on a key with no encoder, so the example defines a small
  `TranslationTrainer` (one overridden method, plus validation without modality dropout); a test
  documents why it is needed and will fail if `Trainer` ever no longer requires it.
- `tests/integration/test_docs_adr_navigation.py`: fails when an ADR file is missing from
  `docs/adr/index.md` or `docs/adr/SUMMARY.md`, when ADR numbers have a gap or a duplicate, when
  a listed link is dead, when an index row is a placeholder, when `mkdocs.yml` stops using the
  directory form for the ADR and changelog nav entries, and (if the `docs` extra is installed)
  when the *built* sidebar does not list every ADR.
- `docs/adr/SUMMARY.md`: one sidebar entry per ADR (see "Fixed").
- `docs/adr/0019-decouple-encoder-inputs-from-decoder-targets.md`: `GlobalVae.selectEncoderInputs`,
  the shared restriction every framework entry point (`Trainer`, `evaluation.evaluate.evaluate`,
  the `visualization` collectors) now applies to a raw per-modality batch before its own forward
  pass, so a decoder's reconstruction target no longer needs to also be one of the model's
  encoder inputs anywhere in the framework, not only inside a caller's own workaround. See
  "Fixed" for what this replaces.
- `tests/integration/test_trainer.py` gained `TestTargetOnlyDecoderKeys` (a decoder target with
  no matching encoder trains and evaluates through the stock `Trainer`, `GlobalVae.forward`
  raises a clear `KeyError` if that restriction is skipped, `selectEncoderInputs` raises
  `ValueError` if nothing in a batch matches any encoder) and
  `TestModalityDropout.test_evaluate_never_applies_modality_dropout`.
- `tests/integration/test_signal_image_example.py::TestTrainerHandlesTargetOnlyKeys` (replaces
  `TestTranslationTrainer`, see "Fixed"): the same scenarios, against the stock `Trainer`.
- `scripts/evaluate.py` gained `--inverse-transform-factory`: an optional
  `"module.path:function_name"` returning a `dict[str, Callable[[Tensor], Tensor]]`, forwarded to
  `exportEvaluationFigures`'/`exportCrossModalFigures`'s own `inverse_transforms` so
  reconstruction figures can show original-scale values instead of whatever `data.transforms`
  preprocessing (spec §6.2) was applied before training. `tests/integration/_script_fixtures.py`
  gained `buildInverseTransformsForScript`; `test_evaluate_script.py` gained
  `TestInverseTransformFactory`.
- `scripts/visualize_latent.py` gained `--history-no-twin-axis`, `--history-log-scale`, and
  `--history-twin-log-scale`: the loss-curve plot now defaults to splitting regularization loss
  onto its own, independently-scaled secondary axis (`plotLossCurves`'s own `twin_metrics`,
  already used by `examples/03_signal_image_to_image.py` but previously unused by this script;
  spec §2.3 already notes regularization is frequently orders of magnitude smaller than
  reconstruction/total loss). `tests/integration/test_visualize_latent_script.py` gained
  `TestLossCurveAxes`.

### Changed

- `docs/adr/index.md` now lists ADRs 0017 and 0018, which existed but were missing. Its rows for
  0015 and 0016 had been committed as "*Guess:*" placeholders written without reading the ADRs;
  they now describe the ADRs as written.
- `utils/conv_math.py` gained a module docstring (it had none) and two single-line summaries
  where a wrapped one violated `D205`. No formula changed.
- `examples/README.md` and `docs/getting-started.md` describe example 03; the README no longer
  calls the single-modality pipeline "the only configuration the framework fully supports".
- `TwoDCnnResidualEncoder.py`, `test_residual_image_encoder.py` and
  `test_residual_image_decoder.py` formatted with `ruff format`.
- `examples/03_signal_image_to_image.py` no longer defines its own `TranslationTrainer`: the
  stock `Trainer` now handles a decoder target with no matching encoder, and never applies
  modality dropout during `evaluate()`, itself (ADR 0019). `main()` now builds a plain `Trainer`.
- `CheckpointCallback`/`BestCheckpointCallback` moved from `training/checkpoint.py`, where
  they lived together in one file, to two separate files:
  `training/callbacks/checkpoint.py` (registered `checkpoint`) and
  `training/callbacks/best_checkpoint.py` (registered `best_checkpoint`), one class per
  file per spec §10's "Modularity" rule. The checkpoint file format itself
  (`saveCheckpoint`/`loadCheckpoint`/`CheckpointMetadata`, imported the same way as before)
  is unchanged and stays in `training/checkpoint.py`. `configs/training/default.yaml`
  updated to the new `callbacks:` shape for these two entries, reproducing its previous
  behavior exactly; its `loggers:` list is untouched. See `docs/adr/0022-callback-registry.md`.

### Removed

- `TrainingConfig.checkpoint` (`CheckpointConfig`): replaced by the registry-driven
  `TrainingConfig.callbacks` field (see Changed, above, and
  `docs/adr/0022-callback-registry.md`). `TrainingConfig.loggers` (`LoggerEntryConfig`) is
  unaffected and was not removed.

### Fixed

- `TwoDCnnDecoder` and `TwoDCnnResidualDecoder` rejected a correct configuration when
  `output_shape` (or `seed_shape`) came from a YAML/Hydra config: YAML has no tuple type, so
  `[64, 64]` arrived as a `list`, and the exact-shape check compared the computed `tuple` against
  it (`(64, 64) != [64, 64]`). Both decoders now normalize the two shapes to tuples up front (and
  `computeOutputShape` does the same for `seed_shape`). The same root cause also affected
  `utils.stage_config.resolveSpatialShape`, which silently turned `[64, 64]` into `([64, 64],
  [64, 64])`: it now accepts a `list` as an explicit shape exactly like a `tuple`, which fixes a
  per-stage entry written `[[3, 5], [3, 5]]` in YAML and `ResampleTransform(target_size=[16,
  16])`. A top-level `list` is still the per-stage wrapper in `broadcastPerStageShape`,
  unchanged. Covered by `tests/integration/test_yaml_shapes.py`, which fails without the fix.
- The ADRs (and the per-version changelog pages) were not visible in the site sidebar.
  `mkdocs.yml` pointed the nav at a file (`adr/index.md`, `changelog/SUMMARY.md`), and
  `literate-nav` only expands a `SUMMARY.md` when the nav entry points at its *directory*: the
  ADR section showed a single page, and the changelog entry rendered `SUMMARY.md` as one ordinary
  page titled "SUMMARY". Both entries now use the directory form (`adr/`, `changelog/`); all 18
  ADRs and the 3 changelog pages appear in the built sidebar, and `mkdocs build --strict` passes.
- **`Trainer` (and every other framework entry point) could not train or evaluate a model
  whose decoder target is not also an encoder input** (a translation-style
  `image_in -> image_out` model): `GlobalVae.forward` raised `KeyError` on the target-only
  key the moment the whole batch was handed to it as `inputs`. Fixed by
  `GlobalVae.selectEncoderInputs`, now called before every framework forward pass over a
  raw batch (`Trainer.computeLosses`, `evaluation.evaluate.evaluate`,
  `visualization.latent_plot.collectLatentParams`,
  `visualization.reconstruction_plot.collectReconstructions`); see ADR 0019. This closes
  `examples/03_signal_image_to_image.py`'s own `TranslationTrainer` workaround, removed
  (see "Changed").
- **`Trainer.evaluate` applied modality dropout during validation**, so the reported
  `"val/..."` metrics depended on which modalities a given validation pass happened to
  randomly hide, adding spurious noise to whatever `BestCheckpointCallback` or an
  early-stopping rule compares across epochs. Fixed by `Trainer.computeLosses`'s new
  `apply_dropout` parameter, which `evaluate()` sets to `False` (ADR 0019, which also
  measures the effect: about a 45% reduction in run-to-run spread on a
  two-modality/`modality_dropout_p=0.9` smoke case).
- **`visualization.latent_plot`'s PCA projection was silently non-deterministic**:
  `_pcaProject` used `torch.pca_lowrank`, a *randomized* low-rank SVD that draws an
  internal `torch.randn(...)` sketch matrix, so two calls on the exact same latent
  vectors could (and, measured directly, did: max absolute difference `5.6` on a toy
  16-dimensional example, versus the data's own scale of order `1`) return visibly
  different 2D projections depending only on the ambient RNG state at call time, not on
  the data. This was the leading cause of `scripts/evaluate.py`'s and
  `scripts/visualize_latent.py`'s latent scatter plots looking different for the exact
  same checkpoint and dataloader, despite both collecting the identical `mu` values.
  `_pcaProject` now uses `torch.linalg.svd` (exact, not randomized): repeated calls on the
  same input are now bit-for-bit identical, matching what `projectLatentSamples`'s own
  `seed` parameter docstring already (and now correctly) claimed for `"pca"`.

**Verification note (ADR 0019).** `Trainer`'s and `GlobalVae`'s new behavior was checked with a
real two-encoder/PoE model on real PyTorch: `computeLosses`/`evaluate` train and evaluate
correctly on a batch carrying a decoder-only key, `forward` raises the documented `KeyError`
when that restriction is skipped, and ten repeated `evaluate()` calls (no seed reset between
them, mirroring how a real training loop's RNG stream keeps advancing epoch to epoch) show a
measurably smaller spread than the old dropout-in-eval behavior on the same model and batch.
The `torch.pca_lowrank` non-determinism above was confirmed directly (two calls, identical
input, materially different output) before, and ruled out (identical output) after, switching
to `torch.linalg.svd`. The full test suite (`pytest tests/`, 1535 tests) passes against real
PyTorch 2.14 (CPU), apart from the same pre-existing `umap-learn`-dependent skip noted above.

**Verification note.** The whole test suite was run against real PyTorch 2.14 (CPU) for the first time for the 2D
pair, which ADRs 0017 and 0018 record as only verified against a shape-tracking stand-in:
every test passes, including the gradient-flow and unconstrained-output tests the stand-in
could not run, apart from `test_latent_plot.py::test_umap_projects_to_the_requested_dimensionality`
(needs the optional `umap-learn`, absent from the environment). `mypy --strict` reports no
issue on the package. The new tests were mutation-checked: deliberately breaking a formula,
a per-axis check, or a documented behavior makes them fail.
