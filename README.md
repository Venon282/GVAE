# Global Multimodal VAE

**Current release: v0.1.0 (2026-08-29)** — see `CHANGELOG.md` for the full release
notes. This is the framework's first release: spec §6.1 milestone 1 (a
single-modality signal VAE, trained/checkpointed/evaluated/visualized end to end)
works, along with everything else described below. It is not, and does not claim to
be, a finished framework (see "What's deliberately not built yet").

A modular, extensible multimodal Variational Autoencoder framework.
Ground truth for the project's architecture and decisions is
`global-vae-project-specification.md` (kept alongside this repo) — read
it first if anything below is unclear.

Want to see it work before reading further? `python examples/01_signal_vae_pipeline.py`
runs the entire pipeline (data, transforms, model, training, evaluation,
visualization) end to end on synthetic data, no setup required beyond
`pip install -e ".[dev]"`. `examples/02_config_driven_pipeline.py` runs the same
pipeline driven entirely from `configs/` YAML instead, training and comparing two
named experiment variants side by side. See `examples/README.md`.

## Status

This is an initial scaffold, not a finished framework, but the training loop and
generic data-preprocessing pipeline described in earlier drafts of this document as
"differed" are **both implemented as of this release**, not pending: `training/
trainer.py`'s raw PyTorch loop was a formally decided question (checklist item A3,
`docs/adr/0005-training-loop.md`), and `data/transforms/`'s generic transforms
(checklist item covered by `docs/adr/0012-generic-data-transforms.md` and
`docs/adr/0013-coordinate-aware-resampling.md`) are real, tested code, not a
placeholder. The only part of the data pipeline still out of scope,
**permanently, not temporarily**, is dataset loading/pairing/splitting itself
(`datamodule.py`, spec §6.2) — see "What's deliberately not built yet" below for
exactly what that boundary does and does not cover. What's built:

- The extension points: `AbstractEncoder`, `AbstractDecoder`,
  `AbstractFusion`, `AbstractAssembler`, each with a self-registration
  registry.
- The latent routing graph (`latent/base.py`) and the two structural
  constraints from spec §2.2 (no orphan latent spaces; `sum`/`average`
  assemblers require matching dimensionality), enforced at
  model-construction time.
- `GlobalVae`, assembling any routing graph (spec §2.2) from
  `encoder_configs`/`decoder_configs`/`routing_graph`, plus
  `createSingleLatent(...)`, a convenience constructor for the
  **`EN-L1-DN`** configuration (spec's recommended Phase-1 default:
  per-modality encoders, one fused latent, per-modality decoders) and
  for the single-modality `signal -> z -> signal` case (spec §6.1
  milestone 1), which needs no fusion strategy at all.
- Concrete implementations: `OneDCnnEncoder`/`OneDCnnDecoder`
  (`1d_cnn_encoder_v1`/`1d_cnn_decoder_v1`, spec §6's 1D signal
  modality, length-agnostic on the encoder side via adaptive pooling)
  and `ProductOfExperts` (`poe`, spec §4's MVAE-style fusion strategy).
- Pluggable latent regularization (`losses/regularizers/`:
  `kl_standard_normal`, `free_bits_kl`, `mmd`) and pluggable
  beta-weighting schedules (`training/beta_schedules/`: `constant`,
  `linear_warmup`, `cyclical_annealing`), both spec §2.3.
- Generic, invertible data transforms (`data/transforms/`, spec §6.2):
  `log`/`standardize`/`resample`, each fully generic across
  dimensionality (a single `ResampleTransform` handles 1D/2D/3D data
  through a `num_spatial_dims` parameter, no per-dimensionality
  subclasses), plus `ComposeTransform` for chaining several into one
  invertible pipeline. `resample` additionally supports coordinate-aware
  resampling (`interpolation="scipy"`): explicit `source_coords`/
  `target_coords`, shared or per-sample, and a choice of interpolation
  method beyond evenly-spaced grids (`scipy.interpolate`'s `interp1d`
  kinds, plus `cubic_spline`/`pchip`/`akima` splines) — see
  `docs/adr/0013-coordinate-aware-resampling.md`. `DataConfig.transforms`
  (`config/data.py`) is a list of these, resolved by
  `buildTransformPipeline`; dataset loading, pairing, and splitting
  remain entirely out of scope, permanently (`data/NOTE.md`). See
  `docs/adr/0012-generic-data-transforms.md`.
- `training/trainer.py`: `Trainer`, a raw PyTorch training loop
  (forward, reconstruction + regularization loss, backward, optimizer
  step, device placement, optional modality dropout, per-step/per-epoch
  metrics via `TrainerCallback` hooks). See
  `docs/adr/0005-training-loop.md`.
- Reproducibility (spec §10): `utils/seed.py`'s `setGlobalSeed` (global
  RNG seeding, a documented deterministic-mode flag) and
  `training/checkpoint.py`'s `saveCheckpoint`/`loadCheckpoint`,
  `CheckpointCallback` (periodic, for resuming an interrupted run), and
  `BestCheckpointCallback` (saves only on improvement of a monitored
  metric, so evaluation or visualization can run on the best model
  without retraining). See
  `docs/adr/0006-reproducibility-seed-and-checkpointing.md` and
  `docs/adr/0007-best-checkpoint-callback.md`.
- Experiment tracking (spec §10): `training/loggers/`'s
  `AbstractExperimentLogger` (self-registered like every other
  pluggable strategy) with two built-in backends, `CsvLogger` (a
  single long/tidy-format file, no extra dependency) and
  `TensorBoardLogger` (`tensorboard` is an optional extra,
  `pip install -e ".[tensorboard]"`). Every logger is itself a
  `TrainerCallback`, so no `Trainer` changes were needed to support it,
  and running several loggers at once needs no dedicated "composite"
  class (`callbacks=[CsvLogger(...), TensorBoardLogger(...)]` just
  works). See `docs/adr/0008-experiment-loggers.md`.
- Visualization (spec §10, spec §6.1 milestone 1): `visualization/`'s
  latent-space projection and scatter plots (`"pca"`/`"tsne"`/`"umap"`),
  a per-dimension KL bar chart for spotting posterior collapse,
  reconstruction overlay plots with an `inverse_transform` hook for
  the caller's own preprocessing (now directly fillable with
  `buildTransformPipeline(...).inverse`, see above), and
  loss/step/beta-schedule curves.
  Requires the `visualization` extra (`pip install -e ".[visualization]"`).
  Every function returns a plain `matplotlib.figure.Figure`; nothing
  displays, saves, or logs it, so it composes directly with
  `training/loggers/`'s `logFigure`. See
  `docs/adr/0009-visualization.md`.
- Evaluation (spec's C8 requirement): `evaluation/`'s `evaluate(model,
  dataloader, ...)`, needing only a `GlobalVae` and a dataloader (no
  `Trainer`): reconstruction metrics per modality (mse/rmse/mae/r2/
  pearson_r by default), a regularization value per latent space that
  always includes a `kl_standard_normal` number directly comparable
  across runs regardless of which regularizer actually trained the
  model, and a JSON/console report. `exportEvaluationFigures` saves
  reconstruction and latent-space figures via `visualization/`.
  `scripts/evaluate.py` is the CLI entry point
  (`python scripts/evaluate.py --checkpoint ... --model-factory ...
  --dataloader-factory ...`), with model/data construction supplied as
  your own factory functions rather than hardcoded. See
  `docs/adr/0010-evaluation.md`.
- Config management (spec §10): `global_vae/config/` is a Hydra-driven, dataclass-validated
  config layer covering the model, data, and training domains
  (`ModelConfig`, `DataConfig`, `TrainingConfig`, composed into one
  `ExperimentConfig`). `buildModelFromConfig`/`buildTrainerFromConfig`/
  `buildDataloadersFromConfig` turn a validated config into a real
  `GlobalVae`/`Trainer`/dataloaders; `DataConfig` is a schema-only
  contract (paths, batch size, split, a generic transform pipeline, and a
  `loader_factory` reference to your own data-loading callable), never
  a dataset implementation, matching this framework's data-pipeline
  scope boundary. `scripts/train.py` is the Hydra CLI entry point
  (`python scripts/train.py data.train_path=... data.loader_factory=...`),
  running `configs/experiment/signal_vae.yaml` (the spec §6.1
  milestone 1 single-modality signal VAE) by default. See
  `docs/adr/0011-hydra-config-layer.md`.
- `scripts/visualize_latent.py`: a standalone CLI for quickly inspecting a checkpoint's
  latent space and training curves without running a full evaluation pass, distinct
  from `scripts/evaluate.py`'s own figure export. Saves a latent-space scatter plot and
  a per-dimension KL bar chart per latent space, plus a training-curve plot whenever
  the checkpoint carries one. Supports coloring the scatter plot by an arbitrary batch
  field (`--label-key`), restricting which latent spaces get plotted
  (`--latent-names`), and plotting realized samples instead of the posterior mean
  (`--use-samples`). Model/dataloader construction supplied as your own factory
  functions, the same convention as `scripts/evaluate.py`.
- A unit-test suite for the registries and the routing-graph validator,
  plus end-to-end integration tests for the `EN-L1-DN` configuration
  (with dummy encoders/decoders/fusion, see
  `docs/adr/0001-phase1-default-configuration.md` for why `EN-L1-DN`
  first) and for `Trainer`; a dedicated real-module integration test for
  spec §6.1 milestone 1 (`OneDCnnEncoder`/`OneDCnnDecoder` via
  `GlobalVae.createSingleLatent`, no fusion,
  `tests/integration/test_signal_vae_milestone.py`); a small trainer
  smoke test (`tests/integration/test_trainer_smoke.py`); and
  invertibility tests for the generic data transforms
  (`tests/integration/test_transforms.py`). See spec §10's Testing
  bullet (checklist item **C11**) for the full requirement list.

What's deliberately **not** built yet, and why (see `NOTE.md` in each
directory): concrete image encoders/decoders, the MoE/concat_mlp/
cross-attention fusion strategies, an image-comparison reconstruction
plot (needs an image decoder to exist first), and the dataset-loading half
of the data pipeline — `datamodule.py`, i.e. reading files, matching/
pairing samples, and splitting — which is out of this framework's scope
by permanent design (spec §6.2; the person building on this framework
owns their own data loading). Generic preprocessing (`data/transforms/`)
is *not* in this list: it is implemented, see above. Each remaining item
either depends on an open question flagged in spec §11 that hasn't been
decided yet, or is simply the next not-yet-reached milestone — per spec
§12, an open question is a reason to ask, not to guess.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the signal-VAE milestone (spec §6.1 milestone 1)

```bash
python scripts/train.py \
    data.train_path=/path/to/your/data \
    data.loader_factory=my_project.data:buildSignalDataloaders
```

`data.loader_factory` must point at your own `(DataConfig) -> DataloaderBundle`
callable (spec: data loading stays your own responsibility). See
`configs/experiment/signal_vae.yaml` for the full default config and
`global_vae/config/data.py` for the exact contract, including the generic
`transforms` pipeline (`log`/`standardize`/`resample`, spec §6.2) your own
`loader_factory` can call via `buildTransformPipeline(config.data)` if it
wants to. Override any hyperparameter from the command line, e.g.
`training.num_epochs=50 training.optimizer.kwargs.lr=0.0003`.

Inspect the result once trained:

```bash
python scripts/visualize_latent.py \
    --checkpoint outputs/signal_vae/checkpoints/best.pt \
    --model-factory my_project.data:buildSignalModel \
    --dataloader-factory my_project.data:buildSignalDataloaders

python scripts/evaluate.py \
    --checkpoint outputs/signal_vae/checkpoints/best.pt \
    --model-factory my_project.data:buildSignalModel \
    --dataloader-factory my_project.data:buildSignalTestDataloader \
    --output-dir results/
```

## Running checks

```bash
ruff check .
ruff format --check .
mypy
pytest
```

## Repository structure

See spec §8 for the target layout; `src/global_vae/` mirrors it.

## Naming convention (deviates from PEP8 — read this before contributing)

- Classes → `CamelCase` (e.g. `GlobalVae`, `SignalEncoder`).
- Variables → `snake_case` (e.g. `latent_dim`, `batch_size`).
- **Functions and methods → `camelCase`** (e.g. `registerEncoder`,
  `computeKlLoss`), not PEP8's usual `snake_case`. This is intentional
  (spec §10). `ruff`'s `N802`/`N803`/`N806` naming rules are disabled
  in `pyproject.toml` specifically so linting doesn't silently "fix"
  this back to snake_case. Framework-mandated overrides
  (`forward`, `__init__`, and other PyTorch/Python dunder or
  base-class-required names) are the only exception — leave those as
  the base class defines them.

## Adding a new modality (spec §10 checklist)

1. Subclass `AbstractEncoder` in `encoders/`, decorate it with
   `@registerEncoder("your_encoder_name")`.
2. Subclass `AbstractDecoder` in `decoders/`, decorate it with
   `@registerDecoder("your_decoder_name")`.
3. Register both (the decorator does this — nothing else to wire up).
4. Add a config entry referencing the two registry names (see
   `configs/model/default.yaml` for the shape, once config loading is
   wired up).
5. Add a test — a unit test for the encoder/decoder shapes, and ideally
   an entry in the relevant integration test.

No core framework file should need to change. If it does, that's a
signal the registry pattern is being bypassed somewhere — flag it
rather than special-casing the new modality into `GlobalVae`.

## Adding a new fusion, assembler, or data transform strategy

Same pattern: subclass `AbstractFusion` (`fusion/`), `AbstractAssembler`
(`assemblers/`), or `AbstractTransform` (`data/transforms/`), register with
`@registerFusion("name")` / `@registerAssembler("name")` /
`@registerTransform("name")`. A new transform must stay fully generic
across dimensionality (spec §6.2): no per-modality or per-dataset logic.

## Extending beyond `EN-L1-DN`

The routing-graph machinery (`latent/base.py`, `latent/routing_graph_builders/`)
already supports arbitrary encoder-latent-decoder
topologies, including multiple independent latent spaces. `GlobalVae`
currently only *drives* the single-fused-latent case end-to-end; growing
it (or introducing sibling model classes) to cover the other 7
configurations in spec §2.1 is the next milestone. See
`docs/adr/0001-phase1-default-configuration.md`.
