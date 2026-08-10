# Global Multimodal VAE

A modular, extensible multimodal Variational Autoencoder framework.
Ground truth for the project's architecture and decisions is
`global-vae-project-specification.md` (kept alongside this repo) — read
it first if anything below is unclear.

## Status

This is an initial scaffold, not a finished framework. What's built:

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
  the caller's own preprocessing, and loss/step/beta-schedule curves.
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
- A unit-test suite for the registries and the routing-graph validator,
  plus end-to-end integration tests for the `EN-L1-DN` configuration
  and for `Trainer` (with dummy encoders/decoders/fusion — see
  `docs/adr/0001-phase1-default-configuration.md` for why `EN-L1-DN`
  first).

What's deliberately **not** built yet, and why (see `NOTE.md` in each
directory): concrete image encoders/decoders, the MoE/concat_mlp/
cross-attention fusion strategies, an image-comparison reconstruction
plot (needs an image decoder to exist first), and the data pipeline
(out of this framework's scope by design; the person building on it
owns their own data loading). Each of these either depends on an open
question flagged in spec §11 that hasn't been decided yet, or is
simply the next not-yet-reached milestone — per spec §12, an open
question is a reason to ask, not to guess.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
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

## Adding a new fusion or assembler strategy

Same pattern: subclass `AbstractFusion` (`fusion/`) or
`AbstractAssembler` (`latent/assembler.py`), register with
`@registerFusion("name")` / `@registerAssembler("name")`.

## Extending beyond `EN-L1-DN`

The routing-graph machinery (`latent/base.py`, `latent/single.py`,
`latent/factorized.py`) already supports arbitrary encoder-latent-decoder
topologies, including multiple independent latent spaces. `GlobalVae`
currently only *drives* the single-fused-latent case end-to-end; growing
it (or introducing sibling model classes) to cover the other 7
configurations in spec §2.1 is the next milestone. See
`docs/adr/0001-phase1-default-configuration.md`.
