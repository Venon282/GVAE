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
  `training/checkpoint.py`'s `saveCheckpoint`/`loadCheckpoint`/
  `CheckpointCallback` (model + optimizer + step/epoch/history + an
  arbitrary config snapshot + RNG state, so evaluation or visualization
  can run on a trained model without retraining it). See
  `docs/adr/0006-reproducibility-seed-and-checkpointing.md`.
- A unit-test suite for the registries and the routing-graph validator,
  plus end-to-end integration tests for the `EN-L1-DN` configuration
  and for `Trainer` (with dummy encoders/decoders/fusion — see
  `docs/adr/0001-phase1-default-configuration.md` for why `EN-L1-DN`
  first).

What's deliberately **not** built yet, and why (see `NOTE.md` in each
directory): concrete image encoders/decoders, the MoE/concat_mlp/
cross-attention fusion strategies, the data pipeline (out of this
framework's scope by design; the person building on it owns their own
data loading), and concrete experiment loggers (TensorBoard/CSV/W&B/
MLflow; `TrainerCallback` is the seam they plug into, exactly as
`CheckpointCallback` already demonstrates for checkpointing). Each of
these either depends on an open question flagged in spec §11 that
hasn't been decided yet, or is simply the next not-yet-reached
milestone — per spec §12, an open question is a reason to ask, not to
guess.

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
