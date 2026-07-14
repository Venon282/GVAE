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
- `GlobalVae`, assembling the **`EN-L1-DN`** configuration (spec's
  recommended Phase-1 default: per-modality encoders, one fused latent,
  per-modality decoders) from a config dict.
- A unit-test suite for the registries and the routing-graph validator,
  plus the first of the 8 end-to-end integration tests spec §10 asks
  for (`EN-L1-DN`, with dummy encoders/decoders/fusion — see
  `docs/adr/0001-phase1-default-configuration.md` for why this one
  first).

What's deliberately **not** built yet, and why (see `NOTE.md` in each
directory): concrete signal/image encoders and decoders, the PoE/MoE/
cross-attention fusion strategies, the data pipeline, and the training
loop. Each of these depends on an open question flagged in spec §11
(first joint dataset/task, Lightning vs. raw loops, loss-weighting
schedule) that hasn't been decided yet — per spec §12, that's a reason
to ask, not to guess.

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
