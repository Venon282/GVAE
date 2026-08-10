# 0006 — Reproducibility: global seed management and checkpointing

**Status:** accepted
**Date:** 2026-08-03

## Context

Spec §10's "Reproducibility" bullet asks for three things: global seed
management, a documented deterministic-mode flag, and config
snapshotted with every run. None of these existed yet. Separately, the
practical need to re-run evaluation or visualization on an already
trained model without retraining it (raised directly in this
conversation) had no support either: `Trainer` had no way to save or
restore its state.

## Decision

### `utils/seed.py`: `setGlobalSeed(seed, deterministic=False, warn_only=False)`

A standalone function, not a `Trainer`/`GlobalVae` method: seeding
must happen before model construction (weights are initialized the
moment a layer is built), so it belongs in the caller's script, before
anything else in this framework is touched.

- Seeds Python's `random`, NumPy (soft dependency: only if installed),
  and PyTorch's CPU and CUDA generators (all devices) with the same
  `seed`.
- `deterministic=True` calls `torch.use_deterministic_algorithms(True,
  warn_only=warn_only)`, disables cuDNN's auto-tuning
  (`torch.backends.cudnn.benchmark = False`,
  `torch.backends.cudnn.deterministic = True`), and best-effort sets
  `CUBLAS_WORKSPACE_CONFIG` (via `os.environ.setdefault`, so an
  existing caller-set value is respected, not overwritten) since
  PyTorch's own documentation requires it for some CUDA operations to
  be deterministic at all. `deterministic=False` explicitly resets
  every one of these flags too, so the setting cannot silently leak
  from an earlier call within the same process (e.g. a notebook
  switching between a fast, non-deterministic run and a slow,
  reproducible one).
- `warn_only` is exposed because `torch.use_deterministic_algorithms(True)`
  otherwise raises `RuntimeError` outright the first time an operation
  with no deterministic implementation is used, which is the correct
  strict default (surface the problem instead of silently producing a
  non-reproducible run) but not always what a caller wants mid-training.

### `training/checkpoint.py`: `saveCheckpoint`/`loadCheckpoint` plus `CheckpointCallback`

- Plain module-level functions, `saveCheckpoint(path, model,
  optimizer=None, global_step=0, start_epoch=0, history=None,
  config=None, include_rng_state=True)` and `loadCheckpoint(path,
  model, optimizer=None, map_location=None, restore_rng_state=True,
  strict=True) -> CheckpointMetadata`, not `Trainer` methods
  themselves. Loading a checkpoint for evaluation only needs a model,
  not an optimizer or a `Trainer` instance at all; tying checkpoint I/O
  to `Trainer` would force constructing an unused optimizer just to
  load weights for inference. `Trainer.saveCheckpoint`/
  `Trainer.loadCheckpoint` are thin convenience wrappers around these,
  using the trainer's own model/optimizer/step/epoch/history, mirroring
  how `Trainer` already delegates to standalone functions elsewhere
  (`computeTotalReconstructionLoss`, `resolveBetaSchedules`).
- A checkpoint (`torch.save`/`torch.load`) bundles: the model's
  `state_dict` (always), the optimizer's `state_dict` (only if an
  optimizer was passed in, so an eval-only checkpoint carries no unused
  optimizer state), `global_step`/`start_epoch`/`history` (`Trainer`'s
  own bookkeeping, kept as plain instance state since
  `docs/adr/0005-training-loop.md` specifically for this purpose), an
  arbitrary `config` object (spec §10's "config snapshotted with every
  run": this framework does not define or enforce a config schema, spec
  §11 leaves that open, so `config` is stored and returned unchanged,
  whatever shape the caller's own setup uses), and RNG state
  (Python/NumPy/PyTorch), so a resumed run continues drawing from the
  same random sequence instead of silently re-seeding.
- `loadCheckpoint` uses `torch.load(..., weights_only=False)`, since a
  checkpoint intentionally carries non-tensor metadata (config, RNG
  state) that the safer tensors-only loading mode does not support;
  the module docstring documents the accompanying trust caveat
  (`torch.load` is a pickle underneath: only load checkpoints from
  sources you trust).
- `CheckpointCallback(directory, every_n_epochs=1, config=None,
  keep_last_n=None, filename_pattern=...)`: a concrete
  `TrainerCallback` making the exact example
  `training/callbacks.py`'s own docstring already used ("a checkpointer
  only overrides `onEpochEnd`") real. `keep_last_n` deletes older
  checkpoints by save order once more than `keep_last_n` exist;
  keeping the *best* `N` by some validation metric is a natural future
  extension, not built here (choosing a metric and a comparison
  direction is not something this callback can know generically).

### Bug found and fixed while building `CheckpointCallback`

`Trainer.fit` updated `self.start_epoch = epoch + 1` *after* calling
every `onEpochEnd` callback. A checkpoint saved from inside
`onEpochEnd` would therefore serialize the *previous* `start_epoch`,
one epoch behind where training should actually resume. Fixed by
moving the `self.start_epoch` update before the `onEpochEnd` callback
loop, so any callback reading `trainer.start_epoch` (not just
`CheckpointCallback`) sees the correct resume point. Caught by
`tests/integration/test_checkpoint.py::TestCheckpointCallback::test_last_checkpoint_can_be_loaded_and_matches_final_state`.

## Consequences

- Spec §10's "Reproducibility" bullet is now fully addressed: seed
  management, a documented deterministic flag, and config snapshotting
  (via `saveCheckpoint`'s `config` parameter) all exist.
- Re-running evaluation or visualization on a trained model without
  retraining (the practical motivation raised alongside spec §10) is
  now possible: `saveCheckpoint`/`loadCheckpoint` need only a `GlobalVae`
  instance, no `Trainer`.
- `tests/integration/test_seed.py` covers: reproducibility across
  Python/NumPy/PyTorch/model-initialization, that different seeds
  actually differ, the deterministic flag's effect on PyTorch's
  backend flags (both directions), the `CUBLAS_WORKSPACE_CONFIG`
  environment variable (set when missing, respected when already
  present), and `warn_only`.
- `tests/integration/test_checkpoint.py` covers: exact weight and
  optimizer-state roundtripping, config roundtripping, `Trainer`
  step/epoch/history roundtripping and resumed epoch numbering, RNG
  state roundtripping (including the "restore disabled" and "not
  saved" cases), the relevant error paths (missing file, requesting
  optimizer restore from an optimizer-less checkpoint), and
  `CheckpointCallback`'s periodic saving, `keep_last_n` pruning, and
  invalid-argument error paths.
- `training/NOTE.md` updated: checkpointing and global seed management
  are no longer listed as deferred. Experiment tracking (a concrete
  TensorBoard/CSV/W&B/MLflow logger) remains the one item from spec
  §10's list not yet built; `TrainerCallback` is still the seam it
  will plug into, exactly as `CheckpointCallback` now demonstrates for
  checkpointing.
