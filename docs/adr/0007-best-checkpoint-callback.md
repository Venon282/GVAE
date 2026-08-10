# 0007 — Best-model checkpointing (`BestCheckpointCallback`)

**Status:** accepted
**Date:** 2026-08-03

## Context

`docs/adr/0006-reproducibility-seed-and-checkpointing.md` shipped
`CheckpointCallback`: periodic, `every_n_epochs`, optionally pruned to
the last `keep_last_n` saves by save order. Reviewing it surfaced a
real gap: that callback's actual value is narrow. It serves resuming
an interrupted training run (a crash, a cluster preemption, a manual
stop), where you want the **most recently saved** state, with
`global_step`, optimizer momentum, and RNG state intact, so the run
does not restart from scratch. It does not serve, and was never meant
to serve, "give me the best model to evaluate or visualize", which is
what this conversation's stated goal for checkpointing actually was
("nécessaire pour pouvoir relancer l'éval/visualisation sans
réentraîner"). The most recent epoch is not necessarily the best one;
`keep_last_n` prunes by recency, not by validation performance. Without
a metric-aware companion, `CheckpointCallback` alone mostly spends disk
space on a use case (long-run resume) that may not even be a current
priority for a single, short MVP training run.

## Decision

Add `BestCheckpointCallback` to `training/checkpoint.py`, alongside
`CheckpointCallback` rather than folding metric-awareness into it as
extra flags: the two solve genuinely different problems (resume a
long run vs. select the best model), and giving them separate,
single-purpose classes keeps each one's behavior unambiguous, matching
this project's "one responsibility per file/class" convention (spec
§10) rather than growing one class with interacting modes.

- `BestCheckpointCallback(path, monitor="val/loss/total", mode="min",
  config=None)`: tracks `monitor` in each epoch's metrics dict, and
  saves **only when it improves**, always overwriting the same single
  file at `path`. There is no pruning logic to reason about: `path` is
  always exactly the best model seen so far.
- `mode` is `"min"` (default, e.g. a loss) or `"max"` (e.g. an
  accuracy, or any custom metric a `TrainerCallback` might contribute
  to the metrics dict later).
- Raises `KeyError` if `monitor` is missing from an epoch's metrics
  (the common mistake: monitoring a `"val/..."` key without passing
  `val_dataloader` to `Trainer.fit`), rather than silently never saving
  anything.
- `CheckpointCallback`'s docstring was rewritten to state its actual,
  narrower purpose explicitly (resuming, not model selection) and
  point at `BestCheckpointCallback` for the other case, so the
  distinction is documented at the point someone would reach for
  either one, not only in this ADR.
- Both callbacks delegate to the exact same `Trainer.saveCheckpoint` /
  `saveCheckpoint` used everywhere else: same checkpoint format, same
  `loadCheckpoint` path, whether saved periodically, saved on
  improvement, or saved by calling `Trainer.saveCheckpoint` directly.

## Consequences

- The practical need this conversation raised for checkpointing
  ("re-run eval/visualization without retraining") is now actually
  served: `BestCheckpointCallback` plus `loadCheckpoint` gives a single
  always-current-best file to load at any time.
- `CheckpointCallback` remains available for its narrower, real purpose
  (resuming a long run); nothing about it changed except its
  docstring. It is not required for a short MVP training run and can
  be omitted from `Trainer`'s `callbacks` list entirely without losing
  anything relevant to model selection.
- `tests/integration/test_checkpoint.py::TestBestCheckpointCallback`
  covers: saving only on improvement (both `mode="min"` and
  `mode="max"`), always overwriting a single file, that the loaded
  checkpoint's history actually matches the best (not the last) epoch,
  the missing-`monitor`-key error path, and working correctly with a
  validation dataloader.
- Not built: keeping the top-`K` best checkpoints (`K > 1`) rather than
  only the single best. A natural extension of the same mechanism
  (a small sorted list of `(value, path)` pairs, evicting the worst
  once more than `K` exist), not added now since the single-best case
  is what was actually asked for and is enough for the current
  single-run MVP workflow.
