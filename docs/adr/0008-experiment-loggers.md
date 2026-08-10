# 0008 — Pluggable experiment loggers (`training/loggers/`)

**Status:** accepted
**Date:** 2026-08-04

## Context

Spec §10's "Experiment tracking" bullet asks for logging losses,
latent-space visualizations, and reconstructions per run, naming
Weights & Biases or MLflow as examples. Earlier in this project's
history (informal decision "A4"), the choice was made to keep this
user-selectable rather than hardcoding one backend, starting with
TensorBoard and CSV. `TrainerCallback` (`docs/adr/0005-training-loop.md`)
and `CheckpointCallback` (`docs/adr/0006-reproducibility-seed-and-checkpointing.md`)
both already anticipated this: "a metrics logger... `TrainerCallback`
is the seam this plugs into" was the running example used in both.

## Decision

- `training/loggers/base.py`: `AbstractExperimentLogger(TrainerCallback, ABC)`.
  Combines `TrainerCallback`'s plain hook interface with `ABC`'s
  enforcement of one mandatory method, `logScalar`. `onStepEnd` and
  `onEpochEnd` are pre-wired (not left to each backend to reimplement)
  to call `logScalars(metrics, x, tag=...)`, tagged `"step"` or
  `"epoch"` since the two carry differently-shaped metric dicts (raw
  `"loss/..."` keys vs. epoch-averaged, `"train/"`/`"val/"`-prefixed
  ones); `onTrainEnd` calls `close()`. `logImage`/`logFigure`/`close`
  have sensible defaults (a warning-and-skip for the first two,
  a no-op for `close`) so a backend that cannot or need not support
  them (e.g. a purely tabular one) is not forced to.
- `training/loggers/registry.py`: `registerLogger`/`getLoggerClass`/
  `listRegisteredLoggers`, the same self-registration pattern as every
  other pluggable strategy in this codebase (encoders, fusion,
  regularizers, beta schedules, ...). This is a different shape of
  extension point from `TrainerCallback` itself: callbacks compose (a
  plain list, several active at once), loggers are picked by name one
  at a time from a registry, exactly the shape spec §10's own phrasing
  implies ("Weights & Biases *or* MLflow"). No dedicated
  "CompositeLogger" class exists to combine several loggers: none is
  needed, since `Trainer.callbacks` already accepts a list and every
  logger already is a `TrainerCallback`
  (`callbacks=[CsvLogger(...), TensorBoardLogger(...)]` just works).
- `training/loggers/csv_logger.py`: `CsvLogger`, a single CSV file in
  **long/tidy format**: one row per `(x, tag, metric, value)`, instead
  of one wide row per step with one column per metric. A wide format
  would need its column set to match exactly across every row, but the
  metric-key set genuinely differs between `"step"` and `"epoch"`
  entries and can even change between one `Trainer.fit` call and the
  next (e.g. adding a validation dataloader partway through a
  notebook session); CSV has no clean way to add a column to an
  already-started file. The fixed 4-column long format never runs
  into this, and is trivial to pivot back into a wide table with
  pandas (`df.pivot(index=["x", "tag"], columns="metric",
  values="value")`). Does not override `logImage`/`logFigure`: a CSV
  has no honest way to represent visual content, so the base class's
  warn-and-skip default is the correct behavior, not a gap to fill.
- `training/loggers/tensorboard_logger.py`: `TensorBoardLogger`, wrapping
  `torch.utils.tensorboard.SummaryWriter`. Supports `logImage`
  (`add_image`) and `logFigure` (`add_figure`, `figure` typed `Any`
  so this module never requires matplotlib as a dependency merely to
  type-check). The `tensorboard` PyPI package is a genuine soft
  dependency: `SummaryWriter` is imported inside
  `TensorBoardLogger.__init__`, not at module level, so importing
  `training.loggers` (which imports `tensorboard_logger.py` for its
  `@registerLogger` side effect, per this codebase's established
  registration convention) never requires `tensorboard` to be
  installed; only actually constructing a `TensorBoardLogger` does,
  at which point a missing package raises a clear, actionable
  `ImportError` instead of a confusing one from deep inside PyTorch.
  `pyproject.toml` gained a `tensorboard` extra (`pip install
  -e ".[tensorboard]"`) and also lists it under `dev`, so this
  project's own test suite exercises `TensorBoardLogger` against the
  real package rather than a mock, matching this codebase's existing
  testing style.

## Consequences

- Spec §10's "Experiment tracking" bullet is now addressed for its two
  named backends (TensorBoard, CSV); adding a third (W&B, MLflow) is
  purely additive: one new file implementing `logScalar` (and
  optionally `logImage`/`logFigure`/`close`), registered in
  `training/loggers/__init__.py`, no changes to `Trainer` or to any
  existing logger.
- `Trainer` itself needed zero changes to support this: every logger
  is already a `TrainerCallback`, so `callbacks=[SomeLogger(...)]`
  (or several at once) was already the entire integration surface
  `docs/adr/0005-training-loop.md` set up.
- `tests/integration/test_loggers.py` covers: the registry pattern,
  `CsvLogger`'s row format and file lifecycle (including
  `logImage`/`logFigure` warning rather than raising), `TensorBoardLogger`
  against the real package (scalars, images, figures, the missing-package
  `ImportError` path via a `sys.modules` patch), full `Trainer.fit`
  runs through each logger with the expected row/event counts, and
  running two loggers simultaneously with no dedicated composition
  code.
- `training/NOTE.md`'s last remaining "deferred" item under spec §10
  ("Experiment tracking") is resolved.
