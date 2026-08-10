# Status

`trainer.py` (`Trainer`) is implemented: a raw PyTorch loop (spec
§10's resolved raw-loop-vs-Lightning question), covering forward pass,
reconstruction + regularization loss (weighted by beta, including
per-latent-space schedules via `beta_schedule_resolution.py`),
optimizer step, device placement, modality dropout (spec §5), and a
`TrainerCallback` hook seam (`callbacks.py`) for per-step/per-epoch
metrics, called once per optimizer step and once per epoch.

`beta_schedules/` predates `trainer.py` and does not depend on it: see
`beta_schedules/base.py`, `beta_schedule_resolution.py`, and
`docs/adr/0004-pluggable-beta-schedules.md`. `Trainer` calls
`resolveBetaSchedules(...)` internally each step and passes the result
straight into `GlobalVae.computeRegularizationLoss(..., beta=...)`, per
that ADR's own stated plan; `GlobalVae.computeRegularizationLoss` was
retrofitted with a `beta` parameter to make that call actually possible
(it did not expose one yet when ADR 0004 was written).

`checkpoint.py` (`saveCheckpoint`/`loadCheckpoint`/`CheckpointCallback`/
`BestCheckpointCallback`) and `../utils/seed.py` (`setGlobalSeed`) are
implemented, covering all three parts of spec §10's "Reproducibility"
bullet (global seed management, a documented deterministic-mode flag,
config snapshotted with every run). `CheckpointCallback` (periodic,
for resuming an interrupted run) and `BestCheckpointCallback` (saves
only on improvement of a monitored metric, for evaluating/visualizing
the best model without retraining) are separate, single-purpose
callbacks: see `docs/adr/0006-reproducibility-seed-and-checkpointing.md`
and `docs/adr/0007-best-checkpoint-callback.md`.

`loggers/` (`AbstractExperimentLogger`, `CsvLogger`, `TensorBoardLogger`)
is implemented, covering spec §10's "Experiment tracking" bullet.
Every concrete logger is itself a `TrainerCallback`, so no change to
`Trainer` was needed to support it (`callbacks=[CsvLogger(...)]`, or
several loggers at once, just works). See
`docs/adr/0008-experiment-loggers.md`.

# Nothing currently deferred under spec §10 for this subpackage.

`trainer.py` itself may still grow (e.g. mixed precision, multi-GPU,
gradient accumulation) as real training needs surface, and migrating
to PyTorch Lightning remains the eventual plan once the architecture
stabilizes (spec §10), but every item spec §10 explicitly lists for
the training loop (raw loop, reconstruction + regularization loss,
optimizer, device placement, logging, reproducibility, experiment
tracking, checkpointing) is now built. Keeping the top-`K` best
checkpoints (`K > 1`, rather than only the single best
`BestCheckpointCallback` keeps) is a natural, not-yet-built extension,
noted in `docs/adr/0007-best-checkpoint-callback.md`.
