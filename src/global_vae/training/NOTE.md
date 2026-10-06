# Status

`trainer.py` (`Trainer`) is implemented: a raw PyTorch loop (spec
§10's resolved raw-loop-vs-Lightning question), covering forward pass,
reconstruction + regularization loss (weighted by beta, including
per-latent-space schedules via `beta_schedule_resolution.py`),
optimizer step, device placement, modality dropout (spec §5), and a
`TrainerCallback` hook seam (`callbacks/base.py`) for per-step/per-epoch
metrics, called once per optimizer step and once per epoch.
`Trainer.should_stop` (reset at the start of every `fit()` call, checked
right after each epoch's `onEpochEnd` callbacks have all run) is the
lever a callback uses to end a run early (`callbacks/early_stopping.py`).

`beta_schedules/` predates `trainer.py` and does not depend on it: see
`beta_schedules/base.py`, `beta_schedule_resolution.py`, and
`docs/adr/0004-pluggable-beta-schedules.md`. `Trainer` calls
`resolveBetaSchedules(...)` internally each step and passes the result
straight into `GlobalVae.computeRegularizationLoss(..., beta=...)`, per
that ADR's own stated plan; `GlobalVae.computeRegularizationLoss` was
retrofitted with a `beta` parameter to make that call actually possible
(it did not expose one yet when ADR 0004 was written).

`checkpoint.py` (`saveCheckpoint`/`loadCheckpoint`/`CheckpointMetadata`)
and `../utils/seed.py` (`setGlobalSeed`) are implemented, covering all
three parts of spec §10's "Reproducibility" bullet (global seed
management, a documented deterministic-mode flag, config snapshotted
with every run). `checkpoint.py` owns only the checkpoint file format;
the two `TrainerCallback`s that call into it (`CheckpointCallback`,
periodic, for resuming an interrupted run, and `BestCheckpointCallback`,
saves only on improvement of a monitored metric, for evaluating/
visualizing the best model without retraining) live in their own files,
`callbacks/checkpoint.py` and `callbacks/best_checkpoint.py`,
registered as `"checkpoint"`/`"best_checkpoint"` (see
`docs/adr/0006-reproducibility-seed-and-checkpointing.md`,
`docs/adr/0007-best-checkpoint-callback.md`, and
`docs/adr/0022-callback-registry.md` for why the split).

`callbacks/` (ADR 0022) holds `TrainerCallback` itself plus every
checkpoint/scheduling-style callback, self-registered into one unified
`training.callbacks` registry (`registerCallback`/`getCallbackClass`/
`listRegisteredCallbacks`, mirroring every other pluggable strategy in
this codebase), selected from `TrainingConfig.callbacks` (registry name
-> constructor kwargs) instead of config having one hardcoded field per
kind of callback:
- `checkpoint.py`/`best_checkpoint.py`: `CheckpointCallback`/
  `BestCheckpointCallback`, above.
- `early_stopping.py` (`EarlyStopping`, `"early_stopping"`) and
  `reduce_lr_on_plateau.py` (`ReduceLrOnPlateau`,
  `"reduce_lr_on_plateau"`): both reuse the shared plateau-detection
  algorithm in the private `_plateau.py` (`PlateauTracker`), the same
  `mode`/`threshold`/`threshold_mode`/`cooldown`/`patience` semantics as
  `torch.optim.lr_scheduler.ReduceLROnPlateau`. `ReduceLrOnPlateau`
  mirrors that torch class directly (lowers `trainer.optimizer`'s
  learning rate on a plateau); `EarlyStopping` reacts to the identical
  plateau signal by setting `trainer.should_stop = True` instead, since
  plain `torch` ships no early-stopping class of its own. See
  `docs/adr/0022-callback-registry.md`.

`loggers/` (`AbstractExperimentLogger`, `CsvLogger`, `TensorBoardLogger`)
is implemented, covering spec §10's "Experiment tracking" bullet, and is
a separate subsystem from `callbacks/` above: it keeps its own
`registerLogger`/`getLoggerClass`/`listRegisteredLoggers` registry and
its own `TrainingConfig.loggers` config field, untouched by ADR 0022. A
`Logger` is a journalling service, not a callback selected from
`TrainingConfig.callbacks`; `AbstractExperimentLogger`'s `TrainerCallback`
inheritance is structural plumbing (so `Trainer.callbacks` can call its
hooks uniformly), not membership in that registry. Every concrete logger
is itself a `TrainerCallback`, so no change to `Trainer` was needed to
support it (`callbacks=[CsvLogger(...)]`, or several loggers at once,
just works). See `docs/adr/0008-experiment-loggers.md` and
`docs/adr/0022-callback-registry.md`'s own note on why the two stay
apart.

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
