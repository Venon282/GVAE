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

`learning_rate.py` (ADR 0028) holds the keys under which a learning rate is recorded
(`learningRateMetrics`, `isLearningRateKey`: `lr` per step, `train/lr` per epoch, `lr/group<i>`
with several parameter groups). A callback that changes the rate sets the class attribute
`TrainerCallback.acts_on_learning_rate` (`ReduceLrOnPlateau` does), and `Trainer.logs_learning_rate`
is then `True`: `fitEpoch` adds the rate each step trained with to the metrics, so it shows in the
epoch log line, `Trainer.history`, every logger and the checkpoint history. Without such a
callback nothing is added.

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

`trainer.py` has three opt-in capabilities, all off by default (ADR 0031): `mixed_precision`
(`"bf16"` or `"fp16"`; autocast around the forward pass, the losses computed in `float32`, a
`torch.amp.GradScaler` for `fp16`, kept in `Trainer.scaler` and saved in the checkpoint),
`accumulate_grad_batches` (the gradients of `N` batches added before one optimizer step; a *step*
is an update, so `global_step`, beta schedules and `onStepEnd` count updates, and the partial last
group of an epoch is rescaled to a mean) and `compile_model` (the forward pass goes through
`torch.compile`, `self.model` is untouched so checkpoint keys stay the same).

Resuming (ADR 0032): `TrainerCallback.stateDict()`/`loadStateDict()` (no-ops by default) let a
callback keep what decides the future of a run. `Trainer.saveCheckpoint` writes them under
`callbacks_state` (`{"class", "state"}` per callback, in order) and `Trainer.loadCheckpoint` hands
each to the callback of the same class. `config/training.py` has `resumeTrainerFromConfig`, which
`scripts/train.py` calls for `training.resume_from`: `num_epochs` is the total of the run there,
while `Trainer.fit(num_epochs)` stays an increment. A resumed deterministic run reproduces the
uninterrupted loss curve to a relative `1e-6` (`tests/integration/test_resume.py`).

`callbacks/` also holds `top_k_checkpoint.py` (`TopKCheckpointCallback`, the `k` best epochs, the
extension ADR 0007 listed), and `linear_warmup_lr.py` and `cosine_lr.py`, built on the private
`_lr_schedule.py` (`AbstractLrSchedule`: the rate is `initial_lr * factor(step)`, `initial_lr`
kept in the optimizer's parameter groups, so a schedule has no state of its own). See ADR 0033.

`checkpoint.py` reads a file with `weights_only=True` by default and writes only plain values
(NumPy RNG state as a list, a dataclass `config` as a dict); `upgradeCheckpoint` rewrites a file
from an earlier version (ADR 0034).

# Nothing currently deferred under spec §10 for this subpackage.

Every item spec §10 explicitly lists for the training loop (raw loop, reconstruction +
regularization loss, optimizer, device placement, logging, reproducibility, experiment tracking,
checkpointing) is built. Multi-GPU and migrating to PyTorch Lightning remain the eventual plan once
the architecture stabilizes (spec §10).
