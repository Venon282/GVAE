# 0022: Training callback registry, and two new callbacks

**Status:** accepted
**Date:** 2026-10-03

## Context

Every other pluggable piece of this codebase (encoders, decoders, fusion,
assemblers, regularizers, beta schedules) follows the same shape: an
`AbstractX` interface, a registry (`registerX`/`getXClass`), and a config
field naming, by string, exactly which registered strategies to use. The
checkpoint callbacks did not. `TrainerCallback` (`training/callbacks.py`)
was a single, un-registered base class, and `CheckpointCallback`/
`BestCheckpointCallback` (`training/checkpoint.py`) were wired into config
through one hand-written, hardcoded dataclass, `CheckpointConfig`, that
`TrainingConfig` always carried whether or not a run wanted either callback.
Adding a third kind of this sort of callback (a learning-rate scheduler, an
early-stopping rule) meant adding a third hardcoded config field and a third
branch in `buildCallbacksFromConfig`, not registering a new strategy the way
every other extension point in this framework already works.

Separately, a real gap: nothing in `Trainer` gave a callback a way to stop
training early, and no learning-rate scheduling of any kind existed. Both
are common enough (spec §10's own training-loop concerns) that they belong
as built-in callbacks, not something every project using this framework has
to write itself.

`training/checkpoint.py` also held `CheckpointCallback` and
`BestCheckpointCallback` side by side in one file, which spec §10's
"Modularity" rule ("one class per file... no god-files") does not actually
exempt: the rule names encoders/decoders/fusion strategies/assemblers/
regularizers/transforms explicitly, but its reasoning (a base class, its
registry, and every concrete strategy each get their own file) applies
exactly as well to two independent `TrainerCallback`s that happen to share a
module today only because nothing moved them out.

**Explicitly out of scope: experiment loggers.** `training/loggers/`
(`AbstractExperimentLogger`, `CsvLogger`, `TensorBoardLogger`, ADR 0008) is
not touched by this ADR, and a first draft of this decision that folded
loggers into the same unified registry was reverted before being merged. A
`Logger` is a journalling service: it receives events or metric values and
writes them somewhere (console, file, a tracking server). A
`TrainerCallback` is a hook a component calls later when something happens.
`AbstractExperimentLogger(TrainerCallback, ABC)` subclasses `TrainerCallback`
*structurally*, so `Trainer.callbacks: list[TrainerCallback]` can call
`onStepEnd`/`onEpochEnd` on a logger exactly like any other callback
(ADR 0008, predating this ADR); that inheritance is plumbing, not evidence
that a logger belongs in a callback *selection* registry next to checkpointing
or early stopping. Concretely: this ADR does not move `training/loggers/`,
does not change `registerLogger`/`getLoggerClass`/`listRegisteredLoggers`,
does not change `TrainingConfig.loggers`/`LoggerEntryConfig`, and does not
make `"csv"`/`"tensorboard"` resolvable through the registry this ADR adds.
Loggers and the callbacks below are combined into one `Trainer.callbacks`
list (as they always were), but selected from config through two separate,
independent mechanisms.

## Decision

- **One registry for checkpointing, early stopping, and LR scheduling.**
  `TrainerCallback` and a `registerCallback`/`getCallbackClass`/
  `listRegisteredCallbacks` registry (mirroring `fusion/registry.py`
  exactly) move into a new `training/callbacks/` subpackage (`base.py`,
  `registry.py`). `CheckpointCallback`/`BestCheckpointCallback` (previously
  config-visible only through the hardcoded `CheckpointConfig`) now
  self-register by name with `@registerCallback(...)`, exactly like an
  encoder or a fusion strategy. `training/callbacks/__init__.py` imports
  every built-in module for that registration side effect, the same
  convention every other registry-based subpackage's `__init__.py` already
  follows (spec §10).
- **Checkpoint format, and the two checkpoint callbacks, are three separate
  files.** `training/checkpoint.py` keeps exactly `saveCheckpoint`/
  `loadCheckpoint`/`CheckpointMetadata` and the RNG-state helpers: the file
  format, independent of `Trainer` or any callback. `CheckpointCallback`
  moves to `training/callbacks/checkpoint.py`, registered `"checkpoint"`.
  `BestCheckpointCallback` moves to its own sibling file,
  `training/callbacks/best_checkpoint.py`, registered `"best_checkpoint"`:
  one class per file, per spec §10's "Modularity" rule, rather than the two
  callbacks continuing to share a module the way they did in
  `training/checkpoint.py` before this ADR. Nothing about either class's own
  behavior changes.
- **`TrainingConfig.callbacks` replaces `checkpoint: CheckpointConfig`.**
  One new field, `callbacks: dict[str, dict[str, Any] | None]`: registry
  name -> constructor kwargs (`None`, or an empty mapping, meaning "every
  default"). Empty by default: nothing is enabled unless a person names it,
  which is the actual fix for "on est obligé de les avoir" (forced to have
  callbacks one did not ask for) and "on ne peut pas choisir" (config's
  previous shape hardcoded which checkpoint callbacks existed, not which
  ones a run wanted). `TrainingConfig.loggers`/`LoggerEntryConfig` are
  untouched (see Context, above). Example::

      loggers:
        - name: csv
          kwargs:
            path: ${output_dir}/metrics.csv
      callbacks:
        checkpoint:
          directory: ${output_dir}/checkpoints
          every_n_epochs: 10
        early_stopping:
          monitor: val/loss/total
          patience: 5

  `buildCallbacksFromConfig` resolves `config.loggers` through
  `getLoggerClass` exactly as before this ADR, then resolves each
  `config.callbacks` entry through `getCallbackClass` and instantiates it,
  in the mapping's own order (a plain `dict`, so YAML key order); `Trainer`
  calls the combined list, loggers first, in that same order. A callback
  whose constructor declares a `config` parameter, and whose own kwargs did
  not already supply one, is passed the run's `config_snapshot`
  automatically (checked via `inspect.signature`, not by special-casing
  `CheckpointCallback`/`BestCheckpointCallback` by class identity): this is
  what keeps "config snapshotted with every run" (spec §10) working for the
  checkpoint callbacks without this function needing to know either class by
  name, and lets a person's own future callback opt into the same mechanism
  by simply declaring the same parameter. Loggers never receive
  `config_snapshot`: their constructors have no such parameter, matching
  their behavior before this ADR. `configs/training/default.yaml` is updated
  to the new shape, reproducing its previous behavior exactly (csv +
  tensorboard, unchanged, plus periodic checkpoint + best checkpoint, now
  under `callbacks:`).
- **`Trainer.should_stop`.** A new `bool` attribute, `False` by default and
  reset to `False` at the start of every `fit()` call. `Trainer.fit` checks it
  immediately after every `onEpochEnd` callback has run for the current
  epoch (so a callback listed after the one that requested a stop still sees
  that epoch) and ends the run before starting the next one. This is the one
  `Trainer` change this ADR needed: every other callback capability (reading
  `trainer.optimizer`, `trainer.saveCheckpoint`) already existed.
- **`ReduceLrOnPlateau`** (`training/callbacks/reduce_lr_on_plateau.py`,
  registered `"reduce_lr_on_plateau"`): the exact algorithm and constructor
  arguments of `torch.optim.lr_scheduler.ReduceLROnPlateau` (`mode`, `factor`,
  `patience`, `threshold`, `threshold_mode`, `cooldown`, `min_lr`, `eps`),
  reimplemented as a `TrainerCallback` instead of a manually-`.step()`ped
  object, so it plugs into `Trainer.fit` like every other callback: `monitor`
  names the epoch-metrics key to watch (`torch`'s own scheduler has no such
  concept, since it is handed the value directly), and `onEpochEnd` lowers
  every one of `trainer.optimizer.param_groups`'s learning rates on a
  plateau.
- **`EarlyStopping`** (`training/callbacks/early_stopping.py`, registered
  `"early_stopping"`): plain `torch` ships no early-stopping class of its own;
  this callback reuses the identical plateau-detection algorithm "the torch
  one" has (`ReduceLROnPlateau`'s `mode`/`threshold`/`threshold_mode`/
  `cooldown`/`patience`), and reacts to a plateau by setting
  `trainer.should_stop = True` instead of lowering a learning rate.
- **`PlateauTracker`** (`training/callbacks/_plateau.py`, private): the shared
  `mode`/`threshold`/`threshold_mode`/`cooldown`/`patience`-tracking algorithm
  both callbacks above need, written once so the two stay in exact agreement
  instead of each re-deriving the same floating-point-sensitive comparison
  logic.

## Consequences

- Adding a new checkpointing/scheduling-style callback (a warm-restart LR
  schedule, a gradient-norm-based stopping rule, a caller's own
  project-specific callback) is now exactly "subclass `TrainerCallback`,
  decorate with `@registerCallback("name")`, reference `name` from
  `TrainingConfig.callbacks`": no change to `config/training.py`, `Trainer`,
  or any existing callback.
- A run opts into precisely the checkpoint/scheduling-style callbacks it
  wants, in any combination, in an order it controls; a run that wants none
  at all leaves `callbacks: {}` (the default) rather than carrying a config
  field it has to remember to disable. Loggers keep working exactly as
  before, through their own, separate `loggers:` list.
- The same registry name cannot appear twice in `TrainingConfig.callbacks`
  (it is a `dict`), so two instances of the same callback type (e.g. two
  `CheckpointCallback`s writing to different directories) are not expressible
  from config; build a `Trainer` directly with an explicit `callbacks=[...]`
  list for that rare case, exactly as before this ADR for anything not
  expressible from config.
- `CheckpointCallback`/`BestCheckpointCallback`'s own behavior, and the
  checkpoint file format, are unchanged; only their module location (now two
  files, not one, and not the same file as the format code), and how they
  are selected from config, changed. Existing checkpoints written before
  this ADR still load with `loadCheckpoint` unmodified.
- Loggers (`training/loggers/`, `TrainingConfig.loggers`) are unchanged by
  this ADR in every respect: location, registry, config field, and
  behavior. A future ADR could revisit that boundary, but this one does not.
- `tests/integration/test_callback_registry.py` covers the registry
  mechanism itself (mirroring every other registry's own test file) and
  that every built-in callback is actually registered under the name these
  docs and `configs/training/default.yaml` use (explicitly asserting
  `"csv"`/`"tensorboard"` are *not* registered here, since they belong to
  `training.loggers`'s own registry instead). `test_checkpoint.py` now
  covers only the checkpoint file format; `test_checkpoint_callbacks.py`,
  `test_plateau_tracker.py`, `test_early_stopping.py`, and
  `test_reduce_lr_on_plateau.py` cover, respectively, the two checkpoint
  callbacks, the shared plateau algorithm's own value correctness, and each
  of the two new callbacks' behavior both via direct calls and through a
  real `Trainer.fit()` run. `test_loggers.py` is unaffected by this ADR.
  `test_trainer.py` gained a `TestShouldStop` class covering the new
  attribute and `fit()`'s handling of it, independent of any specific
  callback.
- Not built: a `top_k` best-checkpoint callback (still noted as a natural
  future extension in ADR 0007, unaffected by this one), and a cooldown-aware
  resume story for `EarlyStopping` across separate `fit()` calls beyond
  `should_stop` simply resetting (a caller wanting to resume past an early
  stop already has to re-inspect `trainer.history` and decide for itself
  whether resuming is warranted; this ADR does not add policy for that
  decision).
