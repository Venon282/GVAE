# 0032: Resuming a run from the training script

**Status:** accepted (roadmap item P2-3; `training.resume_from` is `null` by default, so a run that does not set it is unchanged)
**Date:** 2026-10-18

## Context

`Trainer.saveCheckpoint` and `Trainer.loadCheckpoint` exist (ADR 0005), and a checkpoint holds the
model, the optimizer, `global_step`, `start_epoch`, the history and the RNG state. But
`scripts/train.py` never called `loadCheckpoint`, so an interrupted run could be resumed only from
Python. Wiring it in raises four questions, and the roadmap asks for them to be answered:

1. Is `num_epochs` a total or an increment? `Trainer.fit(num_epochs)` is an increment (it is
   documented as "train 5 more epochs"), and a config says `num_epochs: 200`.
2. What else besides the model has state that decides the future of a run? The state of the
   callbacks does: `EarlyStopping` and `ReduceLrOnPlateau` count bad epochs, `BestCheckpointCallback`
   remembers the best value, `CheckpointCallback` remembers the files it may prune. A resumed run
   that starts them from scratch stops later, reduces the rate later, and can overwrite a better
   best checkpoint with a worse model (its best value starts at infinity again).
3. What happens to the files the first part wrote? `CsvLogger` opened its file in `"w"` mode, so a
   resumed run erased the metrics of the first part.
4. What does "reproduces the uninterrupted run" mean, with which tolerance, under which conditions?

## Decision

**1. `training.resume_from`, a path or `null`.** A new `TrainingConfig` field, `None` by default.
`scripts/train.py` builds the model, the dataloaders and the trainer as before, then calls
`resumeTrainerFromConfig(trainer, config.training)`, which loads the checkpoint into the trainer
(model, optimizer, `global_step`, `start_epoch`, history, RNG state, scaler, callback state) and
returns the number of epochs to run:

```bash
python scripts/train.py training.num_epochs=200 \
    training.resume_from=runs/signal_vae/checkpoints/checkpoint_epoch_0099.pt
```

A path that is not a file is a `FileNotFoundError` that starts with `training.resume_from`. The
checkpoint must come from the same model: `load_state_dict` is strict, so a different architecture
fails with PyTorch's own message.

**2. `num_epochs` is the total of the run.** `resumeTrainerFromConfig` returns
`num_epochs - start_epoch`. So the command line of a run, with `resume_from` added, completes that
run, and the history it ends with is the one the uninterrupted run would have had. When the
checkpoint already holds `num_epochs` epochs or more, nothing is left: it returns `0`, the script
logs "Nothing to train" and exits successfully, so re-running a finished command is harmless.
`Trainer.fit(num_epochs)` is not changed: in Python it stays an increment, as ADR 0005 documents,
and the config layer is where the total lives.

**3. Callbacks have a state.** `TrainerCallback` gets `stateDict() -> dict` (default `{}`) and
`loadStateDict(state)` (default: nothing). The built-in callbacks that carry something implement
them:

| Callback | State |
|---|---|
| `EarlyStopping`, `ReduceLrOnPlateau` | the plateau tracker: `best`, `num_bad_epochs`, `cooldown_counter` |
| `BestCheckpointCallback` | `best_value` |
| `CheckpointCallback` | `saved_paths`, so `keep_last_n` goes on pruning |
| `TopKCheckpointCallback` (ADR 0033) | the kept `(value, path)` pairs |

`Trainer.saveCheckpoint` stores them under `callbacks_state` as `{"class": name, "state": dict}`
entries in the order of the trainer's callbacks, and `loadCheckpoint` hands each one to the
callback of the same class, in order. A saved state with no matching callback is dropped with a
warning that names it; a callback with no saved state keeps its fresh one (a checkpoint from before
this ADR loads as it did). The states are plain values, which is what keeps the file readable in
the safe mode (ADR 0034). A caller's own callback opts in by overriding the two methods.

A checkpoint written inside `onEpochEnd` of a callback holds the state of the callbacks **after**
it as they were before this epoch (they have not run yet). The checkpoint you resume from should
therefore come from the last callback of the list: the shipped configs list `checkpoint` last, and
the how-to says so. `CheckpointCallback`, `BestCheckpointCallback` and `TopKCheckpointCallback`
update their own state before they save, so the file they write holds it.

**4. Loggers continue their file.** `CsvLogger(path, append=False)` gains `append`: when `True` and
the file exists and is not empty, rows are added and no second header is written.
`buildCallbacksFromConfig` passes `append=True` to a logger whose constructor accepts `append` and
whose kwargs do not give it, when `resume_from` is set. The rows the interrupted run wrote after its
last checkpoint stay in the file, so a step can appear twice, and the later row is the resumed
one. TensorBoard already adds an events file to its directory.

**5. The guarantee, and its tolerance.** *A resumed deterministic run reproduces the loss curve of
the uninterrupted run: every metric of every epoch agrees to a relative `1e-6` (absolute `1e-9`
for values near zero), and the final weights to `1e-6`.* "Deterministic" means: the same config and
seed, a data order that is the same in both runs, and either the CPU or `deterministic: true`. On
the CPU the two runs execute the same operations in the same order, and the tests find them equal
beyond that margin; the number is what the guarantee promises, so that a change of PyTorch's kernel
selection does not break it by itself. Six configurations are tested, each interrupted after three
of six epochs: a plain run, a validation set with `reduce_lr_on_plateau`, a beta schedule with
`cosine_lr`, modality dropout, accumulation with `bf16` and clipping, and all of them together,
and one run of `scripts/train.py` as a subprocess.

What the checkpoint restores to make that hold: the Python, NumPy, PyTorch (and CUDA) RNG states,
so modality dropout and the reparameterization noise continue the same sequence; the optimizer
state; the step counter (beta schedules and learning-rate schedules are functions of it); the
callback state; the scaler. A test removes the RNG state from a checkpoint and checks that the
curve then differs, so the tolerance test is not passing vacuously.

What it does not cover: a different PyTorch, hardware or number of threads; a GPU run without
`deterministic: true`; and a dataloader whose order is not reproducible. A `DataLoader` that draws
its shuffling from the global PyTorch RNG is reproduced, since that state is restored. One with its
own `generator`, or workers seeded by something else, is not, and its state is the caller's to
save.

## Alternatives considered

- **`num_epochs` as an increment** (`training.num_epochs=100` means "100 more"). It matches `fit`,
  but a resumed command line then differs from the original in its meaning, a finished run run
  twice trains twice, and "the run of 200 epochs" is no longer a property of the config. Lightning
  and most trainers use a total (`max_epochs`) for the same reason.
- **A separate `resume` script, or `resume_from: auto`** (the latest checkpoint of `output_dir`).
  The first duplicates the construction code; the second guesses, and which file is "latest" is
  not always the one wanted (a `top_k` directory, a best file). An explicit path is the smallest
  thing that works, and `auto` can be added on top.
- **Restore callback state by position only, or by a name chosen by the user.** Position breaks
  when a callback is inserted; a user name needs a new config field. The class name, in order,
  needs nothing from the user and fails visibly (a warning) when the lists differ.
- **Pickle the callbacks whole.** It would restore everything, including the caller's own
  callbacks, but it ties the file to the code, and it is what the safe mode forbids (ADR 0034).
- **Truncate the metrics file at the checkpoint's step on resume.** It removes the duplicated
  rows, but it deletes data the user did not ask to delete, and the later row already wins.
- **Save the dataloader's state.** There is no standard way to do it for an arbitrary iterable;
  the caller makes the data order reproducible, and the how-to says which seeds to set.

## Consequences

- A run that does not set `resume_from` is unchanged; `fit` still raises on `num_epochs <= 0`, and
  the script never reaches it with `0` epochs left.
- A checkpoint now holds `callbacks_state` (and `scaler_state_dict` for `fp16`) when there is any.
  Older checkpoints lack them and load as before.
- The `config` snapshot of a resumed run's checkpoints includes `resume_from`, which is accurate:
  that is the config it ran with.
- A callback whose state is not plain values breaks the safe loading of every checkpoint of the
  runs that use it. The base-class docstring says so.
- With the default order of a config that lists `checkpoint` before `best_checkpoint`, a resumed
  `best_checkpoint` can be one epoch behind. The shipped configs list `best_checkpoint` first.
- Not decided here: `resume_from: auto`, saving the dataloader's state, resuming with another
  optimizer or learning rate (`Trainer.loadCheckpoint(restore_optimizer=False)` does it in
  Python), and a different number of devices.
