# 0033: Top-K checkpoints and learning-rate schedules as registered callbacks

**Status:** accepted (roadmap item P2-3; three new callbacks, none in a shipped config, so a run is unchanged)
**Date:** 2026-10-18

## Context

[ADR 0007](0007-best-checkpoint-callback.md) built `BestCheckpointCallback`, which overwrites one
file with the single best model, and listed "keep the top K" as a natural extension. The only
learning-rate callback was `ReduceLrOnPlateau` (ADR 0022). A run of any length wants a warm-up (the
first steps of Adam at the full rate are the unstable ones) and a decay to a floor, and a tuned run
wants more than the single best epoch kept, since the best validation loss and the best model for a
downstream use are not always the same epoch.

Both fit the callback registry of ADR 0022: a callback is chosen by name in
`TrainingConfig.callbacks`, takes constructor kwargs from the config, and sees the trainer at four
points. The question is only how each behaves, in particular together with resume (ADR 0032),
accumulation (ADR 0031) and the learning-rate metrics (ADR 0028).

## Decision

**1. `top_k_checkpoint`: `TopKCheckpointCallback(directory, k=3, monitor="val/loss/total",
mode="min", filename_pattern="top_k_epoch_{epoch:04d}.pt", config=None)`.** At the end of each
epoch it ranks the monitored value against the kept ones. If fewer than `k` are kept, or the value
is strictly better than the worst kept, it writes a full training checkpoint of that epoch
(`Trainer.saveCheckpoint`, so any of them can be evaluated or resumed from) and then deletes the
file of the entry it evicted. Details that decide its behavior:

- a value equal to the worst kept does not displace it, and equal values stay in epoch order, so
  on ties the earlier epoch is the better one (`BestCheckpointCallback` also needs a strict
  improvement);
- a non-finite value is never kept, and it is logged, because `nan` compares as neither better nor
  worse than anything and would otherwise be kept or break the order;
- the new file is written before the old one is removed, so a crash between the two leaves `k + 1`
  files and never `k - 1`;
- `filename_pattern` must give different names to different epochs, which is checked at
  construction, and it receives `epoch`;
- a missing monitor key is a `KeyError` that names the keys present, as in the other callbacks;
- `entries` and `paths` list the kept `(value, path)` pairs and files, best first, as copies;
- its state is the list of kept pairs (`stateDict`), so a resumed run keeps ranking against the
  same `k` and deletes the same files.

It does not replace `best_checkpoint`: that one writes a fixed path that a script can name
(`best.pt`), while the paths here carry the epoch.

**2. Learning-rate schedules are functions of the optimizer step.** Two callbacks share
`AbstractLrSchedule`, which sets every parameter group's rate to `initial_lr * factor(step)`:

- `linear_warmup_lr(warmup_steps)`: `(step + 1) / warmup_steps` for `step < warmup_steps`, then it
  stops touching the rate (`isActive` is false), so `reduce_lr_on_plateau` or a hand-set rate takes
  over from the peak;
- `cosine_lr(total_steps, warmup_steps=0, min_lr_ratio=0.0)`: the same linear warm-up, then
  `min_lr_ratio + (1 - min_lr_ratio) * (1 + cos(pi * p)) / 2` with
  `p = (step - warmup_steps) / (total_steps - warmup_steps)`, and `min_lr_ratio` for every step from
  `total_steps` on. `0 <= warmup_steps < total_steps` and `0 <= min_lr_ratio <= 1` are checked.

`onTrainBegin` records each group's starting rate under `initial_lr` in the optimizer's own
parameter group (with `setdefault`, the key and the idiom `torch.optim.lr_scheduler` uses) and sets
the rate of the first step to run; `onStepEnd` sets the rate of the next one. Because the rate is a
function of the step and of `initial_lr` alone:

- **a resume needs no schedule state.** `Trainer.loadCheckpoint` restores `global_step` and the
  optimizer, whose groups carry `initial_lr`, and `onTrainBegin` sets the rate for that step. The
  schedule scales the rate the run started with, not the already-scheduled rate the checkpoint
  holds, so it does not decay twice. A test resumes in the middle of a cosine and of a warm-up and
  gets the rates of the uninterrupted run;
- **it is exact for any number of batches per epoch, and with accumulation**, because the step is
  the optimizer step of ADR 0031. `total_steps` is then `epochs * ceil(batches / N)`; it is given
  in steps and not epochs for that reason;
- **per-group rates keep their ratios**, since each group is scaled from its own `initial_lr`.

Both set `acts_on_learning_rate`, so ADR 0028 applies and the rate is in the step metrics, the
epoch metrics (`train/lr`), the loggers and the checkpoint history without anything else. `cosine_lr`
owns the rate for the whole run: when another callback that acts on it is present
(`ReduceLrOnPlateau`, another schedule), it warns at the start of training that those changes will
be overwritten, and the warm-up does not, because it leaves the rate alone after it ends.

## Alternatives considered

- **Wrap `torch.optim.lr_scheduler` classes.** They keep their own state in a `state_dict` that
  would have to join the checkpoint, count epochs or calls to `step()` depending on the class (a
  frequent source of off-by-one), and `ReduceLROnPlateau` and a cosine both mutate the same field.
  Closed-form functions of the step have none of these problems and are a few lines each.
- **Schedules in epochs.** The number of updates per epoch depends on the dataloader and on
  accumulation, so an epoch schedule cannot be exact for a warm-up of 500 steps. A caller who thinks
  in epochs computes `total_steps` once.
- **Keep the `k` best in memory and write them at the end.** It would not survive an interruption,
  which is when the checkpoints matter.
- **Delete the evicted file first.** It frees the disk sooner but can leave fewer than `k` files if
  the save fails; the order chosen costs one extra file for a moment.
- **More shapes** (cosine with restarts, one-cycle, polynomial, exponential). Each is a subclass
  that implements `factor`, which is what `AbstractLrSchedule` is for; none is needed yet.
- **A `Trainer` argument instead of callbacks.** Callbacks are how `reduce_lr_on_plateau` is
  chosen, and a config can already name them.

## Consequences

- A run that lists none of the three is unchanged.
- `cosine_lr` and `reduce_lr_on_plateau` together is allowed and warned about: the schedule wins at
  every step. Use `linear_warmup_lr` with `reduce_lr_on_plateau` to combine a warm-up with plateau
  reductions.
- A schedule that is added to a resumed run (it was not in the first part) starts from the
  optimizer's current rate as `initial_lr`, which is the rate the checkpoint holds.
- The `initial_lr` key now appears in the optimizer's `param_groups` of a scheduled run and in its
  checkpoints, like with PyTorch's schedulers.
- Each kept top-K file is a full checkpoint, with the optimizer, so a model that is large costs
  `k` times that in disk. `k` is small by default and the how-to says so.
- Not decided here: a schedule on the beta weight beyond the existing `beta_schedules`, a schedule
  per parameter group, and keeping the top K by several metrics.
