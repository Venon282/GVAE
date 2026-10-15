# 0028: The learning rate in the metrics, the logs and a plot

**Status:** accepted
**Date:** 2026-10-15

This is a feature the owner asked for, not a roadmap item. No roadmap decision depends on it.

## Context

`ReduceLrOnPlateau` lowers `trainer.optimizer`'s learning rate and writes one line to the log when
it does. Nothing else records the change. The epoch log line, `Trainer.history`, the loggers and
the checkpoint history carry losses only, so after a run there was no way to see when a reduction
happened or what rate an epoch trained with. `scripts/visualize_latent.py` plots the loss curves
from the checkpoint history, so it could not plot the rate either.

Two requests followed:

- when a callback acts on the learning rate, the per-epoch metrics shown in the logs include it;
- in the same case, the script that plots the losses also plots the learning rate.

Three questions had to be answered: which runs record the rate, under which key, and where the
plot goes.

## Decision

**A callback says that it acts on the rate.** `TrainerCallback` gains the class attribute
`acts_on_learning_rate: ClassVar[bool] = False`. `ReduceLrOnPlateau` sets it to `True`, and so does
any callback of yours that changes `optimizer.param_groups[...]["lr"]` (a warmup, a cosine
schedule). `Trainer.logs_learning_rate` is `True` when any callback of the run sets it. This is the
same convention as `AbstractAssembler.requires_matching_dims` (ADR 0026): a class flag the
framework reads, so no list of names lives anywhere.

**The rate is recorded only when the flag is set.** `Trainer.fitEpoch` then reads the rate of each
parameter group just before `optimizer.step()` and adds it to that step's metrics. A run without
such a callback records nothing new: its history, its log lines, its logger rows and its checkpoint
files are the ones it had before.

**Keys.** `training/learning_rate.py` holds them, so the trainer, the log formatting and the plots
agree:

| Where | One parameter group | Several parameter groups |
|---|---|---|
| Step metrics | `lr` | `lr/group0`, `lr/group1`, ... |
| Epoch metrics | `train/lr` | `train/lr/group0`, `train/lr/group1`, ... |

The epoch value is the mean of the rates the epoch's steps used, like every other `train/` metric.
`isLearningRateKey` is `True` for any key with an `lr` path component.

**The rate is read before the step.** It is the value the step trained with. A reduction that a
callback makes in `onEpochEnd` of epoch `n` therefore shows from epoch `n + 1`, and a reduction in
`onStepEnd` shows from the next step. The epoch the callback acted at keeps the old value, which is
the honest one: that epoch did train with it.

**Log formatting.** Losses keep four decimals. A rate is written with `.6g` (`1e-05`, `0.0003`),
since four decimals would print `1e-05` as `0.0000`. Both the epoch line and the per-step line use
one helper, `_formatMetrics`.

**The plot.** `plotLearningRate(history, keys=None, log_scale=True, ...)` in
`visualization/loss_curves.py` reuses the series helpers of the loss plots and draws one figure
with a log y-axis by default, since a reduction is a factor. It raises `ValueError` when the
history has no learning-rate key, and the message names `acts_on_learning_rate`.
`scripts/visualize_latent.py` writes it to `lr_curve.png` next to `loss_curves.png`, only when the
checkpoint history has a learning-rate key. The loss plot leaves the rate keys out, so the two
figures do not share an axis. `--skip-history` skips both.

Nothing else changed. The rate reaches `CsvLogger` and the TensorBoard logger through the ordinary
epoch metrics, and the checkpoint history through `Trainer.history`, with no code of their own.

## Alternatives considered

- **Record the rate on every run.** Simple, and one more metric for everyone. Rejected: it changes
  the log lines, the logger output and the checkpoint history of runs that have a constant rate
  and no use for the value, and a constant line is noise in a plot.
- **Detect that the rate changed.** Compare the rate before and after each epoch and record it
  only if it moved. Rejected: the first epochs would have no rate and the key would appear
  mid-run, so the history would be ragged and the plot would start late, and a run that never
  reduces would show nothing, which is the case where the user wants to see a flat line.
- **Put the rate on the loss plot** as a second y-axis. Rejected: two scales on one figure
  mislead, and the loss plot is also used with no callback at all.
- **Log the rate at the end of the epoch**, after the callbacks ran. It would show the new value
  on the epoch that produced it, but a mean of the rates the steps used is the same kind of
  quantity as the other `train/` metrics, and an end-of-epoch value would not be the rate any
  step trained with.
- **A list of callback names** in the trainer (`"reduce_lr_on_plateau"`). Rejected: it would not
  cover a callback of your own, and it is the special-casing the registry pattern exists to avoid.

## Consequences

- A callback that changes the rate and does not set the flag works as before and is not recorded.
  The documentation of `TrainerCallback` and the how-to for callbacks say to set it.
- A run that records the rate has more keys in `Trainer.history` and its loggers. A consumer that
  reads every key of the history must accept `train/lr`. `visualize_latent.py` does.
- `fitEpoch` reads the optimizer once per step when the flag is set, which is negligible next to a
  forward and backward pass.
- The rate is the one of the optimizer the trainer holds. A scheduler that you step yourself
  outside the callbacks changes the same `param_groups` and is recorded, but only if some callback
  of the run sets the flag.
