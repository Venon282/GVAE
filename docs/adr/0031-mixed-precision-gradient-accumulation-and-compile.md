# 0031: Mixed precision, gradient accumulation and `torch.compile`, each opt-in

**Status:** accepted (roadmap item P2-3; every option is off by default, so a run that sets none of them is unchanged)
**Date:** 2026-10-18

## Context

`Trainer` ran every forward pass in `float32`, took one optimizer step per batch and ran the model
eagerly. The three things a longer or larger run asks for first are not there:

- a lower-precision forward pass (autocast), which is faster on a GPU and halves the activation
  memory;
- an effective batch larger than the one that fits in memory, by adding the gradients of several
  batches before one step;
- compiling the model.

Each changes what a "step" is or what the optimizer sees, so each needs a stated meaning, a default
that keeps the existing behavior to the last bit, and a test of that claim. The roadmap item asks
for "each one opt-in and off by default, each covered by a smoke test in the style of
`test_trainer_smoke.py`".

## Decision

**1. Three constructor arguments and three config fields, all off by default.**

| `Trainer` argument | `TrainingConfig` field | Default | Meaning |
|---|---|---|---|
| `mixed_precision` | `mixed_precision` | `None` | `None`, `"bf16"` or `"fp16"` |
| `accumulate_grad_batches` | `accumulate_grad_batches` | `1` | batches per optimizer step, at least 1 |
| `compile_model`, `compile_kwargs` | `compile_model`, `compile_kwargs` | `False`, `{}` | wrap the forward pass in `torch.compile` |

A value of `mixed_precision` other than `None`, `"bf16"` and `"fp16"` and an
`accumulate_grad_batches` below 1 are a `ValueError` at construction, which names the supported
values. A test builds a trainer with every default spelled out and checks that its history and
weights equal those of a trainer built with no argument.

**2. Mixed precision is autocast on the trainer's device, with `float32` losses.**
`computeLosses` runs the forward pass inside `torch.autocast(device_type, dtype)`. `bf16` needs no
gradient scaler, because its range is that of `float32`. `fp16` creates a `torch.amp.GradScaler`
for the device, scales the loss before `backward`, and takes the step through the scaler, after
`unscale_` when gradient clipping is on so that the clip threshold is in the unscaled units. The
scaler's state is saved in the checkpoint and restored by `Trainer.loadCheckpoint`, so a resumed
`fp16` run does not restart its scale search.

The model's outputs are cast back to `float32` before the reconstruction loss and the regularizer
are computed (`_castOutputsToFloat32`: the reconstructions and the `(mu, logvar)` of each latent
space). Those two terms are where reduced precision costs accuracy: the KL takes an exponential
of a log-variance and sums over dimensions, and a sum-reduced reconstruction (ADR 0029) sums over
every element. The parameters stay `float32`, so the optimizer state and the checkpoints are the
same as without autocast, and a checkpoint of a `bf16` run loads into a full-precision trainer
(a test does it). `evaluate` runs under the same autocast, so a validation loss is measured the
way training is.

**3. A step is an optimizer update.** With `accumulate_grad_batches = N`, the trainer adds the
gradients of `N` batches, each loss divided by `N` so that the sum is the mean, and takes one
step. From there on, everything that was counted per batch is counted per update:

- `Trainer.global_step`, and the `step` that beta schedules and `onStepEnd` receive;
- the step metrics (the mean over the `N` batches of the group) and `log_every_n_steps`;
- the epoch metrics, which are the mean over the updates of the epoch;
- learning-rate schedules (ADR 0033): a cosine over `total_steps` ends after that many updates, so
  for `E` epochs of `B` batches `total_steps = E * ceil(B / N)`.

With `N = 1` (the default) an update is a batch and nothing changes.

**4. The last group of an epoch is a partial group.** When `B` is not a multiple of `N`, the batches
left at the end of an epoch are one more update, not carried into the next epoch (a carried
gradient would mix two epochs, and would be lost if the run ended there). Their gradient holds
`pending / N` of a mean, so it is multiplied by `N / pending` before the step, which makes it the
mean of the batches it holds. A test compares both the full and the partial group with a manual
loop of plain steps.

Gradient clipping applies to the accumulated gradient, once per update. An empty dataloader is
still a `ValueError` from `fitEpoch`, as it is from `evaluate`.

**5. `torch.compile` wraps the forward pass; it does not replace the module.** The trainer keeps
`self.model` as given and calls `torch.compile(self.model, **compile_kwargs)` for the forward pass
only. `state_dict` keys therefore never carry the `_orig_mod.` prefix, a checkpoint of a compiled
run loads into an uncompiled model and the reverse (a test checks the keys), and everything that
touches the model besides the forward pass (`computeRegularizationLoss`, `selectEncoderInputs`,
checkpoints) is unaffected. `compile_kwargs` is passed through, so `{"mode": "reduce-overhead"}` or
`{"backend": "eager"}` work. Modality dropout feeds the model inputs with different keys, which
compile to other graphs: compiling costs more recompilations with dropout on, and the how-to says
so.

## Alternatives considered

- **Always autocast on a GPU.** It changes numbers for every existing run, and the roadmap asks for
  off by default.
- **Compute the losses under autocast too.** Simpler, but the KL and a summed reconstruction lose
  accuracy in `bf16`, and nothing is gained: those terms are a small part of the cost of a step.
- **A scaler for `bf16`.** Not needed (same exponent range as `float32`); an unused scaler only
  adds a skipped-step path.
- **Count a step per batch and let accumulation skip some updates.** It keeps `global_step`
  stable but makes a schedule over steps depend on `N` in a way the user cannot see, and a beta
  warm-up of "1000 steps" would silently mean `1000 * N` batches.
- **Carry a partial group into the next epoch.** It would make a resumed run depend on a gradient
  that no checkpoint holds, since the optimizer is stepped and its gradient is not saved.
- **Replace `self.model` by the compiled module.** The usual one-liner, but it renames every
  `state_dict` key and breaks the checkpoints of either kind in the other direction.

## Consequences

- A run that sets none of the three options is bit-for-bit unchanged, and a test pins it.
- `accumulate_grad_batches > 1` changes the meaning of every "step" setting for that run. The
  how-to lists them: `log_every_n_steps`, the warm-up of a beta schedule, `total_steps` of a
  learning-rate schedule.
- A `bf16` or `fp16` run is not expected to match a `float32` run beyond a few percent in the loss
  (the tests check `5e-2` on the first loss), and `fp16` can skip a step when the scaler finds
  an infinite gradient, which is the scaler's normal behavior.
- `fp16` on the CPU needs PyTorch 2.3 or later for `torch.amp.GradScaler("cpu")`; the package
  requires 2.2, so the tests that need it skip themselves on 2.2. On a GPU nothing is needed.
- The three options are in the checkpoint's config snapshot (they are config fields), but only the
  scaler has state of its own to save.
- Not decided here: `fp8`, a distributed strategy, activation checkpointing, and a per-step
  gradient norm in the metrics.
