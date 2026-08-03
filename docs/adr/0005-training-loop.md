# 0005 — Raw PyTorch training loop (`Trainer`)

**Status:** accepted
**Date:** 2026-08-02

## Context

Spec §10 resolves the raw-loop-vs-Lightning open question in favor of
a raw PyTorch loop for now, migrating to Lightning (or Lightning
Fabric as an intermediate step) once the architecture stabilizes. No
training loop existed yet; `training/NOTE.md` tracked it as the one
remaining piece blocking spec §6.1 milestone 1 (a single-modality
signal VAE trained end to end). Building it surfaced one real gap in
already-shipped code: `GlobalVae.computeRegularizationLoss` did not
expose a `beta` parameter at all, even though ADR 0004 already
documented the plan for a trainer to call
`model.computeRegularizationLoss(latent_params, beta=...)` with a
per-step, per-latent-space resolved beta. Data pipeline concerns
(datasets, transforms, splitting, batching) are explicitly out of this
framework's scope by design (the person building on this framework
owns their own data loading); the loop's contract with the outside
world had to be drawn accordingly.

## Decision

- **Fix the pre-existing gap:** `GlobalVae.computeRegularizationLoss`
  gained a `beta: dict[str, float] | float = 1.0` parameter, forwarded
  unchanged to `losses.regularization.computeTotalRegularizationLoss`.
  This is not a new capability, just wiring an already-designed
  parameter through, matching what ADR 0004 already stated as this
  method's expected shape.
- **Batch contract:** `Trainer` consumes any `Iterable` yielding
  `dict[str, torch.Tensor]` batches (modality name -> raw tensor for
  the whole batch), the same per-modality convention
  `GlobalVae.forward` already uses. That single dict is used as both
  the encoder input (after modality dropout, if enabled) and the
  reconstruction target, unmodified. This makes the common case (plain
  autoencoding, spec §6.1 milestone 1) require nothing from the
  caller beyond yielding raw per-modality tensors: no separate
  "targets" object to construct, and modality dropout (spec §5) needs
  no extra plumbing on the caller's side since the full original batch
  is always available as the target regardless of what the encoders
  actually saw.
- **`TrainerCallback` (`training/callbacks.py`) is a plain composable
  list, not a registry-selected strategy.** Every other pluggable
  piece in this codebase (encoders, decoders, fusion, assemblers,
  regularizers, beta schedules) is a mutually-exclusive choice for one
  role, selected by name from a registry. Callbacks are not: a
  training run commonly wants several active simultaneously (a metrics
  logger and a checkpointer, say), so `Trainer` takes
  `callbacks: list[TrainerCallback] | None`. `TrainerCallback` itself
  is a plain class (not an `ABC` with `@abstractmethod` hooks): every
  hook (`onTrainBegin`, `onEpochBegin`, `onStepEnd`, `onEpochEnd`,
  `onTrainEnd`) is a no-op by default, so a concrete callback overrides
  only the events it cares about, mirroring the callback pattern used
  by Keras, PyTorch Lightning, and Hugging Face `Trainer`.
- **Modality dropout (spec §5) lives in `Trainer`, not in the data
  pipeline.** It decides, per step, which already-loaded modalities to
  hide from the encoders; it never touches how data is loaded or
  preprocessed, so it does not cross the data/framework boundary. Each
  modality is dropped independently with probability
  `modality_dropout_p` (default `0.0`, a no-op, correct for the
  single-modality milestone where there is nothing to drop); at least
  one modality is always kept.
- **Beta resolution merges a base `beta` with any per-latent-space
  `beta_schedules`** (`Trainer._computeBeta`): every latent space in
  `beta_schedules` gets that schedule's value for the current step,
  every other latent space gets `beta` (broadcast to every latent
  space first if `beta` is a single float, so a partial `beta_schedules`
  can never silently leave a latent space using the wrong shared
  value). This is the "orthogonal, composable" design spec §2.3 and
  ADR 0004 already called for; `Trainer` is simply where it is finally
  exercised end to end.
- **Optimizer configurability mirrors the "value or factory" pattern
  already used elsewhere** (e.g. `OneDCnnEncoder`'s
  `activations`/`normalizations`): `optimizer` accepts either an
  already-constructed `torch.optim.Optimizer` (full control, e.g.
  custom per-parameter-group learning rates) or an optimizer class
  (default `torch.optim.Adam`), instantiated internally as
  `optimizer(model.parameters(), **optimizer_kwargs)`.
- **Console progress uses the standard `logging` module** (never
  `print`, spec §10), independent of `TrainerCallback`: the two serve
  different purposes (human-readable progress vs. structured metric
  events for downstream consumers) and are not meant to be merged into
  one mechanism.
- **`Trainer.fit` runs in a `try`/`finally`**, so every callback's
  `onTrainEnd` fires even when training exits early via an exception,
  letting a file-based logger (once one exists) always get a chance to
  flush/close cleanly.
- **`global_step`, `start_epoch`, and `history` are plain instance
  state, persisting across multiple `fit()` calls** on the same
  `Trainer` (e.g. "train 5 more epochs"), rather than being reset each
  call. This is deliberately what a future checkpoint-resume feature
  is expected to serialize; `Trainer` does not implement checkpointing
  itself yet.

## Consequences

- Spec §6.1 milestone 1 ("a working single-modality signal VAE...
  trained end to end... with the ability to inspect training curves")
  is now reachable: `GlobalVae.createSingleLatent(...)` plus `Trainer`
  covers forward, reconstruction + regularization loss, backward,
  optimizer step, and per-step/per-epoch metrics, on whatever data the
  caller provides.
- `tests/integration/test_trainer.py` covers: loss actually decreasing
  over epochs on a fixed toy dataset, optimizer configurability
  (instance and class), device placement, beta-schedule resolution and
  override precedence, callback firing counts and metric keys
  (including the `onTrainEnd`-still-fires-on-exception case), modality
  dropout (disabled by default, always keeps at least one modality,
  a no-op for a single-modality model), gradient clipping, multi-epoch
  history and resumed epoch/step numbering, and the relevant error
  paths.
- `tests/integration/test_en_l1_dn_default.py` gained
  `test_beta_scales_the_regularization_loss`, covering the
  `computeRegularizationLoss` fix above (both a shared float and a
  per-latent-space dict).
- Still deliberately **not** built, tracked in `training/NOTE.md`:
  checkpointing (model/optimizer/step state save-restore), any
  concrete experiment logger (TensorBoard, CSV, W&B, MLflow; the
  callback seam this plugs into already exists), and global seed/
  deterministic-mode management (needs to run before model
  construction, so it is out of `Trainer`'s own scope regardless of
  when it is added).
- `training/NOTE.md` updated to reflect the above instead of "trainer.py
  itself is not yet implemented".
