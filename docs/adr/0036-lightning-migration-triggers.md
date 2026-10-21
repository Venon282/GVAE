# 0036: When to migrate the training loop to Lightning or Fabric

**Status:** accepted (roadmap item P2-5; it records when to migrate and what the migration must keep, and changes no code)
**Date:** 2026-10-21

## Context

Spec §10 keeps a raw PyTorch loop (`training/trainer.py`) "for now" and says to migrate to PyTorch
Lightning, or Lightning Fabric as an intermediate step, "once the model design stabilizes and
multi-GPU/scaling needs become concrete". [ADR 0005](0005-training-loop.md) repeated the sentence.
Neither says how to tell that the moment has come, so the question would be reopened by whoever
next feels the loop is in the way.

The loop has also grown since ADR 0005. It now has mixed precision, gradient accumulation and
`torch.compile` ([ADR 0031](0031-mixed-precision-gradient-accumulation-and-compile.md)), resume with
the state of the callbacks ([ADR 0032](0032-resume-from-the-training-script.md)), top-K checkpoints
and learning-rate schedules ([ADR 0033](0033-top-k-checkpoints-and-learning-rate-schedules.md)) and
checkpoints that the restricted unpickler reads ([ADR 0034](0034-safe-checkpoint-loading-by-default.md)).
Lightning and Fabric provide most of these themselves. Migrating before there is a need rebuilds a
tested loop for nothing, and migrating late means porting more of it, so the triggers should be
checkable, and not a matter of taste.

## Decision

**1. Migrate only when both triggers hold.**

- **A. The model design is stable: Phase 1 of the roadmap is complete.** In practice, P1-1 to P1-6
  are done, which means `tests/integration/test_configuration_matrix.py` passes for all 8
  configurations of spec §2.1. *State today: met.* The defaults taken for decisions D-1 to D-3 still
  await the owner's confirmation, and the ADR that starts a migration checks that they stand.
- **B. A real dataset or model needs more than one GPU, measured and not forecast.** One run on the
  real data (roadmap P3-1) or with a real backbone (P3-5), on one GPU with the levers that already
  ship (`mixed_precision: bf16`, `accumulate_grad_batches`, `compile_model`), shows one of two
  things: the model with a batch of the smallest size the owner accepts does not fit in the memory
  of one GPU, or the epochs the run needs take longer than the owner accepts. The second limit is a
  number only the owner can give (spec §12: asked, never guessed). The measurement (device, batch
  size, peak memory, seconds per step, epochs needed) goes into the ADR that starts the migration.
  *State today: not met.* No run on real data exists, and the test suite runs on one device.

Not triggers: wanting Lightning's ecosystem, a hyperparameter sweep or a set of independent runs
(run one process per GPU), a long single-GPU run that the levers above can still shorten, and a new
experiment logger (loggers are registered here, ADR 0008 and ADR 0035).

**2. What the migration must keep.** It changes how the loop runs, not what the framework is:

- **The configuration.** `TrainingConfig` and the shipped YAML files stay valid. A new optional
  field (`devices`, `strategy`) defaults to today's behavior. The existing fields map onto
  Lightning's `Trainer` as `num_epochs` -> `max_epochs`, `mixed_precision` -> `precision`
  (`bf16-mixed`, `16-mixed`), `accumulate_grad_batches` and `log_every_n_steps` by name,
  `grad_clip_norm` -> `gradient_clip_val` with the norm algorithm, `device` -> `accelerator` and
  `devices`, `resume_from` -> the `ckpt_path` of `fit`, and `compile_model` -> compiling the module
  before it is handed over.
- **The callback seam.** Each registered callback and logger keeps its name and its kwargs. The
  expected shape is one adapter, a Lightning `Callback` that holds a `TrainerCallback` and forwards:

  | `TrainerCallback` | Lightning `Callback` |
  |---|---|
  | `onTrainBegin` | `on_train_start` |
  | `onEpochBegin` | `on_train_epoch_start` |
  | `onStepEnd(step, metrics)` | `on_train_batch_end`, which runs for every batch: forward only when the batch ended an optimizer update, which is what a step is here (ADR 0031) |
  | `onEpochEnd(epoch, metrics)` | the epoch-end hook that runs after validation, with train and validation metrics together |
  | `onTrainEnd` | `on_train_end`, and `on_exception` so that a failed run is still closed |
  | `stateDict` and `loadStateDict` | `state_dict` and `load_state_dict` (with a `state_key`) |
  | `trainer.should_stop = True` | `trainer.should_stop = True` |

  The hook names are those of the Lightning 2.x documentation. The order of the two epoch-end hooks
  and the metrics each one sees are checked against the installed version by a test, not assumed.
- **The checkpoint and the guarantee.** The migrated loop either writes the file of
  `training/checkpoint.py`, or ships a converter in the manner of `upgradeCheckpoint`, so that
  `loadCheckpoint(weights_only=True)`, `scripts/evaluate.py` and `scripts/visualize_latent.py`
  keep working (ADR 0034). On one device, the resume guarantee of ADR 0032 holds again, and the
  per-epoch metrics of the synthetic configurations agree with those of the raw loop to a tolerance
  that the migration's ADR states.
- **What Lightning does not provide stays ours.** Modality dropout (spec §5), the per-latent-space
  beta schedules (ADR 0004), the per-modality losses and reductions (ADR 0029, ADR 0030) and
  `computeLosses` move over unchanged, into the training step.
- **The data boundary.** Datasets, pairing and splitting stay the caller's `loader_factory` (spec
  §6.2); no `DataModule` enters `src/`. Under a distributed strategy the caller's loaders must be
  `torch.utils.data.DataLoader` objects, which Lightning and Fabric can shard. A plain list or
  generator, as the tests use, cannot be sharded.

**3. Fabric first.** When both triggers hold, start with Fabric, the "intermediate step" of spec
§10. It takes the loop as it is (`fabric.setup`, `fabric.backward`, `fabric.clip_gradients`,
`fabric.save` and `fabric.load`) and adds device placement, precision, strategy and launch, while
the callbacks, the loggers, the config and the checkpoint code stay where they are. Moving on to
Lightning's own `Trainer` is a separate question, worth asking only if the hand-written parts that
Lightning also provides (resume, top-K, the gradient scaler) become a burden. This order is a
default, and the ADR that starts the migration may change it.

## Alternatives considered

- **Migrate now.** Phase 1 is complete, but nothing needs a second GPU. The raw loop is where the
  features of ADRs 0031 to 0034 were built and tested, and a migration with no need would change the
  loop and the checkpoint layout for no measured gain.
- **Never migrate, and write distributed training in the raw loop.** `DistributedDataParallel`, a
  launcher, samplers, rank-aware logging and checkpointing, and per-rank RNG state are what Fabric
  exists to provide, and each would be ours to maintain. It stays an option only if the migration
  turns out to cost more than that.
- **A date or a version as the trigger** ("at 2.0"). It is not tied to a need, and it would fire
  whether or not anyone had a second GPU.
- **Trigger B as a forecast** ("the real data will be large"). A wrong forecast costs a migration
  nobody needed, or a late one. One measured run on the real data is cheap, and P3-1 needs it
  anyway.

## Consequences

- Roadmap P2-5 is closed with a document and no code. The "Not planned" entry of roadmap §8 (a
  migration before the triggers hold) now points here.
- Trigger A holds, and B cannot be judged until P3-1 or P3-5 produces a real run. That run is when
  to record the measurement above, whatever its result.
- Until then, new training behavior goes through the `TrainerCallback` seam, `computeLosses` and the
  config, which the migration keeps, and not into the body of `Trainer.fit`, which it replaces. That
  keeps the size of the migration bounded.
- Not decided here: Fabric or Lightning in the end, the strategy (DDP, FSDP), several nodes, the
  checkpoint layout, the tolerance against the raw loop, and the owner's wall-time limit.
