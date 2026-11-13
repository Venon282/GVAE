# 0052: Warm start of the pair model from two solo checkpoints

**Status:** accepted (roadmap item P6-2; it adds an optional `training.warm_start` list, a pair model file, a reader of the
weights of a checkpoint and one optional field of the stage manifest, and changes no number that a run without
`training.warm_start` produced)
**Date:** 2026-11-10

## Context

Stage 3 of the staged recipe (roadmap Phase 6) trains the encoders and the decoders of two modalities together, and it
starts from the two solo checkpoints of stages 1 and 2: the pair phase of PairVAE builds on the solo-trained VAEs. Three
things stand in the way today.

- **There is no pair model file.** The pair model is two encoders, two latent spaces of the same width each fed by its own
  encoder through an identity edge, and one decoder per modality that reads its own space and nothing else. The schema of
  ADR 0025 can say it (a decoder that reads one space needs no assembler), but no file does.
- **There is no way to start a run from another run's modules.** `loadCheckpoint` loads a whole checkpoint into the model
  that wrote it, and refuses anything else under `strict=True`. The pair model has the encoders and decoders of two
  checkpoints and the latent spaces of neither.
- **A wrong file fails late or not at all.** Two solo models of the same architecture have the same keys and shapes, so
  the image checkpoint loaded where the signal one was meant can load without an error whenever the two modalities have
  the same module layout. A solo model that collapsed (every sample encoded to the same code) loads too, and stage 3
  then spends its pairs aligning two latent spaces that carry nothing, which minimizes any distance (roadmap I-8).

The solo latent spaces carry no weights and have the same name, `z_fused`, in the shipped single-latent models, so what
moves from a solo checkpoint to the pair model is the weights of the encoder and the decoder of one modality, and the
spaces are renamed by the pair model file.

Constraints from earlier decisions: a checkpoint is read with `torch.load(weights_only=True)` by default (ADR 0034); a
run that asks for nothing is the run it was (the opt-in rule of ADR 0047, 0048 and 0051); each stage is an ordinary,
resumable run, and the framework does not orchestrate the three (roadmap D-15); the manifest of ADR 0051 is made of plain
values, and a reader ignores the keys it does not know.

## Decision

**1. A pair model file, `configs/model/signal_image_pair.yaml`.** `latent_mode: several`, with the encoders `signal`
(`1d_cnn_encoder_v1`) and `image` (`2d_cnn_encoder_v1`) and the decoders `signal` (`1d_cnn_decoder_v1`) and `image`
(`2d_cnn_decoder_v1`) **under the names the solo models give them** (`encoders.signal`, `decoders.signal`, ...), the same
registry names and the same kwargs as `signal_single_latent.yaml` and `image_single_latent.yaml`. Two latent spaces,
`z_signal` and `z_image`, of the same width (16, the width of the solo models; roadmap D-16 proposes equal widths), each fed
by its own encoder through an identity edge, with the `kl_standard_normal` regularizer of the solo models. Each decoder
consumes its own space, with no assembler. The names of the spaces are the keys of `training.beta_schedules`.

**2. `training.warm_start`, a list of entries, empty by default.** Each entry is:

| Field | Type | Meaning |
|---|---|---|
| `checkpoint` | `str` | The solo checkpoint to read. |
| `modules` | `list[str]` | What to copy. A modality name (`signal`) stands for `encoders.signal` and `decoders.signal`; `encoders.signal` or `decoders.signal` names one module. A module has the same name in both models. |
| `latent_space` | `str or null` | The space of the pair model that the solo latent space becomes. `null` (default) takes the one space the copied encoder feeds, and is an error when the encoder feeds several. |
| `check_manifest` | `bool` | `true` (default): the stage manifest is required and checked (decision 4). `false`: it is not read, and only the shape checks of decision 5 remain, for a file saved before the manifest existed. |
| `min_active_units` | `int` | `1` (default): the collapse guard of decision 6 refuses a solo encoder with fewer active latent units than this. `0` switches the guard off. |

An empty list is the default: no entry, no read, no change. A module may be loaded by one entry only.

**3. The reader takes the weights and nothing else.** `readCheckpointWeights(path)` (in `training/checkpoint.py`) reads the
file in the safe mode (`weights_only=True`, ADR 0034; there is no option to read it otherwise, and the error of a file the
safe mode refuses names `upgradeCheckpoint`) and returns the model's state dict and the stage manifest, or `None` when the
file has none. `loadCheckpoint` and the reader share one function that opens the file. **Nothing else of the file is
used**: not the optimizer state, the step and epoch counters, the history, the random-number state, the callbacks' state
or the `config`. Stage 3 is a new run with its own optimizer, its own seed and a step counter at 0; its checkpoints
count from there.

**4. The manifest is checked before a weight is copied.** For an entry with `check_manifest: true`, in this order, each
with a message that names the file and the entry:

1. the file has a manifest, and it describes a solo model: one modality and one latent space;
2. each copied module belongs to that modality (the image checkpoint where `signal` was meant is refused here, by name);
3. the width of the solo latent space equals the width of the pair model's `latent_space`;
4. the config of each copied module is the one the checkpoint was trained with (the digest, below).

**The digest of ADR 0051 cannot do check 4.** `config_digest` fingerprints the whole solo model config, and the pair model
config is a different config by construction, so the two can never be equal. The manifest therefore gains one optional
field, `module_digests`: module path (`encoders.signal`, `decoders.signal`) to `computeConfigDigest` of
`{"name": registry name, "kwargs": kwargs}` of that module, **after resolution** (the `latent_dim` that
`buildModelFromConfig` fills in is part of the kwargs, so the width of the space it reads or feeds is part of the digest).
A solo run records it, for every encoder and decoder, whenever it records a manifest (`training.stage` set), and the
pair run computes the same digests from its own model config and compares the entries it copies. `config_digest` stays as
it is. The field is optional in `StageManifest.fromDict` (a file written by ADR 0051 reads, with `{}`), absent digests make
check 4 fail with a message that says to retrain the solo stage or to set `check_manifest: false`, and
`CheckpointMetadata` gains `module_digests` with the default `{}`. The digest is the fingerprint of the config as written:
the same encoder spelled with another kwargs layout gives another digest, which errs on the side of refusing.

**5. The copy is strict and exact.** For each module, the keys the checkpoint holds under its path and the keys the model
holds (parameters and buffers, `BatchNorm` running statistics included) must be the same set, and each shape must agree.
A missing key, an unexpected key or a shape mismatch is an error that names the key, the two shapes and the file, and
nothing is copied. The copy is `load_state_dict(strict=True)` of each module, so a copied module is bit-identical to its
source, and every other module, and the latent spaces, are untouched. If anything fails after the first copy (the guard of
decision 6), the modules already copied are restored, so a failed call leaves the model as it found it.

**6. A collapsed solo checkpoint is refused, by probing the loaded encoder.** Posterior collapse is a property of what the
encoder does to data, and the weights alone do not show it. When an entry copies an encoder and `min_active_units` is above
0, and the caller passes data, the loader encodes up to 256 samples of the pair's training loader with the **posterior
mean** (`encode(use_mean=True)`, eval mode, no gradient) and counts the active units of the entry's latent space with
`computeLatentDiagnostics` (the criterion of Burda et al.: a variance of the posterior mean above 0.01, `DEFAULT_ACTIVE_UNIT_THRESHOLD`). Fewer
than `min_active_units` is an error that says which file and space, how many of how many dimensions are active over how
many samples, and that the usual cause is a `beta` too large for the data shape (`docs/how-to/choose-beta.md`). The probe
runs inside `torch.random.fork_rng()`, restores the training mode of the model, and builds no graph, so it changes neither
the random-number stream nor any weight. With no data passed, the guard is not run, and the report says so. The default
`1` refuses only a total collapse, the case the roadmap names; a stricter bar is a larger `min_active_units`, per entry.

**7. The latent spaces are renamed by the model file, and beta stays constant by default.** The solo space `z_fused`
becomes `z_signal` or `z_image` because the pair model file says so, and the copied weights do not depend on it. The
per-space settings follow the names in the file: the regularizer is declared with each space, and
`training.beta_schedules` is keyed by space name. Stage 3 should **not** restart a warm-up: the solo models already went
through one, and a warm-up from step 0 would train the first steps of stage 3 under a `beta` other than the one the
solo models settled at. The default is therefore no schedule for the pair spaces, which the trainer treats as the constant
`training.beta` (1.0 in every shipped training group, the end value of the solo warm-up). `configs/training/several_latents.yaml`
already has an empty `beta_schedules`; with `training: default` the `z_fused` schedule names no space of the pair model and
the trainer says so in a warning. A solo stage that ended at another `end_value` needs the same value in `training.beta`
for stage 3: the manifest does not record it, and this ADR does not add that.

**8. `scripts/train.py` applies it; `training.resume_from` skips it.** After the model and the dataloaders are built and
before the `Trainer` is, `applyWarmStartFromConfig(model, training_config, model_config, probe_dataloader)` runs the
entries and logs one line per entry (the file, the stage label, the modules, the number of tensors, the active units).
Applying them before the trainer means the optimizer is built over the weights the run starts from; the copy is in place,
so an optimizer built earlier sees them too. A run that resumes (`training.resume_from` set) does not apply them: the
checkpoint it resumes holds the weights of the stage-3 run, which already started from the solo files, and the solo files
need not exist any more. The skip is inside `applyWarmStartFromConfig`, so every caller gets it, and a log line says so.
The solo run records `module_digests` through `buildStageManifestFromConfig`, which computes them with
`computeModuleDigests(model_config)` when it is given a `ModelConfig` (a plain mapping cannot be resolved into modules, and
records none).
Without this wiring the field would be a setting that does nothing, so it is part of this item, ahead of the experiment
files, the how-to and the end-to-end test of P6-7.

## Alternatives considered

- **Compare `config_digest` of the pair model config with the solo one.** Never equal, as decision 4 explains. Rejected.
- **Point each entry at the solo model file and digest it.** It needs a model file or an experiment composed again at
  build time, and it is wrong whenever the solo run was launched with a command-line override of the model, which the
  digest of the run (taken from the resolved config) includes. The digest of the modules, recorded by the run itself and
  recomputed from the pair run's own config, has neither problem. Rejected.
- **Detect collapse from the weights or from the history in the checkpoint.** The weights do not show it, and the history
  (a list of loss values, absent from many files) does not say how many units are active. The probe is the same
  measurement the evaluation reports, on the data stage 3 is about to use. Rejected.
- **Compute active units at the end of the solo run and store them in the manifest.** It would cost a pass over the
  validation set at every checkpoint, or only fill the last one, and it would not cover a file written before it existed.
  Rejected.
- **Probe on the validation loader, or on a loader the entry names.** The pair stage is not guaranteed a validation split,
  and one more field for a diagnostic is not worth its surface. The training loader is the data the run is about to use.
  Rejected.
- **Warm-start by loading the whole solo checkpoint with `strict=False`.** It cannot rename or select modules, and it would
  drop a mismatch with a warning instead of an error. Rejected; the strictness is the point.
- **A `warm_start` that also copies the optimizer or the step.** Stage 3 is a new run (roadmap D-15) with an optimizer over
  different parameters; moving state across would break resumption. Rejected.

## Consequences

- With `training.warm_start` empty (the default) nothing is read, nothing is probed, and no run changes. A solo run that
  records a manifest now also records `module_digests`, one more key in a file that already had the block of ADR 0051.
- A wrong modality, a wrong latent width, a changed module config, a missing or unexpected key, a wrong shape and a
  collapsed solo model are six different errors, each raised before the first step of stage 3 and each naming the file.
- With both solo checkpoints loaded, in eval mode, the pair model reconstructs each modality from its own input, with
  `use_mean=True`, exactly as the solo model does (a test pins it bit for bit).
- A solo checkpoint written before ADR 0051, or by a run with no `training.stage`, needs `check_manifest: false` and so
  loses checks 1 to 4; the shape checks and the collapse guard remain.
- The probe reads at most 256 samples of the training loader once, before training. It does not touch the global random
  number stream, so the uninterrupted run and a resumed one stay equal (ADR 0032).
- The warm start copies modules, not state: the `Trainer` of stage 3 starts at step 0 with a fresh optimizer, and its
  checkpoints carry its own stage manifest.
- Checking a pair model against a different pair of solo checkpoints (a new modality joined to an existing model), freezing
  the copied modules and a learning-rate factor for them are P6-5 and P6-8, not built here.
