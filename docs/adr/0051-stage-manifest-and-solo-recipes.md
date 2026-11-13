# 0051: Stage manifest and solo recipes

**Status:** accepted (roadmap item P6-1; it adds an optional block to the checkpoint, an optional `training.stage`
field and two experiment files, and changes no number that a run of 2.0.0 produced)
**Date:** 2026-11-10

## Context

Phase 6 trains a pair of modalities in three stages (a solo run for each modality, then a pair run that starts from the
two solo checkpoints). A solo stage is an ordinary run of a single-modality model, so it needs no new training code.
What is missing is a record. A checkpoint written by `saveCheckpoint` holds weights, optimizer state, counters and a
`config` snapshot, but nothing says in a form a program can read:

- which modality the file covers,
- which latent space it has and how wide it is,
- which model config produced it,
- which stage of a recipe it is.

The warm start of P6-2 has to check the first three before it copies a weight, because the failure of a wrong file is
late and unclear. A pair run that loads the image checkpoint where the signal one was meant fails at the first shape
mismatch with a message about a tensor, or, worse, does not fail: two solo models of the same width and the same
architecture have the same state dict keys and shapes, and a latent width that happens to agree lets a swapped file load.
The `config` snapshot could answer these questions, but it is free-form (ADR 0034 lets a caller save any plain value
there), it is the whole experiment and not the model, and `Trainer.saveCheckpoint(path)` called with no `config` saves
none.

Three constraints come from earlier decisions:

- **Plain values only.** A checkpoint is read with `torch.load(weights_only=True)` by default (ADR 0034), so the block
  may hold only strings, numbers, `None`, lists and dicts.
- **Absent by default.** 2.0.0 checkpoints must load, and a run that asks for nothing must write the file it wrote
  (the rule of ADR 0047 and ADR 0048: opt-in, bit-identical default).
- **No orchestration here.** Decision D-15 of the roadmap keeps each stage a normal, resumable, reproducible run. The
  manifest describes a run, it does not drive one.

## Decision

**1. A `StageManifest`, in a module of its own: `training/stage_manifest.py`.** A frozen dataclass of plain values:

| Field | Type | Meaning |
|---|---|---|
| `stage` | `str \| None` | A free label (`solo_signal`, `pair`, ...). The framework never interprets it. |
| `modalities` | `tuple[str, ...]` | `GlobalVae.input_modalities`: the batch keys the encoders ingest, in encoder order. |
| `latent_spaces` | `dict[str, int]` | Latent space name to its width (`LatentSpace.dim`), in the order of the model. |
| `config_digest` | `str \| None` | The digest of the model config (decision 3), `None` when the builder was not given one. |

`StageManifest.toDict()` returns a plain dict (lists, not tuples, so that the file looks the same before and after a
round trip through JSON), and `StageManifest.fromDict` is its inverse. `buildStageManifest(model, stage, model_config)`
reads the model and, when given, the model config. Input modalities and not output modalities are recorded: for a solo
stage they are the same, and the pair model of P6-2 will read what a checkpoint was fed. A model whose decoder
reconstructs something no encoder ingests (the `signal_image_to_image` model) is a different recipe, and a field for it
can be added to the manifest later without breaking a file, because the reader ignores keys it does not know.

**2. The checkpoint carries it under one key, `stage_manifest`, written only when there is one.**
`saveCheckpoint(..., stage_manifest=None)` takes a `StageManifest`, adds nothing to the file when it is `None`, and
otherwise stores its `toDict()` (a dict of plain values) as `checkpoint["stage_manifest"]`. `loadCheckpoint` reads it with
`checkpoint.get("stage_manifest")`, so a file written before this ADR loads exactly as before and yields the defaults.
`CheckpointMetadata` gains four optional fields, with the defaults of "nothing recorded": `stage` (`None`),
`modalities` (`[]`), `latent_spaces` (`{}`) and `config_digest` (`None`). They are flat fields and not a nested object,
which is how the roadmap words the item, and they are what P6-2 will read. `upgradeCheckpoint` needs no change: it
copies every key it does not touch.

The block is stored as plain values and read back as plain values, so a file with a manifest still loads with
`weights_only=True`. A test pins this.

**3. The digest is the SHA-256 of the canonical JSON of the model config.** `computeConfigDigest(config)` takes a
dataclass instance (`ModelConfig`) or a mapping of plain values, converts a dataclass with `dataclasses.asdict`, dumps it
with sorted keys and no insignificant whitespace, encodes it as UTF-8 and returns the lowercase hexadecimal digest. Two
configs with the same content give the same digest whatever the order the keys were written in, in any process and on
any machine, and a changed value (a kernel size, a latent width, an activation) changes it. A value that JSON cannot
represent is an error that names it, never a silent `str()`, because a digest of a `repr` that contains a memory address
would differ on every run. It covers the **model** config and not the whole experiment: the seed, the learning rate, the
output directory or the number of epochs may change between two runs that build the same network, and a pair run must
not refuse a checkpoint for that. The function is public so that P6-2 can compute the digest of the solo model config
file it is told to expect and compare it with the one in the checkpoint, without building the model.

**4. `training.stage`, opt-in.** `TrainingConfig.stage: str | None = None`. `buildTrainerFromConfig(model, config,
config_snapshot=None, model_config=None)` builds a manifest when `config.stage` is set, from the model and from
`model_config` (which `scripts/train.py` passes), and gives it to the `Trainer`. With `stage` unset nothing is built and
nothing is written. An empty string is refused with a message that names `training.stage`. The label is the switch, not
a decoration: a run that wants the record sets one, and a run that does not is the run of 2.0.0.

**5. The `Trainer` holds the manifest, so every writer of a checkpoint writes it.** `Trainer(stage_manifest=None)` keeps
it as `Trainer.stage_manifest`, and `Trainer.saveCheckpoint` passes it to `saveCheckpoint`. The checkpoint callbacks
(`checkpoint`, `best_checkpoint`, `top_k_checkpoint`) all call `Trainer.saveCheckpoint`, so none of them changes, and a
file written by any of them, by a direct call or by a resumed run carries the same block. `Trainer.loadCheckpoint` does
not read it back: resuming a run restores the trainer's own state, and the check of a manifest against a model is the
work of P6-2, where the question "is this the checkpoint I meant" arises.

**6. One experiment file per solo stage of the shipped synthetic pair.** `configs/experiment/pair_solo_signal.yaml` and
`pair_solo_image.yaml` compose the single-latent signal and image models, the data contract of the signal plus image
pair (`configs/data/signal_image.yaml`, read through `examples/04_paired_loader.py`, which already pairs the shipped
synthetic data), and the default training config, and set `training.stage` and a checkpoint callback. They are ordinary
experiments: `python scripts/train.py --config-name experiment/pair_solo_signal ...`. The third stage has its own file in
P6-7, once the warm start and the pair loss terms exist. `docs/how-to/train-in-stages.md` gives the recipe, and says
plainly which stage is not available yet.

## Alternatives considered

- **Put the label and the digest in the `config` snapshot.** It costs no new key, but the snapshot is free-form and may
  be absent, so a reader could never tell "no manifest" from "no config", and the whole experiment (seed, paths, epochs)
  would be part of what is compared. Rejected.
- **A sidecar JSON file next to the checkpoint.** It is easy to read without PyTorch, but it can be lost, renamed or
  edited apart from the weights, which is the failure the manifest is meant to catch. The checkpoint is the one object
  that travels with the weights. Rejected.
- **Always write the manifest.** It would add a key to every file and change the bytes of every run, against the rule of
  the opt-in default, and a run with no stage has no label to write. Rejected; the derived fields (modalities, widths,
  digest) are only worth their keys when a recipe asks for them.
- **A digest of the whole experiment config.** It would make a pair run refuse a solo checkpoint that differs only by a
  seed or a learning rate. Rejected for the reason in decision 3.
- **A digest of `repr(model)`.** It reads the built network and not the config, so it follows a change of code and not of
  intent, and it differs between versions of PyTorch that print a layer differently. Rejected.
- **A nested `manifest` object in `CheckpointMetadata`.** Equivalent, but the roadmap names three flat optional fields
  and the reader of P6-2 only needs those. Rejected for the smaller surface.

## Consequences

- A solo run with `training.stage` set writes `stage_manifest` in every checkpoint, and `loadCheckpoint` returns it in
  four fields of `CheckpointMetadata`. A run without it, and every 2.0.0 file, is unchanged: same keys, same bytes of the
  weights, same metadata defaults.
- The digest and the widths give P6-2 a refusal before the first copy of a weight: a wrong modality, a wrong latent width
  and a different model config are three different messages that name the file.
- The digest is a fingerprint of the config as written. Two configs that build the same network in two ways (an
  argument left to its default in one and written out in the other) have different digests, which errs on the side of
  refusing. P6-2 may relax it by comparing digests of resolved configs if this proves too strict; that is a
  decision for it, with its own ADR.
- The label is free. Two stages may share one, and the framework does not check an order of stages: D-15 keeps the
  orchestration out of the framework.
- `Trainer` gains one optional argument and one attribute, `buildTrainerFromConfig` one optional argument, and
  `TrainingConfig` one optional field. Nothing that built before fails to build.
