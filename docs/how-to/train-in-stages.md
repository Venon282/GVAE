# Train in stages

A staged recipe trains a pair of modalities in three runs that hand checkpoints to each other:

| Stage | Trains | Data | Experiment file | Available |
|---|---|---|---|---|
| 1, solo A | encoder A and decoder A | every sample that has modality A | `pair_solo_signal` | yes |
| 2, solo B | encoder B and decoder B | every sample that has modality B | `pair_solo_image` | yes |
| 3, pair | both, from the two solo checkpoints | paired samples only | `pair_align` | warm start only (see below) |

Stages 1 and 2 are ordinary runs of a single-modality model, so there is no new training code in
them. This page covers what they add, a **stage manifest** written into every checkpoint, so that a
later stage can tell which modality, which latent space and which model config a file belongs to,
and the **warm start** that builds the pair model of stage 3 from the two solo checkpoints and
checks them against that manifest. The motivation and the full plan are in
[Phase 6 of the roadmap](../roadmap.md), in [ADR 0051](../adr/0051-stage-manifest-and-solo-recipes.md)
and in [ADR 0052](../adr/0052-warm-start-from-solo-checkpoints.md).

!!! note "Stage 3 is only half shipped"
    The warm start, the pair model file and the guard against a collapsed solo model are here
    (roadmap P6-2) and the distances are registered (P6-3). The pair loss terms and the `pair_align`
    experiment are roadmap items P6-4 to P6-7, so a stage 3 run today trains the two modalities from their solo weights with the
    reconstruction and regularization terms only, and no term that pulls the two latent spaces
    together. Nothing on this page pretends otherwise. The distances those terms will use are
    already registered (roadmap P6-3): [Choose a latent distance](choose-a-latent-distance.md)
    says what each one compares and which to try first.

**On this page**

- [Run the two solo stages](#run-the-two-solo-stages)
- [What each stage needs](#what-each-stage-needs)
- [Read the manifest](#read-the-manifest)
- [Use it in your own experiment](#use-it-in-your-own-experiment)
- [Start the pair stage from the two solo checkpoints](#start-the-pair-stage-from-the-two-solo-checkpoints)
- [What the warm start refuses](#what-the-warm-start-refuses)
- [What it copies, and what it does not](#what-it-copies-and-what-it-does-not)

## Run the two solo stages

The two solo experiments read the data contract of the signal plus image pair
(`configs/data/signal_image.yaml`), so one loader and one data folder serve every stage. Each model
uses its own key (`signal` or `image`) and ignores the other. With the reference loader of
`examples/04_paired_loader.py` and a folder in the layout it describes, from the repository root:

```bash
# Stage 1: a solo VAE of the signal
PYTHONPATH=. python scripts/train.py --config-name experiment/pair_solo_signal \
    data.loader_factory=examples.04_paired_loader:buildPairedDataloaders \
    data.train_path=/path/to/paired_dataset

# Stage 2: a solo VAE of the image
PYTHONPATH=. python scripts/train.py --config-name experiment/pair_solo_image \
    data.loader_factory=examples.04_paired_loader:buildPairedDataloaders \
    data.train_path=/path/to/paired_dataset
```

Running `python examples/04_paired_loader.py` with no argument writes a small synthetic paired
dataset you can point `data.train_path` at. Each run writes its checkpoints to
`outputs/pair_solo_signal/checkpoints/` and `outputs/pair_solo_image/checkpoints/`: `best.pt` (the
epoch with the lowest validation loss) and the last three periodic files. `best.pt` is the file the
pair stage should take.

Both runs are resumable like any other (`training.resume_from=...`, see
[Train a model](train.md#resume-an-interrupted-run)). Resume with the same experiment file, and the
manifest is in the checkpoints of the resumed part too.

## What each stage needs

- **Stages 1 and 2: every sample of the modality**, paired or not. This is the point of staging: the
  solo VAEs use all the data of their modality, and the scarce pairs are kept for stage 3. The
  loader is yours (project specification, §6.2), so give each solo stage a `loader_factory` that
  yields every sample that has the modality. The reference loader yields the **pairs only**, which
  is enough to run the recipe and to read the manifest, but not what a real run wants.
- **Stage 3: paired samples only**, in the row order that makes row `i` of every modality one
  sample.
- **The loss scale of the image.** `pair_solo_image` sets `reconstruction_reduction: sum` and
  `reconstruction_weight: 0.5`, as `image_vae` does, because a 64 x 64 image under the default `mean`
  reduction collapses the model ([Choose beta and the loss scale](choose-beta.md)).
- **Judge each solo model before going on.** Evaluate it alone
  ([Evaluate and visualize a checkpoint](evaluate.md)): a collapsed or poor solo VAE is found
  cheaply here, and not after the pair stage.

## Read the manifest

The manifest is on only when `training.stage` is set, and both solo experiments set it
(`solo_signal` and `solo_image`). Every checkpoint the run writes, whichever callback writes it,
holds a `stage_manifest` entry of plain values, so it is read in the safe mode like the rest of the
file ([ADR 0034](../adr/0034-safe-checkpoint-loading-by-default.md)).
`loadCheckpoint` returns it in five optional fields of `CheckpointMetadata`:

```python
from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.training.checkpoint import loadCheckpoint

experiment = loadExperimentConfig(
    "experiment/pair_solo_signal",
    overrides=[
        "data.loader_factory=examples.04_paired_loader:buildPairedDataloaders",
        "data.train_path=/path/to/paired_dataset",
    ],
)
model = buildModelFromConfig(experiment.model)
metadata = loadCheckpoint("outputs/pair_solo_signal/checkpoints/best.pt", model)

metadata.stage  # "solo_signal"
metadata.modalities  # ["signal"]
metadata.latent_spaces  # {"z_fused": 16}
metadata.config_digest  # 64 hexadecimal characters
metadata.module_digests  # {"encoders.signal": "...", "decoders.signal": "..."}
```

| Field | Meaning |
|---|---|
| `stage` | The free label of `training.stage`. The framework never interprets it. |
| `modalities` | The batch keys the model's encoders ingest, in encoder order. |
| `latent_spaces` | Each latent space and its width. |
| `config_digest` | The SHA-256 of the model config as canonical JSON. It covers the **model** only, so a different seed, learning rate or output folder does not change it, and a different kernel size, latent width or activation does. |
| `module_digests` | The same fingerprint for each encoder and decoder alone (`encoders.signal`, `decoders.signal`): the registry name and the kwargs after resolution, so the width of the latent space the module touches is part of it. The pair stage compares these, because the config of a pair model is never the config of a solo one. `{}` for a file written before ADR 0052. |

A checkpoint saved with no `training.stage`, and every 2.0.0 file, has none of this: the fields keep
their defaults (`None`, `[]`, `{}`, `None`, `{}`) and the file loads exactly as before.

To check a file against the model config you expect, without building the model, compare digests:

```python
from global_vae.training.stage_manifest import computeConfigDigest

computeConfigDigest(experiment.model) == metadata.config_digest
```

The digest is the fingerprint of the config **as written**: an argument left to its default in one
file and spelled out in another gives two digests, which errs on the side of refusing.

## Use it in your own experiment

Set the label in any experiment file, or from the command line:

```yaml
training:
  stage: solo_spectrum
```

```bash
python scripts/train.py --config-name experiment/signal_vae training.stage=solo_spectrum \
    data.loader_factory=my_project.data:buildSignalDataloaders data.train_path=/path/to/data
```

A run that leaves `stage` at `null` writes no manifest: its checkpoints have no `stage_manifest`
key, and nothing else about them changed. The label must be a non-empty string.

## Start the pair stage from the two solo checkpoints

Stage 3 trains both modalities in one model, `configs/model/signal_image_pair.yaml`:

```text
signal -> encoder "signal" -> z_signal -> decoder "signal" -> signal
image  -> encoder "image"  -> z_image  -> decoder "image"  -> image
```

The encoders and the decoders carry the names the solo models give them, with the same registry
names and kwargs as `signal_single_latent.yaml` and `image_single_latent.yaml`. The solo latent
space `z_fused` has no weights, so it is simply renamed: the pair file declares two spaces of the
same width (16), `z_signal` and `z_image`, each fed by its own encoder and read by its own decoder.
If you change a kwarg in a solo model file, change it in the pair file too, or the warm start
refuses the checkpoint (below).

`training.warm_start` is a list of entries, one per solo checkpoint, and is empty by default. Until
`pair_align` lands, compose the pair stage from the solo experiment file and the pair model:

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/pair_solo_signal \
    model=signal_image_pair training=several_latents training.stage=pair \
    output_dir=outputs/pair \
    data.loader_factory=examples.04_paired_loader:buildPairedDataloaders \
    data.train_path=/path/to/paired_dataset \
    'training.warm_start=[{checkpoint: outputs/pair_solo_signal/checkpoints/best.pt, modules: [signal]},{checkpoint: outputs/pair_solo_image/checkpoints/best.pt, modules: [image]}]'
```

Or in a YAML file:

```yaml
training:
  stage: pair
  warm_start:
    - checkpoint: outputs/pair_solo_signal/checkpoints/best.pt
      modules: [signal]
    - checkpoint: outputs/pair_solo_image/checkpoints/best.pt
      modules: [image]
```

| Field | Default | Meaning |
|---|---|---|
| `checkpoint` | required | The solo checkpoint (`best.pt` of a solo stage). |
| `modules` | required | A modality name (`signal`) copies `encoders.signal` and `decoders.signal`; `encoders.signal` or `decoders.signal` copies one module. One modality per entry. |
| `latent_space` | `null` | The pair model's space the solo space becomes. `null` takes the one space the copied encoder feeds, and is an error when it feeds several. |
| `check_manifest` | `true` | Require the manifest and check it. `false` is for a file written before the manifest existed: the shape checks and the collapse guard remain. |
| `min_active_units` | `1` | The fewest active latent units the copied encoder may show. `0` switches the guard off. |

`scripts/train.py` applies the entries after it builds the model and the data loaders and before it
builds the trainer, and logs one line per entry (the file, its stage label, the modules, how many
tensors, how many latent units are active). With both files loaded, in eval mode, the pair model's
`use_mean=True` reconstruction of each modality from its own input equals the solo model's bit for
bit; the test suite pins it.

**Beta stays constant.** The solo runs already went through their warm-up, and a warm-up restarted
at step 0 would pull a trained posterior back toward the prior. `training: several_latents` has no
`beta_schedules`, so both spaces are weighted by the constant `training.beta` (1.0, the end value of
the solo warm-up). If a solo stage ended at another value, set `training.beta` to it. A schedule
left over from `training: default` is keyed by `z_fused`, names no space of the pair model, and the
trainer says so in a warning. Per-space settings follow the new names: the regularizer is declared
with each space, and `training.beta_schedules` is keyed by `z_signal` and `z_image`.

A run that resumes (`training.resume_from`) does not apply `warm_start`: the checkpoint it resumes
holds the weights of the pair run, which already started from the solo files, and those files need
not exist any more. A log line says so.

## What the warm start refuses

Each of these stops the run before the first step, with a message that names the entry
(`training.warm_start[1]`) and the file, and leaves the model as it was:

| You get | Because |
|---|---|
| `has no stage manifest` | The file was written by a run with no `training.stage`. Train the solo stage again with it set, or use `check_manifest: false`. |
| `is not a solo model` | The file has several modalities or latent spaces, for example a pair checkpoint. |
| `is the solo model of ['image'], but the entry asks for ... 'signal'` | The wrong file. |
| `is 16 wide, but the pair model's 'z_signal' is 8 wide` | The two latent spaces must have equal widths. |
| `the config of 'encoders.signal' ... is not the one the checkpoint ... was trained with` | The pair file builds the module another way than the solo file did, even when every shape agrees (a pooling, an activation). |
| `no tensor for N key(s) of ...`, `N key(s) that ... does not have`, `shape mismatch for '...'` | A key is missing, unexpected or of another shape. Every parameter and buffer of a copied module must load, and the message names the first key. |
| `looks collapsed` | The solo encoder gives every training sample nearly the same code: 0 of 16 latent dimensions have a posterior-mean variance above 0.01. Aligning it with another modality would align nothing. The usual cause is a beta too large for the data ([Choose beta and the loss scale](choose-beta.md)); train the solo stage again. |

The collapse guard encodes up to 256 samples of the training loader once, with the posterior mean
and without a gradient, and counts active units with the same measure as the evaluation
([Evaluate and visualize a checkpoint](evaluate.md)). It does not touch the random-number stream,
so a run and its resumption stay equal. `min_active_units: 1` refuses a total collapse; ask for more
with a larger number, or switch the guard off with `0` when you mean to.

## What it copies, and what it does not

Only the **model weights** of a checkpoint are read, in the safe mode
([ADR 0034](../adr/0034-safe-checkpoint-loading-by-default.md)), and only the encoder and the
decoder you name, parameters and buffers (the running statistics of a batch normalization
included). Not copied: the optimizer state, the step and epoch counters, the history, the
random-number state and the callbacks' state. Stage 3 is a new run, with its own optimizer, its
own seed and a step counter at 0, and its checkpoints carry its own manifest (`stage: pair`, two
modalities, two spaces). Freezing the copied modules, a lower learning rate for them and the pair
loss terms are later roadmap items (P6-4 and P6-5).
