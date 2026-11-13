# Train in stages

A staged recipe trains a pair of modalities in three runs that hand checkpoints to each other:

| Stage | Trains | Data | Experiment file | Available |
|---|---|---|---|---|
| 1, solo A | encoder A and decoder A | every sample that has modality A | `pair_solo_signal` | yes |
| 2, solo B | encoder B and decoder B | every sample that has modality B | `pair_solo_image` | yes |
| 3, pair | both, from the two solo checkpoints | paired samples only | `pair_align` | not yet |

Stages 1 and 2 are ordinary runs of a single-modality model, so there is no new training code in
them. This page covers what they add: a **stage manifest** written into every checkpoint, so that a
later stage can tell which modality, which latent space and which model config a file belongs to.
The motivation and the full plan are in [Phase 6 of the roadmap](../roadmap.md) and in
[ADR 0051](../adr/0051-stage-manifest-and-solo-recipes.md).

!!! note "Stage 3 is not shipped yet"
    The warm start that builds the pair model from the two solo checkpoints, the pair loss terms and
    the `pair_align` experiment are roadmap items P6-2 to P6-7. Until they land, stages 1 and 2 give
    you two trained solo VAEs and the manifest the pair stage will check. Nothing on this page
    pretends otherwise.

**On this page**

- [Run the two solo stages](#run-the-two-solo-stages)
- [What each stage needs](#what-each-stage-needs)
- [Read the manifest](#read-the-manifest)
- [Use it in your own experiment](#use-it-in-your-own-experiment)

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
`loadCheckpoint` returns it in four optional fields of `CheckpointMetadata`:

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
```

| Field | Meaning |
|---|---|
| `stage` | The free label of `training.stage`. The framework never interprets it. |
| `modalities` | The batch keys the model's encoders ingest, in encoder order. |
| `latent_spaces` | Each latent space and its width. |
| `config_digest` | The SHA-256 of the model config as canonical JSON. It covers the **model** only, so a different seed, learning rate or output folder does not change it, and a different kernel size, latent width or activation does. |

A checkpoint saved with no `training.stage`, and every 2.0.0 file, has none of this: the fields keep
their defaults (`None`, `[]`, `{}`, `None`) and the file loads exactly as before.

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
