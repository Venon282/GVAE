# Train in stages

A staged recipe trains a pair of modalities in three runs that hand checkpoints to each other:

| Stage | Trains | Data | Experiment file | Available |
|---|---|---|---|---|
| 1, solo A | encoder A and decoder A | every sample that has modality A | `pair_solo_signal` | yes |
| 2, solo B | encoder B and decoder B | every sample that has modality B | `pair_solo_image` | yes |
| 3, pair | both, from the two solo checkpoints | paired samples only | `pair_align` | yes |

All three are ordinary runs of `scripts/train.py`: they resume, seed and log like any other, and
the stages hand each other files, not state (roadmap decision D-15). What the staging adds is a
**stage manifest** written into every checkpoint, so that a later stage can tell which modality,
which latent space and which model config a file belongs to, the **warm start** that builds the
pair model of stage 3 from the two solo checkpoints and checks them against that manifest, and the
two **pair loss terms** that make the two latent spaces say the same thing about a sample. The
motivation and the full plan are in
[Phase 6 of the roadmap](../roadmap.md), in [ADR 0051](../adr/0051-stage-manifest-and-solo-recipes.md)
and in [ADR 0052](../adr/0052-warm-start-from-solo-checkpoints.md).

!!! note "What it does and does not tell you"
    The commands below run end to end on the synthetic dataset of `examples/04_paired_loader.py`
    (about six minutes on two CPU cores for the three stages). That dataset is a toy, and what it
    shows is that the stages chain and what each guard sees, not how the recipe does on your pairs
    ([ADR 0057](../adr/0057-staged-training-end-to-end.md) has the numbers, and the caveats).
    [Choose a latent distance](choose-a-latent-distance.md) says what each distance compares and
    which to try first.

**On this page**

- [Run the three stages](#run-the-three-stages)
- [What each stage needs](#what-each-stage-needs)
- [What to look at between the stages](#what-to-look-at-between-the-stages)
- [Read the manifest](#read-the-manifest)
- [Use it in your own experiment](#use-it-in-your-own-experiment)
- [Start the pair stage from the two solo checkpoints](#start-the-pair-stage-from-the-two-solo-checkpoints)
- [What the warm start refuses](#what-the-warm-start-refuses)
- [What it copies, and what it does not](#what-it-copies-and-what-it-does-not)
- [Pull the two latent spaces together](#pull-the-two-latent-spaces-together)
- [Which distance to try first](#which-distance-to-try-first)
- [Keep the solo solutions: freeze or slow down](#keep-the-solo-solutions-freeze-or-slow-down)
- [Read the result against the baselines](#read-the-result-against-the-baselines)

## Run the three stages

The three experiments read the data contract of the signal plus image pair
(`configs/data/signal_image.yaml`), so one loader and one data folder serve every stage. Each solo
model uses its own key (`signal` or `image`) and ignores the other. With the reference loader of
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

# Stage 3: the pair model, from the two solo checkpoints, with the two pair loss terms
PYTHONPATH=. python scripts/train.py --config-name experiment/pair_align \
    data.loader_factory=examples.04_paired_loader:buildPairedDataloaders \
    data.train_path=/path/to/paired_dataset
```

Running `python examples/04_paired_loader.py` with no argument writes a small synthetic paired
dataset to `examples/outputs/04_paired_loader/data` (git-ignored), which is what `data.train_path`
can point at to try the three commands. Each run writes its checkpoints to
`outputs/<experiment>/checkpoints/`: `best.pt` (the epoch with the lowest validation total loss)
and the last three periodic files. Stage 3 needs no path on its command line: `pair_align` lists
`outputs/pair_solo_signal/checkpoints/best.pt` and `outputs/pair_solo_image/checkpoints/best.pt` as
its warm start, which are where the first two commands write, so run the three from the same folder.
If you gave a solo stage another `output_dir`, name the files in `training.warm_start`
([below](#start-the-pair-stage-from-the-two-solo-checkpoints)).

All three are resumable like any other run (`training.resume_from=...`, see
[Train a model](train.md#resume-an-interrupted-run)). Resume with the same experiment file, and the
manifest is in the checkpoints of the resumed part too. A stage 3 that resumes does not read the solo
files again.

## What each stage needs

- **Stages 1 and 2: every sample of the modality**, paired or not. This is the point of staging: the
  solo VAEs use all the data of their modality, and the scarce pairs are kept for stage 3. The
  loader is yours (project specification, §6.2), so give each solo stage a `loader_factory` that
  yields every sample that has the modality. The reference loader yields the **pairs only**, which
  is enough to run the recipe and to read the manifest, but not what a real run wants.
- **Stage 3: paired samples only**, in the row order that makes row `i` of every modality one
  sample.
- **The loss scale, the same in all three.** Both solo experiments sum their reconstruction
  (`reconstruction_reduction: sum`), and the image has a weight of 0.5, as `image_vae` does
  ([Choose beta and the loss scale](choose-beta.md)); `pair_align` has the same. An averaged
  reconstruction weighs 256 times less for a signal of 256 points against a beta that warms up to 1, and
  on the reference dataset that ended with every latent dimension of the signal model inactive
  ([ADR 0057](../adr/0057-staged-training-end-to-end.md)). If you trained a solo stage at another
  scale, give stage 3 the same one (`training.reconstruction_reduction`, `reconstruction_weight`), or
  the pair stage pulls the copied weights toward another objective than the one they were trained for.
- **Judge each solo model before going on.** Evaluate it alone
  ([Evaluate and visualize a checkpoint](evaluate.md)): a collapsed or poor solo VAE is found
  cheaply here, and not after the pair stage.

## What to look at between the stages

Each stage ends with a short list of things worth reading before the next one starts; none of them
needs more than the log and `scripts/evaluate.py`.

| After | Look at | What it tells you |
|---|---|---|
| Stage 1 or 2 | The last `New best val/loss/total=... at epoch N` line of the log. | The epoch of `best.pt`. The total includes a beta that is still rising during the warm-up, so it favours early epochs: with a run of 200 epochs, a best epoch of 25 holds a model that never saw the final beta, and stage 3 continues it at beta 1. Train longer, or shorten the warm-up, until the best epoch is near the end. |
| Stage 1 or 2 | The solo model on its own ([Evaluate and visualize a checkpoint](evaluate.md)): reconstructions, latent diagnostics. | A solo model that reconstructs badly or uses 0 latent units is cheaper to find here. The warm start refuses a collapsed one. |
| Stage 3, at the start | The two `Warm start: copied ...` lines. | Which files it took, their stage labels and **how many latent units are active** in each space. Keep these two numbers: they are the reference of the collapse guard of the evaluation. |
| Stage 3, during | `train/loss/latent_alignment` and `train/loss/cross_reconstruction` in the log, next to the same two under `val/`. | The values are after the weight, so they start at exactly 0 (the ramp) and rise before they fall. A validation term far above the training one (on the reference dataset, 1.4 against 0.09) is a stage that learns its pairs: with few pairs, stop earlier or freeze ([below](#keep-the-solo-solutions-freeze-or-slow-down)). |
| Stage 3, at the end | `scripts/evaluate.py --latent-pair` on held-out pairs ([below](#read-the-result-against-the-baselines)). | Whether the spaces say the same thing: the retrieval of the partner against chance, the cross reconstruction, the self reconstruction against what the solo models had, the active units. A distance that fell is not the answer. |

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

`training.warm_start` is a list of entries, one per solo checkpoint, and is empty by default.
`pair_align` sets it to the `best.pt` of the two solo experiments at the folders they write to, so
the third command above needs nothing more. A solo stage that wrote somewhere else (an `output_dir`
of your own), or a file you would rather start from, is named on the command line. The list is
replaced as a whole, so give both entries, and quote it, or the shell splits the braces:

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/pair_align \
    data.loader_factory=examples.04_paired_loader:buildPairedDataloaders \
    data.train_path=/path/to/paired_dataset \
    'training.warm_start=[{checkpoint: runs/signal/checkpoints/best.pt, modules: [signal]},{checkpoint: runs/image/checkpoints/best.pt, modules: [image]}]'
```

In a YAML file of your own, the same two entries are:

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
`beta_schedules`, so a pair stage built on it alone weights both spaces by the constant `training.beta`
(1.0, the end value of the solo warm-up). `pair_align` writes that out: each of `z_signal` and `z_image`
has a `constant` schedule whose value is `${training.beta}`, because every shipped experiment names a
schedule for every latent space, and so `training.beta` stays the one setting. If a solo stage ended at
another value, set `training.beta` to it and both spaces follow. A schedule left over from
`training: default` is keyed by `z_fused`, names no space of the pair model, and the trainer says so in
a warning. Per-space settings follow the new names: the regularizer is declared with each space, and
`training.beta_schedules` is keyed by `z_signal` and `z_image`.

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
modalities, two spaces). Keeping the copied modules fixed, or at a lower learning rate, is
[a section of its own](#keep-the-solo-solutions-freeze-or-slow-down); the pair loss terms are
[the next section](#pull-the-two-latent-spaces-together).

## Pull the two latent spaces together

Warm-started and left alone, the two spaces of the pair model stay where the solo runs put them:
nothing in the loss asks `z_signal` and `z_image` to agree about the same sample. Two loss terms
do, and you list them in `training.loss_terms` like any other term
([ADR 0047](../adr/0047-extra-loss-terms.md), [ADR 0054](../adr/0054-pair-stage-loss-terms.md)).
`pair_align` lists both, and this is the part of it that sets them up:

```yaml
training:
  stage: pair
  modality_dropout_p: 0.0              # required: see "What the terms refuse"
  reconstruction_reduction: sum        # the scale of the solo stages (finding I-4)
  reconstruction_weight: {signal: 1.0, image: 0.5}
  loss_terms:
    - name: latent_alignment
      kwargs:
        pairs: [[z_signal, z_image]]
        distance: wasserstein2_diag    # a name; {name, kwargs} when the distance has options
      schedule: {strategy: linear_warmup, kwargs: {warmup_steps: 100, end_value: 0.3}}
    - name: cross_reconstruction
      kwargs:
        pairs: [[z_signal, z_image], [z_image, z_signal]]
      schedule: {strategy: linear_warmup, kwargs: {warmup_steps: 100, end_value: 0.3}}
```

The weight `lambda` of a term is the `weight` of its entry, or a `schedule` instead (never both); a
ramp from zero protects the solo solutions at the start, because a distance that is large at the
start would otherwise pull on weights that are already good.

| Term | What it adds | Arguments |
|---|---|---|
| `latent_alignment` | For each pair, the distance between the two posteriors of the **same sample**, averaged over the batch, summed over the pairs. Nothing is encoded again: it reads the posteriors the step already computed. | `pairs`: a list of `[first, second]` latent space names (or `anchor` and `others`, see below). `distance`: required, a name from [Choose a latent distance](choose-a-latent-distance.md) or `{name, kwargs}`. `stop_gradient`: `none` (default), `source` or `target`. |
| `cross_reconstruction` | For each **ordered** pair `[source, target]`, the sample of the source space is decoded by the decoders that read the target space, and compared with the target modality of the batch. The loss, the weight and the reduction are the trainer's own for that modality (`training.reconstruction_loss`, `reconstruction_weight`, `reconstruction_reduction`), so a modality is scored the same way whichever space it is rebuilt from. | `pairs`: a list of `[source, target]`, both orders for both directions (or `anchor`, `others` and `directions`, see below). `stop_gradient`: `none` (default), `source` or `target`. |

`stop_gradient: source` cuts the gradient on the first name of every pair, `target` on the second,
which makes one side an anchor that the other is trained toward. For `cross_reconstruction`,
`source` detaches the sample (only the decoders learn to read the other space) and `target` keeps
the decoders out of this term's gradient (only the encoders learn to write what the decoders read).
The setting is one per term; a different one for each pair is not offered.

### Three spaces or more: anchor mode

With three modalities that each have a latent space, listing every pair costs `K (K - 1) / 2` distances and
`K (K - 1)` decodings. Anchor mode aligns every space to one of them instead, in `K - 1` and `2 (K - 1)`
([ADR 0059](../adr/0059-more-than-two-models-anchor-mode.md)): give a term `anchor` and `others` in place of `pairs`.

```yaml
training:
  loss_terms:
    - name: latent_alignment
      kwargs:
        anchor: z_signal
        others: [z_image, z_text]
        distance: wasserstein2_diag
        stop_gradient: target          # the anchor stays where it is; the others move to it
    - name: cross_reconstruction
      kwargs:
        anchor: z_signal
        others: [z_image, z_text]
        directions: both               # both, to_anchor or from_anchor
```

`latent_alignment` builds `[other, anchor]` for each of `others`, so the anchor is the **second** space of every pair
(`stop_gradient: target` keeps it). `cross_reconstruction` decodes the anchor from every other's sample and every other
from the anchor's, unless `directions` says one way. `pairs` and `anchor` are not mixed in one term, and each of the two
terms of a stage names its anchor. Anchor mode aligns each space to the hub and **not** the others to each other: how far
two non-anchor spaces agree through it is read with `--latent-set` in
[Evaluate and visualize a checkpoint](evaluate.md#across-three-latent-spaces-or-more).

### Spaces of different widths: adapters

Two solo models trained at the width that suits each modality (16 and 8) do not have to be retrained at one width to be
aligned ([ADR 0060](../adr/0060-unequal-latent-widths-adapters.md)). Give `latent_alignment` an adapter on one of the
spaces: a latent head that only this term puts between the space and the distance.

```yaml
training:
  loss_terms:
    - name: latent_alignment
      kwargs:
        pairs: [[z_signal, z_image]]        # widths 16 and 8
        distance: wasserstein2_diag
        stop_gradient: target               # z_signal stays; the adapter and z_image move
        adapters:
          z_image: 16                       # linear, 8 -> 16; or {name: linear, out_dim: 16}
```

The spaces, their encoders and their decoders keep their widths, and nothing else reads the adapter. It is a parameter of
the term: it trains with the term, is saved in the checkpoint under `loss_terms_state_dict` and comes back when a run
resumes. The width compared is the adapter's `out_dim`, so the two spaces of a pair must still be read at one width, and
the error says so when the adapters leave two.

- **Adapt one space toward the other.** Two trainable adapters can make the distance small by shrinking both outputs. On
  the synthetic problem of the ADR, one adapter retrieved the true partner at recall@1 of 0.22 to 0.55 (chance 0.003)
  and kept the spread of what it read; two adapters without a stop gradient retrieved 0.05 to 0.09 and kept 1 to 7 % of
  it, which the active-unit guard did not flag. Use one adapter and `stop_gradient` on the other space.
- **Keeping both VAEs and training only the adapter** is `freeze_modules` with `modules: ["*"]` and `adapters` (the table above): the
  quickest way to align two finished models, and the one that cannot touch what the solo stages settled.
- **`cross_reconstruction` does not take adapters**: it hands the sample of one space to the decoders of the other, which
  an adapter cannot make fit. In a stage with unequal widths use `latent_alignment` alone.
- Read the result with `--latent-pair` and `--alignment-adapter`
  ([Evaluate and visualize a checkpoint](evaluate.md#spaces-of-different-widths)).

### Few pairs, many unpaired samples

When a few dozen samples carry every modality and thousands carry one, a pair stage that sees only the pairs stops
refining each VAE on its own data. Put the unpaired batches in the same stream
([ADR 0061](../adr/0061-few-pairs-many-unpaired.md)). Two settings go together: the pair terms are told to expect batches
that lack a modality, and the loaders are mixed.

```yaml
training:
  loss_terms:
    - name: latent_alignment
      kwargs:
        pairs: [[z_signal, z_image]]
        distance: wasserstein2_diag
        on_unpaired: skip              # a batch without both modalities adds a zero and is counted
```

```python
from global_vae.data.interleave import interleaveBatches

stream = interleaveBatches(
    paired_loader,                                      # batches with signal and image
    {"signal": signal_loader, "image": image_loader},   # batches with one modality each
    unpaired_per_paired=2,
)
trainer.fit(stream, num_epochs=150)
```

An epoch is one pass over the paired loader, each of its batches followed by `unpaired_per_paired` unpaired ones (`0.5`
is one after every second paired batch) taken from the unpaired loaders in turn. The unpaired loaders keep their position
from one epoch to the next and start again when they run out, so every unpaired sample is used over the epochs. The
reconstruction and the KL of an unpaired batch train its own VAE as in the solo stage; the pair term sees the paired
ones. `on_unpaired` is `raise` by default, and `skip` is a choice of the stage: a loader that was meant to pair its
samples and does not would otherwise train on nothing without a trace. `num_applied` and `num_skipped` on each term say
how many batches it was computed on.

On the synthetic problem of the ADR (16 pairs, 2000 unpaired samples per modality, three seeds), this halved the
reconstruction error of held-out signals (0.35 to 0.17) at a retrieval that did not fall. Start here.

**A frozen flow prior** is the second tool, with a caveat. It asks one space to be distributed like the other space's
posterior means over the unpaired samples (Li et al., [arXiv 2506.05240](https://arxiv.org/abs/2506.05240)), which needs no
pair. Fit it once from the solo checkpoint of the space it is fitted on, then pull the other space toward it:

```bash
python scripts/fit_flow_prior.py \
    --checkpoint runs/signal_solo/checkpoints/best.pt \
    --model-factory my_project.models:buildPairModel \
    --dataloader-factory my_project.data:buildUnpairedSignalLoader \
    --space z_signal --output priors/z_signal.pt          # add --dim N if the other space has another width
```

```yaml
    - name: flow_prior_alignment
      kwargs:
        space: z_image
        prior: priors/z_signal.pt
        on_missing: skip
      weight: 0.01
```

- It does **not** pair anything. A distribution has symmetries, and on the synthetic problem a run with the prior and no pair
  term retrieved the true partner at chance. The pairs say which latent goes with which.
- It is a surrogate, not a likelihood: a cloud pulled toward the middle of the target scores below the target's own
  points, so the term shrinks the space it moves. Active units stay at their full count and the collapse guard stays quiet
  while the spread of `z_image` fell from 1.2 to 1.8 down to 0.2 to 0.3 at weight 1.0, and retrieval fell as the weight rose (0.066, 0.051, 0.027 at 0.01, 0.1
  and 1.0). It did not improve retrieval on the toy worlds of the ADR. Start at 0.01, compare the spread of the pulled
  space with the same stage without the term, and read `prior.alignmentGap(y)` (the loss of `y` over the loss of the
  target's own points; a gap below zero means `y` is scoring better than the target).
- The prior is a frozen submodule of the term, kept in the checkpoint's loss-term state, so a resumed run uses the prior it
  began with whatever the file now holds.

### What the terms refuse

Each of these stops the run with a message that says what to change. The first three are refused
when the trainer is built, before the first step:

| You get | Because |
|---|---|
| `modality dropout must be off in this stage` | `several_latents` sets `modality_dropout_p: 0.15`. A pair stage has no fusion to be robust for, and a term that compares two spaces needs both encoded in every step. Set `training.modality_dropout_p=0`. |
| `has latent widths 16 and 8 ... must have the same width (roadmap D-16)` | The two spaces of a pair have different `dim`. Set the same width for both in the model's `latent.spaces`, or, for `latent_alignment`, keep the widths and give `adapters` ([below](#spaces-of-different-widths-adapters)). `cross_reconstruction` takes no adapter: it still needs equal widths. |
| `names 'z_picture', which is not a latent space of this model` | A typo, or a space of another model. The message lists the real ones. |
| `needs paired batches` | A batch that lacks a modality of a pair. The term raises instead of skipping the pair, because a skipped term would change the objective without a trace in the numbers, and the likeliest cause is a loader that does not pair its samples. The message names the missing modality and says which pair needs it. It applies to the validation loader too. A stage that mixes unpaired batches on purpose says `on_unpaired: skip` ([above](#few-pairs-many-unpaired-samples)). |

Every batch of both loaders must therefore carry both modalities, with row `i` of one describing
the same sample as row `i` of the other. A loader that pairs by sample id, as
`examples/04_paired_loader.py` does, gives that. A `distance` is not defaulted either: leave it out
and the term raises a `TypeError` naming it.

**Let the loader say that the rows are the same samples.** Nothing in two tensors says that row `i` of
one is the sample of row `i` of the other, and a loader that joins two sources by position breaks that
quietly (one side sorted differently, one side filtered), after which stage 3 trains on strangers and
reads as a retrieval at chance. A loader that knows where each row came from can add one entry per
modality to the batch, `sample_id.<modality>`, holding the id that *that modality's own source* gave to
the row (an integer tensor of shape `(batch,)`; `hashSampleIds` makes ids from file names or record
keys, the same in every process). The two terms and `scripts/evaluate.py --latent-pair` compare them and
stop on the first batch where they differ ([ADR 0058](../adr/0058-pair-integrity-sample-ids.md)):

```python
from global_vae.data.sample_ids import hashSampleIds


def __getitem__(self, index):
    return {
        "signal": self.signal[index],
        "image": self.image[index],
        "sample_id.signal": hashSampleIds([self.signal_names[index]])[0],
        "sample_id.image": hashSampleIds([self.image_names[index]])[0],
    }
```

The error reads like this (the ids depend on your names):

```text
ValueError: Loss term 'latent_alignment': row i of 'signal' and row i of 'image' are not the same
sample: their ids differ at 8 of 8 rows (row 0 (signal 212308146065094374, image 5871203397045512106),
...). A pair loss on this batch would train on strangers. Check how the loader pairs its sources: a sort,
a filter or a shuffle applied to one modality only.
```

The entries are optional (a batch without them is trusted, as before), a batch that has them for one
modality and not the other is refused, and `check_sample_ids: false` in a term's `kwargs` turns the
check off. The ids prove what the loader says and no more: the statistical check of the evaluation
([below](#read-the-result-against-the-baselines)) is still what catches a join that gives the same ids
to the wrong rows.

### Watch it

Each term is logged as `train/loss/latent_alignment`, `val/loss/latent_alignment` and the same for
`cross_reconstruction` (in the history, the CSV file and any other logger), and it is added to
`loss/total`. The logged value is the one **after the weight**, so with a ramp it starts near zero
even though the distance is largest then; to read the distance itself, measure it on held-out
batches with the posterior means, or watch the curve with a constant weight.

A falling distance is not enough to call the spaces aligned. Under the KL term both spaces also
drift toward the prior, so the distance falls without the terms too, and an alignment that has
collapsed both spaces drives it to zero. What separates an aligned pair is that **a modality can be
rebuilt from the other's code**: the cross reconstruction error, and the self reconstruction errors
staying where the solo stages left them. Measure the three before and after the stage
([Evaluate and visualize a checkpoint](evaluate.md)).

!!! warning "What the measurements say, and do not say"
    On a small synthetic pair (a signal and an image that describe the same frequency and phase),
    stage 3 with the two terms brought the latent distance to under a tenth of its start and under
    half of the same stage without them, kept both self reconstructions within 10 % of where they
    started, and rebuilt the image from the signal's code far better than the control did
    ([ADR 0054](../adr/0054-pair-stage-loss-terms.md) gives the table and the four seeds). These are
    the numbers of a toy, not a recommendation for a weight or a distance: a constant weight of 1 on
    both terms at the default `beta` of 1 was worse. Which distance to train with is roadmap decision
    D-13, still open: the comparison exists ([Read the result against the
    baselines](#read-the-result-against-the-baselines)) and the synthetic pair does not decide it.

Two things to know. The cross reconstruction runs the decoders in training mode, so their batch
normalization layers update their running statistics with activations that come from the other
modality's codes; a long stage with a small momentum should check the self reconstruction in
evaluation mode, or [freeze the decoders](#keep-the-solo-solutions-freeze-or-slow-down), which puts them in
evaluation mode and keeps their statistics where the solo stage left them. And a run
with the terms resumes like any other (`training.resume_from`): the terms have no parameters, and
the random-number state of the checkpoint is what makes a stochastic distance such as `mixture_jsd`
resume to the same curve.

## Which distance to try first

`pair_align` trains with **`wasserstein2_diag`**, in its default squared form: it compares the two
posteriors, means and variances, in closed form, with a gradient everywhere
([Choose a latent distance](choose-a-latent-distance.md#which-to-try-first) says why, and what each of
the other five ignores). The ablation to run next is **`mean_mse`**, the squared gap of the two
posterior means: if it does as well on your pairs, the variances were not what mattered. The distance
is a name in the file, so the command line replaces it, and a different `output_dir` keeps the first
run's files:

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/pair_align \
    data.loader_factory=examples.04_paired_loader:buildPairedDataloaders \
    data.train_path=/path/to/paired_dataset \
    training.loss_terms.0.kwargs.distance=mean_mse output_dir=outputs/pair_align_mean_mse
```

`loss_terms.0` is `latent_alignment`, the first entry of the list. Compare the two runs with the
evaluation below, on the same held-out pairs, and not by their training curves: the two distances are
on different scales, so a lower value says nothing about the better run.

D-13 stays open. That order is where to start and not a result: no distance has been measured on a real
pair, and on the reference dataset `mean_mse` did as well as `wasserstein2_diag` within what one seed on 24
held-out pairs can tell (the numbers are [below](#read-the-result-against-the-baselines)). The six
distances, and the comparison that settles it from your own pairs, are in
[Choose a latent distance](choose-a-latent-distance.md).

## Keep the solo solutions: freeze or slow down

With a few pairs and small batches the pair stage can undo what the solo stages learned: every weight is free to
move, and the `BatchNorm` layers of the CNN blocks move their running statistics at every forward pass in training
mode, whether the weights train or not. Two settings, both off by default (nothing frozen, one learning rate, as in
the paper), hold the solo solutions back ([ADR 0055](../adr/0055-freeze-and-learning-rate-groups.md)).

### Freeze modules

`freeze_modules` is a callback, chosen in `training.callbacks` like the others:

```yaml
training:
  callbacks:
    freeze_modules:
      modules: [decoders.*]     # name patterns, see below
      until_epoch: 10           # optional: frozen in epochs 0 to 9, training from epoch 10
```

or from the command line (quote it, the shell would expand the `*`):

```bash
python scripts/train.py ... '+training.callbacks.freeze_modules.modules=[decoders.*]' \
  +training.callbacks.freeze_modules.until_epoch=10
```

A frozen module is three things at once: its parameters do not train (and no optimizer moves them, weight decay
included), it is in **evaluation mode** every epoch (so its `BatchNorm` statistics and any dropout stand still, which
`requires_grad=False` alone does not do), and it **stays in the optimizer**, so the checkpoint has the same layout
whether a module is frozen or not. `until_epoch` is the first epoch in which the modules train; without it they stay
frozen for the whole run. A run interrupted before, at or after that epoch and resumed is frozen, or released, exactly
as the uninterrupted one would have been. When the run ends, every `requires_grad` goes back as it was.

The patterns match the dotted names of the model's modules (`encoders.signal`, `decoders.image`, a layer inside one, as
in `encoders.signal.conv.0`), with `*` standing for any characters, dots included. A module stands for its own parameters
and for every module inside it. A pattern that matches no module stops the run with a message that lists real names,
because a typo that freezes nothing looks like a stage that works.

| You want | `modules` |
|---|---|
| The decoders fixed, the encoders adapt to them | `[decoders.*]` |
| One modality as an anchor, the other moves toward it | `[encoders.signal, decoders.signal]` |
| Only the encoder of one modality fixed | `[encoders.image]` |
| Both VAEs fixed and only an alignment head trained | `["*"]`, with `adapters` on `latent_alignment` ([below](#spaces-of-different-widths-adapters)): the adapter is a parameter of the term, not of the model, so it keeps training. Without an adapter there is nothing left to train, and the run stops before the first step and says so. |

All the patterns of one `freeze_modules` share its `until_epoch`.

### Slow modules down

`training.param_groups` gives some modules a fraction of the learning rate:

```yaml
training:
  optimizer: {name: adamw, kwargs: {lr: 0.0003}}
  param_groups:
    - modules: [encoders.signal, decoders.signal]
      lr_factor: 0.1              # a tenth of the rate, 0.00003
      name: signal                # optional, shown in the checkpoint and the messages
```

The trainer builds the optimizer from one group per entry, in order, and a last group named `default` with everything
that no entry took (and the parameters of the loss terms) at the full rate. The first entry that matches a parameter
takes it, so put a narrow entry before a wide one. An entry that matches no module, or whose modules were all taken by
earlier entries, is an error. `lr_factor` is above 0 (to stop a module, freeze it: a rate of 0 would leave its
`BatchNorm` statistics moving). The schedule callbacks (`linear_warmup_lr`, `cosine_lr`) keep the ratio between the
groups all through the run, and `lr/group<i>` in the logs shows each rate. Without `param_groups` the optimizer is
built as before.

The groups are written to the checkpoint with the optimizer, and a resumed run must be given the same ones: PyTorch
restores the saved rates, so a config that says 0.5 where the file says 0.1 would be ignored without a word, and
`loadCheckpoint` refuses it instead (the message names `training.param_groups` and `restore_optimizer=False`, which
loads the weights only and starts a new optimizer).

!!! warning "What the measurements say, and do not say"
    On the small synthetic pair of [ADR 0055](../adr/0055-freeze-and-learning-rate-groups.md) (30 epochs of stage
    3 on 64 pairs, three seeds), the image self reconstruction of the paper's setting ended anywhere between 57 and
    385 for a start of 84.5, and with the decoders frozen, or both sides at a tenth of the rate, it ended within 1 of
    each other, at or below the start. The price was on the other side: the solo signal model was far from converged,
    the free run kept improving it, and a frozen or slowed side did not. A frozen anchor aligned less in 30 epochs
    (a distance of 28 to 39 against 7), and a release in the middle of the stage undid part of the protection.
    These are the numbers of a toy and not a recommendation: D-14 stays at "nothing frozen" until a real pair says
    otherwise (finding I-15).

## Read the result against the baselines

A latent distance that falls during stage 3 is not a result: two collapsed spaces are also at distance 0 from each
other (roadmap I-8). Three things make the number mean something
([ADR 0056](../adr/0056-evaluation-across-two-latent-spaces.md)), all described in
[evaluate](evaluate.md#across-two-separate-latent-spaces):

- **Evaluate the pair model across its two spaces.** `scripts/evaluate.py --latent-pair z_signal z_image` gives the
  retrieval of each direction against its chance level, every registered distance on paired and on mismatched
  samples, the cross reconstruction next to the self reconstruction, and the active units of both spaces. Pass the
  active units that each solo stage had (`--solo-active-units`, the numbers of the two `Warm start` lines of stage 3):
  a space that ends with fewer is flagged, which is what a stage 3 that traded information for agreement looks like.
- **Check that the pairs are pairs.** Pass the training pairs (`--train-dataloader-factory`): the orthogonal map fitted
  on them must recover the partner above chance, or row `i` of the two modalities may not be one sample.
- **Compare with what stage 3 has to beat.** The same report holds the Procrustes row: a closed-form orthogonal map
  between the two spaces, fitted on the training pairs and applied to the held-out ones. `examples/05_alignment_baselines.py`
  runs, on one split, the solo spaces as they are (expected at chance), that map, stage 3 once for each registered
  distance, and joint training from scratch with a shared latent space, over repeated random splits and seeds. If the
  Procrustes row matches a stage 3 row on your pairs, the learned alignment is not buying anything yet.

### The command

`scripts/evaluate.py` takes three factories, written as `module:function`: the pair model, the held-out pairs and the
training pairs. For the reference dataset they are a few lines, which build both from `pair_align` and the loader of the
three commands (save it as `my_project/pair_eval.py`, and set `DATA`):

```python
"""The three factories `scripts/evaluate.py` takes for the pair model of the staged recipe."""

import importlib

from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.models.global_vae import GlobalVae

LOADER = "examples.04_paired_loader"
DATA = "/path/to/paired_dataset"


def _experiment() -> ExperimentConfig:
    return loadExperimentConfig(
        "experiment/pair_align",
        overrides=[
            f"data.loader_factory={LOADER}:buildPairedDataloaders",
            f"data.train_path={DATA}",
        ],
    )


def buildPairModel() -> GlobalVae:
    return buildModelFromConfig(_experiment().model)


def buildTestPairs() -> list:
    return list(importlib.import_module(LOADER).buildPairedDataloaders(_experiment().data).test)


def buildTrainPairs() -> list:
    return list(importlib.import_module(LOADER).buildPairedDataloaders(_experiment().data).train)
```

```bash
PYTHONPATH=.:my_project python scripts/evaluate.py \
    --checkpoint outputs/pair_align/checkpoints/best.pt \
    --model-factory pair_eval:buildPairModel \
    --dataloader-factory pair_eval:buildTestPairs \
    --train-dataloader-factory pair_eval:buildTrainPairs \
    --latent-pair z_signal z_image \
    --solo-active-units z_signal=1 z_image=15 \
    --output-dir results/pair_align/
```

`z_signal=1 z_image=15` are the numbers the two `Warm start` lines printed on the reference dataset; use yours. The
report is `results/pair_align/latent_alignment.md` (and `.json`). Read it from the top: the flags, then the retrieval
against its chance level, then the distance ratios, then the cross reconstruction next to the self reconstruction.
The self reconstruction error each modality had before stage 3 is the figure to hold its error after stage 3 against
(evaluate the two solo checkpoints for it, [Evaluate and visualize a checkpoint](evaluate.md)): a modality whose own
reconstruction got worse paid for the agreement.

### What it reads on the reference dataset

One run of the three commands above on the dataset `examples/04_paired_loader.py` writes (200 epochs each, the shipped
settings, 192 training pairs), read on its **24 held-out pairs** ([ADR 0057](../adr/0057-staged-training-end-to-end.md)
has the table and the controls):

| | start of stage 3 | end, `wasserstein2_diag` | end, `mean_mse` |
|---|---|---|---|
| `recall@5` of the partner, signal to image / image to signal (chance 0.21) | 0.33 / 0.08 | 0.54 / 0.46 | 0.54 / 0.50 |
| self error, signal / image | 0.0014 / 0.0032 | 0.0011 / 0.0040 | 0.0013 / 0.0028 |
| R2 of the cross reconstruction, image from the signal / signal from the image | 0.21 / 0.56 | 0.36 / 0.92 | 0.39 / 0.98 |
| active units, `z_signal` / `z_image` | 1 / 15 | 7 / 14 | 7 / 12 |

What to take from it, and what not:

- **The recipe moves the right numbers and the baseline is not beaten on retrieval.** The retrieval rises from near chance to
  0.5, and the signal is rebuilt from the image's code. The Procrustes row of the same report reads `recall@5` 0.54 / 0.46, the
  same as stage 3 (one pair is 0.04 of recall at this size), and rebuilds the image from the signal's code as well (R2
  0.36); it rebuilds the signal from the image's code worse (R2 0.83 against 0.92). On this dataset stage 3 matches
  a closed-form map for the retrieval and beats it for one cross reconstruction.
- **The image paid for it.** Its own reconstruction error ends 23 % above the solo model's with `wasserstein2_diag`
  (0.0040 against 0.0032), and `scripts/evaluate.py` flags `z_image` for 14 active units against the 15 the solo stage had;
  the 24 held-out samples can account for a unit of difference, and the flag is worth reading, not obeying. With `mean_mse`
  the image error went the other way (0.0028), and two more units went quiet (12). If the image's own reconstruction matters
  to you, this is the number to watch, and [freezing the image side or slowing it down](#keep-the-solo-solutions-freeze-or-slow-down)
  is what the stage offers.
- **192 pairs are few for 200 epochs.** The alignment term is 0.09 on the training pairs and 1.37 on the validation ones.
  Stop earlier, or freeze, before you trust the curve.
- **The two distances do not separate** here: the same recall, and the differences in the other rows go both ways. That is one
  seed on 24 pairs of a synthetic dataset where one factor of three is shared (finding I-15), not a verdict on either; D-13
  stays open.

If a run of yours ends with the retrieval at chance, or with the Procrustes row above stage 3 on both the retrieval and
the cross reconstruction, look first at the solo stages (the best epoch and the active units, in
[What to look at between the stages](#what-to-look-at-between-the-stages)), before the loss terms: a solo model that
learned little leaves the pair stage nothing to align.

### If a solo file is missing

Stage 3 stops before the first step with `FileNotFoundError: No checkpoint found at
'outputs/pair_solo_signal/checkpoints/best.pt'`. The file named is the one of `training.warm_start` that does not exist, so
either a solo stage has not been run from this folder, or it wrote to another `output_dir`: run the stage, or name its file
([above](#start-the-pair-stage-from-the-two-solo-checkpoints)).
