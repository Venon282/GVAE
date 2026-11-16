# 0057: Staged training end to end: the pair experiment, the solo handoff and the integration test

**Status:** accepted (roadmap item P6-7, which closes decision D-15; it adds an experiment file, one setting in
another, a variant of the configuration matrix, an integration test and the how-to, and changes nothing in `src/`)
**Date:** 2026-11-15

## Context

P6-1 to P6-6 delivered the parts of the staged recipe, and each was tested on its own: the manifest and the two solo
experiments, the warm start, the distances, the two pair loss terms, freezing, the evaluation across two spaces. The
tests of the parts run in one process and on tiny models. What no test did was chain the three stages the way a person
does, as three commands that hand files to each other, and read the result with the instruments that exist for it. This
item does that, and it is where the roadmap asks D-15 to be settled: three ordinary runs chained by checkpoints (A), or
one driver (B), "revisit B after P6-7".

The wiring the roadmap lists was already there. `scripts/train.py` applies `training.warm_start` after it builds the
model and the loaders and before it builds the trainer (ADR 0052), and `buildTrainerFromConfig` builds
`training.loss_terms` (ADR 0047, 0054), so a pair experiment file needs no new code. What this item adds is the file, the
test that pins the wiring through the script, and what chaining the stages on a real-sized run showed.

## Decision

**1. `configs/experiment/pair_align.yaml` is stage 3.** It composes the pair model, the data contract of the signal
plus image pair and `training: several_latents`, and sets what the stage needs, each with its reason in the file:

- `training.warm_start` lists the `best.pt` of the two solo experiments at the folders those experiments write to by
  default, so that the third command needs no path and is as short as the first two. A run with other folders names its
  own entries.
- The loss scale of each modality is the one of its solo stage (decision 2): both sum their reconstruction, the signal
  with weight 1.0 and the image with 0.5. The cross reconstruction scores a modality with that modality's own loss, so the
  same scale serves it.
- `training.beta` stays constant at the 1.0 of `several_latents`, the end value of the solo warm-up (ADR 0052), and is
  written out as a `constant` schedule for each of `z_signal` and `z_image` whose value is `${training.beta}`: the
  existing suite `test_model_config_schema.py` requires every shipped experiment to name a schedule for every latent
  space, which `pair_align` failed in the first full run, and the interpolation keeps `training.beta` the one setting
  (a schedule of its own value would have made the override of the how-to and of the test a second one). The value is
  the same float the trainer already used: stage 3 on the toy with the schedules and with the key removed
  (`~training.beta_schedules`) writes the same `best.pt` bit for bit, so no number of this ADR changes.
- `modality_dropout_p` is 0, which the terms require (ADR 0054).
- The two terms of ADR 0054, each on a ramp from 0 to 0.3 over 100 steps, with `wasserstein2_diag` as the distance. The
  distance is written as a plain name, which is its squared default, so that
  `training.loss_terms.0.kwargs.distance=mean_mse` replaces it from the command line: a `{name, kwargs}` form would have
  kept the old kwargs through the merge, and a distance that does not take `squared` would fail on it.

**2. `pair_solo_signal` sums its reconstruction.** The file leaves the `mean` of `signal_vae.yaml`, and on the reference
dataset (`examples/04_paired_loader.py`, 192 training pairs) that is not a recipe that works: averaged over 256 points the
reconstruction weighs 256 times less against a beta that warms up to 1, the model ends the 200 epochs with 0 of 16 latent
dimensions active, and the collapse guard of the warm start refuses its last epoch. The run did not notice, because
`best.pt` is the epoch with the lowest validation **total** loss, and a total that includes a beta still rising is lowest
early: it was epoch 25 of 200, at a beta of 0.16, which is not the model at the beta that stage 3 continues with, and which
the guard accepted with 1 active dimension. Summed, as the image stage already does, the model keeps one active dimension
(the signal of that dataset carries one factor), its best epoch is 145, at a beta of 0.87, and the last epoch is healthy
too. One setting changes; both solo stages and the pair stage then share the reduction, and only the weight differs.

**3. The integration test runs the stages as the how-to prints them.** `tests/integration/test_staged_training.py`
(marked `slow`) runs `scripts/train.py` three times in subprocesses from one working directory, with **no** `output_dir`
and **no** `training.warm_start` on the command line, so that stage 3 finds the solo files by the default paths of its own
file. The command line adds what a toy needs and nothing about the loss terms, the ramps, the distance, the scales or the
model: the loader, the number of epochs, a solo warm-up shortened to end inside the run (at the beta 0.1 that stage 3
then holds), and `keep_last_n` so that a checkpoint inside the ramp exists. The data
(`_staged_training_fixtures.py`) is the linked synthetic pair of the P6-2 fixtures, in which the image is a function of the
signal's frequency and phase; independent modalities would give retrieval at chance whatever the stage did. The
measurements are taken on 64 held-out pairs no stage saw, with the evaluation of P6-6.

| Roadmap | What the test pins |
|---|---|
| (a) the warm start is exact | The script's log shows both copies; with both solo files loaded, in evaluation mode, the pair model's `use_mean=True` reconstruction of each modality, from its own input alone and from the full batch, is the solo model's bit for bit. |
| (b) the distance falls | `wasserstein2_diag` on held-out pairs ends under a tenth of its start. |
| (c) retrieval | `recall@10` of each direction is within 0.15 of its chance level before stage 3 and at least 0.5 above it after. |
| (d) self reconstruction | Neither ends above 1.10 times its start. |
| (e) cross reconstruction | Each ends under 0.8 times the error of the constant "mean sample" predictor, and is not under it before stage 3. |
| (f) stage 3 resumes | A run stopped at epoch 20 (step 80, inside the 100-step ramp, so the schedules are restored and not only the weights) and resumed ends with the weights of the uninterrupted one within 1e-6, does not apply `warm_start` again, and trains the 80 epochs that remain. |
| (g) existing suites unchanged | The only experiments that set a stage, a warm start, loss terms or learning-rate groups are the three of the recipe, and this item changes no line of `src/`. |

It also pins what the roadmap does not list and a person relies on: each stage writes its manifest, the terms of the
experiment file reach the trainer (`train/loss/latent_alignment` and the other are in every epoch of the history, and the
first step logs both at exactly 0, the start of the ramp), stage 3 started from a folder where stage 1 never ran stops
before the first step with a message that names the missing file, no space ends with fewer active units than its solo
stage, and `scripts/evaluate.py --latent-pair` reads the checkpoint of stage 3 and flags nothing.

**4. The pair model is one more variant of the configuration matrix.** `EN-LN-DN+pair` in
`tests/integration/test_configuration_matrix.py` is the model of `signal_image_pair.yaml`, composed from `pair_align`,
with only the sizes cut down to the matrix's (a signal of 32 points, a 32 x 32 image, two small layers), so the topology
under test is the file's. It has no assembler and no fusion, so the matrix runs it once and not once for each assembler,
and a separate class pins its topology (two spaces, identity edges, each decoder reading its own space, the decoding
seam of ADR 0046 turning a code of one space into the other modality, equal widths per D-16). The explicit listing test
fails when the variant is dropped (checked by removing it).

**5. D-15 stays at A.** Three ordinary runs chained by checkpoints is what the test runs and what the how-to prints, and
what the work found does not argue for a driver. Each stage resumes, seeds and logs like any run; a failure in stage 3 does
not repeat stages 1 and 2; the paths chain by default. And the things a person has to read **between** the stages are the
things a driver would have to stop for: the epoch the best checkpoint of a solo stage comes from, the active units the warm
start logs, whether the solo model is any good alone. A driver would run them without reading them, which is how the
solo signal of decision 2 went through unnoticed. B is revisited only if a recipe needs the stages to share state that a
file cannot carry.

## Alternatives considered

- **A driver script (`scripts/train_stages.py`, B of D-15).** Discussed above: it saves typing two commands and removes the
  place where the checks are read.
- **Keep `mean` for the signal and take the last epoch of the solo stage.** The last epoch is the collapsed model and the
  guard refuses it, which is the guard working; the recipe would not run.
- **Monitor `val/loss/reconstruction` for the best checkpoint of the solo stages.** It picks an epoch without regard to the
  regularization the model will be held to. It was not tried: summing the signal's reconstruction removes the case in
  which the total loss picked an early epoch, and a different monitor is a change to the default callbacks for a problem
  the how-to can name.
- **Select the best epoch only after the beta warm-up has ended.** It is the right fix for the cause, in the callback, and
  a change to `src/`. This item changes none; the how-to tells the reader to look at the epoch (below).
- **Name the `training.warm_start` entry in the message for a missing file.** The error is
  `No checkpoint found at 'outputs/pair_solo_signal/checkpoints/best.pt'`, which names the file and not the entry the other
  refusals name. It is clear enough to act on (stage 1 has not been run from this folder) and a change to the wording would
  be a change to `src/`; the test pins the message as it is.
- **Evaluate (c) and (e) on the training pairs.** They are what a loss curve shows. The held-out pairs are what separates
  alignment from memory.

## Measurements

**The toy the test runs on** (the linked pair, 64 training pairs, 20 epochs of each solo stage and 100 of stage 3, 64
held-out pairs; seed 0 gives the same numbers on Python 3.11 and 3.13). The solo spaces are the "before", the checkpoint of
stage 3 the "after", three seeds of stage 3, and one stage 3 with no loss terms as the control. `recall@10` has a chance
level of 0.156 and is given for `z_signal->z_image` / `z_image->z_signal`; the errors of the cross reconstruction are
divided by that of the constant predictor, so that under 1 means the code carries information about the partner.

| | distance | recall@10 | self error, signal | self error, image | cross / constant, image | cross / constant, signal | active units |
|---|---:|---:|---:|---:|---:|---:|---:|
| before stage 3 | 54.13 | 0.22 / 0.16 | 0.497 | 0.0186 | 1.28 | 1.09 | 16 / 7 |
| after, seed 0 | 2.09 | 0.94 / 0.89 | 0.256 | 0.0078 | 0.53 | 0.54 | 16 / 16 |
| after, seed 1 | 1.87 | 0.95 / 0.94 | 0.249 | 0.0070 | 0.44 | 0.58 | 16 / 16 |
| after, seed 2 | 1.91 | 0.97 / 0.95 | 0.240 | 0.0068 | 0.47 | 0.62 | 16 / 16 |
| control, no terms | 11.58 | 0.19 / 0.14 | 0.245 | 0.0061 | 5.46 | 1.40 | 16 / 16 |

- **The terms are what aligns.** Without them the distance still falls to 0.21 of its start (the KL term pulls both spaces
  toward the prior, as ADR 0054 found), and a test of (b) alone would nearly pass; retrieval stays at chance and the cross
  reconstruction is five times worse than the constant predictor for the image. (c) and (e) carry the weight.
- **(d) is not binding on this toy.** The self reconstruction ends at 0.37 to 0.51 times its start in the three seeds, 0.33
  to 0.49 in the control and 0.43 and 0.44 in a run with a weight 100 times larger: the solo stages are short, and stage 3
  goes on training both models.
  The tolerance guards a regime in which the solo models are converged, which this toy does not reach.
- The margins of the test sit well inside the seeds: a distance fraction of 0.1 against 0.035 to 0.039, a recall at least
  0.5 above chance against 0.73 to 0.81, a cross fraction of 0.8 against 0.44 to 0.62.

**The reference dataset, run as the how-to prints it** (200 epochs for each stage, the shipped settings, 192 training pairs,
**24 held-out pairs of the test split, one seed**; `recall@5` has a chance level of 0.208 and a standard error of about
0.08 at this size). The signal of this dataset carries one factor of the image's three (the size; the image also has the
centre), so the two modalities share one factor and no code of the signal can match all the coordinates of the image's.

| | solo signal | start of stage 3 | end of stage 3 |
|---|---|---|---|
| as shipped before this item (`mean`) | last epoch 0 of 16 active, `best.pt` from epoch 25 (beta 0.16) | recall@5 0.29 / 0.21, signal self error 0.0169, image 0.0032, active 1 / 15 | recall@5 0.21 / 0.21 (chance), distance 8.60 to 8.22, **signal self error 0.1032 (6.1 times)**, image 0.0040, active 3 / 15 |
| same, stage 3 with no terms | | same | recall@5 0.08 / 0.00, distance 6.50, signal self error 0.329, active 0 / 6 |
| `sum` (this item) | last epoch 1 of 16 active, `best.pt` from epoch 145 (beta 0.87) | recall@5 0.33 / 0.08, signal self error 0.0014, image 0.0032, active 1 / 15 | **recall@5 0.54 / 0.46**, distance 9.48 to 6.07, signal self error 0.0011, image 0.0040 (**+23 %**), cross R2 of the image from the signal 0.21 to 0.36, of the signal from the image 0.56 to 0.92, active 7 / 14 |
| `sum`, `mean_mse` in place of `wasserstein2_diag` (the ablation of the how-to) | same | same | recall@5 0.54 / 0.50, distance (read as `wasserstein2_diag`) 9.48 to 5.12, signal self error 0.0013, image 0.0028, cross R2 of the image 0.39, of the signal 0.98, active 7 / 12 |

The shipped recipe before this item did not align this dataset and damaged the signal's self reconstruction; after it, held-out
retrieval is above chance in both directions, the signal is rebuilt from the image's code (R2 0.92), the signal's own
reconstruction improves, and the image's own reconstruction is 23 % worse than the solo image model's, which the
how-to tells the reader to look for. The validation alignment term (1.37) stays far above the training one (0.09): 192 pairs
are few for 200 epochs. All of this is one seed on 24 pairs of a synthetic dataset (finding I-15); it says that the recipe
as printed runs, chains and moves the right numbers on a dataset where the answer is known, and not how it does on a real
pair. `scripts/evaluate.py --latent-pair` on that checkpoint flags `z_image`: 14 active units on the 24 held-out samples
against the 15 the warm start counted on 192 training samples, a difference the sample size can produce, and the how-to says so.

**The baseline of the same report is not beaten on retrieval.** The Procrustes row (an orthogonal map between the two stage 3
spaces, fitted on the 192 training pairs, applied to the 24 held-out ones) reads `recall@5` 0.54 / 0.46, the same as stage 3,
and `recall@10` 0.79 / 0.75 against 0.83 / 0.71; it rebuilds the image from the signal's code as well as stage 3 does (R2
0.36 against 0.36, error 0.0265 against 0.0267) and the signal from the image's code worse (R2 0.83 against 0.92, error 0.126
against 0.059). The integrity check says `ok` (the map recovers the partner on the training pairs, `recall@10` 0.135 against 0.052
for chance and 0.091 for shuffled pairs). On this dataset, then, stage 3 matches a closed-form map for the retrieval and beats it
for the signal's cross reconstruction, which is a weaker claim than the toy of the table above supports (ADR 0056 found
Procrustes well short of stage 3 there), and the how-to states it as it is. The `mean_mse` run reads the same retrieval, a better
image self error and signal cross reconstruction and fewer active units in `z_image` (12): the two distances do not separate
on one seed and 24 pairs, which is what ADR 0056 found on the toy, and D-13 stays where the roadmap has it.

## Consequences

- **No run changes unless it asks.** Every experiment but the three of the recipe leaves the stage, the warm start, the
  loss terms and the groups at their defaults (a test lists them all), and no line of `src/` changed.
- **`pair_solo_signal` changes its loss scale.** A solo signal checkpoint trained with the old file (averaged) still loads,
  because the manifest digest covers the model and not the training; stage 3 then continues it at another scale, and the
  pair file says what to set when a solo stage used another one.
- **The test costs about 80 seconds** (three runs of 20, 20 and 100 epochs, a resumed run of 80, an evaluation) and is
  marked `slow` like the other script tests. It passes on Python 3.11 and 3.13.
- **Limit: the choice of the best epoch is not fixed.** A solo stage with a beta warm-up still takes the epoch with the
  lowest validation total, which favours early epochs. With the summed signal the best epochs are 145 and 195 of 200; a run
  whose log says `New best ... at epoch 25` of 200 holds a model that never saw the final beta, and the how-to names the line.
- **Limit: one toy, one reference dataset, one seed on the second.** None of the margins is a recommendation, D-13 (which
  distance) and D-14 (what to freeze) stay open for the reason ADR 0054 and 0055 gave, and the image self reconstruction that
  got worse on the reference dataset is the number to watch on a real pair.
