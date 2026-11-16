# 0056: Evaluation across two separate latent spaces, with baselines

**Status:** accepted (roadmap item P6-6; it adds an evaluation module, flags of `scripts/evaluate.py`, one argument of
`buildMetricsTable` and a baselines harness with an example, and nothing that a run without them touches)
**Date:** 2026-11-14

## Context

Stage 3 of the staged recipe trains a pair model with one latent space per modality (`z_signal`, `z_image`) and asks
the two to agree (ADR 0054). Roadmap finding I-8 says what the existing report cannot do: `paired_alignment.py`
compares the posterior means of a latent space that **both** modalities reach, so two separate spaces cannot be
compared, and it assumes the pairs (row `i` of every modality is one sample) instead of checking them.

A number that falls during stage 3 does not answer the question either. Every latent distance of ADR 0053 is minimized
by two spaces that have collapsed to a point, so a distance that goes to zero proves nothing by itself, and a
loss that trades the information of each space for agreement looks like progress. What the roadmap asks for is a
reading of the pair that has something to be read against:

- the same numbers on **mismatched** samples, so that "small" can be told from "small for everyone";
- retrieval with its **chance level**, in both directions;
- the cross reconstruction next to the self reconstruction, with the gap;
- a **collapse guard**, a space that has fewer active units than its solo stage;
- a check that the **pairs are pairs**;
- and **baselines on the same split**: the solo spaces as they are, a closed-form orthogonal map between them, stage 3
  for each distance, and the joint model with a shared latent space that the framework already had, over repeated
  random splits and seeds, never one run.

## Decision

**1. A view is a latent space together with the modalities it is read from.** `LatentView(space, inputs)`, written
`space@input+input` (`parseLatentView`, `resolveLatentView`). For two separate spaces the inputs are the modalities
whose encoders feed the space (`z_signal` is read from `signal`), and nothing has to be said. A shared space is read
from one modality at a time, which makes it a pair as well: `z_fused@signal` and `z_fused@image` are the same space
through two doors, and the report names such views by their inputs (`signal->image`). Every view is encoded from its own
inputs alone with `GlobalVae.encode(..., use_mean=True)`, the encode-only seam of ADR 0046, so no decoder runs for the
posteriors. `pairLabels` refuses the same view twice, and two views of different widths are refused with a message that
names D-16 (a distance, a retrieval and an orthogonal map are undefined between them).

**2. `evaluation/latent_alignment.py`, `evaluateLatentAlignment`.** It reads the held-out batches **once** and, when
given, the training pairs once (posteriors only). It returns a `LatentAlignmentReport` (`toDict`, `save`, `summary`,
`toMarkdown`, `saveMarkdown`) that holds, for a pair of views `a` and `b`:

- `retrieval["a->b"]` and `["b->a"]`: `computeRetrievalMetrics` of P3-6 unchanged (`recall@k`, mean and median rank,
  MRR, each with `chance_recall@k`);
- `distances`: every registered latent distance (item 3);
- `self_reconstruction`, `cross_reconstruction` and `reconstruction_gap` (item 4);
- `diagnostics` and `collapse`: the latent diagnostics of P5-2 for both views, and the guard (item 5);
- `procrustes` and `pair_integrity`: the baseline and the check (items 6 and 7);
- `flags`: one sentence for each thing that needs a look, `[]` when nothing was flagged.

Opt-in: `evaluate()`, `EvaluationResults` and `--paired` are unchanged, and every number is computed from the posteriors
and the streaming metrics of P3-6, so nothing is pooled except the posteriors (`samples x latent_dim`), which the
diagnostics need anyway.

**3. Every distance, on paired and on mismatched samples.** `computeDistanceTable` builds each registered distance with
its default arguments and calls it on consecutive chunks of `chunk_size` samples (the value is the mean over the
samples; a trailing chunk smaller than the distance's `min_batch_size` is merged into the one before). The mismatched
value is the same distance between sample `i` of one view and sample `i + 1` of the other (wrapping round), which no
alignment can make small. `DistanceResult` carries `paired`, `mismatched`, their `ratio` (below 1: partners are closer
than strangers) and `separates`: **False when the mismatched value is 0** (below `1e-9 * (1 + |paired|)`), because then a
small paired value says nothing, and that is exactly what two collapsed spaces give. A stochastic distance
(`mixture_jsd`) runs under a private generator seeded by `seed`, and the global generator is left as it was. `info_nce`
takes its negatives from the other samples of the call, so it is read at the size of a chunk. A distance that needs more
samples than there are, or cannot be built without arguments, is **skipped with the reason** in the report. An unknown
name is a `KeyError` that lists the real ones.

**4. Cross reconstruction next to self reconstruction, and the gap.** The **self** reconstruction of a view decodes its
own code through its own space (`model.decode({space: code})`) and scores the modalities the view was read from. The
**cross** reconstruction of view `b` from view `a` decodes the code of `a` passed **under the name of `b`'s space**
(`decode` of ADR 0046 takes any code of the right width, which D-16 guarantees) and scores the modalities of `b` against
the sample's own. `computeReconstructionGap` is `cross - self` per metric, against the self value of the target view, so
a gap near 0 means that the decoder of one modality reads the code of the other as well as its own. The five default
metrics are accumulated batch by batch (the streaming metrics of P3-6), and a direction whose target space is reached
only through a `concat` assembler (no decoder can run from it alone) has no entry. `buildMetricsTable` takes `gaps`: one
more row for each cell, after every other row, with the same `inputs` label and the modality written
`image (cross minus self)`. The cross rows themselves reach the table in the shape that `cross_modal` already has
(input subset of the source view -> modality -> metrics), so the table of P3-6 "gets the cross rows" without a second
format, and a cell that `--paired` already measured (a shared space) is kept as it is.

**5. The collapse guard.** `checkCollapse` flags a view whose active units (P5-2, a posterior-mean variance above
`--active-unit-threshold`) are **fewer than required**: the active units of its solo stage when they are given
(`--solo-active-units z_signal=12 z_image=9`, measured with the same threshold), else an absolute floor
`--min-active-units` (default 1). The comparison is with the solo stage on purpose: a space that has lost units since
its solo stage has traded information for agreement, whatever its distance says. The flag is a sentence in the report,
a warning in the log and a line of the console summary. In the baselines harness (item 9) the reference of stage 3 is
the active units of the solo spaces in the same split.

**6. Baseline 1, post-hoc orthogonal Procrustes.** `fitOrthogonalProcrustes(first, second)` fits the orthogonal `R`
that minimizes `||X R - Y||` on the **training** posterior means in closed form (the SVD of `X^T Y`, `R = U V^T`):
nothing to train, no hyperparameter, no initialization. `OrthogonalMap.toSecond` and `toFirst` carry means in either
direction with the one fit (a reflection is allowed), and `--procrustes-center` centers each space on its training mean
first (off by default: the roadmap's baseline is one orthogonal map). It is **applied to the held-out pairs**:
`ProcrustesResult` has its retrieval in both directions, its alignment metrics and its cross reconstruction (the code of
one view carried by the map, then decoded through the other's space), with the gap. It is the reference of the
alignment literature the roadmap cites (Maystre et al., arXiv 2510.13406; Achara et al. for more than two models,
arXiv 2602.06205): if it matches stage 3 on your pairs, the learned alignment is not buying anything.

**7. The pair-integrity check.** `checkPairIntegrity` reads the map of item 6 **on the training pairs it was fitted
on**. If row `i` of the two modalities is one sample, the map recovers the partner and the recall of the true partner is
above chance; if the join is wrong, no map does. The recall is read at the largest `k` whose chance level is at most 25%
(a recall at a `k` near the number of samples is high for any ranking), averaged over the two directions, and it has to
exceed **both** `chance + margin` (`--integrity-margin`, default 0.05) **and the largest recall that the map reaches on
`--integrity-shuffles` shuffled copies of the second view** (default 20). The second condition is the reason the check
is not just a threshold: an orthogonal map in a wide space has many free parameters and partly fits pairs that do not
match, and the check is in-sample on purpose (the map has nothing else to be tried on if the pairs are the suspect).
The result is `ok`, `suspect` (with a message that names the three likely causes: the join, a collapsed space, spaces
that no orthogonal map relates) or `not_checked` (no training pairs, or fewer than two), and `suspect` is a flag.

**8. `scripts/evaluate.py --latent-pair FIRST SECOND`** (repeatable). With it:

- the held-out pairs are `--dataloader-factory`, the training pairs are `--train-dataloader-factory` (one pass over
  each per pair, so the factory is called again, as for `--paired`);
- `--latent-distances`, `--solo-active-units`, `--min-active-units`, `--procrustes-center`, `--integrity-shuffles`,
  `--integrity-margin`, `--max-train-samples` and `--alignment-seed` set what items 3 to 7 say; `--retrieval-k`,
  `--retrieval-similarity`, `--max-samples` and `--active-unit-threshold` are the existing ones;
- the report is written next to `results.json` as `latent_alignment.json` and `latent_alignment.md` (`latent_alignment_
  <first>_<second>` for several pairs), the cross and gap rows join `metrics_table.md` and `.csv`, and the flags are
  logged as warnings;
- an invalid pair, a missing modality in the batches, an unknown view in `--solo-active-units` or a training factory
  without a pair is a `parser.error` with the message of the cause.

**9. Four baselines in one command.** `evaluation/alignment_baselines.py`, `runAlignmentBaselines(pool, recipe,
settings)`, and `examples/05_alignment_baselines.py` on a synthetic linked pair. For each of `repeats` random splits
(seed `seed + r` for the split and every model): the two solo stages are trained (stage 1 and 2), and

- **0** is the pair model warm-started from the two solo checkpoints (ADR 0052) and not trained;
- **1** is the Procrustes map fitted on the training pairs of that very model and applied to the held-out pairs;
- **2** is stage 3, once per distance: the same warm start, `latent_alignment` with that distance and
  `cross_reconstruction` (ADR 0054), each ramped from zero, trained on the training pairs;
- **3** is the joint model of `configs/model/default.yaml` (one shared latent space, `poe`) at the width of the pair
  spaces (D-16) and with modality dropout 0.3 (so each modality alone has a posterior of its own), trained from scratch
  on the training pairs for **as many epochs as the staged route spends** (solo plus pair, so that the comparison is at
  equal training), scored as `z_fused@signal` against `z_fused@image`.

Every row is scored on the **same held-out pairs** by `evaluateLatentAlignment`. The solo spaces' active units are the
reference of the collapse guard of the later rows (the joint model's, by modality). The result is a `BaselineComparison`:
for every baseline and every number, the mean, the sample standard deviation, the range and the count over the repeats
(`Spread`), `nan` left out, as a Markdown table (`comparison.md`) and as every number of every repeat
(`comparison.json`). The pair-integrity status is counted over the repeats. The reference paper uses five seeds on 72
pairs, which are the defaults. The models are built from the shipped experiment configs (`pair_solo_signal`,
`pair_solo_image`, `signal_image_pair`), with the callbacks cleared (the baselines hold no validation data: the held-out
pairs must not select anything) and the data a placeholder (spec section 6.2: the dataset is the caller's), and another
pair of modalities writes its own `BaselineRecipe`. A solo stage that collapsed is refused by the warm start (ADR 0052)
and the run stops with its message, instead of scoring a pair that cannot mean anything.

## Alternatives considered

- **Evaluate on the training pairs.** It is what a loss curve shows. A map with 120 free parameters fits 54 pairs of
  16 dimensions, and so does a decoder pair: the held-out pairs are what separates alignment from memory. The integrity
  check is the one place where the training pairs are read, because there the question is about the join.
- **A permutation test for the integrity check.** The rank of the true recall among many shuffled ones is the principled
  form. It needs hundreds of fits for a useful p-value, and the output people act on is one status. The maximum of 20
  shuffles is the cheap version of the same idea and is enough to refuse a join that a wide map can partly fit.
- **A learned distance, or a scaled Procrustes, as the baseline.** The roadmap asks for the orthogonal map (the
  reference of the literature), and a map with more freedom is a different baseline that would be easier to beat by
  accident. The centered variant is one flag, and the adapter of P6-8 is where a freer map belongs.
- **Compare with a distance only.** Every distance of ADR 0053 can be driven to 0 by collapse. The mismatched samples and
  the guard are what make it readable, and `separates` is the label that says so in the report itself.
- **Report the best seed.** The roadmap says never one run. The comparison keeps every repeat, and the table shows the
  spread next to the mean.
- **Put the baselines in `scripts/evaluate.py`.** The script scores one checkpoint. Training four kinds of model on
  repeated splits is a driver, and spec section 6.2 keeps the dataset outside the package, so it is an example that
  builds a pool and a library function that takes any pool.
- **A joint baseline with a separate latent space per modality and shared decoders.** It is a different model. The
  roadmap names the framework's existing route, a shared latent space with `poe`, and the pair `z_fused@signal`,
  `z_fused@image` is the way a shared space is compared with itself.

## Measurements

The numbers are those of the synthetic linked pair of `examples/05_alignment_baselines.py` (a sine of a random frequency
and phase, and a Gaussian blob whose place is set by that frequency and that phase), **the defaults of the command**:
72 pairs, 54 for training and 18 held out in each of five random splits and seeds, 40 epochs for each solo stage, 100
for each stage 3, 140 for the joint model, latent width 16, six distances (12 minutes on two CPU cores). Every setting is
a placeholder and the data are synthetic (finding I-15): this section says what the harness reads and what its guards
catch, **not which route is best**. With 18 held-out pairs `recall@1` moves in steps of 1/18, and the chance levels are
0.056 for `recall@1`, 0.278 for `recall@5` and 9.5 for the mean rank.

| baseline | recall@1 | recall@5 | mean rank | cross R2 | self R2 | active units | flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 solo spaces | 0.078 +/- 0.036 | 0.272 +/- 0.063 | 9.37 +/- 0.83 | -0.415 +/- 0.133 | 0.129 +/- 0.086 | 11.0 +/- 2.0 | 0/5 |
| 1 procrustes | 0.178 +/- 0.042 | 0.539 +/- 0.025 | 6.56 +/- 0.49 | -0.349 +/- 0.158 | 0.129 +/- 0.086 | - | - |
| 2 stage 3, cosine | 0.467 +/- 0.077 | 0.894 +/- 0.099 | 2.56 +/- 0.69 | 0.199 +/- 0.048 | 0.315 +/- 0.091 | 16.0 +/- 0.0 | 0/5 |
| 2 stage 3, info_nce | 0.489 +/- 0.114 | 0.950 +/- 0.046 | 2.19 +/- 0.50 | 0.233 +/- 0.070 | 0.360 +/- 0.077 | 16.0 +/- 0.0 | 0/5 |
| 2 stage 3, mean_mse | 0.461 +/- 0.191 | 0.883 +/- 0.103 | 2.59 +/- 1.16 | 0.227 +/- 0.104 | 0.244 +/- 0.306 | 16.0 +/- 0.0 | 0/5 |
| 2 stage 3, mixture_jsd | 0.361 +/- 0.065 | 0.844 +/- 0.046 | 2.97 +/- 0.33 | 0.174 +/- 0.037 | 0.329 +/- 0.088 | 16.0 +/- 0.0 | 0/5 |
| 2 stage 3, symmetric_kl | 0.489 +/- 0.125 | 0.878 +/- 0.134 | 2.57 +/- 1.11 | 0.184 +/- 0.122 | -0.188 +/- 1.025 | 16.0 +/- 0.0 | 0/5 |
| 2 stage 3, wasserstein2_diag | 0.561 +/- 0.110 | 0.956 +/- 0.058 | 2.03 +/- 0.48 | 0.273 +/- 0.084 | 0.258 +/- 0.451 | 16.0 +/- 0.0 | 0/5 |
| 3 joint, poe | 0.367 +/- 0.114 | 0.861 +/- 0.086 | 3.12 +/- 0.87 | 0.141 +/- 0.067 | 0.255 +/- 0.203 | 15.9 +/- 0.2 | 0/5 |

Mean +/- sample standard deviation over the five repeats. The pair-integrity check passed in 5 of 5. What the table shows:

- **Baseline 0 is at chance**, as the roadmap expects: `recall@5` 0.272 against 0.278, mean rank 9.37 against 9.5, and
  every distance has a paired over mismatched ratio of 0.998 to 1.012 (except `mixture_jsd`, below): two solo spaces
  that were never aligned put a partner no nearer than a stranger. The cross reconstruction is worse than predicting
  the mean (`R2` -0.415), against a self reconstruction `R2` of 0.129.
- **Baseline 1 is above chance and well short of stage 3**: `recall@5` 0.539 for the closed-form map. This is the number
  that a learned alignment has to beat, and on this toy it does by a margin larger than the spread. The map barely
  helps the cross reconstruction (`R2` -0.349 against -0.415): a rotation fitted on the means does not by itself make
  the other decoder read the code.
- **Stage 3 beats both baselines in every distance**, `recall@5` 0.84 to 0.96, with `R2` of the cross reconstruction
  at 0.17 to 0.27 (positive: the other modality's code reconstructs better than the mean) and a gap to self
  reconstruction of -0.16 to +0.37. **The distances do not separate from each other**: the standard deviations
  (0.07 to 0.19 for `recall@1`, over 18 pairs) are as large as the differences between the means, so this run does not
  say which distance to train with, and D-13 stays open.
- **The joint model is in the range of the weaker stage 3 distances** (`recall@5` 0.861) at equal training.
- **The collapse guard does not fire on any of these runs**: stage 3 ends with 16 active units in both spaces, more than
  the 11.0 of the solo stages. These runs give no reason to say why.
- **`mixture_jsd` has a ratio of 1.000 in every row**, stage 3 trained with it included. The divergence is bounded by
  `ln 2` and reaches it as soon as two posteriors do not overlap, so a ratio of exactly 1 says that the posteriors of
  these models do not overlap (this run did not measure their widths). On synthetic posteriors in 16 dimensions with
  partners 0.3 apart in the means, the paired value goes 0.41, 0.62, 0.692 and 0.693 and the mismatched value 0.665,
  0.693, 0.693 and 0.693 as the log-variance goes 0, -2, -4 and -6, while `wasserstein2_diag` keeps a ratio of about
  0.05. Stage 3 trained with it still retrieves (`recall@1` 0.361 against 0.056), so the saturation is a limit of the
  distance as a **reading** of narrow posteriors, not proof that the term does not train; the ratio of the table is
  what shows it.
- **One bad repeat shows in the spread, not in the mean**: the `self R2` of `symmetric_kl` is -0.188 +/- 1.025 because one
  of its five repeats reconstructs its own modality much worse than the mean (-1.98), and that repeat is also its worst
  one for retrieval (`recall@5` 0.67 against 0.83 to 1.00) and for the cross reconstruction (`R2` 0.00 against 0.12 to
  0.31). A mean without its spread would hide it, and five repeats are too few to say whether the distance is unstable
  or the split was unlucky.

Two runs on the same pool and the same settings show what the guards catch (five repeats each, `wasserstein2_diag`
only, no joint model):

| run | pair integrity | recall@5 of baseline 1 | recall@5 of stage 3 | active units of stage 3 | flagged |
|---|---|---:|---:|---:|---|
| the pairs as they are | ok 5/5 | 0.539 +/- 0.025 | 0.956 +/- 0.058 | 16.0 | 0/5 |
| the image rows permuted (a bad join) | **suspect 5/5** | 0.267 +/- 0.075 | 0.272 +/- 0.103 | 16.0 | 5/5, by the pairs |
| `beta = 300` in stage 3 (a collapsed run) | ok 5/5 | 0.539 +/- 0.025 | 0.278 +/- 0.034 | **0.0 +/- 0.0** (solo 11.0) | **both spaces, 5/5** |

- **A bad join is caught by the integrity check and by nothing else in the table.** With the image rows in a different
  order from the signal rows the Procrustes baseline falls to chance (0.267 against 0.278), so does stage 3 (0.272), and
  the check says `suspect` in five of five repeats. Without it the table would read as "alignment does not work".
- **A collapsed stage 3 is caught by the guard, not by its distance.** Under an enormous KL weight both spaces end with
  0 active units where the solo stages had 11.0, the guard flags both views in every repeat, retrieval is at chance
  (`recall@5` 0.278 against 0.278), and the distance ratio is 1.02: the distance between two collapsed spaces does not
  separate paired from mismatched samples, which is the point of `separates`. The solo baseline is unflagged in the same
  runs.

The tests pin the collapsed run and the refused solo stage at a small size (`tests/integration/test_alignment_baselines.py`,
and the bad join on the tiny model in `tests/integration/test_latent_alignment_script.py`). The numbers above are not
asserted: the first table is the command with its defaults, and the two runs are `PairPool`s built from the same pool
(one with the image rows permuted) or `BaselineSettings(beta=300.0)`, with `distances=("wasserstein2_diag",)` and
`include_joint=False`.

## Consequences

- **A run that does not pass `--latent-pair` is unchanged**, and `evaluate()`, `EvaluationResults`, `--paired` and every
  existing report keep their output. `buildMetricsTable` without `gaps` is what it was.
- **The report is written next to `results.json`** for a pair model, with the flags in the log and on the console.
- **The collapse guard fires on a deliberately collapsed run**, at two levels that the tests pin: a tiny pair model whose
  encoders are collapsed by hand (`tests/unit/test_latent_alignment.py`, and the script, in
  `tests/integration/test_latent_alignment_script.py`), and the real pipeline, where a stage 3 under an enormous KL weight
  ends with fewer active units than the solo stages it started from and is flagged in both spaces while the solo
  baseline is not (`tests/integration/test_alignment_baselines.py`).
- **The four baselines run in one command on the synthetic pair** (`python examples/05_alignment_baselines.py`, a quick
  form in `tests/integration/test_alignment_baselines.py`, marked `slow`).
- **The equal-width rule of D-16 is checked here too**, with the message of the cause.
- **Limit: the integrity check is in-sample**, and so only as strong as its shuffled control. A set of pairs that is
  mostly right and partly shuffled is reported as `ok` if the map still recovers more than shuffled pairs do, which is
  the right answer for the map and may not be the one a person wants; the report gives the recall and the shuffled
  maximum so that it can be read.
- **Limit: a view that no decoder can read alone** has no cross reconstruction. The report leaves the direction out.
- **Limit: two views only.** More than two models is the anchor mode and the multi-way baseline of P6-8 and P7-4. The
  Procrustes map of item 6 extends to it, and the table here does not.
- **Limit: the baselines run one model at a time on a CPU**, about eleven models per repeat for the six distances and
  the joint model, which the defaults make a long run. Fewer repeats, epochs and distances make a quick check.
- **No number here is a measurement of real data** (finding I-15). D-13 (which distance to train with) stays open for the
  same reason: this item gives the means to measure it, and the synthetic pair does not decide it.
