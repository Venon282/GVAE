# Status

`evaluation/` and `scripts/evaluate.py` are implemented, covering the "script/mode
d'éval distinct de l'entraînement" requirement: reconstruction metrics on a test set
(`mse`/`rmse`/`mae`/`r2`/`pearson_r` by default), a regularization/KL value per latent
space (always including a `kl_standard_normal` number directly comparable across
runs, regardless of which regularizer strategy actually trained the model), and
optional reconstruction/latent-space figure export. See
`docs/adr/0010-evaluation.md`.

Neither `evaluate()` nor `exportEvaluationFigures()` needs a `Trainer`: both take
only a `GlobalVae` and a dataloader, so the intended flow is `build a model`,
`training.checkpoint.loadCheckpoint(path, model=model)`, then `evaluate(model,
test_dataloader)`, entirely independent of the training run that produced the
checkpoint.

# Paired data and very large test sets (roadmap P3-6)

Three additions, none of which changes `GlobalVae` or the routing (`docs/adr/0038-paired-data-evaluation.md`):

- `paired_alignment.py`: cross-modal retrieval (`recall@k`, mean and median rank, MRR, each with
  the value of a random ranking) and alignment between the **modality-specific posterior means**,
  the `mu` a latent space gets when the model is queried with one modality alone. Opt-in, like
  the cross-modal reconstruction reports of `cross_modal.py`: `evaluate()` and
  `EvaluationResults` are unchanged, and `scripts/evaluate.py --paired` runs it. Ties between
  the partner and other candidates count as a random ordering would, so a collapsed latent space
  scores chance and not a perfect retrieval.
- `metrics_table.py`: `MetricsTable`, one row per modality (and per input subset, for the
  cross-modal numbers) and one column per metric, as Markdown, CSV or JSON.
- `streaming_metrics.py`: the five reconstruction metrics accumulated batch by batch.
  `evaluate(..., streaming=True)` uses them and never pools the originals and the reconstructions.
  An earlier version of this note said that `r2` and `pearson_r`, unlike `mse`, `rmse` and `mae`,
  need the whole pooled tensor. They do not: both are functions of a few running sums (the
  moments of the target, the sum of squared residuals, and the co-moment), merged by the
  parallel algorithm of Chan et al., so all five stream.

Still pooled, on purpose: the latent posteriors `(mu, logvar)` (`num_samples x latent_dim` per
latent space, which the latent diagnostics and the KL values need), the reconstructions that
`scripts/evaluate.py --paired` collects for the cross-modal table (bounded by `--max-samples`),
and the data of the figures.

# Metric registry and per-modality sets (roadmap P5-3)

`metrics.py` holds a name -> function table (`registerMetric`, `getMetric`, `listRegisteredMetrics`) with a declared
default set, `DEFAULT_METRIC_NAMES` (`mse`, `rmse`, `mae`, `r2`, `pearson_r`): metrics are not exclusive strategies, so it is
a table and not a class registry (`docs/adr/0048-reconstruction-loss-and-metric-registries.md`). It also registers the
`masked_*` forms (a metric on the values whose target is not `NaN`, made by `maskUnobserved`), and `modality_metrics.py`
registers `chamfer`, `precision`, `recall`, `f_score` (clouds), `edge_auc` (graphs) and `token_accuracy` (text) on the scores
that live in `utils/`. `metric_sets.py` resolves what `evaluate(reconstruction_metrics=...)` is given (`None`, a dict of
functions, a list of names, or a mapping by reconstruction name) into one set per modality; with nothing given, the set follows
the loss (`default_metrics`, else the masked five under their plain names for a loss that reads `NaN` as unobserved, else the
five). `streaming_metrics.py` streams the ten array metrics and refuses the others by name.

Not done on purpose: the cross-modal and paired reports keep the five default metrics, `edge_auc` has no node counts to leave
padding out (call `computeEdgeScores` with them), and a metric of a single modality is not told apart by its data.

# Across two separate latent spaces (roadmap P6-6)

`latent_alignment.py` evaluates a pair of latent **views** (a space and the modalities it is read from: `z_signal`, or
`z_fused@signal` for a shared space read from one modality) on held-out pairs, where `paired_alignment.py` needs one
space that both modalities reach (`docs/adr/0056-evaluation-across-two-latent-spaces.md`). `evaluateLatentAlignment`
reads the held-out batches once, and the training pairs once (posteriors only, through `GlobalVae.encode`), and gives:
retrieval in both directions with the chance level (`computeRetrievalMetrics`), every registered latent distance on
paired and on mismatched samples (`computeDistanceTable`: a distance that is 0 for strangers too is marked
`separates=False`), cross reconstruction (`GlobalVae.decode` of the code of one view under the name of the other's
space) next to self reconstruction and their gap, the latent diagnostics of both views with the collapse guard
(`checkCollapse`: fewer active units than the solo stage, else a floor), the post-hoc orthogonal Procrustes baseline
(`fitOrthogonalProcrustes`, closed form, fitted on the training pairs and applied to the held-out ones) and the
pair-integrity check (`checkPairIntegrity`: the Procrustes recall on the training pairs must beat chance plus a margin
and the best of a shuffled control, or the pairs are suspect, roadmap I-8). Opt-in like the other reports:
`scripts/evaluate.py --latent-pair` runs it, writes `latent_alignment.json` and `.md` next to `results.json`, and adds
the cross rows and their gap to the metrics table (`buildMetricsTable(..., gaps=...)`). `evaluate()` is unchanged.

`alignment_baselines.py` runs the four baselines of the roadmap on one random split per repeat (`runAlignmentBaselines`:
the solo spaces, Procrustes, stage 3 once per distance, joint training with a shared latent) and reports a mean and a
spread over the repeats (`BaselineComparison`); `examples/05_alignment_baselines.py` is its command on a synthetic
linked pair.

Not done on purpose: the two spaces must have equal widths (roadmap D-16, a distance, a retrieval and an orthogonal map
are undefined otherwise); a direction whose target space is read only through a `concat` assembler has no cross
reconstruction; the integrity check is in-sample and so only as strong as its shuffled control; no number here is a
measurement of real data (roadmap I-15).

# Nothing else is currently deferred.
