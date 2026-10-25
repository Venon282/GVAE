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

# Nothing else is currently deferred.
