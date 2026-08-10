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

# Nothing currently deferred.

`evaluate`'s pooled-dataset accumulation (module docstring of `evaluate.py`) trades
memory for correctness on very large test sets; `max_samples` bounds it today. A
streaming/chunked variant of the metrics that do not need the whole pooled tensor at
once (`mse`/`rmse`/`mae`, unlike `r2`/`pearson_r`) would be a natural future
optimization if evaluation memory ever becomes a real constraint, not built now since
it is not a problem yet.
