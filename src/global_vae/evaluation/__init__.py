"""Evaluation subpackage of global_vae: a standalone evaluation pass, distinct from training.

See spec §10's "MSE at minimum... KL value... reconstructions for visual inspection".

Nothing here needs a `Trainer`: `evaluate()` only needs a `GlobalVae` and a
dataloader (the same `dict[str, torch.Tensor]` batch convention used everywhere else
in this framework), matching `training/checkpoint.py`'s own "loading for evaluation
does not need an optimizer" design. This is what makes re-running evaluation without
retraining (the concern that originally motivated `saveCheckpoint`/`loadCheckpoint`,
`docs/adr/0006-reproducibility-seed-and-checkpointing.md`) actually possible: build a
model, `loadCheckpoint(path, model=model)`, then `evaluate(model, test_dataloader)`.

For paired data, `paired_alignment` measures retrieval and alignment between the modalities'
posterior means, `metrics_table` lays the metrics out as a per-modality table, and
`streaming_metrics` lets `evaluate(streaming=True)` skip the pooled dataset (roadmap P3-6).

`metrics` holds the five default reconstruction metrics and the name -> function table of metrics
with a declared default set (roadmap P5-3, ADR 0048); `modality_metrics` registers the scores of
point clouds, graphs and text in it, and is imported here so that they are there whenever the
package is.
"""

from global_vae.evaluation import modality_metrics  # noqa: F401
