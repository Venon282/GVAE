"""evaluation subpackage of global_vae: a standalone evaluation pass, distinct from
training (spec §10's "MSE at minimum... KL value... reconstructions for visual
inspection").

Nothing here needs a `Trainer`: `evaluate()` only needs a `GlobalVae` and a
dataloader (the same `dict[str, torch.Tensor]` batch convention used everywhere else
in this framework), matching `training/checkpoint.py`'s own "loading for evaluation
does not need an optimizer" design. This is what makes re-running evaluation without
retraining (the concern that originally motivated `saveCheckpoint`/`loadCheckpoint`,
`docs/adr/0006-reproducibility-seed-and-checkpointing.md`) actually possible: build a
model, `loadCheckpoint(path, model=model)`, then `evaluate(model, test_dataloader)`.
"""
