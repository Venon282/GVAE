# Status

`trainer.py` (`Trainer`) is now implemented: a raw PyTorch loop (spec
§10's resolved raw-loop-vs-Lightning question), covering forward pass,
reconstruction + regularization loss (weighted by beta, including
per-latent-space schedules via `beta_schedule_resolution.py`),
optimizer step, device placement, modality dropout (spec §5), and a
`TrainerCallback` hook seam (`callbacks.py`) for per-step/per-epoch
metrics, called once per optimizer step and once per epoch.

`beta_schedules/` predates `trainer.py` and does not depend on it: see
`beta_schedules/base.py`, `beta_schedule_resolution.py`, and
`docs/adr/0004-pluggable-beta-schedules.md`. `Trainer` calls
`resolveBetaSchedules(...)` internally each step and passes the result
straight into `GlobalVae.computeRegularizationLoss(..., beta=...)`, per
that ADR's own stated plan; `GlobalVae.computeRegularizationLoss` was
retrofitted with a `beta` parameter to make that call actually possible
(it did not expose one yet when ADR 0004 was written).

# Deferred

- **Checkpointing** (spec §10, "Reproducibility"): saving/restoring
  model + optimizer + step/epoch state is not built yet. `Trainer`'s
  `global_step`/`start_epoch`/`history` state is deliberately kept
  simple and instance-level so a future checkpoint feature has
  something clean to serialize, without needing to touch `Trainer`'s
  loop logic itself.
- **Experiment tracking** (spec §10: "Weights & Biases or MLflow,
  logging losses, latent-space visualizations, and reconstructions per
  run"): `TrainerCallback` is the seam this plugs into (a logger
  implemented as, or wrapped by, a callback), but no concrete logger
  (TensorBoard, CSV, W&B, MLflow) is implemented yet.
- **Global seed management / deterministic-mode flag** (spec §10,
  "Reproducibility"): not yet a dedicated utility; needs to run before
  model construction, so it is out of `Trainer`'s own scope regardless
  of when it is added.
