# Status

Both halves of the objective are implemented here.

- `reconstruction.py`: `computeTotalReconstructionLoss`, the per-modality reconstruction
  term. It takes one loss function for every modality or a per-modality dict (default
  `torch.nn.functional.mse_loss`), per-modality weights, and raises `ValueError` when a
  reconstruction and its target differ in shape instead of letting `mse_loss` broadcast.
- `regularization.py` and `regularizers/`: `computeTotalRegularizationLoss` and the
  pluggable per-latent-space penalty (`AbstractLatentRegularizer`, registry in
  `regularizers/registry.py`). Registered strategies: `kl_standard_normal` (the default),
  `free_bits_kl` and `mmd`. `GlobalVae.computeRegularizationLoss` is the model-side entry
  point and delegates here. See `docs/adr/0003-pluggable-latent-regularization.md`.

Beta weighting lives outside this package, in `training/beta_schedules/` (`constant`,
`linear_warmup`, `cyclical_annealing`; see `docs/adr/0004-pluggable-beta-schedules.md`).
`Trainer` resolves the schedules each step and passes the result to
`computeRegularizationLoss(..., beta=...)`, so nothing in `losses/` knows a schedule exists.

# Scale convention

The reconstruction term is a per-element mean by default while the regularizer is a
per-sample sum over latent dimensions, so under the default `beta = 1` weights the KL
`N / 2` times more than the usual ELBO does (`N`: values per sample of the modality).
`computeTotalReconstructionLoss(..., reduction=...)` takes `"mean"` (the default, nothing
changes) or `"sum"` (the per-sample sum, averaged over the batch), as one name or a
modality -> name mapping; `TrainingConfig.reconstruction_reduction` selects it from config.
See `docs/adr/0029-loss-scale-convention.md` and `docs/how-to/choose-beta.md`.

# Deferred

The values of `beta`, of the warm-up and of the other schedule hyperparameters (roadmap
D-6) are empirical and wait for the first real run (P3-1). A learned decoder variance, and
per-modality `reconstruction_loss` and `reconstruction_weight` in config (roadmap P2-2),
are not part of this.
