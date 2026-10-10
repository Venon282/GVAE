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

# Deferred

The loss scale convention is still open (spec §11, "Precise loss weighting / β-VAE
annealing schedule"; roadmap P2-1). The reconstruction term is a per-element mean while the
regularizer is a per-sample sum over latent dimensions, so `beta = 1` is not the usual
ELBO and a useful `beta` depends on how many values a modality reconstructs. A
config-selectable reduction per modality and a how-to on choosing `beta` are planned;
nothing here changes silently until that decision is made.
