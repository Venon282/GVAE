# Status

Both halves of the objective are implemented here.

- `reconstruction/`: a package (roadmap P5-3, `docs/adr/0048-reconstruction-loss-and-metric-registries.md`).
  `aggregate.py` holds `computeTotalReconstructionLoss`, the per-modality reconstruction
  term. It takes one loss function for every modality or a per-modality dict (default
  `torch.nn.functional.mse_loss`), per-modality weights, and raises `ValueError` when a
  reconstruction and its target differ in shape instead of letting `mse_loss` broadcast, unless the
  loss opts out. `base.py` has `AbstractReconstructionLoss` (an `nn.Module`, so a loss can own
  parameters), `registry.py` the registry (`registerReconstructionLoss`, `getReconstructionLossClass`,
  `listRegisteredReconstructionLosses`), and one file per loss: `mse`, `l1`, `bce`, `smooth_l1` (the
  numbers of 2.0.0), `bce_with_logits`, `masked_mse`, `token_cross_entropy`, `chamfer`,
  `tabular_mixed`. A loss returns a mean over its terms and declares, as class attributes,
  `requires_matching_shapes`, `reads_nan_as_unobserved`, `default_metrics`, `decoder_arguments`, and
  `countTerms`, the number that `sum` multiplies by. `Trainer.reconstruction_losses` holds the
  modules, so their parameters are in the optimizer and saved under `reconstruction_loss_state_dict`.
- `regularization.py` and `regularizers/`: `computeTotalRegularizationLoss` and the
  pluggable per-latent-space penalty (`AbstractLatentRegularizer`, registry in
  `regularizers/registry.py`). Registered strategies: `kl_standard_normal` (the default),
  `free_bits_kl` and `mmd`. `GlobalVae.computeRegularizationLoss` is the model-side entry
  point and delegates here. See `docs/adr/0003-pluggable-latent-regularization.md`.

- `terms/`: the extra terms of the training loss (`AbstractLossTerm`, registry in
  `terms/registry.py`, roadmap P5-6, `docs/adr/0047-extra-loss-terms.md`). The registry ships
  empty: the terms are the objectives of Phases 6 and 7. `Trainer.computeLosses` weights each
  one, adds it to the total and logs it as `loss/<name>`.

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
Every registered loss is a mean over its terms, and `sum` multiplies it by `countTerms`.
See `docs/adr/0029-loss-scale-convention.md` and `docs/how-to/choose-beta.md`.

The loss function and the weight of a modality are chosen from config the same way
(`TrainingConfig.reconstruction_loss`, `reconstruction_weight`: one value or a reconstruction ->
value mapping, validated against the model's decoders, `docs/adr/0030-per-modality-reconstruction-loss-and-weight.md`).
The loss is a registered name, an entry `{name, kwargs}` or a mapping of either (ADR 0048).
A modality's term is its loss function, then its reduction, then its weight.

# Deferred

The values of `beta`, of the warm-up and of the other schedule hyperparameters (roadmap
D-6) are empirical and wait for the first real run (P3-1). A learned decoder variance is
not part of this.
