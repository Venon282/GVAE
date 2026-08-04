# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial project scaffold: repository structure per spec §8.
- `AbstractEncoder`, `AbstractDecoder`, `AbstractFusion`, `AbstractAssembler`
  interfaces, each with a self-registration registry (`registerEncoder`,
  `registerDecoder`, `registerFusion`, `registerAssembler`).
- `LatentSpace` and `RoutingGraph` abstractions, plus `validateRoutingGraph`
  enforcing the two structural constraints from spec §2.2 (no orphan
  latent spaces; `sum`/`average` assemblers require matching
  dimensionality).
- `single.py` and `factorized.py` convenience presets for building common
  `RoutingGraph` topologies.
- `GlobalVae` model class assembling the `EN-L1-DN` (Phase-1 recommended
  default) configuration from a config dict.
- Unit tests for the registry pattern and for routing-graph validation.
- First of the 8 end-to-end integration tests called for in spec §10
  (`EN-L1-DN`, with dummy encoders/decoders/fusion).
- ADR 0001 documenting the choice of `EN-L1-DN` as the first
  configuration implemented.
- `pyproject.toml` with `ruff`, `mypy` (strict), and `pytest` configured,
  including the naming-convention override (`N802`/`N803`/`N806`
  disabled) required by spec §10.
- `tests/integration/test_en_l1_dn_default.py` rebuilt against the
  `GlobalVae.createSingleLatent(...)` constructor introduced in ADR
  0002 (the previous version targeted the old `EN-L1-DN`-only
  constructor from ADR 0001 and was flagged as stale by ADR 0002's own
  "Consequences" section). Covers, with dummy per-modality
  encoders/decoders and a dummy PoE fusion: forward-pass shapes for
  every reconstruction and for the fused latent, missing-modality
  robustness (spec §5), KL-loss finiteness, gradient flow into every
  encoder and decoder, and the registry `KeyError` path. Passes `ruff
  check`, `ruff format --check`, and `mypy --strict`.
- `losses/regularizers/` subpackage: `AbstractLatentRegularizer` interface,
  its self-registration registry (`registerRegularizer` /
  `getRegularizerClass`), and the default `kl_standard_normal` strategy,
  extracted from `LatentSpace.klDivergence` so latent regularization is
  pluggable per latent space (spec §2.3), matching the same pattern
  already used for Fusion and Assemblers.
- ADR 0003 documenting the pluggable-regularization retrofit described
  below.
- `tests/integration/test_regularizers.py` (registry pattern and
  `kl_standard_normal` value correctness) and
  `tests/integration/test_regularization_loss.py` (cross-latent-space
  aggregation, beta weighting).
- `GlobalVae.createSingleLatent` gained a `regularizer_strategy`
  parameter (default `"kl_standard_normal"`), and
  `tests/integration/test_en_l1_dn_default.py` gained tests proving the
  default is wired in and that a non-default strategy is actually used,
  not just accepted.
- `training/beta_schedules/` subpackage: `AbstractBetaSchedule` interface
  (a plain `ABC`, not an `nn.Module`: resolving a training step to a
  scalar weight is not an autograd computation), its self-registration
  registry (`registerBetaSchedule` / `getBetaScheduleClass` /
  `listRegisteredBetaSchedules`), and two default strategies,
  `ConstantBetaSchedule` (`"constant"`, the explicit no-annealing case)
  and `LinearWarmupBetaSchedule` (`"linear_warmup"`, ramps linearly
  from `start_value` to `end_value` over `warmup_steps` then holds
  `end_value`), making the beta-weighting schedule spec §2.3 requires
  expressible as a per-latent-space, pluggable strategy instead of a
  hardcoded branch in a future trainer.
- `training/beta_schedule_resolution.py`: `resolveBetaSchedules(schedules,
  step)`, resolving a per-latent-space set of schedules into the plain
  `dict[str, float]` that `losses.regularization
  .computeTotalRegularizationLoss` and `GlobalVae.computeRegularizationLoss`
  already accept as `beta`, with zero changes to either. See ADR 0004.
- ADR 0004 documenting the pluggable beta-schedule addition and why it
  required no changes to `GlobalVae` or `losses/regularization.py`.
- `tests/integration/test_beta_schedules.py`: registry pattern,
  `ConstantBetaSchedule` / `LinearWarmupBetaSchedule` value
  correctness (including negative-step clamping, the exact
  warm-up-boundary case, and the `warmup_steps <= 0` error path), and
  a round-trip test proving `resolveBetaSchedules`'s output feeds
  `computeTotalRegularizationLoss` unchanged.
- `losses/regularizers/free_bits_kl.py`: `FreeBitsKlRegularizer`
  (spec §2.3, §11: named alongside MMD as a candidate strategy). Gives
  each latent dimension a fixed KL budget (`free_bits` nats, default
  `0.5`) it is never penalized for using, which is the standard fix
  for the failure mode where plain KL-to-standard-normal keeps
  pushing an already-collapsed dimension toward the prior with no
  counterbalancing pressure. Supports both the standard per-dimension
  formulation (default) and a coarser aggregate variant
  (`per_dimension=False`) for parity/ablation purposes. At
  `free_bits=0`, the per-dimension variant is mathematically identical
  to `kl_standard_normal` (verified by test).
- `losses/regularizers/mmd.py`: `MmdRegularizer` (spec §2.3, §11: the
  other named candidate strategy), a WAE-MMD/InfoVAE-style batch-level
  kernel two-sample test between reparameterized posterior samples and
  prior samples, computed with a configurable multi-scale kernel
  (`"rbf"`, the common default, or `"imq"`, the original WAE-MMD
  paper's heavier-tailed choice). Unlike KL, MMD only constrains the
  *aggregate* posterior to look like the prior, not every individual
  sample, which is the property this strategy is meant to make
  available where KL's sample-by-sample pressure is suspected of
  contributing to posterior collapse. Its `forward` docstring documents
  the one place this strategy deviates from a literal per-sample
  reading of the shared regularizer interface: MMD returns the same
  batch-level scalar broadcast across the `(batch,)` output, since MMD
  itself has no meaningful per-sample decomposition.
- `training/beta_schedules/cyclical_annealing.py`:
  `CyclicalAnnealingBetaSchedule` (spec §2.3, explicitly named
  alongside "linear warm-up" as one of the three common beta-weighting
  patterns; previously the only implemented pattern was the single
  warm-up case). Repeats a ramp-then-hold pattern every `period`
  steps instead of running it once, optionally holding `end_value`
  indefinitely after `num_cycles` cycles.
- Tests for all three additions above, mirroring the existing registry
  test style (registration, value correctness, edge cases): six new
  test classes across `tests/integration/test_regularizers.py` and
  `tests/integration/test_beta_schedules.py`.
- `tests/integration/test_en_l1_dn_default.py` gained
  `test_single_modality_needs_no_fusion_strategy` and
  `test_missing_fusion_strategy_with_several_modalities_still_raises`,
  covering the `GlobalVae.createSingleLatent` change described below
  under "Changed".
- `training/callbacks.py`: `TrainerCallback`, the hook interface
  `Trainer` calls into at `onTrainBegin`/`onEpochBegin`/`onStepEnd`/
  `onEpochEnd`/`onTrainEnd`. Every hook is a no-op by default. A plain
  composable `list[TrainerCallback]`, not a registry-selected strategy
  (see ADR 0005 for why this extension point is shaped differently
  from encoders/decoders/fusion/assemblers/regularizers/beta
  schedules).
- `training/trainer.py`: `Trainer`, the raw PyTorch training loop spec
  §10 calls for (§6.1 milestone 1: a single-modality signal VAE
  trained end to end). Forward pass, reconstruction loss
  (`computeTotalReconstructionLoss`) plus regularization loss
  (`GlobalVae.computeRegularizationLoss`, weighted by beta, including
  per-latent-space schedules via `resolveBetaSchedules`), backward,
  optimizer step (Adam by default, any `torch.optim.Optimizer` class
  or instance), device placement (explicit or auto-detected
  CPU/GPU), optional gradient-norm clipping, optional modality dropout
  (spec §5), console progress via the standard `logging` module (never
  `print`), and per-step/per-epoch metrics dispatched to every
  `TrainerCallback`. `Trainer.fit` is resumable across multiple calls
  on the same instance (`global_step`/`start_epoch`/`history` persist),
  and always fires `onTrainEnd` even when training exits early via an
  exception. See ADR 0005 for the full set of design decisions.
- `docs/adr/0005-training-loop.md` documenting the above.
- `tests/integration/test_trainer.py`: loss-decreases-over-epochs
  sanity check on a fixed toy dataset, optimizer configurability
  (instance and class), device placement, beta-schedule resolution and
  override precedence, callback firing counts and metric keys
  (including `onTrainEnd` still firing when a callback raises),
  modality dropout (default no-op, always keeps at least one modality,
  no-op for a single-modality model, still trains correctly when
  enabled on a two-modality model), gradient clipping, multi-epoch
  history and resumed epoch/step numbering, validation-metrics
  merging, and the relevant error paths (empty batch/dataloader,
  invalid constructor arguments, non-positive `num_epochs`).
- `utils/seed.py`: `setGlobalSeed(seed, deterministic=False, warn_only=False)`
  (spec §10's "global seed management, deterministic-mode flag
  documented"). Seeds Python's `random`, NumPy (if installed), and
  PyTorch's CPU/CUDA generators; `deterministic=True` enables
  `torch.use_deterministic_algorithms`, disables cuDNN auto-tuning,
  and best-effort sets `CUBLAS_WORKSPACE_CONFIG` (via `setdefault`,
  never overwriting an existing value); `deterministic=False` (default)
  explicitly resets those same flags, so the mode never silently leaks
  across calls in the same process.
- `training/checkpoint.py`: `saveCheckpoint`/`loadCheckpoint`
  (model + optional optimizer state + step/epoch/history + an
  arbitrary `config` snapshot, spec §10's "config snapshotted with
  every run", + RNG state for exact resumability) and
  `CheckpointCallback`, a concrete `TrainerCallback` that saves
  periodically (`every_n_epochs`) with optional `keep_last_n` pruning
  of older checkpoints. `Trainer.saveCheckpoint`/`Trainer.loadCheckpoint`
  are thin convenience wrappers using the trainer's own model,
  optimizer, step, epoch, and history. See
  `docs/adr/0006-reproducibility-seed-and-checkpointing.md`.
- `tests/integration/test_seed.py` and `tests/integration/test_checkpoint.py`
  covering the above; see ADR 0006 for the full list.
- `docs/adr/0006-reproducibility-seed-and-checkpointing.md` documenting
  both additions above and the `Trainer.fit` bug fix described below
  under "Fixed".

### Changed

- `GlobalVae.computeRegularizationLoss` gained a
  `beta: dict[str, float] | float = 1.0` parameter, forwarded
  unchanged to `computeTotalRegularizationLoss`. This method did not
  expose `beta` at all before this change, even though ADR 0004 already
  documented a trainer calling it as
  `model.computeRegularizationLoss(latent_params, beta=...)`; that call
  shape was not actually possible until now. See ADR 0005.
- `tests/integration/test_en_l1_dn_default.py` gained
  `test_beta_scales_the_regularization_loss`, covering the fix above.
- `training/NOTE.md` rewritten to reflect that `Trainer` now exists,
  and to track what is still deferred (checkpointing, experiment
  loggers, global seed management) instead of "trainer.py itself is
  not yet implemented".

- `GlobalVae.createSingleLatent`'s `fusion_strategy` parameter is now
  `str | None = None` (previously a required `str`), and `latent_dim`
  moved ahead of it in the parameter list. A latent space fed by
  exactly one encoder never calls Fusion at all (spec §4), so the
  single-modality `signal -> z -> signal` case (spec §6.1 milestone 1)
  no longer has to pass a fusion strategy name that is never used.
  Passing `None` with more than one modality still raises `ValueError`
  from `__init__`, unchanged (delegates to the existing
  `fusion_strategies` validation instead of duplicating it).

- Renamed `latent/factorized.py` to `latent/shared_private.py` and
  `buildFactorizedRoutingGraph` to `buildSharedPrivateRoutingGraph`,
  since "several latent spaces" is a general routing graph, not a
  hardcoded factorization scheme. And deplace them to `routing_graph_builders` directory
- Split `latent/assembler.py` (base class, registry, and all three
  concrete assemblers in one file) into its own `assemblers/`
  subpackage, one class per file, matching the rest of the codebase's
  modularity rule.
- `GlobalVae` is now built from an explicit `RoutingGraph` instead of
  assuming a single fused latent space fed by every encoder.
- `GlobalVae.__init__` takes separate `encoder_configs` and
  `decoder_configs` instead of one `modality_configs` dict, so a single
  shared decoder can be registered under its own name (the `*-D1` rows
  of spec §2.1) instead of reusing a modality name that may not exist.
- Fusion strategy is now selected per latent space (only required for
  latent spaces fed by more than one encoder), not once globally for
  the whole model.
- `GlobalVae.__init__` now accepts `regularizer_strategies` and
  `regularizer_kwargs`, and builds `self.regularizers` (an
  `nn.ModuleDict`, one entry per latent space, symmetric to
  `self.fusions`), instead of the model class going straight to
  `LatentSpace.klDivergence` for every latent space unconditionally.
- `losses/kl.py` is superseded by `losses/regularization.py`:
  `computeTotalRegularizationLoss` aggregates via each latent space's
  own regularizer module instead of calling `LatentSpace.klDivergence`
  directly.
- `GlobalVae.computeKlLoss` renamed to `computeRegularizationLoss`: it
  was never only about KL divergence in spirit (spec §2.3 always
  described the regularization term as pluggable), and now it isn't in
  code either.
- `LatentSpace` (`latent/base.py`) no longer exposes `klDivergence`;
  its module docstring no longer refers to "its own KL term" but "its
  own regularization term, not necessarily KL divergence".
- `models/global_vae.py` no longer imports the unused `LatentSpace`
  name from `latent.base` (a pre-existing dead import, `ruff` F401,
  noticed while editing this file's imports for the change above).
- `training/NOTE.md` updated: the raw-loop-vs-Lightning open question
  it cited is resolved (raw PyTorch loop for now, per spec §10);
  `trainer.py` itself remains a separate, not-yet-built milestone, now
  decoupled from the beta-schedule work described above.

### Fixed
- `Trainer.fit` updated `self.start_epoch` *after* invoking every
  `onEpochEnd` callback for that epoch, so a callback that saves
  `trainer.start_epoch` (such as `CheckpointCallback`) would persist a
  resume point one epoch behind where training should actually
  resume. Fixed by updating `self.start_epoch` before the `onEpochEnd`
  callback loop. Caught by
  `tests/integration/test_checkpoint.py::TestCheckpointCallback::test_last_checkpoint_can_be_loaded_and_matches_final_state`;
  see `docs/adr/0006-reproducibility-seed-and-checkpointing.md`.
- `validateRoutingGraph` now rejects a decoder that consumes more than
  one latent space but has no assembler assigned, instead of silently
  skipping the dimensionality check for it.
- Fixed a typo in the decoder registry's error message ("Unknow" to
  "Unknown").
- `GlobalVae.__init__` no longer crashes on construction: the fusion
  strategy name was being called as a function before being resolved
  to a class.
- `GlobalVae.__init__` now explicitly rejects, with a clear
  `NotImplementedError`, an encoder assigned to more than one latent
  space, instead of silently reusing its output at the wrong
  dimension for the second latent space. Found by running the
  shared-plus-private preset end to end; see ADR 0002.

### Removed
- `assemblers/assembler.py`: a leftover duplicate of `AbstractAssembler`
  and the assembler registry, left behind at the new subpackage path
  after the split described above under "Changed".
- `losses/kl.py`: superseded by `losses/regularization.py` (see
  "Changed" above). Should be deleted from the repository.
- `LatentSpace.klDivergence`: superseded by
  `losses.regularizers.kl_standard_normal.KlStandardNormalRegularizer`,
  which is now the only place this exact computation lives.
