# Architecture Decision Records

Each ADR captures one architectural decision and the reasoning behind it *at
the time it was made*. A changed decision gets a **new** ADR rather than an
edit to an old one in place (see the
[project specification](../global-vae-project-specification.md), §10).

!!! note
    The one-line topics below were reconstructed from how the codebase
    itself cross-references each ADR (docstrings, README), not copied from
    the ADR files directly — double-check they still match your actual
    file titles, and correct anything that's drifted.

| ADR | Topic |
|---|---|
| [0001 — Phase 1 default configuration](0001-phase1-default-configuration.md) | Why `EN-L1-DN` (per-modality encoders, one fused latent, per-modality decoders) is the recommended Phase-1 default among the 8 valid configurations (spec §2.1). |
| [0002 — Generalizing GlobalVae to a routing graph](0002-generalize-global-vae-to-routing-graph.md) | Building `GlobalVae` from an explicit `RoutingGraph` instead of a fixed shared/private split; documents the current encoder fan-out limitation. |
| [0003 — Pluggable latent regularization](0003-pluggable-latent-regularization.md) | Moving KL-to-standard-normal out of the model class and into a registry (`AbstractLatentRegularizer`), so MMD / free-bits / learned priors can be swapped in per latent space. |
| [0004 — Pluggable beta schedules](0004-pluggable-beta-schedules.md) | Beta-weighting schedules (constant, linear warm-up, cyclical annealing) as a strategy orthogonal to which regularizer is used. |
| [0005 — Training loop](0005-training-loop.md) | Choosing a raw PyTorch loop (`Trainer`) over Lightning for now, migrating later once the architecture stabilizes. |
| [0006 — Reproducibility: seed and checkpointing](0006-reproducibility-seed-and-checkpointing.md) | Global seed management, a documented deterministic-mode flag, and checkpoint save/restore. |
| [0007 — Best-checkpoint callback](0007-best-checkpoint-callback.md) | Why `BestCheckpointCallback` (saves on metric improvement) is a separate callback from `CheckpointCallback` (saves on a schedule, for resuming). |
| [0008 — Experiment loggers](0008-experiment-loggers.md) | `AbstractExperimentLogger` as a `TrainerCallback`, so CSV/TensorBoard backends need no `Trainer` changes to support, and several can run at once. |
| [0009 — Visualization](0009-visualization.md) | The `visualization/` subpackage: latent-space projections/plots, reconstruction overlays, loss curves. |
| [0010 — Evaluation](0010-evaluation.md) | A standalone evaluation pass (`evaluate()`), distinct from training, needing only a model and a dataloader. |
| [0011 — Hydra config layer](0011-hydra-config-layer.md) | The `global_vae.config` package: dataclass-validated `ModelConfig` / `DataConfig` / `TrainingConfig` / `ExperimentConfig`. |
| [0012 — Generic data transforms](0012-generic-data-transforms.md) | `data/transforms/`: `log` / `standardize` / `resample` as dimensionality-agnostic, invertible, registry-based transforms. |
| [0013 — Coordinate-aware resampling](0013-coordinate-aware-resampling.md) | `ResampleTransform`'s `interpolation="scipy"` mode: explicit source/target positions, not just point counts. |
| [0014 — Residual 1D encoder/decoder](0014-residual-1d-encoder-decoder.md) | `OneDCnnResidualEncoder` / `OneDCnnResidualDecoder`: configurable-depth residual blocks as an alternative backbone (spec §7). |
