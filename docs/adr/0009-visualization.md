# 0009 — Visualization subpackage (`visualization/`)

**Status:** accepted
**Date:** 2026-08-05

## Context

Spec §10's "Experiment tracking" bullet names "latent-space
visualizations, and reconstructions per run" alongside loss logging;
spec §6.1 milestone 1 (a working single-modality signal VAE) explicitly
requires "the ability to inspect training curves and visualize the
latent space". Nothing existed for either yet. `training/loggers/`
(`docs/adr/0008-experiment-loggers.md`) already built `logImage`/
`logFigure` on `AbstractExperimentLogger`, anticipating that something
would eventually produce the images/figures to pass to them; this is
that something.

## Decision

Four new modules, each returning a plain `matplotlib.figure.Figure`
and never displaying, saving, or logging it themselves (that decision,
and which experiment logger if any receives it, is left to the
caller, keeping this subpackage decoupled from both file I/O and any
particular tracking backend):

- `latent_plot.py`: `projectLatentSamples` (`"auto"`, `"pca"`,
  `"tsne"`, `"umap"`, or `"none"`, matching spec's own framing
  "directe si latent_dim=2, sinon PCA/t-SNE/UMAP") and `plotLatentSpace`
  (1D/2D/3D scatter, optionally colored by a continuous or categorical
  label). `"pca"` uses `torch.pca_lowrank`, no extra dependency;
  `"tsne"`/`"umap"` are soft dependencies (scikit-learn/umap-learn),
  imported lazily with a clear `ImportError` if missing, mirroring
  `TensorBoardLogger`'s own pattern for the `tensorboard` package
  (`docs/adr/0008-experiment-loggers.md`). `collectLatentParams`/
  `collectLatentSamples` run a `GlobalVae` over a dataloader to gather
  a latent space's `(mu, logvar)` or realized values, so a caller does
  not have to hand-write that batch loop before plotting.
  `plotPerDimensionKl` is a bar chart of average KL per latent
  dimension: not explicitly requested, but added because it makes
  posterior collapse (the exact failure `free_bits_kl`, `mmd`, and
  `cyclical_annealing` all exist to mitigate, spec §11) directly
  visible instead of only inferable from a scalar training curve, and
  costs nothing beyond the encoder's already-available `(mu, logvar)`
  output regardless of which regularization strategy is actually
  training the model.
- `reconstruction_plot.py`: `plotReconstruction`/`plotReconstructionGrid`,
  a line overlay (not an image comparison: the only concrete decoder
  built so far, `OneDCnnDecoder`, reconstructs a 1D series, spec §6
  Phase 1). `inverse_transform: Callable[[Tensor], Tensor] | None` is
  how this respects the framework/data boundary established earlier
  in this project (data pipeline responsibilities are explicitly out
  of scope): spec §6 keeps preprocessing (e.g. log-scale SAXS
  intensity) in the caller's own pipeline, so this module has no
  built-in transforms of its own to invert, only ever accepting the
  caller's own inverse function as a plain callable.
  `collectReconstructions` mirrors `collectLatentParams`'s role for
  this module.
- `loss_curves.py`: `plotLossCurves` (from `Trainer.history` directly,
  matching spec §6.1's own phrasing "inspect training curves"; default
  metric selection picks up every `"train/loss/"`/`"val/loss/"` key
  `Trainer` itself produces, so the common case needs no argument),
  `plotStepCurves` (step-level, keyed by an explicit `"step"` field
  rather than list position, since a resumed `Trainer`'s `global_step`
  is not simply the list index, `docs/adr/0005-training-loop.md`), and
  `plotBetaSchedule` (plots `beta(step)` for one or several schedules
  independent of any actual training run, for sanity-checking a chosen
  schedule, e.g. `cyclical_annealing`'s period and ramp shape, before
  committing to a long run).
- `history_callback.py`: `HistoryCallback`, a `TrainerCallback`
  (exactly like every `AbstractExperimentLogger`) that accumulates
  step- and epoch-level metrics into plain in-memory lists. `Trainer`
  already retains epoch-level history on its own; this covers the
  step-level case (`Trainer` does not retain per-step history itself)
  without requiring a file-based logger first, producing exactly the
  shape `plotStepCurves` expects.

Dependency handling: `pyproject.toml` gained a `visualization` extra
(`matplotlib`, unconditionally needed: there is no meaningful
"plotting without a plotting library" fallback, unlike every other
soft dependency in this codebase so far) plus separate `tsne`
(`scikit-learn`) and `umap` (`umap-learn`) extras for
`projectLatentSamples`'s optional methods specifically. All three are
listed under `dev` too, so this project's own test suite exercises
every projection method for real, matching this codebase's existing
testing style (`TensorBoardLogger`'s tests against the real
`tensorboard` package, `docs/adr/0008-experiment-loggers.md`).

## Consequences

- Spec §6.1 milestone 1's visualization requirement and spec §10's
  "latent-space visualizations, and reconstructions" bullet are both
  addressed. Combined with `training/loggers/`
  (`docs/adr/0008-experiment-loggers.md`), a full loop is now possible:
  `fig = plotLossCurves(trainer.history)` (or `plotLatentSpace`,
  `plotReconstruction`, ...) then `tb_logger.logFigure("name", fig,
  epoch)`.
- `tests/integration/test_latent_plot.py`,
  `tests/integration/test_reconstruction_plot.py`, and
  `tests/integration/test_loss_curves.py` cover: projection
  correctness (PCA verified against a synthetic dataset with a known
  dominant axis, not just shape checks; t-SNE/UMAP against the real
  packages), scatter-plot labeling (continuous colorbar, categorical
  legend), the collection helpers against a real `GlobalVae` forward
  pass, `plotPerDimensionKl`'s bar heights checked against the KL
  formula directly (not just "does not raise"), reconstruction overlay
  correctness including `inverse_transform` and custom x-axis values,
  grid layout and truncation, loss/step/beta-schedule curve value
  correctness, and `HistoryCallback` end to end through `Trainer.fit`.
- An image-comparison reconstruction plot (side-by-side original/
  reconstruction images, as opposed to `reconstruction_plot.py`'s line
  overlay) is a natural future addition once an image decoder exists
  (spec §6.1 milestone 2), not built here.
