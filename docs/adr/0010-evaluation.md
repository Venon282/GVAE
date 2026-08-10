# 0010 — Standalone evaluation (`evaluation/`, `scripts/evaluate.py`)

**Status:** accepted
**Date:** 2026-08-07

## Context

Spec's C8 requirement: "un script/mode d'éval distinct de l'entraînement" (a
script/eval mode distinct from training), with reconstruction metrics on the test set
(MSE at minimum), a KL value, and optionally exporting reconstructions for visual
inspection. `Trainer.evaluate` (spec §10, `docs/adr/0005-training-loop.md`) already
runs a no-gradient validation pass, but it only ever reports the same three aggregate
losses training itself does (`"val/loss/total"` etc.) and requires a `Trainer`
instance (model, optimizer, and everything else `Trainer.__init__` needs) to exist at
all. That does not serve "distinct from training": re-running a thorough evaluation on
a saved checkpoint, independent of ever having built a `Trainer`, needed its own path.

A first pass at this (`evaluation/metrics.py`, `evaluation/evaluate.py`,
`evaluation/visual_export.py`) had already been started before this ADR, and got
interrupted partway through: `visual_export.py` contained a dead `... if False else
...` branch around a `plotReconstructionGrid` call that passed `title=None`, a
parameter that function did not actually have yet. This ADR's work picked that up,
fixed it properly, and finished the remaining pieces (tests, the CLI script,
documentation) rather than starting over.

## Decision

- **`evaluation/metrics.py`**: plain functions, `computeMse`/`computeRmse`/
  `computeMae`/`computeR2`/`computePearsonR`, each `(reconstruction, target) ->
  float`, plus `DEFAULT_RECONSTRUCTION_METRICS: dict[str, MetricFn]` bundling all
  five. Not a registry (unlike encoders/fusion/regularizers/etc.): a caller wants
  *several* metrics computed together, not one chosen strategy, so `evaluate` takes a
  plain `dict[str, MetricFn]` a caller can extend, trim, or replace outright, mirroring
  how `losses/reconstruction.py`'s own `loss_fn` parameter already works.
  `computeR2`/`computePearsonR` return `nan` (not raise) for a degenerate constant
  input, since a whole test-set metrics report should not abort over one modality's
  edge case; `nan` prints and JSON-serializes fine and is the honest "undefined" value.
- **`evaluation/evaluate.py`**: `evaluate(model, dataloader, ...) -> EvaluationResults`.
  Computes the two aggregate losses as running batch averages (directly comparable to
  `Trainer`'s own training/validation curves), but computes every per-modality
  reconstruction metric and per-latent-space regularization value once over the
  *entire pooled dataset* (every batch concatenated first), not batch-averaged: R^2 and
  Pearson r are not correctly computable as a naive average of per-batch values (both
  are ratios of sums, not sums of ratios), and a naive average would also be subtly
  wrong for any metric once batch sizes differ. This trades memory (the whole
  collected test set held at once) for correctness; `max_samples` bounds it.
  `regularization_metrics` always reports **two** numbers per latent space:
  `"configured"` (whatever `AbstractLatentRegularizer` strategy the model actually
  uses, spec §2.3) and `"kl_standard_normal"` (always the plain KL formula,
  regardless of the configured strategy), so two runs trained with different
  regularizers (say, one `mmd`, one `free_bits_kl`) still have one number that is
  directly comparable between them. `use_mean=True` by default (unlike `Trainer`,
  which always samples): deterministic reconstructions from the posterior mean is the
  standard choice for a reconstruction-quality report, removing sampling noise as a
  source of run-to-run variance in the numbers being reported. This uses
  `GlobalVae.forward`'s new `use_mean` parameter (added alongside this work) rather
  than resampling after the fact.
- **`GlobalVae.forward` gained a `use_mean: bool = False` parameter.** `False`
  preserves existing behavior exactly (every existing caller, including `Trainer`,
  is unaffected); `True` uses each active latent space's posterior mean directly
  instead of calling `reparameterize`, which is what `evaluate`'s own `use_mean=True`
  default needs. This is a model-level capability (skip sampling), not something
  `evaluate` should approximate by discarding `logvar` and hand-rolling the
  mean-selection logic itself.
- **`evaluation/visual_export.py`**: `exportEvaluationFigures(model, dataloader,
  output_dir, ...) -> list[Path]`. Pure wiring, no new plotting logic: reuses
  `visualization/`'s own collection and plotting functions
  (`docs/adr/0009-visualization.md`) and saves the resulting figures as PNG files.
  Takes a `list[dict[str, torch.Tensor]]`, not just any `Iterable`, since it walks the
  same data once per modality and once per latent space (unlike `evaluate`'s single
  pass): a single-use iterator would silently only produce output for the first thing
  collected. `plotReconstructionGrid` gained a `title: str | None = None` parameter
  (`fig.suptitle`) as part of finishing the interrupted work above, so
  `exportEvaluationFigures` can label each modality's grid
  (`f"Reconstructions: {modality_name}"`) without a separate figure-titling mechanism.
- **`scripts/evaluate.py`**: a CLI wrapping `loadCheckpoint` + `evaluate` +
  `exportEvaluationFigures`. Model construction and data loading are dynamically
  imported from `--model-factory`/`--dataloader-factory` arguments shaped
  `"module.path:function_name"`, rather than this script hardcoding either: every
  other part of this framework keeps model construction and data loading as the
  caller's own responsibility (data pipeline concerns are explicitly out of scope; no
  config schema exists yet, spec §11), and a CLI script is not an exception to that
  just because it needs *some* way to obtain a model and a dataloader. Prints a
  human-readable summary always; `--output-dir` additionally saves a JSON report
  (`EvaluationResults.save`) and, unless `--no-figures`, exported figures.

## Consequences

- Spec's C8 requirement is satisfied: `evaluate()` (the "mode", usable from any
  Python code) and `scripts/evaluate.py` (the "script", usable from the command
  line) both work from just a checkpoint and a dataloader, no `Trainer` required.
  MSE is always included (part of `DEFAULT_RECONSTRUCTION_METRICS`); a directly
  comparable KL value is always reported (`regularization_metrics[...]
  ["kl_standard_normal"]`) regardless of which regularizer actually trained the
  model; reconstruction export is available via `--output-dir` (or
  `exportEvaluationFigures` directly).
- `tests/integration/test_evaluate.py`, `test_visual_export.py`, and
  `test_evaluate_script.py` cover: every metric's value correctness (including the
  `nan` edge cases), `EvaluationResults`' JSON round-trip and summary formatting,
  `evaluate`'s aggregate/per-modality/per-latent-space outputs, `beta`/`use_mean`/
  `max_samples`/custom-metrics options, `exportEvaluationFigures`'s file output, and
  the CLI script end to end (including its dynamic-import error paths and the
  `--no-figures`/missing-checkpoint/missing-argument cases). `_script_fixtures.py`
  (the CLI test's model/dataloader factories) is deliberately named outside pytest's
  own `test_*.py` discovery pattern, the same reasoning already documented in
  `test_checkpoint.py` for why sibling test files do not import each other's dummy
  fixtures directly.
- `evaluation/`'s own `NOTE.md` and this ADR replace the interrupted state the
  subpackage was found in; nothing here was rewritten from scratch, only completed
  and corrected (the dead-code fix, the added tests, the CLI script, `use_mean`,
  `plotReconstructionGrid`'s `title`).
