# 0016: Cross-modal reconstruction reporting

**Status:** accepted
**Date:** 2026-09-12

## Context

Spec §5 ("Missing-modality robustness") already states that, with PoE / MoE /
cross-attention fusion, "the model can be trained and queried with any subset of
available modalities." `GlobalVae.forward` already implements exactly this: when a
latent space is fed by more than one encoder, but `inputs` this pass only contains a
subset of them, Fusion (`self.fusions[latent_name]`) is only invoked once
`len(active) > 1`; with exactly one active encoder, that latent space's
`(mu, logvar)` is simply that one encoder's own output (`models/global_vae.py`), and
every decoder consuming the latent space still runs regardless of which encoder(s)
produced it. No code change is needed to make e.g.
`model({"signal": batch})["reconstructions"]["image"]` work today; it already does,
and `tests/integration/test_en_l1_dn_default.py::test_forward_with_missing_modality`
already covers exactly this path.

What was missing was a way to make this **visible and systematic** rather than
something a caller has to reconstruct by hand for every input subset it wants to
compare: running the same batch through several input subsets, collecting what
every decoder produces under each, and laying the results out for comparison (a
matrix of "fed this subset in" x "read this decoder's output"), both qualitatively
(a figure) and quantitatively (a number per cell). This is a reporting/evaluation-
tooling need, not a new architectural capability; it belongs alongside
`visualization/` and `evaluation/`, not inside `GlobalVae` itself.

## Decision

Three new functions in `visualization/reconstruction_plot.py`, reusing the module's
existing `_plotOnePair` helper (the same one `plotReconstruction`/
`plotReconstructionGrid` already use) and following its existing collect-then-plot
split:

- `resolveDefaultInputSubsets(model) -> list[frozenset[str]]`: one singleton subset
  per encoder, plus the full set of every encoder together (only once, only if
  there is more than one encoder). Deliberately not the full power set
  (`2**N - 1` non-empty subsets): for the fusion strategies this framework ships, a
  fused posterior only depends on *which* encoders are active, not on any further
  structure (spec §4), so beyond "each modality alone" and "everything together"
  the remaining combinations grow combinatorially without being independently
  informative as a *default*. A caller wanting a specific further combination
  passes an explicit `input_subsets` instead (`itertools.combinations(model.encoders,
  2)` for every pair, for example).
- `collectCrossModalReconstructions(model, dataloader, input_subsets=None, ...) ->
  dict[frozenset[str], dict[str, tuple[Tensor, Tensor]]]`: for every batch and
  every subset, restricts `inputs` to that subset (mirroring
  `Trainer._applyModalityDropout`'s own restrict-the-input/keep-the-full-batch-as-
  target convention) and pairs every resulting reconstruction against that
  decoder's own ground truth from the *unrestricted* batch, so a decoder's
  reconstruction is comparable against ground truth even when its own modality was
  withheld from that subset. `use_mean=True` by default (matching `evaluate`'s own
  default, unlike the existing single-modality `collectReconstructions`, which
  never sets it): isolating the effect of "which modalities were available" from
  sampling noise is the whole point of this comparison, and the existing default
  would introduce a second, unrelated source of cell-to-cell variance. Unlike
  `evaluation.visual_export.exportEvaluationFigures` (needs a `list`: it walks its
  dataloader once per modality/latent space), this walks its dataloader exactly
  once regardless of how many subsets are requested (every subset is handled
  inside the same per-batch loop), so any `Iterable` works, single-use iterators
  included.
- `plotCrossModalReconstructionMatrix(collected, example_index=0, ...) -> Figure`:
  rows = input subsets (sorted by their formatted label), columns = every decoder
  name appearing in `collected`, one `_plotOnePair` overlay per cell for a single
  chosen example, blank (axis off) wherever that (subset, decoder) pair is absent.
  Same `inverse_transform`/`x_values` parameters as `plotReconstructionGrid`, now
  keyed per decoder name (matching `exportEvaluationFigures`'s own
  `inverse_transforms` convention) since different columns can be different
  modalities with different preprocessing.

A new `evaluation/cross_modal.py`, not an addition to `evaluate.py`/
`visual_export.py` (a deliberate, small deviation from how this idea was originally
pitched, which suggested folding the figure-export half into `visual_export.py`
directly; see "Consequences" for why):

- `computeCrossModalReconstructionMetrics(collected, metrics=None) ->
  dict[frozenset[str], dict[str, dict[str, float]]]`: applies
  `evaluation.metrics.DEFAULT_RECONSTRUCTION_METRICS` (or a caller-supplied set) to
  every cell of `collected`, pooled per cell exactly like `evaluate` itself does
  for the same metrics (and for the same reason: R^2/Pearson r are not a correct
  average of per-batch values). Pairs with the figure above to answer "how much
  does this actually cost", not only "what does it look like": the proposal that
  prompted this ADR asked for "every possibility", and a plot alone cannot quantify
  that, only show it.
- `exportCrossModalFigures(model, dataloader, output_dir, ...) -> list[Path]`: glue
  over the two functions above plus file I/O, mirroring `exportEvaluationFigures`'s
  own shape. Deliberately a **no-op** (returns `[]`, writes nothing) for a model
  with fewer than two encoders, rather than raising: a single-modality model has
  only one possible input subset, so there is nothing cross-modal to report, and
  this keeps a caller free to always call it unconditionally in a reporting
  pipeline without special-casing single-modality models itself. A genuinely
  misused explicit `input_subsets` (e.g. an empty one, or one naming an unknown
  modality) still raises `ValueError` from `collectCrossModalReconstructions`, not
  silently swallowed here. Wired into `scripts/evaluate.py`, called unconditionally
  alongside the existing `exportEvaluationFigures` call whenever `--output-dir` is
  given: free for spec §6.1 milestone 1's single-modality model (a no-op), and
  immediately useful once a second modality exists.

## Consequences

- No change to `GlobalVae`, `AbstractFusion`, or any encoder/decoder: this ADR only
  adds reporting on top of the existing, already-tested spec §5 behavior.
- Deliberately a **separate module** (`evaluation/cross_modal.py`), not folded into
  `evaluate()`'s own always-on pass or into `visual_export.py`: cross-modal
  reporting is opt-in and, for the common single-modality case (spec §6.1 milestone
  1), entirely inapplicable, so every existing `evaluate()`/`exportEvaluationFigures()`
  call site keeps its current behavior unchanged, and a single-modality caller
  never pays for, or has to reason about, a no-op branch inside its own evaluation
  pass. It also gives the quantitative half (`computeCrossModalReconstructionMetrics`)
  a home that does not depend on `visual_export.py`'s own file-I/O concerns, the
  same reasoning that already keeps `metrics.py` and `evaluate.py` themselves
  separate from `visual_export.py` today.
- `tests/integration/test_cross_modal_reconstruction.py` covers
  `resolveDefaultInputSubsets` (singleton-plus-full-set default, the single-encoder
  edge case), `collectCrossModalReconstructions` (subset restriction actually
  reaching the encoder side while ground truth still comes from the full batch,
  `use_mean` determinism, the empty-subset/unknown-name/empty-dataloader/no-subsets
  error paths, `max_samples`), and `plotCrossModalReconstructionMatrix` (row/column
  resolution and its override, blank cells for an absent pair, per-decoder
  `inverse_transform`, `example_index` selection and its out-of-range error path).
  `tests/integration/test_cross_modal_evaluation.py` covers
  `computeCrossModalReconstructionMetrics` (value correctness against
  `evaluation.metrics` directly) and `exportCrossModalFigures` (the single-modality
  no-op, file output for a real two-modality dummy model, the misused-explicit-
  subset error path surviving the no-op check). A three-modality dummy model is
  exercised too, not only the two-modality case the original proposal used as its
  motivating example, since `resolveDefaultInputSubsets`'s "singles plus full set"
  rule, and every function's plain iteration over `frozenset[str]` keys, make no
  assumption about modality count.
- `scripts/evaluate.py`'s own CLI test gained `TestCrossModalFigures`, using two new
  additive dummy fixtures in `tests/integration/_script_fixtures.py`
  (`buildTwoModalityModelForScript`/`buildTwoModalityDataloaderForScript`, mirroring
  how `buildLabeledDataloaderForScript` was added there before), covering both that
  a two-modality model actually gets `cross_modal_reconstructions.png` and that the
  existing single-modality fixture still does not.
- A real end-to-end example (`examples/`) demonstrating this against two real,
  non-dummy modalities awaits spec §6.1 milestone 2 (a real image encoder/decoder);
  building one now, against only synthetic dummy modalities, would either duplicate
  `tests/integration/test_cross_modal_reconstruction.py`'s own dummy fixtures or
  jump ahead of the milestone order (spec §12), so it is left for that milestone
  instead of added here.
