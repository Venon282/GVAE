# 0015 — Per-modality data transforms and sequence length

**Status:** accepted
**Date:** 2026-09-10

## Context

`DataConfig.transforms` (`docs/adr/0012-generic-data-transforms.md`) and
`DataConfig.sequence_length` were both a single, flat value shared by the whole
experiment: `transforms: list[TransformConfig]`, `sequence_length: int | None`.
`buildTransformPipeline(config) -> ComposeTransform` resolved that one list into
one pipeline.

This directly contradicts spec §6's own stated intent for the 1D-signal modality
family: "Other signal sources (spectroscopy, sensor time series, etc.) should slot
into the same 'signal' encoder/decoder family later, with only preprocessing
differing, not the architecture." A second 1D-signal dataset (a different
instrument, a different sensor — e.g. pairing SAXS with an unrelated LES dataset)
is exactly the case this sentence describes, and the flat schema could not express
it: two datasets sharing the `1d_cnn_encoder_v1`/`1d_cnn_decoder_v1` family, each
needing its own `log`/`standardize` statistics (possibly a different set of steps
entirely — one dataset may not need a `log` step at all) and its own resampled
length, had nowhere to configure that independently. The only way around this was
to bypass `DataConfig`/`buildTransformPipeline` entirely and hand-build per-dataset
pipelines directly in Python, which is exactly what
`examples/01_signal_vae_pipeline.py` already does.

This was also an internal inconsistency, not only a missing feature:
`evaluation.visual_export.exportEvaluationFigures` and
`visualization.reconstruction_plot.plotReconstructionGrid` already accept
`inverse_transforms: dict[str, InverseTransform]`, keyed **per modality**. That
parameter only makes sense once the forward pipeline is also per-modality; before
this ADR, the only way to produce a correctly-keyed `inverse_transforms` dict from
`config/data.py` was to build it by hand outside the config layer, since
`buildTransformPipeline` never returned anything keyed by modality at all.

## Decision

- `DataConfig.transforms` is now `dict[str, list[TransformConfig]]`: modality name
  -> that modality's own ordered step list, instead of one flat list shared by
  every modality.
- `DataConfig.sequence_length` is now `dict[str, int] | None`: modality name ->
  that modality's own target resampled length, for the same reason (two
  1D-signal-family datasets can legitimately need different lengths, each still
  required to match that same modality's own decoder `output_length` by hand, per
  `config/model.py`'s own "data domain stays outside this module" boundary,
  unchanged from before this ADR).
- `buildTransformPipeline(config) -> dict[str, ComposeTransform]`: one composed
  pipeline per key of `config.transforms`, not one pipeline for the whole config.
  A modality with an empty step list still resolves to an identity
  `ComposeTransform([])`, exactly as an empty flat list used to. A modality with no
  entry in `config.transforms` at all is simply absent from the returned dict:
  `DataConfig` has no independent notion of which modalities exist in the first
  place (that list lives in `ModelConfig.modalities`, a separate config domain,
  spec §9), so `buildTransformPipeline` cannot invent an identity entry for a
  modality it was never told about. A caller wanting an explicit identity fallback
  does so itself (`pipelines.get(name, ComposeTransform([]))`), the same one-line
  pattern `examples/_synthetic_signal_data.py` now uses for its own single
  `"signal"` modality.
- No change to `TransformConfig` itself (still name + kwargs for one step), to
  `ComposeTransform`, or to any concrete transform (`LogTransform`,
  `StandardizeTransform`, `ResampleTransform`): every transform in
  `data/transforms/` was, and remains, fully generic across dimensionality and
  knows nothing about modalities (spec §6.2's own hard requirement, unchanged).
  The per-modality *mapping* lives one layer up, in `config/data.py` only.
- Deliberately **no shared "default" key** that applies to every modality without
  its own entry. Every other per-modality config in this framework
  (`GlobalVae.createSingleLatent`'s `modality_configs`, `Trainer`'s
  `reconstruction_weights`/`reconstruction_loss_fn` per-modality dicts,
  `evaluation.visual_export`'s `inverse_transforms`) already requires naming each
  modality explicitly, with no implicit fallback; a "default" transforms entry
  would reintroduce exactly the "which pipeline silently applied here" ambiguity
  this ADR removes, for one extra top-level YAML key per modality saved.
- `configs/data/signal.yaml` updated to the new shape: its one `signal` modality's
  steps now nest under a `signal:` key, and `sequence_length: 256` is now
  `sequence_length: {signal: 256}`.

## Consequences

- A second 1D-signal-family dataset (or any further modality) can now be
  configured with its own preprocessing pipeline and its own resampled length
  purely through `configs/data/*.yaml`, with zero Python code and zero changes to
  `GlobalVae`, `Trainer`, or any registry, matching spec §6's own stated intent.
- `buildTransformPipeline`'s return value is now directly usable as
  `evaluation.visual_export.exportEvaluationFigures`'s `inverse_transforms`
  argument (pass the returned dict's `.inverse` methods, keyed the same way both
  already expect), closing the inconsistency described in "Context" above.
- Breaking change to `DataConfig`'s shape: any config file or programmatic
  `DataConfig(...)` construction using the old flat `transforms: list[...]`/
  `sequence_length: int` shape must be updated to nest under a modality key. No
  real dataset depended on the old shape yet (spec §11's pairing-mechanism
  question is still open), so this is a contained, backward-incompatible change
  with no live-data migration to perform.
- `tests/integration/test_transforms.py::TestBuildTransformPipelineFromConfig`
  gained direct coverage of the motivating case (two modalities with independent
  pipelines, `test_two_modalities_get_independent_pipelines`) and of the "absent
  modality is absent from the result, not an implicit identity" behavior
  (`test_modality_absent_from_transforms_is_absent_from_the_result`); every
  pre-existing test in that class still passes, updated only to the new
  `transforms={"signal": [...]}` / `pipelines["signal"]` shape.
- `tests/integration/_train_script_fixtures.py`,
  `tests/integration/test_config.py`, and `examples/_synthetic_signal_data.py`
  (all effectively single-modality) updated to read
  `config.sequence_length`/`buildTransformPipeline(config)` via `"signal"`
  lookups instead of using the value/result directly.
