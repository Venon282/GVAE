# 0019: Decouple encoder inputs from decoder targets in `Trainer` and `GlobalVae`

**Status:** accepted
**Date:** 2026-09-26

## Context

Every framework entry point that walks a raw per-modality batch (`Trainer.computeLosses`,
`evaluation.evaluate.evaluate`, and the `visualization` collectors, `collectLatentParams`/
`collectReconstructions`) used to hand the *whole* batch dict straight to
`GlobalVae.forward` as `inputs`, then use that same whole batch again as the
reconstruction target for `losses.reconstruction.computeTotalReconstructionLoss`. That
is correct whenever every batch key is both an encoder input and a decoder target
(plain autoencoding, spec §6.1's milestones, and the `EN-L1-DN` default, ADR 0001): the
same dict genuinely serves both roles.

It stops being correct the moment a decoder's own reconstruction target is not itself
an encoder input, e.g. a translation-style model with one encoder (`image_in`) and one
decoder (`image_out`), or more generally any decoder name absent from
`GlobalVae.encoders`. `GlobalVae.forward` iterates `inputs.items()` and looks up
`self.encoders[encoder_name]` for every key; a batch carrying `image_out` alongside
`image_in` raised a bare `KeyError('image_out')` the moment it reached `forward`,
regardless of whether the caller was `Trainer`, `evaluation.evaluate.evaluate`, or a
`visualization` collector. Working around this at the call site (a `Trainer` subclass
overriding `_applyModalityDropout` to pre-filter its own input, as
`examples/03_signal_image_to_image.py` used to) fixes exactly one call site, not the
underlying gap; every other place in the framework that runs `model(batch)` over a raw
batch has the identical failure mode.

A second, unrelated bug surfaced while fixing the first: `Trainer.evaluate` shares its
forward/loss logic with `Trainer.fitEpoch` through `computeLosses`, which always applied
`self.modality_dropout_p`. Modality dropout (spec §5) is deliberately a *training-time*
robustness technique; applying it during validation as well made the reported
`val/loss/*` metrics depend on which modalities a given validation pass happened to
randomly hide, rather than on the model and the data alone. Concretely, on a two-encoder
model with `modality_dropout_p=0.9`, ten back-to-back `evaluate()` calls against the
exact same single-batch dataloader spanned a `max - min` loss range of about `0.75`
before this fix and about `0.40` after (the residual spread is the reparameterization
sample `Trainer` still draws for its ELBO estimate, unaffected by this ADR, see the
"what stays unchanged" section below). Left uncorrected, this adds spurious noise to
whatever `val/loss/total` (or any other `"val/..."` key) is used for: early-stopping
decisions, `BestCheckpointCallback`'s monitored metric, and any epoch-to-epoch
comparison a caller draws from `Trainer.history`.

## Decision

- `GlobalVae` gains `selectEncoderInputs(batch) -> dict[str, torch.Tensor]`: the one
  place a raw per-modality batch is restricted to the keys naming one of `self.encoders`.
  Raises `ValueError` if nothing in `batch` matches any encoder (nothing to encode);
  never raises for a batch that also carries decoder-only keys, and never mutates its
  argument.
- `GlobalVae.forward` itself now raises a clear, actionable `KeyError` naming the
  offending key(s) and pointing at `selectEncoderInputs` when `inputs` still contains a
  key with no matching encoder, instead of the bare `KeyError` a plain dict lookup gave
  before. `forward`'s own contract is unchanged: `inputs` must already be restricted to
  encoder names; this only makes violating that contract diagnosable.
- `Trainer.computeLosses` calls `self.model.selectEncoderInputs(batch)` before modality
  dropout and before the forward pass; the unrestricted `batch` is still what
  `computeTotalReconstructionLoss` matches reconstructions against. A translation-style
  batch `{"image_in": ..., "image_out": ...}` now trains and evaluates with no
  special-casing anywhere in the caller's own code.
- `Trainer.computeLosses` gains `apply_dropout: bool = True`. `Trainer.fitEpoch` keeps
  the default (dropout applies during training, unchanged); `Trainer.evaluate` passes
  `apply_dropout=False`, so a validation pass never applies modality dropout regardless
  of `self.modality_dropout_p`.
- `evaluation.evaluate.evaluate`, `visualization.latent_plot.collectLatentParams`, and
  `visualization.reconstruction_plot.collectReconstructions` all call
  `model.selectEncoderInputs(batch)` before their own forward pass, for the identical
  reason. `visualization.reconstruction_plot.collectCrossModalReconstructions` already
  restricted to an explicit, pre-validated subset of `model.encoders` per input subset,
  so it needed no change. `scripts/visualize_latent.py`'s own
  `_collectLatentParamsAndLabels` already filtered inline before this ADR; it is
  unchanged besides a comment cross-referencing `selectEncoderInputs`.
- `examples/03_signal_image_to_image.py`'s `TranslationTrainer` subclass is removed: the
  stock `Trainer` now does everything it existed to work around.
  `tests/integration/test_signal_image_example.py::TestTranslationTrainer` (which
  documented, and asserted, the old `KeyError`) is replaced by
  `TestTrainerHandlesTargetOnlyKeys`, exercising the same scenarios against the stock
  `Trainer`.

## What stays unchanged

`Trainer.fitEpoch`/`Trainer.evaluate` still call `self.model(inputs)` with
`use_mean=False` (the default), i.e. both still estimate the same sampled-`z` ELBO
`fitEpoch` trains against; `Trainer.evaluate` was never meant to be the deterministic,
posterior-mean report `evaluation.evaluate.evaluate` already is for exactly that purpose
(`use_mean=True` by default, see that function's own docstring). This ADR removes
modality dropout as a source of noise in `Trainer.evaluate`'s reported loss; it does not
make that loss bit-for-bit reproducible across calls, and is not intended to: an
unweighted, sampled ELBO estimate genuinely has some residual variance by construction,
the same variance the training objective itself has. A completely deterministic
reconstruction-quality report already exists (`evaluation.evaluate.evaluate`) and is
unaffected by this decision beyond the same `selectEncoderInputs` restriction above.

## Consequences

- A decoder's reconstruction target no longer needs to also be one of the model's
  encoder inputs anywhere in this framework, not only in `Trainer`: spec §2.1's routing
  graph already allowed a decoder to consume a latent space with no encoder of its own
  name; this closes the one remaining place (raw batch handling) that assumed otherwise.
- `Trainer.evaluate`'s reported `"val/..."` metrics no longer depend on
  `self.modality_dropout_p` at all: two calls against the same non-shuffled dataloader
  now differ only by the residual reparameterization-sampling noise described above, not
  by which modalities a given pass happened to drop.
- `GlobalVae.forward`'s error message for a genuinely unrecognized input key is now
  actionable (names every offending key, lists the available encoders, and points at
  `selectEncoderInputs`) instead of a bare `KeyError` with only the first missing key.
