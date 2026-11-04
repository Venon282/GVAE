# 0045: Loss settings are keyed by reconstruction name

**Status:** accepted (roadmap item P5-4; it changes what the keys of three mappings are in
`config/training.py` and no other module; it supersedes in part [ADR 0029](0029-loss-scale-convention.md)
and [ADR 0030](0030-per-modality-reconstruction-loss-and-weight.md), which keyed them by decoder name)
**Date:** 2026-11-04

## Context

`training.reconstruction_loss`, `training.reconstruction_weight` and
`training.reconstruction_reduction` each take one value or a mapping. ADR 0029 and ADR 0030 keyed the
mapping by **decoder name** and validated the keys against `model.decoders`. The code that applies the
settings, `computeTotalReconstructionLoss`, walks `outputs["reconstructions"]` and looks each setting
up by the **reconstruction name**, which is what `GlobalVae.output_modalities` lists.

For a decoder that handles one modality the two names are the same string, so nothing was wrong. For a
decoder shared by several modalities ([ADR 0027](0027-shared-encoders-and-decoders.md)) they differ: in
`signal_image_shared_transformer` the decoder is `backbone_out` and the reconstructions are `signal`
and `image`. The two layers then disagreed (roadmap finding I-1):

| Key written | Weight or reduction | Loss function |
|---|---|---|
| `signal` (a modality) | refused: "names decoder(s) ['signal'] that the model does not have" | refused, the same way |
| `backbone_out` (the shared decoder) | accepted, then **not applied** (the loss never looks `backbone_out` up) | accepted, then `KeyError` at the first step |

A model with a signal and an image could therefore not weight them separately, which is what it needs
(one weight cannot balance the two, see P3-3 in the 2.0.0 changelog), and a config that looked right
silently did nothing.

## Decision

**1. The keys of the three mappings are reconstruction names.** A reconstruction name is the name of a
decoder that handles one modality, or one entry of `output_modalities` of a decoder that handles
several. `config/training.py` validates against `model.output_modalities` and nothing else changes in
what the mappings mean: the three fields still take one value for every modality or a mapping, and
`reconstruction_loss`, `reconstruction_weight` and `reconstruction_reduction` read the same way.

**2. The name of a shared decoder is refused, not accepted as a shorthand for its modalities.** One
namespace, one rule. A shorthand would also have to say what happens when a decoder and a modality
share a name, or when a mapping names both. The error says what to write instead, and lists both sets
of names, since the shared decoder's name is the likeliest mistake:

```
training.reconstruction_weight names decoder(s) ['backbone_out'] that are not reconstruction names of
the model (decoders: ['backbone_out'], reconstruction names: ['image', 'signal']). Key a setting by
reconstruction name: the name of a decoder of one modality, or one modality of a shared decoder. A
reconstruction left out of the mapping keeps 1.0, so a mistyped name would change nothing and say
nothing.
```

The start of the message (`<field> names decoder(s) [...]`) and the `decoders: [...]` listing are what
they were, so a caller or a test that reads them keeps working.

**3. The rule for a name left out is unchanged** (ADR 0030, point 2): `"mse"` for the loss, `1.0` for
the weight and `"mean"` for the reduction. The loss resolver writes the default out as an entry for
**every reconstruction** of the model (it was every decoder), because `Trainer` raises `KeyError` for a
reconstruction that has no entry. The weight and the reduction resolvers keep only the configured
entries, because `computeTotalReconstructionLoss` supplies the default.

**4. Two modalities of one shared decoder may differ in all three settings.** The loss reads each
setting per reconstruction, so no change is needed there: a test sets a different loss, weight and
reduction for the signal and the image of one `shared_transformer` decoder and checks the sum against
the value computed by hand.

## Consequences

- **A config without a shared decoder gives the same results.** For a decoder of one modality the
  decoder name and the reconstruction name are the same, so every config that worked keeps its meaning,
  to the last bit (the tests of ADR 0029 and ADR 0030 pass unchanged).
- **A config that keyed a shared decoder by its own name changes from "silently ignored" (weight,
  reduction) or "crashes at the first step" (loss) to a clear error at build time.** That is the
  point of the item, and no shipped file did it: `signal_image_shared_transformer_vae` uses one value
  for both modalities.
- `scripts/evaluate.py` reads the same resolvers, so `--reconstruction-weight`,
  `--reconstruction-loss` and `--reconstruction-reduction` follow without a change.
- The two `strict` `xfail` tests of `test_transformer_model_configs.py` are ordinary tests, and five
  tests state the new behavior (a mapping by modality is accepted, the shared decoder's name is
  refused with both lists, and a weight, a loss and a reduction are each applied per modality), plus
  one that composes the shared experiment from YAML overrides and checks the value that reaches the
  loss.
- `computeTotalReconstructionLoss`, `Trainer`, `evaluate` and the model are untouched.

## Not decided here

How a loss becomes a registered name with arguments (roadmap P5-3) resolves its names against the same
set of reconstruction names, so it needs nothing from this ADR but the namespace.
