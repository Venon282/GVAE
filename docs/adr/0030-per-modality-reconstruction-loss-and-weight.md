# 0030: A reconstruction loss and weight per modality, from config

**Status:** accepted (roadmap item P2-2; the defaults are unchanged, so nothing changes silently)
**Date:** 2026-10-17

## Context

`TrainingConfig.reconstruction_loss` was a `str` and `reconstruction_weight` a `float`, one value
shared by every modality. `computeTotalReconstructionLoss` and `Trainer` already took a
per-modality `dict` for both, so a run that needed a different loss for one modality (the
specification's example: `binary_cross_entropy` for a segmentation target next to `mse_loss` for a
continuous one) or a different weight (to rebalance modalities, ADR 0029) had to build the `Trainer`
in Python and could not be described by a config file, a Hydra override or a checkpoint's config
snapshot. [ADR 0029](0029-loss-scale-convention.md) made the reduction a `str | dict[str, str]`
and left these two to this item.

The two functions treat a mapping differently, and the difference matters for config:

- a weight mapping that does not name a decoder gives it the weight `1.0`;
- a loss mapping that does not name a decoder is a `KeyError` ("No loss function provided for
  reconstruction ..."), raised at the first batch that reconstructs it, which is after the model,
  the optimizer and the callbacks are built and, with modality dropout, possibly not on the first
  step at all.

Nothing checks that a mapping key is a decoder of the model. A mistyped key in a weight mapping
changes nothing, and the decoder it was meant for trains on a weight of 1.0.

## Decision

**1. Both fields take a single value or a mapping decoder name -> value.**
`reconstruction_loss` is `str | dict[str, str]` and `reconstruction_weight` is
`float | dict[str, float]`. They are typed `Any` in the structured config, because it cannot
express such a union (the reason of `LatentSpaceConfig.fed_by` and of `reconstruction_reduction`),
and `buildTrainerFromConfig` validates them. The defaults stay `"mse"` and `1.0`, so a run that
sets neither computes exactly the loss it computed before, to the last bit (a test compares the
two).

**2. A decoder left out of a mapping keeps the default**: `"mse"` for the loss, `1.0` for the
weight, `"mean"` for the reduction. It is the rule the weights already had, and the one
ADR 0029 chose for the reduction, so the three fields read the same way. The resolver for the loss
writes the default out as an entry for every decoder of the model, because the `Trainer` would
otherwise raise for the decoders the mapping leaves out. The resolver for the weight keeps only the
entries that were configured, because `computeTotalReconstructionLoss` already supplies `1.0`.

**3. Validation, with the field's path at the start of every message.**

| Problem | Error |
|---|---|
| A loss name that is not one of `listSupportedReconstructionLossNames()` | `KeyError`, naming the decoder for a mapping and listing the supported names |
| A loss that is not a name or a mapping, a weight that is not a number, a mapping key that is not a string | `ValueError` |
| A weight that is negative, `nan` or infinite | `ValueError` |
| A `bool` as a weight | `ValueError` (it is an `int` to Python, and `reconstruction_weight: true` is a typo, not `1`) |
| A mapping key that is not a decoder of the model | `ValueError` listing the decoders and what the decoder would have kept |

A weight of exactly `0` is accepted: it takes a modality out of the loss while its decoder still
runs, which is a legitimate way to freeze what a decoder learns from the reconstruction term. A
negative weight is refused because it would make the optimizer maximize a reconstruction error.

A mapping key that names no decoder is an error and not a warning, as it is for the reduction
(ADR 0029): the typo would leave the intended decoder on its default and say nothing. This is
stricter than `beta_schedules`, whose unmatched keys log a warning.

**4. One set of resolvers serves training and evaluation.** `validateReconstructionLoss`,
`validateReconstructionWeight`, `resolveReconstructionLoss`, `resolveReconstructionWeight` and
`resolveReconstructionReduction` (which gains an optional `field_name` for its messages) are in
`config/training.py`. `resolveReconstructionLossFn(name)` stays as it was.

**5. `scripts/evaluate.py` takes the same three settings.** ADR 0029 added
`--reconstruction-reduction` so that the `total_reconstruction_loss` of a report is on the scale of
the training curves. That holds only if the loss and the weight are the training ones too, and the
script used `mse` and `1.0` whatever the model was trained with. It now has `--reconstruction-loss`
and `--reconstruction-weight`, each a bare value or `modality=value` pairs separated by commas
(`signal=mse,mask=bce`), because a command line cannot carry a mapping as a config can. All three
flags go through the resolvers above, so a modality the model does not have is a usage error there
too (it was silently accepted for the reduction until now). The defaults of the flags are the
defaults of the fields.

## Alternatives considered

- **Add plural fields** (`reconstruction_losses`, `reconstruction_weights`) and keep the singular
  ones. Two fields for one setting need a precedence rule and a conflict check, and every reader has
  to know which one wins. One field that takes either shape is what the reduction already does.
- **Check the keys in `Trainer`.** The `Trainer` is also built in Python, with a weight mapping that
  may name modalities freely (a target that no decoder reconstructs yet, a mapping shared between
  models). Where the keys are checked against a model is where a config is turned into a trainer,
  which is also where ADR 0029 checks them for the reduction.
- **Warn instead of raise for an unknown key.** A warning is read after the run has started, and a
  run that trains the wrong objective for hours is the cost. The check is cheap and the fix is a
  rename.
- **Normalize the weights** (divide by their sum). It would change what a single value means and
  would break the reading of ADR 0029: `0.5` with `mse` and `sum` is a fixed-variance Gaussian, and
  that is a statement about each modality, not about their share.
- **A loss given as an import path** (`module:function`). It is a different feature, with the
  security and reproducibility questions of running a named function from a config. The four
  built-in names cover the case that motivated this item, and a caller with another loss builds a
  `Trainer` in Python, as the module docstring of `config/training.py` says.
- **Record the loss and the weights in `EvaluationResults`, like the reduction.** Not done here: it
  extends the report format, and the report already depends on the caller for the weights it was
  given. See below.

## Consequences

- A run that sets neither field is unchanged. A weight written as an integer in YAML (`2`) reaches
  the `Trainer` as `2.0`, which multiplies to the same value.
- A value of the wrong type is now reported by `buildTrainerFromConfig` with the field's path and
  not by the structured config, whose fields were `str` and `float` and are now `Any`.
- The order of the three settings of a modality is fixed: the loss function, then the reduction
  (ADR 0029), then the weight. They are independent mappings and may name different decoders.
- `bce` needs a reconstruction and a target inside `[0, 1]`, so a decoder trained with it must end
  in a sigmoid or an equivalent. PyTorch raises otherwise, and the config does not check it.
- The report written by `scripts/evaluate.py` records the reduction and not the loss and the
  weights, so the caller has to give the three again, as the how-to says. A model trained from a
  config has them in its checkpoint's config snapshot.
- The per-modality reconstruction metrics of a report (`mse`, `r2`, ...) do not depend on any of
  the three settings.
- Not decided here: recording the loss and the weights in the report, a weight that changes during
  training, weights learned from the data (uncertainty weighting), and a loss given by import path.
