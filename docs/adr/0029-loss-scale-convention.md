# 0029: Loss scale convention and a reconstruction reduction per modality

**Status:** accepted (roadmap item P2-1, implemented as the roadmap words it: `mean` stays the
default, so nothing changes silently)
**Date:** 2026-10-16

No empirical number appears in this ADR. The values of `beta`, the warm-up length and the other
schedule hyperparameters are roadmap decision D-6, which waits for the real run of P3-1.

## Context

The objective that `Trainer` minimizes is

```
total = sum over decoders m of  w_m * R_m   +   sum over latent spaces s of  beta_s * KL_s
```

and the two sums were built on different scales:

- `R_m` is what `torch.nn.functional.mse_loss` (or `l1_loss`, `binary_cross_entropy`,
  `smooth_l1_loss`) returns with its default `reduction="mean"`: the mean over **every element of
  the batch**. With `N` the number of values one sample reconstructs (256 for a 256-point signal,
  1024 for a 32 by 32 image) and `S` the per-sample sum of squared errors, `R_m = S / N` averaged
  over the batch.
- `KL_s` is `KlStandardNormalRegularizer`: a **sum over the latent dimensions** of each sample,
  averaged over the batch.

The usual ELBO for a Gaussian decoder of unit variance is `S / 2 + KL`, both per sample. Rewrite
the implemented objective as that:

```
S / N + beta * KL  =  (2 / N) * ( S / 2  +  (beta * N / 2) * KL )
```

Up to the global factor `2 / N`, which rescales the gradient but does not move the optimum, the
implemented objective is the ELBO with an effective weight `beta' = beta * N / 2` on the KL.
This is the factor 128 for a 256-point signal and 512 for a 32 by 32 image. Three consequences:

- `beta = 1` is not the usual ELBO. It weights the KL `N / 2` times more than the number says.
- The useful `beta` depends on the data shape. The same value that works for a 256-point signal is
  four times stronger, in ELBO terms, for a 1024-pixel image.
- A strong KL early in training makes posterior collapse (every dimension of the posterior equal
  to the prior) likelier. The default training config guards against it with a warm-up, which
  treats the symptom and not the scale.

This was the open "loss weighting" question of the specification (§11) and of
`src/global_vae/losses/NOTE.md`. The question is not whether the mean is wrong: a per-element mean
has a real advantage, listed below. It is that nothing stated the convention, and nothing let a
user move to the other one without editing code.

## Decision

**1. The convention is stated.** The reconstruction term of a modality is its loss function's
output scaled by that modality's **reduction**, then by its weight `w_m`. The regularization term
is, for every latent space, the sum over the latent dimensions (in nats) averaged over the batch,
multiplied by `beta_s`. The two reductions are:

| Reduction | `R_m` | A reading of `beta = 1` |
|---|---|---|
| `mean` (default) | the loss function's value: the mean over every element of the batch | the ELBO weight `beta' = N / 2` on the KL, for `mse` |
| `sum` | `N` times that value: the per-sample sum over the modality's values, averaged over the batch | the ELBO of a decoder whose likelihood is the loss: Bernoulli for `bce`, Laplace of scale 1 for `l1`, Gaussian of variance 1/2 for `mse` |

For `mse` and `sum`, the unit-variance Gaussian ELBO needs `reconstruction_weight: 0.5`, because
`S / 2` is the Gaussian negative log-likelihood. A decoder variance `sigma^2` other than 1 is a
weight of `1 / (2 * sigma^2)`. Only a fixed variance is expressible: a learned variance is not
decided here.

**2. The default stays `mean` for every modality.** A run that does not mention the new field
computes exactly the loss it computed before, to the last bit (a test compares the two). A new
default would need a minor release and a changelog note, and its own ADR.

**3. The reduction is chosen per modality, from config.** `TrainingConfig.reconstruction_reduction`
is a name (`mean` or `sum`) for every modality, or a mapping decoder name -> name. A decoder left
out of the mapping keeps `mean`, the way a missing weight keeps `1.0`. The field is typed `Any`
because a structured config cannot express `str | dict[str, str]` (the same reason as
`LatentSpaceConfig.fed_by`), and `buildTrainerFromConfig` validates it: an unsupported name, or a
mapping key that is not a decoder of the model, is a `ValueError` that starts with
`training.reconstruction_reduction`. A mistyped decoder name is an error and not a warning, as the
unmatched `beta_schedules` keys are, because the typo would leave that decoder on `mean` and say
nothing, which is the silent change of scale this ADR exists to prevent.

**4. How `sum` is computed.** `computeTotalReconstructionLoss(..., reduction=...)` multiplies the
loss function's value by the product of the reconstruction's axes after the batch axis. This works
for any loss function that returns an element-wise mean, which is the contract `LossFn` already
documents and what every built-in does. A loss function that does not honour the contract is not
made right by `sum`.

**5. The same reduction reaches every place that reports the loss.** `Trainer` takes
`reconstruction_reduction` (the last argument, so positional callers keep working) and uses it in
`computeLosses`, hence for the training loss and for `Trainer.evaluate`. `evaluate()` takes it as
well, records it in `EvaluationResults.reconstruction_reduction` and prints it in the summary when
it is not `mean`, and `scripts/evaluate.py` has `--reconstruction-reduction`. Without that, the
`total_reconstruction_loss` of a report would be on another scale than the curves it is read next
to.

**6. Latent diagnostics are part of the evaluation report.** `evaluate()` computes, per latent
space and over the whole pooled evaluation set, a `LatentDiagnostics`
(`evaluation/latent_diagnostics.py`):

- `kl_per_dimension`: the mean over samples of each dimension's KL to a standard normal, in nats;
- `mu_variance_per_dimension` and `active_units`: a dimension is **active** when the variance of
  its posterior mean over the samples is above a threshold, 0.01 by default (Burda, Grosse and
  Salakhutdinov, "Importance Weighted Autoencoders", 2016). The threshold is a parameter
  (`active_unit_threshold`, `--active-unit-threshold`) and is stored in the report.

They are in `EvaluationResults.latent_diagnostics`, in the JSON file and in `summary()`. They come
from `(mu, logvar)` alone, so they mean the same whichever regularizer trained the model, which is
what makes two runs comparable. The KL formula is one function,
`computeKlPerDimension`, used by the regularizer, by `plotPerDimensionKl` and by the diagnostics,
so the numbers of the report and the bars of the figure cannot drift apart.

**7. A how-to explains how to choose `beta`**, with a sweep recipe and no number from a real run:
[Choose beta](../how-to/choose-beta.md).

## Alternatives considered

- **Make `sum` the default now.** It is the scale on which `beta = 1` is a likelihood. Rejected for
  this ADR: it changes the value of every loss curve, the meaning of every `beta` already tuned and
  the length of the warm-up that suits it, for every existing run and checkpoint, with no code
  change to warn anyone. The roadmap asks for a minor release and a changelog note.
- **Divide the KL by `N` instead** (a "per-element KL"), keeping `mean` for the reconstruction.
  It makes `beta` comparable with the mean, but there is no single `N`: a latent space can be fed
  by several modalities of different sizes, and `beta` is per latent space. It also hides the ELBO
  rather than restoring it.
- **A third reduction, `gaussian_nll` or `sum_half`** (`0.5 * sum`). Rejected: `sum` with a weight
  of 0.5 gives it, and the 0.5 is a statement about a fixed variance that a learned variance
  would replace. A name for it would hide that assumption.
- **Call the loss function with `reduction="sum"`.** `LossFn` is `(reconstruction, target) ->
  scalar`, and a caller's own function need not take the argument. It would also sum over the batch
  and then need a division by the batch size, so the result would be the same number by a less
  general route.
- **Read `N` from the config** (`sequence_length`, image size). The shape of the tensor is the
  truth, including for a decoder whose output shape no config states.

## Consequences

- A run without `reconstruction_reduction` is unchanged. The new field is in both shipped training
  YAML files, written as `mean`, so it is found.
- `sum` makes the reconstruction term `N` times larger. A loss curve of a `sum` run and one of a
  `mean` run are not comparable, and neither are two runs with different `beta`: compare the
  reconstruction metrics and the latent diagnostics instead, which the how-to does.
- With `sum`, a modality that reconstructs more values weighs more in the reconstruction term,
  where `mean` keeps a 256-point signal and a 1024-pixel image on the same footing. The reduction
  is per modality so that one can keep `mean` for some and use `sum` for others, and the weight is
  the lever for the rest. `reconstruction_weight` is one value for every modality in config until
  roadmap item P2-2; a per-modality weight is available in Python through `Trainer`.
- A checkpoint saved by a callback carries the config snapshot of a config-driven run (spec §10),
  hence its reduction. A `Trainer` built in Python records nothing of the kind, so the caller says
  which reduction to use at evaluation time.
- The active-unit count has the variance of a sample statistic: it is noisy on a small evaluation
  set, and `max_samples` truncates the pooled data it is computed from. A dimension with a KL near
  zero and a flat mean carries no information; a dimension with a KL above zero and a flat mean is
  spending it on an offset or a variance that is not 1. The report shows both so that the two
  cases can be told apart.
- `plotPerDimensionKl` and `KlStandardNormalRegularizer.forward` now call
  `computeKlPerDimension`. Their results are unchanged.
- Not decided here: the default reduction of a future release, the value of `beta` and of the
  warm-up (D-6, after P3-1), a learned decoder variance, and per-modality `reconstruction_loss` and
  `reconstruction_weight` in config (P2-2).
