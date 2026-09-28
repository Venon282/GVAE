# 0020: Additional fusion strategies (`moe`, `concat_mlp`, `cross_attention`)

**Status:** accepted
**Date:** 2026-09-28

## Context

Spec §4 lists four fusion strategies for a latent space fed by more than one encoder:
Product-of-Experts (PoE), Mixture-of-Experts (MoE), concatenation plus MLP, and
cross-attention / transformer fusion. Only `poe` existed (`fusion/poe.py`); the other three
were named in `AbstractFusion`'s own docstring (`fusion/base.py`) and in the spec table, but
not built. Milestone 2 of spec §6.1 (the paired signal + image setup) is the first place
Fusion is exercised for real, and a single strategy makes it impossible to compare how the
choice of fusion affects reconstruction quality or missing-modality robustness (spec §5).

Two further gaps showed up while planning this work. `poe.py` had no direct test of its own
(only dummy `AbstractFusion` subclasses in unrelated tests), even though spec §10 explicitly
requires a registry unit test for fusion. And `GlobalVae` had never been exercised with a
real fusion module other than through those dummies.

## Decision

Three strategies are added, one file each (spec §10), each self-registering through
`@registerFusion(name)` and imported from `fusion/__init__.py` so it actually registers.

| Registry name | Class | Handles missing modalities natively (`handlesMissingModalities`) |
|---|---|---|
| `poe` | `ProductOfExperts` (already existed) | Yes |
| `moe` | `MixtureOfExperts` | Yes |
| `concat_mlp` | `ConcatMlpFusion` | No (explicit imputation scheme, see below) |
| `cross_attention` | `CrossAttentionFusion` | Yes |

No file outside `fusion/` needed to change: `GlobalVae`, `buildModelFromConfig`, and
`FusionConfig` already resolve a fusion by registry name and forward its kwargs unchanged.

### `moe`: closed-form moment matching

A true MMVAE mixture is a mixture of Gaussians, not a Gaussian, but every consumer of a
Fusion output (`LatentSpace.reparameterize`, `AbstractLatentRegularizer.forward`) expects a
single `(mu, logvar)` pair, and `AbstractFusion.forward` is contractually a deterministic
function of `params`. `MixtureOfExperts` therefore returns the single Gaussian that matches
the mixture's first two moments:

```
mu_fused  = sum_i(w_i * mu_i)
var_fused = sum_i(w_i * (var_i + mu_i^2)) - mu_fused^2
```

This keeps the intended MoE behaviour (fused variance grows when experts disagree, unlike PoE,
which only ever sharpens) without sampling inside Fusion. It is a documented adaptation to the
interface, in the same spirit as `MmdRegularizer`'s. Weights default to uniform over the
*active* experts and are renormalized per call, so a missing modality just drops its component;
an explicit `modality_weights` mapping overrides this (an unlisted modality falls back to `1.0`
before renormalization). With one active expert the result is exactly that expert.

### `concat_mlp`: an explicit imputation scheme, not a silent patch

Concatenation has one fixed slot per modality, so the input width must be known at
construction: `modality_dims` (modality name to encoder output width) is required, and slot
order is the mapping's insertion order, never re-sorted. Spec §5 requires that this strategy's
inability to skip a slot be handled by "an explicit masking/imputation scheme" and documented
rather than hidden. The scheme implemented is:

- a missing modality's `(mu, logvar)` slot is filled with `0` (the standard-normal prior's own
  parameters);
- by default (`use_presence_mask=True`) one `1.0`/`0.0` presence scalar per modality is
  appended, so the MLP can tell "missing and imputed" apart from "present and genuinely near
  zero".

`handlesMissingModalities` stays `False`: this is a scheme the class provides, not the free
tolerance the other three have. A modality name not declared in `modality_dims` raises
`KeyError` rather than being ignored.

### `cross_attention`: one token per active modality

Each active modality's `(mu, logvar)` is concatenated, linearly projected to `d_model`
(default `latent_dim`), optionally offset by a learned per-modality embedding
(`known_modalities`), passed through `num_layers` `nn.TransformerEncoderLayer`s, mean-pooled
over the token axis, and projected to the fused `(mu, logvar)`. Mean pooling makes the output
defined for any non-empty token count, which is what makes missing modalities "simply omitted
tokens" (spec §4). If `known_modalities` is omitted the fusion is purely content-based and
accepts any modality name; if it is given, an unlisted name raises `ValueError`.
`num_heads` must divide `d_model`, checked at construction with a clear message.

### Constructor arguments come from the caller

`ConcatMlpFusion` and `CrossAttentionFusion` need `latent_dim` (and `modality_dims` for the
former), while `poe` and `moe` accept neither. `buildModelFromConfig` deliberately does not
auto-fill these into `fusion.kwargs`, unlike `latent_dim` for encoders and decoders: every
encoder and decoder needs it, but fusion strategies have unrelated signatures, so an
unconditional injection would break `poe`/`moe` with an unexpected-keyword `TypeError`. This
matches how regularizer kwargs (`free_bits`, `kernel`, ...) are already passed through
untouched.

## Consequences

- Fusion is now a real, comparable axis of the configuration space: switching strategy is a
  one-line config change (`latent.fusion.strategy`), no code change.
- New tests in `tests/integration/test_fusion.py` cover the registry (registration, lookup,
  duplicate and unknown names), value correctness against hand-computed formulas for `poe` and
  `moe`, missing-modality behaviour, gradient flow for all four, and each new strategy wired
  through the real `GlobalVae.createSingleLatent` path, with and without a modality missing.
- `moe`'s moment-matched output is a unimodal approximation: it cannot represent a genuinely
  bimodal posterior. A sampling-based MMVAE objective would need `GlobalVae.forward` to handle
  several candidate posteriors, which is out of scope here.
- `concat_mlp` trained without modality dropout has never seen an imputed slot; enabling
  `modality_dropout_p` (spec §5) is what teaches it to use the presence mask.
- The per-assignment `residual` flag that spec §4 and the illustrative YAML in spec §9 mention
  is still not implemented for any fusion strategy, `poe` included. It is left for a separate
  decision rather than being added to only some strategies.
- Learned mixture weights for `moe`, and a learned query token instead of mean pooling for
  `cross_attention`, are natural extensions that were not needed to meet spec §4.
