# 0021: Residual connection around a fusion strategy (`fusion.residual`)

**Status:** accepted
**Date:** 2026-09-28

## Context

Spec §4 says each fusion assignment "can optionally use residual connections (a config flag
alongside it)", and the illustrative YAML of spec §9 spells it `residual: true` next to
`strategy: poe`. The spec's glossary also lists a "residual flag" as part of what determines a
model instance. The flag was never built: not for `poe`, and not for the three strategies added
in ADR 0020, which named it as a bullet left for a separate decision.

The spec does not say what a residual connection *is* for a fusion. That is the real decision
here, and it is not obvious, because a residual connection is normally written `y = x + F(x)`,
where `F` outputs a correction. Every fusion strategy in this codebase does the opposite: it
returns an **absolute** posterior `(mu, logvar)`, not a delta.

## Decision

A single wrapper, `fusion.residual.ResidualFusion`, adds the connection to **any**
`AbstractFusion`, so all four strategies (and any future one) behave identically with respect
to the flag. The skip path is the parameter-free mean of the active experts, and the strategy
proposes a correction relative to it, scaled by a learned gate:

```
skip  = mean over the active experts of (mu_i, logvar_i)
fused = skip + gate * (strategy(params) - skip)
```

There is one scalar gate for `mu` and one for `logvar`, both initialized to `0.0` (the ReZero /
LayerScale initialization). This yields the two properties that make a connection residual:

- **identity at initialization:** the fused posterior is exactly the mean of the active experts,
  whichever strategy is wrapped;
- **nothing lost by enabling it:** `gate = 1` reproduces the wrapped strategy exactly.

The flag is per latent space, like the strategy itself (spec §4: fusion is chosen per latent
space): `GlobalVae(..., fusion_residual={"z": True})`, `createSingleLatent(...,
fusion_residual=True)`, and `residual: true` in `FusionConfig`, spelled exactly as in spec §9.
It defaults to off, and off leaves the strategy unwrapped, so every existing model, config and
checkpoint is unchanged. `GlobalVae` rejects the flag on a latent space that has no fusion
(unknown, or fed by fewer than two encoders), at construction and before any module is built.

`ResidualFusion` is not registered under `@registerFusion`: like `ComposeTransform`, it is a
combinator over an already-built instance, not a strategy selected by name.

## Alternatives considered

- **Literal `strategy(x) + mean(x)`.** The standard form, and the simplest. Rejected because
  every strategy's output is an absolute posterior, so it counts the experts twice: two identical
  `poe` experts of mean `m` fuse to `2m` (checked numerically while designing this). Any
  interpretation of the fused value as a posterior over the experts' shared variable breaks.
- **A residual defined separately inside each strategy** (a skip inside the MLP, inside the
  Transformer, and something for `poe`/`moe`). It has no single meaning for the closed-form
  strategies, which have nothing to skip over, and every future strategy would have to invent
  its own. The flag would exist on some strategies and mean different things on others.
- **A fixed convex blend** `(1 - w) * strategy + w * skip`. Well-defined for every strategy,
  but `w` is a hyperparameter to tune rather than something learned, and it has no identity
  at initialization, so it is not a residual in the usual sense.

## Consequences

- **This is an interpretation.** The spec names the flag but does not define it, so this ADR
  fixes one definition. If the intended meaning was a different one, only `residual.py` and its
  tests change: the flag, the config field, and the `GlobalVae` wiring are independent of it.
- **Enabling the flag is an architecture change.** The strategy moves under `inner` and two gate
  parameters appear, so a checkpoint saved with the flag off does not load into a model built
  with it on, and vice versa.
- **The wrapped strategy trains through the gate.** With gates at `0`, the strategy's own
  parameters receive zero gradient on the very first step, and the gates receive a non-zero one
  and open from there. Adam is insensitive to the gradient's scale, so this is harmless in
  practice; plain SGD starts the strategy slowly. Measured on a toy regression with a wrapped
  `concat_mlp` (learning rate `1e-2`, 60 steps): with Adam the strategy's weights move from the
  second step and the loss falls from 1.33 to 0.003; with SGD they move about 100 times less
  and the loss only reaches 1.05. `ResidualFusion(initial_gate=...)`
  exposes the starting value; it is not surfaced in `FusionConfig`, to keep the config a plain
  flag as the spec describes, and can be added if a use for it appears.
- **`poe` and `moe` gain two learnable scalars when the flag is on.** They are otherwise
  parameter-free. This is inherent to a learned gate, not specific to them.
- **The skip path needs the experts and the fused posterior to share one shape**, which the
  single-latent architecture already guarantees. A strategy that changes dimensionality (a
  `concat_mlp` whose `latent_dim` differs from its experts' width) raises a clear `ValueError`
  at `forward`, since there is nowhere for the skip to land.
- **Missing modalities are unaffected.** The skip is a mean over whichever experts are present,
  so `handlesMissingModalities` is inherited from the wrapped strategy unchanged.
