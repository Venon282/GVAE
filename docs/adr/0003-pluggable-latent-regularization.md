# 0003 — Pluggable latent regularization strategies

**Status:** accepted
**Date:** 2026-07-31

## Context

Spec §10 is explicit: `LatentSpace.klDivergence` (or an equivalent
KL-only computation) must never be hardcoded as the model class's only
regularization path; it must go through an `AbstractLatentRegularizer`
registry (spec §2.3), exactly like Fusion and Assemblers. ADR 0002's
rewrite of `GlobalVae` around an explicit `RoutingGraph` did not
address this: `GlobalVae.computeKlLoss` still called
`losses.kl.computeTotalKlLoss`, which called `LatentSpace.klDivergence`
directly and unconditionally for every latent space. This was exactly
the anti-pattern spec §10 rules out, and the `losses/regularizers/`
subpackage the repository layout (spec §8) already reserves for it did
not exist yet.

## Decision

- Introduce `losses/regularizers/`: `AbstractLatentRegularizer` (an
  `nn.Module` ABC, `forward(mu, logvar) -> Tensor` per-sample), its
  registry (`registerRegularizer` / `getRegularizerClass`, mirroring
  `fusion/registry.py`), and the default strategy,
  `kl_standard_normal.KlStandardNormalRegularizer`, whose computation
  is exactly what `LatentSpace.klDivergence` used to do.
- `GlobalVae.__init__` gains `regularizer_strategies` (latent space
  name -> registry name) and `regularizer_kwargs`, and builds
  `self.regularizers`, an `nn.ModuleDict` with **one entry per latent
  space** in the routing graph (unlike `self.fusions`, which only has
  entries for latent spaces fed by more than one encoder). Any latent
  space absent from `regularizer_strategies` defaults to
  `"kl_standard_normal"`, so the common case needs no extra config.
- `GlobalVae.createSingleLatent` gains a `regularizer_strategy`
  parameter (default `"kl_standard_normal"`) for the `EN-L1-DN` case.
- `losses/kl.py` is superseded by `losses/regularization.py`:
  `computeTotalRegularizationLoss(regularizers, latent_params, beta)`
  sums each latent space's own regularizer output instead of calling
  `LatentSpace.klDivergence` directly. `regularizers` is typed as
  `nn.ModuleDict`, not `Mapping[str, AbstractLatentRegularizer]`:
  verified with `mypy --strict` that `nn.ModuleDict` is not a nominal
  subtype of `Mapping`/`dict`, even though it is needed here (rather
  than a plain `dict`) so PyTorch registers each strategy's parameters.
- `GlobalVae.computeKlLoss` is renamed to `computeRegularizationLoss`.
- `LatentSpace.klDivergence` is removed: keeping it alongside
  `KlStandardNormalRegularizer` would leave two implementations of the
  same formula that could silently drift apart.

## Consequences

- Default behavior is unchanged: every latent space is still
  regularized by KL-to-standard-normal unless configured otherwise, so
  no existing config needs to change.
- Adding a new regularization strategy (MMD, free-bits KL, a learned
  prior, spec §7/§11) is now purely additive: one new file under
  `losses/regularizers/`, registered in its `__init__.py`, with zero
  changes to `GlobalVae` or `LatentSpace`, matching the "no touching
  the core" rule (spec §10, §12).
- `tests/integration/test_en_l1_dn_default.py` updated its two
  `computeKlLoss` call sites to `computeRegularizationLoss`, and gained
  two tests specifically covering the new wiring: that the default is
  `kl_standard_normal` when unspecified, and that an explicit
  `regularizer_strategy` is actually used (a dummy always-zero
  regularizer), not merely accepted without effect.
- This is not a beta/annealing-schedule decision (spec §11, still
  open): `beta` in `computeTotalRegularizationLoss` keeps the same
  `float | dict[str, float]` signature as before. Only *which*
  strategy computes each latent space's raw penalty is now pluggable.
- `losses/kl.py` and `LatentSpace.klDivergence` should be deleted from
  the repository (see CHANGELOG "Removed").
