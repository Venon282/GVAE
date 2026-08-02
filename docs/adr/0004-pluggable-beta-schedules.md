# 0004 — Pluggable beta-weighting schedules

**Status:** accepted
**Date:** 2026-08-01

## Context

Spec §2.3 requires that a latent space's beta weight be expressible as
a single constant, a global schedule (linear warm-up, cyclical
annealing), or a per-latent-space value or schedule, all "through the
same config mechanism, not mutually exclusive code paths." Before this
change, `losses.regularization.computeTotalRegularizationLoss` (and
`GlobalVae.computeRegularizationLoss`, which delegates to it) only
accepted an already-resolved `beta: dict[str, float] | float`. Nothing
in the codebase computed that value as a function of training step: a
caller wanting warm-up annealing would have had to hand-roll it
outside the framework, or a future `trainer.py` would have been
tempted to hardcode an `if schedule == "linear_warmup": ...` branch,
exactly the "mutually exclusive code paths" spec §2.3 rules out. This
also matters operationally, not just architecturally: with beta fixed
at a constant (typically `1.0`), the KL term is trivial to drive to
zero early in training (`mu -> 0`, `logvar -> 0`), which can collapse
the posterior to the prior before the decoder has learned to use `z`
at all (posterior collapse / KL vanishing, Bowman et al., 2016).
Warm-up annealing is the standard mitigation, so the schedule
abstraction is needed for the upcoming first real training run, not
only for spec compliance.

## Decision

Add `training/beta_schedules/`, following the exact registry pattern
already used for encoders, decoders, fusion, assemblers, and latent
regularizers:

- `AbstractBetaSchedule`: a plain Python `ABC` (not an `nn.Module`,
  since resolving a step to a float is not an autograd computation),
  with a single `__call__(self, step: int) -> float`.
- `registry.py`: `registerBetaSchedule` / `getBetaScheduleClass` /
  `listRegisteredBetaSchedules`, mirroring
  `losses/regularizers/registry.py`.
- `constant.py`: `ConstantBetaSchedule`, making "no annealing" an
  explicit instance of the same mechanism instead of a special case.
- `linear_warmup.py`: `LinearWarmupBetaSchedule(warmup_steps,
  start_value, end_value)`, ramping linearly then holding
  `end_value`. Rejects `warmup_steps <= 0` at construction.
- `training/beta_schedule_resolution.py`:
  `resolveBetaSchedules(schedules, step) -> dict[str, float]`, the
  single bridge between a set of schedules and the plain `beta` dict
  `computeTotalRegularizationLoss` already accepts. Lives as a sibling
  of `beta_schedules/`, not inside it, mirroring the existing
  relationship between `losses/regularization.py` (the aggregator) and
  `losses/regularizers/` (the strategy subpackage it aggregates over).

Deliberately **not** changed: `losses/regularization.py`,
`models/global_vae.py`, and the existing `beta: dict[str, float] |
float = 1.0` signature everywhere it appears. A caller that wants
schedules resolves them for the current step via
`resolveBetaSchedules(...)` and passes the resulting plain dict as
`beta`, exactly as it would pass any hand-written `dict[str, float]`.
This is what spec §2.3 means by "orthogonal": the schedule mechanism
has zero coupling to `GlobalVae` or to which regularization strategy a
latent space uses, and every existing caller of
`computeRegularizationLoss` (including every test in
`tests/integration/test_en_l1_dn_default.py` and
`tests/integration/test_regularization_loss.py`) keeps working
unmodified, verified by running the full test suite after this change
(80 passed, no regressions).

## Consequences

- `beta: dict[str, float] | float = 1.0` remains the one way any
  caller supplies weights to `computeTotalRegularizationLoss` /
  `GlobalVae.computeRegularizationLoss`. Schedules are strictly
  additive: nothing under `losses/` changed, at the source level or in
  behavior.
- `training/trainer.py` (still not built, see `training/NOTE.md`) is
  expected to hold one `AbstractBetaSchedule` instance per latent
  space that needs annealing, call `resolveBetaSchedules(...)` once
  per step, and pass the result straight into
  `model.computeRegularizationLoss(latent_params, beta=...)`. No
  change to either `training/beta_schedules/` or
  `losses/regularization.py` will be needed to wire that up.
- Adding a new schedule (cyclical annealing, spec §11) is purely
  additive: one new file under `training/beta_schedules/`, registered
  in its `__init__.py`, with zero changes to `GlobalVae` or
  `losses/regularization.py`.
- `training/NOTE.md` updated: the raw-loop-vs-Lightning open question
  it used to cite is resolved (raw loop for now, per spec §10);
  `trainer.py` itself is simply not built yet, a separate milestone
  from this change.
- `tests/integration/test_beta_schedules.py` covers the registry
  pattern (mirroring `test_regularizers.py`), both schedules' value
  correctness including edge cases (negative step, exact warm-up
  boundary, holding past warm-up, non-positive `warmup_steps`), and a
  round-trip test proving `resolveBetaSchedules`'s output feeds
  `computeTotalRegularizationLoss` correctly with no changes to that
  function.
- `mypy --strict` and `ruff check` / `ruff format --check` pass clean
  on every new file. Running `mypy` on the full package surfaced 17
  pre-existing errors in `OneDCnnEncoder.py`, `OneDCnnDecoder.py`, and
  `utils/builders.py`, unrelated to this change and left untouched
  here.
