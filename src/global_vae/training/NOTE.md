# Deferred

`trainer.py` itself is not yet implemented (a separate milestone from
this project's MVP task list, not blocked on any open question
anymore). The open question in spec §11 this package used to be
blocked on ("raw PyTorch loops or PyTorch Lightning") is resolved: a
raw PyTorch loop for now, migrating to Lightning (or Lightning Fabric
as an intermediate step) once the architecture stabilizes (spec §10).

`beta_schedules/` does not depend on `trainer.py` existing and is
already implemented: see `beta_schedules/base.py`,
`beta_schedule_resolution.py`, and
`docs/adr/0004-pluggable-beta-schedules.md`. It resolves a
per-latent-space beta value for a given training step; `trainer.py`,
once built, is expected to call `resolveBetaSchedules(...)` each step
and pass the result straight into
`GlobalVae.computeRegularizationLoss(..., beta=...)`, with no changes
needed to either.
