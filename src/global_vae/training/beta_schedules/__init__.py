"""beta_schedules subpackage of global_vae.training.

Importing this package registers every built-in beta schedule
(`constant`, `linear_warmup`, `cyclical_annealing`) via their
`@registerBetaSchedule` decorators. A `@registerX(...)` decorator only
runs once its module is imported; without these imports,
`getBetaScheduleClass("constant")` would raise `KeyError` even though
`constant.py` exists on disk.
"""

import global_vae.training.beta_schedules.constant  # noqa: F401
import global_vae.training.beta_schedules.cyclical_annealing  # noqa: F401
import global_vae.training.beta_schedules.linear_warmup  # noqa: F401
