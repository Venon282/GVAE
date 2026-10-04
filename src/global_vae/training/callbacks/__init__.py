"""callbacks subpackage of global_vae.training (spec §10, ADR 0022).

Every training-loop hook (`TrainerCallback`, `base.py`) and every concrete
callback *selected by registry name from config* lives here, self-registered
in the unified `registry.py` registry, exactly like every other pluggable
strategy in this codebase (encoders, decoders, fusion, assemblers,
regularizers, beta schedules): `TrainingConfig.callbacks` (`config/training.py`)
picks any number of them by name, and adding a new one (built-in or the
caller's own) never requires touching `Trainer` or the config layer.

Importing this package registers every built-in registry-driven callback
(`"checkpoint"`, `"best_checkpoint"`, `"early_stopping"`,
`"reduce_lr_on_plateau"`) via each module's own `@registerCallback`
decorator. A `@registerX(...)` decorator only runs once its module is
imported; without these imports, `getCallbackClass("checkpoint")` would raise
`KeyError` even though `checkpoint.py` exists on disk.

Checkpoint *format* code (`saveCheckpoint`/`loadCheckpoint`/`CheckpointMetadata`)
stays in `training/checkpoint.py`, unmoved: only the two `TrainerCallback`s that
call into it on a schedule (`CheckpointCallback`, `checkpoint.py`) or on metric
improvement (`BestCheckpointCallback`, `best_checkpoint.py`) live here, one
class per file (spec §10 "Modularity").

Scope note: experiment loggers (`training/loggers/`, `AbstractExperimentLogger`)
are **not** part of this package or this registry. A logger is a journalling
service that happens to subclass `TrainerCallback` so `Trainer.callbacks` can
call its hooks uniformly (ADR 0008, predating this subpackage); it is not a
callback *strategy* a run picks by registry name the way checkpointing or
early stopping are. Loggers keep their own, separate `training.loggers`
subpackage, their own `registerLogger`/`getLoggerClass` registry, and their
own `TrainingConfig.loggers` config field; see `docs/adr/0022-callback-registry.md`
for why the two stay apart.
"""

import global_vae.training.callbacks.best_checkpoint  # noqa: F401
import global_vae.training.callbacks.checkpoint  # noqa: F401
import global_vae.training.callbacks.early_stopping  # noqa: F401
import global_vae.training.callbacks.reduce_lr_on_plateau  # noqa: F401
