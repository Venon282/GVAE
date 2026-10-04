"""`CheckpointCallback`, self-registered in the `training.callbacks` registry.

Spec §10 "Reproducibility", ADR 0022. Wraps the plain, `Trainer`-independent save/load functions in
`training/checkpoint.py` (`saveCheckpoint`/`loadCheckpoint`, unchanged and
unmoved: see that module's own docstring for why checkpoint *format* code and
checkpoint *callback* code are deliberately two different files): this
module owns only the "when does a checkpoint get written during a
`Trainer.fit` run" policy, not the file format itself.

`BestCheckpointCallback` (the other checkpoint callback) lives in its own
sibling module, `best_checkpoint.py`, one class per file per spec §10's
"Modularity" rule, the same convention `fusion/`, `assemblers/`, `heads/`,
`losses/regularizers/`, and `data/transforms/` already follow.

Registered as `"checkpoint"` (see `@registerCallback` below), so it is
selected from `TrainingConfig.callbacks` by name, exactly like any other
callback (`docs/adr/0022-callback-registry.md`); see
`docs/adr/0006-reproducibility-seed-and-checkpointing.md` and
`docs/adr/0007-best-checkpoint-callback.md` for why this and
`BestCheckpointCallback` are two separate callbacks rather than one.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import registerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)


@registerCallback("checkpoint")
class CheckpointCallback(TrainerCallback):
    """Periodically saves a training checkpoint via `onEpochEnd`, for resuming an interrupted run.

    This callback's job is narrower than it might first look: it
    exists to let a **long training run be resumed close to where it
    was** after an interruption (a crash, a cluster preemption, a
    manual stop), with `global_step`, optimizer momentum, and RNG state
    intact, so the run does not have to restart from scratch. It is
    **not** a way to recover the best model for evaluation: the most
    recently saved epoch is not necessarily the best one (validation
    performance can get worse in later epochs), and `keep_last_n`
    prunes by save order, not by quality. Resuming from a "best" epoch
    instead of the most recent one would also throw away every epoch
    of progress made after it, defeating the point of resuming at all.

    For "give me the best model to evaluate or visualize", use
    `training.callbacks.best_checkpoint.BestCheckpointCallback` instead
    (or alongside this one; they solve different problems and are not
    mutually exclusive).

    The running example `training/callbacks/base.py`'s own docstring already
    used ("a checkpointer only overrides `onEpochEnd`"), made concrete:
    delegates to `Trainer.saveCheckpoint` (which in turn calls
    `training.checkpoint.saveCheckpoint` with the trainer's own model,
    optimizer, step, epoch, and history), so the checkpoint format is identical
    whether saved through this callback or called directly.
    """

    def __init__(
        self,
        directory: str | Path,
        every_n_epochs: int = 1,
        config: Any = None,
        keep_last_n: int | None = None,
        filename_pattern: str = "checkpoint_epoch_{epoch:04d}.pt",
    ) -> None:
        """Initialize the callback.

        Args:
            directory: Directory to save checkpoints into (created if
                missing).
            every_n_epochs: Save every `N` epochs (`1`, the default,
                saves after every epoch).
            config: Forwarded unchanged to `Trainer.saveCheckpoint`
                every time this callback saves (spec §10: "config
                snapshotted with every run").
            keep_last_n: If given, delete older checkpoints beyond the
                most recently saved `keep_last_n`, so disk usage does
                not grow unbounded over a long run. `None` (default)
                keeps every checkpoint ever saved by this callback.
                Deletes by save order (oldest first); keeping the best
                `N` by some validation metric instead is a natural
                future extension, not built here, since "best" requires
                choosing a metric and a comparison direction this
                callback has no way to know generically.
            filename_pattern: `str.format` pattern for each
                checkpoint's filename, receiving `epoch` as a keyword
                argument (the epoch index that just finished, matching
                `TrainerCallback.onEpochEnd`'s own `epoch` argument).

        Raises:
            ValueError: If `every_n_epochs` is not positive, or if
                `keep_last_n` is given and not positive.
        """
        if every_n_epochs <= 0:
            raise ValueError(f"every_n_epochs must be positive, got {every_n_epochs}.")
        if keep_last_n is not None and keep_last_n <= 0:
            raise ValueError(f"keep_last_n must be positive when given, got {keep_last_n}.")

        self.directory = Path(directory)
        self.every_n_epochs = every_n_epochs
        self.config = config
        self.keep_last_n = keep_last_n
        self.filename_pattern = filename_pattern
        self._saved_paths: list[Path] = []

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Save a checkpoint if `epoch` lands on an `every_n_epochs` boundary.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch that just finished (0-based).
            metrics: Unused; present only to match `TrainerCallback`'s
                signature.
        """
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        path = self.directory / self.filename_pattern.format(epoch=epoch)
        trainer.saveCheckpoint(path, config=self.config)
        self._saved_paths.append(path)

        if self.keep_last_n is not None:
            while len(self._saved_paths) > self.keep_last_n:
                stale_path = self._saved_paths.pop(0)
                stale_path.unlink(missing_ok=True)
