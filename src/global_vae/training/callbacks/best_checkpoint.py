"""`BestCheckpointCallback`, self-registered in the `training.callbacks` registry.

Spec §10 "Reproducibility", ADR 0022. Wraps the plain, `Trainer`-independent save/load functions in
`training/checkpoint.py` (`saveCheckpoint`/`loadCheckpoint`, unchanged and
unmoved): this module owns only the "save on metric improvement" policy, not
the file format itself.

`CheckpointCallback` (the other checkpoint callback) lives in its own sibling
module, `checkpoint.py`, one class per file per spec §10's "Modularity" rule
(see that module's docstring for why checkpoint *format* code and checkpoint
*callback* code are deliberately separate modules too).

Registered as `"best_checkpoint"` (see `@registerCallback` below), so it is
selected from `TrainingConfig.callbacks` by name, exactly like any other
callback (`docs/adr/0022-callback-registry.md`); see
`docs/adr/0006-reproducibility-seed-and-checkpointing.md` and
`docs/adr/0007-best-checkpoint-callback.md` for why this and
`CheckpointCallback` are two separate callbacks rather than one.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import registerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)


@registerCallback("best_checkpoint")
class BestCheckpointCallback(TrainerCallback):
    """Saves a checkpoint only when a monitored metric improves, via `onEpochEnd`.

    This is the "give me the best model" callback: unlike
    `training.callbacks.checkpoint.CheckpointCallback` (which saves on a
    schedule, for resuming an interrupted run), this one saves purely based
    on whether `monitor` improved this epoch, always overwriting the
    **same** file, so `path` is always exactly the best model seen so far,
    no pruning logic needed. Load it at any time via `loadCheckpoint(path,
    model=...)` (or `Trainer.loadCheckpoint(path)`) to evaluate or
    visualize the best model without retraining.

    Typically monitors a validation metric (e.g. `"val/loss/total"`,
    which requires `val_dataloader` to be passed to `Trainer.fit`).
    Monitoring a training metric instead is allowed but usually less
    useful for model selection: training loss tends to keep improving
    even as the model overfits, so "best training loss" is often close
    to just "the last epoch".
    """

    def __init__(
        self,
        path: str | Path,
        monitor: str = "val/loss/total",
        mode: str = "min",
        config: Any = None,
    ) -> None:
        """Initialize the callback.

        Args:
            path: File path for the single best-so-far checkpoint.
                Every improvement overwrites this same file.
            monitor: Metric key to track, as it appears in the epoch
                metrics dict `Trainer.fit` builds (e.g.
                `"val/loss/total"`, `"train/loss/reconstruction"`).
            mode: `"min"` (default; lower is better, e.g. a loss) or
                `"max"` (higher is better, e.g. an accuracy or a
                custom metric a `TrainerCallback` might add).
            config: Forwarded to `Trainer.saveCheckpoint` every time
                this callback saves.

        Raises:
            ValueError: If `mode` is not `"min"` or `"max"`.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")

        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.config = config
        self.best_value: float | None = None

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Save a checkpoint if `self.monitor` improved this epoch.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch that just finished (0-based).
            metrics: This epoch's metrics; must contain `self.monitor`.

        Raises:
            KeyError: If `self.monitor` is not present in `metrics`
                (e.g. monitoring a `"val/..."` key without passing
                `val_dataloader` to `Trainer.fit`).
        """
        if self.monitor not in metrics:
            raise KeyError(
                f"BestCheckpointCallback is monitoring '{self.monitor}', but it is not present "
                f"in this epoch's metrics ({sorted(metrics)}). If you are monitoring a "
                f"'val/...' key, make sure Trainer.fit was called with a val_dataloader."
            )

        value = metrics[self.monitor]
        improved = self.best_value is None or (
            value < self.best_value if self.mode == "min" else value > self.best_value
        )
        if not improved:
            return

        self.best_value = value
        trainer.saveCheckpoint(self.path, config=self.config)
        logger.info(
            "New best %s=%.6f at epoch %d, saved to '%s'.", self.monitor, value, epoch, self.path
        )
