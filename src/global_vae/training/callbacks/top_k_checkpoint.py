"""`TopKCheckpointCallback`, self-registered in the `training.callbacks` registry (roadmap P2-3).

Keeps the `k` best checkpoints of a run by a monitored metric, each in its own file. The
extension ADR 0007 listed as not built: `BestCheckpointCallback` overwrites one file with the
single best model, this one keeps a small sorted list of `(value, path)` pairs and evicts the
worst once there are more than `k`. Registered as `"top_k_checkpoint"` (see
`@registerCallback`); see `docs/adr/0033-top-k-checkpoints-and-learning-rate-schedules.md`.
"""

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import registerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)


@registerCallback("top_k_checkpoint")
class TopKCheckpointCallback(TrainerCallback):
    """Saves a checkpoint when `monitor` is among the `k` best values so far, via `onEpochEnd`.

    Every kept checkpoint is a full training checkpoint (`Trainer.saveCheckpoint`), so any of
    them can be evaluated, or resumed from. When a new epoch beats the worst of the `k`
    kept, its checkpoint is written and the worst one is deleted. A value equal to the worst
    does not displace it, so on ties the earlier epoch is kept. A non-finite value is never
    kept.

    It lists its files in its state (`stateDict`), so a resumed run keeps the same top `k`.
    Use `best_checkpoint` when only the single best model is wanted, in a fixed path.
    """

    def __init__(
        self,
        directory: str | Path,
        k: int = 3,
        monitor: str = "val/loss/total",
        mode: str = "min",
        filename_pattern: str = "top_k_epoch_{epoch:04d}.pt",
        config: Any = None,
    ) -> None:
        """Initialize the callback.

        Args:
            directory: Directory the checkpoints are written to (created if missing).
            k: How many checkpoints to keep.
            monitor: Metric key to rank by, as it appears in the epoch metrics dict (for
                example `"val/loss/total"`, which needs a `val_dataloader`).
            mode: `"min"` (default; lower is better) or `"max"`.
            filename_pattern: `str.format` pattern of each file name, receiving `epoch` (the
                epoch that just finished). It must make the name depend on `epoch`, or two
                epochs would write the same file.
            config: Forwarded to `Trainer.saveCheckpoint` every time this callback saves.

        Raises:
            ValueError: If `k` is not positive, if `mode` is not `"min"` or `"max"`, or if
                `filename_pattern` gives the same name for two epochs.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        if filename_pattern.format(epoch=0) == filename_pattern.format(epoch=1):
            raise ValueError(
                f"filename_pattern must depend on 'epoch', got '{filename_pattern}': two "
                f"epochs would write the same file."
            )
        self.directory = Path(directory)
        self.k = k
        self.monitor = monitor
        self.mode = mode
        self.filename_pattern = filename_pattern
        self.config = config
        self._entries: list[tuple[float, Path]] = []

    @property
    def entries(self) -> list[tuple[float, Path]]:
        """The kept `(value, path)` pairs, best first (a copy)."""
        return list(self._entries)

    @property
    def paths(self) -> list[Path]:
        """The kept checkpoint files, best first."""
        return [path for _, path in self._entries]

    def stateDict(self) -> dict[str, Any]:
        """Return the kept `(value, path)` pairs, so a resumed run keeps the same top `k`."""
        return {"entries": [[value, str(path)] for value, path in self._entries]}

    def loadStateDict(self, state: dict[str, Any]) -> None:
        """Restore the kept pairs. See `stateDict`."""
        self._entries = [(float(value), Path(path)) for value, path in state["entries"]]

    def _isBetter(self, value: float, other: float) -> bool:
        """Whether `value` ranks strictly ahead of `other` under `self.mode`."""
        return value < other if self.mode == "min" else value > other

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Save a checkpoint if `self.monitor` is among the `k` best values so far.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch that just finished (0-based).
            metrics: This epoch's metrics; must contain `self.monitor`.

        Raises:
            KeyError: If `self.monitor` is not present in `metrics` (for example monitoring
                a `"val/..."` key without passing `val_dataloader` to `Trainer.fit`).
        """
        if self.monitor not in metrics:
            raise KeyError(
                f"TopKCheckpointCallback is monitoring '{self.monitor}', but it is not present "
                f"in this epoch's metrics ({sorted(metrics)}). If you are monitoring a "
                f"'val/...' key, make sure Trainer.fit was called with a val_dataloader."
            )
        value = metrics[self.monitor]
        if not math.isfinite(value):
            logger.warning(
                "TopKCheckpointCallback: '%s' is %s at epoch %d, not keeping a checkpoint.",
                self.monitor,
                value,
                epoch,
            )
            return
        if len(self._entries) >= self.k and not self._isBetter(value, self._entries[-1][0]):
            return

        path = self.directory / self.filename_pattern.format(epoch=epoch)
        self._entries.append((value, path))
        # A stable sort, so an earlier epoch stays ahead of a later one with an equal value.
        self._entries.sort(key=lambda entry: entry[0], reverse=self.mode == "max")
        evicted = self._entries[self.k :]
        del self._entries[self.k :]
        trainer.saveCheckpoint(path, config=self.config)
        for _, evicted_path in evicted:
            evicted_path.unlink(missing_ok=True)
        logger.info(
            "TopKCheckpointCallback: %s=%.6f at epoch %d is among the %d best, saved to '%s'.",
            self.monitor,
            value,
            epoch,
            self.k,
            path,
        )
