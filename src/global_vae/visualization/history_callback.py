"""In-memory metrics collection for visualization, without going through a file-based
logger (`training/loggers/`) first.

`HistoryCallback` is a `TrainerCallback` (`training/callbacks.py`), exactly like every
`AbstractExperimentLogger`, but it stores metrics in plain Python lists instead of
writing anywhere. `Trainer.history` already covers the epoch-level case on its own
(`docs/adr/0005-training-loop.md`); `HistoryCallback` additionally covers the
step-level case (`Trainer` does not retain per-step history itself), producing
exactly the shape `visualization.loss_curves.plotStepCurves` expects.
"""

from typing import TYPE_CHECKING

from global_vae.training.callbacks import TrainerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer


class HistoryCallback(TrainerCallback):
    """Accumulates every step's and epoch's metrics into plain in-memory lists.

    Attributes:
        step_history: One dict per step, each `{"step": step, **metrics}`.
        epoch_history: One dict per epoch, each `{"epoch": epoch, **metrics}`.
            Duplicates the information already in `Trainer.history`
            (which has no `"epoch"` key of its own, using list position
            instead); kept here too for a uniform shape with
            `step_history` and for direct use with
            `visualization.loss_curves.plotStepCurves`-style step-keyed
            plotting if ever wanted at epoch granularity.
    """

    def __init__(self) -> None:
        self.step_history: list[dict[str, float]] = []
        self.epoch_history: list[dict[str, float]] = []

    def onStepEnd(self, trainer: "Trainer", step: int, metrics: dict[str, float]) -> None:
        """Record this step's metrics. See `TrainerCallback.onStepEnd`."""
        self.step_history.append({"step": float(step), **metrics})

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Record this epoch's metrics. See `TrainerCallback.onEpochEnd`."""
        self.epoch_history.append({"epoch": float(epoch), **metrics})
