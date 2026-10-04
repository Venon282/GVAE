"""Reduce-learning-rate-on-plateau callback (spec §10), mirroring torch's own scheduler.

`torch.optim.lr_scheduler.ReduceLROnPlateau` is itself not a `TrainerCallback`:
it is a plain object a caller `.step(metric)`s manually once per epoch, outside
`Trainer` entirely, with no hook into `Trainer.fit`'s own epoch loop or its
`"val/loss/total"`-style metrics dict. `ReduceLrOnPlateau` here is that exact
same algorithm (same constructor arguments, same defaults, same
mode/threshold/threshold_mode/cooldown/patience/min_lr/eps semantics),
reimplemented as a `TrainerCallback` so it plugs into `Trainer.fit` the same
way every other callback does: `onEpochEnd` reads `self.monitor` from the
epoch's own metrics dict and lowers `trainer.optimizer`'s learning rate on a
plateau, with no separate manual `.step(...)` call needed anywhere in the
training loop.

Registered as `"reduce_lr_on_plateau"` (see `@registerCallback`), so it is
selected from `TrainingConfig.callbacks` by name like any other callback.
"""

import logging
from typing import TYPE_CHECKING

from global_vae.training.callbacks._plateau import PlateauTracker
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import registerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)


@registerCallback("reduce_lr_on_plateau")
class ReduceLrOnPlateau(TrainerCallback):
    """Lowers `trainer.optimizer`'s learning rate once `self.monitor` stops improving.

    Every constructor argument and every piece of the underlying
    plateau-detection algorithm (`PlateauTracker`, `_plateau.py`) matches
    `torch.optim.lr_scheduler.ReduceLROnPlateau`'s own; only `monitor` is new,
    since this callback reads its metric from `Trainer.fit`'s epoch metrics
    dict instead of receiving it as a direct `.step(metric)` argument.
    """

    def __init__(
        self,
        monitor: str = "val/loss/total",
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float | list[float] = 0.0,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the callback.

        Args:
            monitor: Metric key to track, as it appears in the epoch
                metrics dict `Trainer.fit` builds (e.g. `"val/loss/total"`,
                the default; requires `val_dataloader` to be passed to
                `Trainer.fit`).
            mode: `"min"` (default; lower is better, e.g. a loss) or
                `"max"` (higher is better).
            factor: Multiplicative factor applied to the learning rate on
                a plateau (`new_lr = old_lr * factor`). Must be in
                `(0, 1)`; matches `ReduceLROnPlateau`'s own default of
                `0.1`.
            patience: Number of epochs with no improvement after which the
                learning rate is reduced. Matches `ReduceLROnPlateau`'s
                own default of `10`.
            threshold: Threshold for measuring a new optimum, per
                `threshold_mode`. Matches `ReduceLROnPlateau`'s own
                default of `1e-4`.
            threshold_mode: `"rel"` (default) or `"abs"`; see
                `PlateauTracker`.
            cooldown: Number of epochs to wait before resuming normal
                patience-counting after a reduction. Matches
                `ReduceLROnPlateau`'s own default of `0`.
            min_lr: A scalar floor applied to every parameter group, or a
                list with one floor per `trainer.optimizer.param_groups`
                entry. Matches `ReduceLROnPlateau`'s own default of `0`.
            eps: Minimal learning-rate decay: an update is skipped if the
                difference between the old and new learning rate is
                smaller than `eps`. Matches `ReduceLROnPlateau`'s own
                default of `1e-8`.

        Raises:
            ValueError: If `factor` is not in `(0, 1)`, or (delegated to
                `PlateauTracker`) if `mode`/`threshold_mode`/`patience`/
                `cooldown` are invalid.
        """
        if not (0.0 < factor < 1.0):
            raise ValueError(f"factor must be in (0, 1), got {factor}.")
        self.monitor = monitor
        self.factor = factor
        self.min_lr = min_lr
        self.eps = eps
        self._tracker = PlateauTracker(
            mode=mode,
            threshold=threshold,
            threshold_mode=threshold_mode,
            patience=patience,
            cooldown=cooldown,
        )

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Update the plateau tracker and reduce the learning rate if it has just plateaued.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch that just finished (0-based).
            metrics: This epoch's metrics; must contain `self.monitor`.

        Raises:
            KeyError: If `self.monitor` is not present in `metrics` (e.g.
                monitoring a `"val/..."` key without passing
                `val_dataloader` to `Trainer.fit`).
        """
        if self.monitor not in metrics:
            raise KeyError(
                f"ReduceLrOnPlateau is monitoring '{self.monitor}', but it is not present in "
                f"this epoch's metrics ({sorted(metrics)}). If you are monitoring a 'val/...' "
                f"key, make sure Trainer.fit was called with a val_dataloader."
            )

        if self._tracker.step(metrics[self.monitor]):
            self._reduceLr(trainer, epoch)

    def _reduceLr(self, trainer: "Trainer", epoch: int) -> None:
        """Lower every parameter group's learning rate by `self.factor`, floored at `min_lr`.

        Args:
            trainer: The `Trainer` whose `optimizer` is adjusted in place.
            epoch: Current epoch index, used only for the log message.
        """
        param_groups = trainer.optimizer.param_groups
        floors = self.min_lr if isinstance(self.min_lr, list) else [self.min_lr] * len(param_groups)
        if len(floors) != len(param_groups):
            raise ValueError(
                f"ReduceLrOnPlateau: min_lr has {len(floors)} value(s) but the optimizer has "
                f"{len(param_groups)} parameter group(s); pass either a single shared float or "
                f"one value per parameter group."
            )

        for group, floor in zip(param_groups, floors, strict=True):
            old_lr = float(group["lr"])
            new_lr = max(old_lr * self.factor, floor)
            if old_lr - new_lr > self.eps:
                group["lr"] = new_lr
                logger.info(
                    "ReduceLrOnPlateau: epoch %d, reducing learning rate from %.6g to %.6g.",
                    epoch,
                    old_lr,
                    new_lr,
                )
