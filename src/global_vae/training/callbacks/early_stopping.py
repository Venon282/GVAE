"""Early-stopping callback (spec §10).

Plain `torch` ships no early-stopping class of its own (unlike
`torch.optim.lr_scheduler.ReduceLROnPlateau`, which this project's own
`ReduceLrOnPlateau`, `reduce_lr_on_plateau.py`, mirrors directly): stopping a
training loop once a metric plateaus is instead the standard job of a
callback in every framework that does ship one (Keras' `EarlyStopping`,
PyTorch Lightning's `EarlyStopping`). Rather than inventing a second,
differently-shaped plateau algorithm, `EarlyStopping` reuses the exact same
`PlateauTracker` (`_plateau.py`) `ReduceLrOnPlateau` uses, "the torch one",
i.e. `ReduceLROnPlateau`'s own `mode`/`threshold`/`threshold_mode`/`cooldown`/
`patience` semantics, and reacts to a plateau by requesting that `Trainer.fit`
stop instead of lowering the learning rate.

Registered as `"early_stopping"` (see `@registerCallback`), so it is selected
from `TrainingConfig.callbacks` by name like any other callback.
"""

import logging
from typing import TYPE_CHECKING

from global_vae.training.callbacks._plateau import PlateauTracker
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import registerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)


@registerCallback("early_stopping")
class EarlyStopping(TrainerCallback):
    """Sets `trainer.should_stop = True` once `self.monitor` stops improving.

    `Trainer.fit` checks `trainer.should_stop` right after every epoch's
    `onEpochEnd` callbacks have all run (so a callback listed after this one
    for the same epoch still runs before training actually stops) and ends
    the run before starting the next epoch; see `Trainer.fit`'s own
    docstring. `should_stop` stays `True` afterward: constructing a new
    `Trainer` (or explicitly resetting `trainer.should_stop = False`) is
    required to train further.
    """

    def __init__(
        self,
        monitor: str = "val/loss/total",
        mode: str = "min",
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
    ) -> None:
        """Initialize the callback.

        Every argument here matches `ReduceLrOnPlateau`'s own identically-named
        one (both share `PlateauTracker`): only the reaction to a plateau
        differs (stop training, instead of lowering the learning rate).

        Args:
            monitor: Metric key to track, as it appears in the epoch metrics
                dict `Trainer.fit` builds (e.g. `"val/loss/total"`, the
                default; requires `val_dataloader` to be passed to
                `Trainer.fit`).
            mode: `"min"` (default; lower is better, e.g. a loss) or
                `"max"` (higher is better).
            patience: Number of epochs with no improvement after which
                training is stopped. Matches `ReduceLROnPlateau`'s own
                default of `10`.
            threshold: Threshold for measuring a new optimum, per
                `threshold_mode`. Matches `ReduceLROnPlateau`'s own default
                of `1e-4`.
            threshold_mode: `"rel"` (default) or `"abs"`; see
                `PlateauTracker`.
            cooldown: Number of epochs to wait before resuming normal
                patience-counting after training would otherwise have
                stopped. `0` (default) matches `ReduceLROnPlateau`'s own
                default; a cooldown is only meaningful here if
                `trainer.should_stop` is reset to `False` externally
                (e.g. by a caller inspecting `Trainer.history` between
                `fit()` calls), since `Trainer.fit` itself ends the run
                the moment a plateau is reported.

        Raises:
            ValueError: Delegated to `PlateauTracker`, if `mode`/
                `threshold_mode`/`patience`/`cooldown` are invalid.
        """
        self.monitor = monitor
        self._tracker = PlateauTracker(
            mode=mode,
            threshold=threshold,
            threshold_mode=threshold_mode,
            patience=patience,
            cooldown=cooldown,
        )

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Update the plateau tracker and request a stop if it has just plateaued.

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
                f"EarlyStopping is monitoring '{self.monitor}', but it is not present in this "
                f"epoch's metrics ({sorted(metrics)}). If you are monitoring a 'val/...' key, "
                f"make sure Trainer.fit was called with a val_dataloader."
            )

        if self._tracker.step(metrics[self.monitor]):
            trainer.should_stop = True
            logger.info(
                "EarlyStopping: no improvement in '%s' for %d epoch(s), stopping at epoch %d.",
                self.monitor,
                self._tracker.patience,
                epoch,
            )
