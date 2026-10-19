"""`CosineLr`, self-registered in the `training.callbacks` registry (roadmap P2-3).

Cosine decay of the learning rate over a fixed number of optimizer steps, with an optional
linear warm-up in front. Registered as `"cosine_lr"` (see `@registerCallback`), so it is
selected from `TrainingConfig.callbacks` by name like any other callback; see
`docs/adr/0033-top-k-checkpoints-and-learning-rate-schedules.md`.
"""

import logging
import math
from typing import TYPE_CHECKING

from global_vae.training.callbacks._lr_schedule import AbstractLrSchedule
from global_vae.training.callbacks.registry import registerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)


@registerCallback("cosine_lr")
class CosineLr(AbstractLrSchedule):
    """Decays each group's learning rate from `initial_lr` to `initial_lr * min_lr_ratio`.

    Step `s` (0-based) trains with:

    - `initial_lr * (s + 1) / warmup_steps` while `s < warmup_steps` (a linear warm-up, none
      when `warmup_steps` is `0`);
    - then `initial_lr * (min_lr_ratio + (1 - min_lr_ratio) * (1 + cos(pi * p)) / 2)` with `p
      = (s - warmup_steps) / (total_steps - warmup_steps)`, which is `initial_lr` at the end of
      the warm-up and reaches `initial_lr * min_lr_ratio` at `s = total_steps`;
    - and `initial_lr * min_lr_ratio` for every later step.

    The schedule owns the learning rate for the whole run: it overwrites it at every step, so
    it does not combine with `reduce_lr_on_plateau` (a warning says so when both are present).
    It acts on the learning rate (`acts_on_learning_rate`), so the run records the rate in its
    metrics (ADR 0028). Steps are optimizer steps, so with `accumulate_grad_batches=N` the
    `total_steps` of a run of `E` epochs of `B` batches is `E * ceil(B / N)`.
    """

    def __init__(self, total_steps: int, warmup_steps: int = 0, min_lr_ratio: float = 0.0) -> None:
        """Initialize the callback.

        Args:
            total_steps: Optimizer step at which the rate reaches its floor.
            warmup_steps: Number of linear warm-up steps at the start (`0`, the default, for
                none). Must be below `total_steps`.
            min_lr_ratio: The floor, as a fraction of `initial_lr`, in `[0, 1]` (`0.0`, the
                default, decays to zero).

        Raises:
            ValueError: If `total_steps` is not positive, if `warmup_steps` is negative or not
                below `total_steps`, or if `min_lr_ratio` is not in `[0, 1]`.
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}.")
        if not 0 <= warmup_steps < total_steps:
            raise ValueError(
                f"warmup_steps must be in [0, total_steps), got {warmup_steps} "
                f"with total_steps={total_steps}."
            )
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}.")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

    def factor(self, step: int) -> float:
        """Return the multiplier of `initial_lr` described in the class docstring.

        Args:
            step: The 0-based optimizer step the rate is for.

        Returns:
            A number in `[min_lr_ratio, 1]`.
        """
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = min((step - self.warmup_steps) / (self.total_steps - self.warmup_steps), 1.0)
        cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def onTrainBegin(self, trainer: "Trainer") -> None:
        """Warn about another callback that also sets the rate, then start the schedule.

        Args:
            trainer: The `Trainer` instance running this training run.
        """
        others = sorted(
            type(callback).__name__
            for callback in trainer.callbacks
            if callback is not self and callback.acts_on_learning_rate
        )
        if others:
            logger.warning(
                "CosineLr sets the learning rate at every step, so %s, which also changes it, "
                "will have its changes overwritten.",
                others,
            )
        super().onTrainBegin(trainer)
