"""Shared mechanics of the learning-rate schedule callbacks (roadmap P2-3, ADR 0033).

`LinearWarmupLr` and `CosineLr` set the learning rate of every parameter group of
`trainer.optimizer` to `initial_lr * factor(step)`, where `step` is the optimizer step the
rate is for and `initial_lr` is the group's rate when the schedule first saw it. Two properties
follow from writing the rate as a function of the step and nothing else:

- a resumed run needs no schedule state: `Trainer.loadCheckpoint` restores `global_step`,
  and `onTrainBegin` sets the rate that step trains with;
- the schedule is exact however many batches an epoch has, because the step is the optimizer
  step (`accumulate_grad_batches` included), not the epoch.

`initial_lr` is kept in the optimizer's own parameter groups, under the same key and with the
same `setdefault` that `torch.optim.lr_scheduler` uses, so it is saved in the optimizer state
and survives a resume: the schedule scales the rate the run started with, not the (already
scheduled) rate a checkpoint holds.
"""

from typing import TYPE_CHECKING, ClassVar

from global_vae.training.callbacks.base import TrainerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

INITIAL_LR_KEY = "initial_lr"


class AbstractLrSchedule(TrainerCallback):
    """Base of the callbacks that set the learning rate as a function of the step.

    A subclass implements `factor`, and may override `isActive` to stop touching the rate
    after a point (a warm-up leaves it alone once it is over, so that another callback, a
    plateau reduction for example, can take over).
    """

    acts_on_learning_rate: ClassVar[bool] = True

    def factor(self, step: int) -> float:
        """Return the multiplier of `initial_lr` for optimizer step `step`.

        Args:
            step: The 0-based optimizer step the rate is for.

        Returns:
            A non-negative number.

        Raises:
            NotImplementedError: Always; a subclass defines the schedule.
        """
        raise NotImplementedError

    def isActive(self, step: int) -> bool:
        """Return whether the schedule sets the rate for optimizer step `step`.

        Args:
            step: The 0-based optimizer step the rate is for.

        Returns:
            `True` by default: the schedule owns the rate for the whole run.
        """
        return True

    def onTrainBegin(self, trainer: "Trainer") -> None:
        """Remember each group's starting rate and set the rate of the first step to run.

        Args:
            trainer: The `Trainer` instance running this training run. After a resume its
                `global_step` is the restored one, so the rate set is the right one for it.
        """
        for group in trainer.optimizer.param_groups:
            group.setdefault(INITIAL_LR_KEY, group["lr"])
        self._apply(trainer, trainer.global_step)

    def onStepEnd(self, trainer: "Trainer", step: int, metrics: dict[str, float]) -> None:
        """Set the rate of the next step, now that `step` has been taken.

        Args:
            trainer: The `Trainer` instance running this training run.
            step: The optimizer step that just finished.
            metrics: Unused; present only to match `TrainerCallback`'s signature.
        """
        self._apply(trainer, step + 1)

    def _apply(self, trainer: "Trainer", step: int) -> None:
        """Set every group's rate to `initial_lr * factor(step)` if the schedule is active.

        Args:
            trainer: The `Trainer` whose optimizer is adjusted in place.
            step: The optimizer step the rate is for.
        """
        if not self.isActive(step):
            return
        multiplier = self.factor(step)
        for group in trainer.optimizer.param_groups:
            group["lr"] = group[INITIAL_LR_KEY] * multiplier
