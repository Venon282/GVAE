"""Constant beta schedule (spec §2.3): the explicit no-annealing case."""

from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.beta_schedules.registry import registerBetaSchedule


@registerBetaSchedule("constant")
class ConstantBetaSchedule(AbstractBetaSchedule):
    """Returns the same beta value at every training step.

    Makes "no annealing" expressible through the exact same mechanism
    as every other schedule (spec §2.3), rather than a special case a
    caller has to bypass the schedule machinery for.
    """

    def __init__(self, value: float = 1.0) -> None:
        """Initialize the schedule.

        Args:
            value: The constant beta value returned at every step.
        """
        self.value = value

    def __call__(self, step: int) -> float:
        """Return the constant value, ignoring `step`.

        Args:
            step: Ignored.

        Returns:
            `self.value`.
        """
        return self.value
