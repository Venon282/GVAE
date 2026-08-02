"""Linear warm-up beta schedule (spec §2.3, §11).

The standard mitigation for posterior collapse / KL vanishing: a low
initial beta lets the decoder learn to use `z` for reconstruction
before the regularization term pulls the posterior toward the prior
(Bowman et al., 2016). Ramps linearly from `start_value` to
`end_value` over `warmup_steps` steps, then holds `end_value`
indefinitely.
"""

from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.beta_schedules.registry import registerBetaSchedule


@registerBetaSchedule("linear_warmup")
class LinearWarmupBetaSchedule(AbstractBetaSchedule):
    """Linearly ramps beta from `start_value` to `end_value` over `warmup_steps`.

    Attributes:
        warmup_steps: Number of steps over which the ramp happens.
        start_value: Beta at `step <= 0`.
        end_value: Beta once `step >= warmup_steps`, held afterward.
    """

    def __init__(self, warmup_steps: int, start_value: float = 0.0, end_value: float = 1.0) -> None:
        """Initialize the schedule.

        Args:
            warmup_steps: Number of steps over which beta ramps from
                `start_value` to `end_value`. Must be positive.
            start_value: Beta at `step <= 0`.
            end_value: Beta once `step >= warmup_steps`, held constant
                afterward.

        Raises:
            ValueError: If `warmup_steps` is not positive.
        """
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}.")
        self.warmup_steps = warmup_steps
        self.start_value = start_value
        self.end_value = end_value

    def __call__(self, step: int) -> float:
        """Linearly interpolate beta at `step`, clamped to `[0, warmup_steps]`.

        Args:
            step: Current training step. Values below `0` are treated
                as `0`; values at or above `warmup_steps` are treated
                as `warmup_steps` (i.e. `end_value` is held
                indefinitely past warm-up).

        Returns:
            The interpolated beta value.
        """
        progress = min(max(step, 0), self.warmup_steps) / self.warmup_steps
        return self.start_value + progress * (self.end_value - self.start_value)
