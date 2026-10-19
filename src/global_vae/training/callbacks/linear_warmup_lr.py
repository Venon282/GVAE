"""`LinearWarmupLr`, self-registered in the `training.callbacks` registry (roadmap P2-3).

Raises the learning rate linearly over the first optimizer steps, then leaves it alone.
Registered as `"linear_warmup_lr"` (see `@registerCallback`), so it is selected from
`TrainingConfig.callbacks` by name like any other callback; see
`docs/adr/0033-top-k-checkpoints-and-learning-rate-schedules.md`.
"""

from global_vae.training.callbacks._lr_schedule import AbstractLrSchedule
from global_vae.training.callbacks.registry import registerCallback


@registerCallback("linear_warmup_lr")
class LinearWarmupLr(AbstractLrSchedule):
    """Ramps each group's learning rate from `initial_lr / warmup_steps` up to `initial_lr`.

    Step `s` (0-based) trains with `initial_lr * (s + 1) / warmup_steps` while `s <
    warmup_steps`, so the last warm-up step already trains with the full rate. After that the
    callback does nothing, which is what lets `reduce_lr_on_plateau` take over afterwards. It
    does not decay: for a decay after the warm-up use `cosine_lr` with `warmup_steps`.

    It acts on the learning rate (`acts_on_learning_rate`), so the run records the rate in its
    metrics (ADR 0028). Steps are optimizer steps, so with `accumulate_grad_batches=N` a
    warm-up of `warmup_steps` covers `N * warmup_steps` batches.
    """

    def __init__(self, warmup_steps: int) -> None:
        """Initialize the callback.

        Args:
            warmup_steps: Number of optimizer steps of the ramp.

        Raises:
            ValueError: If `warmup_steps` is not positive.
        """
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}.")
        self.warmup_steps = warmup_steps

    def factor(self, step: int) -> float:
        """Return `(step + 1) / warmup_steps`, capped at 1.

        Args:
            step: The 0-based optimizer step the rate is for.

        Returns:
            The multiplier of `initial_lr`.
        """
        return min((step + 1) / self.warmup_steps, 1.0)

    def isActive(self, step: int) -> bool:
        """Return whether `step` is still a warm-up step.

        Args:
            step: The 0-based optimizer step the rate is for.

        Returns:
            `True` while `step < warmup_steps`.
        """
        return step < self.warmup_steps
