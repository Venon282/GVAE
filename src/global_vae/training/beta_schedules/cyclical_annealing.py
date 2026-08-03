"""Cyclical annealing beta schedule (spec §2.3: named alongside "linear warm-up").

A single warm-up (`linear_warmup.py`) only ever gives the decoder a
low-beta window once, at the very start of training. Fu et al. (2019,
"Cyclical Annealing Schedule: A Simple Approach to Mitigating KL
Vanishing") repeat that low-to-high ramp in cycles throughout the
whole run instead: each cycle starts back at `start_value`, giving the
decoder repeated fresh opportunities to make full use of `z` before
the posterior is pulled toward the prior again, which the paper shows
mitigates KL vanishing more effectively than a single warm-up window
for models trained long enough to have several cycles.
"""

from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.beta_schedules.registry import registerBetaSchedule


@registerBetaSchedule("cyclical_annealing")
class CyclicalAnnealingBetaSchedule(AbstractBetaSchedule):
    """Repeats a linear ramp-then-hold pattern every `period` steps.

    Within each cycle: linearly ramps from `start_value` to
    `end_value` over the first `ramp_proportion` fraction of the
    cycle, then holds `end_value` for the rest of the cycle, exactly
    like `LinearWarmupBetaSchedule` holds past its own `warmup_steps`.
    The cycle then restarts at `start_value`. `ramp_proportion=1.0`
    (ramp for the entire cycle, hold for none of it) makes this
    schedule behave as a plain sawtooth; the paper's own default is
    `0.5` (half ramp, half hold).

    Attributes:
        period: Number of steps in one full cycle.
        ramp_proportion: Fraction of each cycle spent ramping, in
            `(0, 1]`. The remaining `1 - ramp_proportion` fraction
            holds `end_value`.
        start_value: Beta at the start of every cycle.
        end_value: Beta once the ramp completes, held for the rest of
            the cycle.
        num_cycles: If given, the schedule holds `end_value`
            indefinitely once `num_cycles` full cycles have elapsed,
            instead of restarting a new cycle forever. `None` (default)
            cycles for the entire training run.
    """

    def __init__(
        self,
        period: int,
        ramp_proportion: float = 0.5,
        start_value: float = 0.0,
        end_value: float = 1.0,
        num_cycles: int | None = None,
    ) -> None:
        """Initialize the schedule.

        Args:
            period: Number of steps in one full cycle. Must be
                positive.
            ramp_proportion: Fraction of each cycle spent ramping from
                `start_value` to `end_value`. Must be in `(0, 1]`.
            start_value: Beta at the start of every cycle.
            end_value: Beta once the ramp completes within a cycle.
            num_cycles: Number of cycles to run before holding
                `end_value` indefinitely. `None` (default) means
                "cycle forever".

        Raises:
            ValueError: If `period` is not positive, if
                `ramp_proportion` is not in `(0, 1]`, or if
                `num_cycles` is given and not positive.
        """
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}.")
        if not (0.0 < ramp_proportion <= 1.0):
            raise ValueError(f"ramp_proportion must be in (0, 1], got {ramp_proportion}.")
        if num_cycles is not None and num_cycles <= 0:
            raise ValueError(f"num_cycles must be positive when given, got {num_cycles}.")
        self.period = period
        self.ramp_proportion = ramp_proportion
        self.start_value = start_value
        self.end_value = end_value
        self.num_cycles = num_cycles

    def __call__(self, step: int) -> float:
        """Resolve beta at `step`, cycling every `period` steps.

        Args:
            step: Current training step. Negative values are clamped
                to `0`, matching `LinearWarmupBetaSchedule`.

        Returns:
            The beta value for this step: `end_value` once
            `num_cycles` cycles have elapsed (if `num_cycles` is set),
            otherwise the ramp-then-hold value within the current
            cycle.
        """
        clamped_step = max(step, 0)
        if self.num_cycles is not None and clamped_step >= self.period * self.num_cycles:
            return self.end_value

        position_in_cycle = clamped_step % self.period
        ramp_length = self.ramp_proportion * self.period
        if position_in_cycle >= ramp_length:
            return self.end_value

        progress = position_in_cycle / ramp_length
        return self.start_value + progress * (self.end_value - self.start_value)
