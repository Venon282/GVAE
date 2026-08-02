"""Bridge from per-latent-space beta schedules to a plain beta value (spec §2.3).

This is the only integration point between `AbstractBetaSchedule`
(`beta_schedules/`, a function of training step) and
`losses.regularization.computeTotalRegularizationLoss` /
`GlobalVae.computeRegularizationLoss` (which only ever see an
already-resolved `dict[str, float] | float`). Neither of those two
needs to change, or know schedules exist, to support them: a caller
(typically the training loop, `training/trainer.py`, still deferred,
see `training/NOTE.md`) resolves the active schedules for the current
step into a plain dict here, then passes that dict as `beta` exactly
as it would pass any other hand-written `dict[str, float]`. This keeps
"which regularizer" and "what beta schedule" fully orthogonal, per
spec §2.3, and keeps `beta: dict[str, float] | float = 1.0` as the
one, unchanged way every caller supplies weights.
"""

from global_vae.training.beta_schedules.base import AbstractBetaSchedule


def resolveBetaSchedules(schedules: dict[str, AbstractBetaSchedule], step: int) -> dict[str, float]:
    """Evaluate every latent space's beta schedule at a given training step.

    Args:
        schedules: Latent space name -> `AbstractBetaSchedule` instance.
            A latent space with no entry here simply has no schedule;
            the caller is expected to fall back to a plain constant
            (or omit it from the merged `beta` dict, in which case
            `computeTotalRegularizationLoss`'s own default weight of
            `1.0` applies) rather than this function inventing one.
        step: Current training step, forwarded to every schedule.

    Returns:
        Latent space name -> resolved beta value at `step`. Ready to
        pass directly as the `beta` argument of
        `GlobalVae.computeRegularizationLoss` /
        `losses.regularization.computeTotalRegularizationLoss`, either
        as-is (if every latent space has a schedule) or merged with an
        explicit `dict[str, float]` for the latent spaces that do not.
    """
    return {latent_name: schedule(step) for latent_name, schedule in schedules.items()}
