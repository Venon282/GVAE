"""Unit tests for the beta-schedule registry pattern (spec §2.3, §11).

Mirrors the test style already used for the other self-registration
registries in this codebase (encoders, decoders, fusion, assemblers,
regularizers): registration, lookup, duplicate-registration and
unknown-name error paths, plus value-correctness checks for the two
built-in schedules, and a round-trip test proving
`resolveBetaSchedules` feeds `computeTotalRegularizationLoss`'s
existing `beta` argument with zero changes to either.
"""

import pytest
import torch
from torch import nn

from global_vae.losses.regularization import computeTotalRegularizationLoss
from global_vae.losses.regularizers.kl_standard_normal import KlStandardNormalRegularizer
from global_vae.training.beta_schedule_resolution import resolveBetaSchedules
from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.beta_schedules.constant import ConstantBetaSchedule
from global_vae.training.beta_schedules.cyclical_annealing import CyclicalAnnealingBetaSchedule
from global_vae.training.beta_schedules.linear_warmup import LinearWarmupBetaSchedule
from global_vae.training.beta_schedules.registry import (
    getBetaScheduleClass,
    listRegisteredBetaSchedules,
    registerBetaSchedule,
)


def test_constant_and_linear_warmup_are_registered_by_default() -> None:
    assert "constant" in listRegisteredBetaSchedules()
    assert "linear_warmup" in listRegisteredBetaSchedules()
    assert "cyclical_annealing" in listRegisteredBetaSchedules()
    assert getBetaScheduleClass("constant") is ConstantBetaSchedule
    assert getBetaScheduleClass("linear_warmup") is LinearWarmupBetaSchedule
    assert getBetaScheduleClass("cyclical_annealing") is CyclicalAnnealingBetaSchedule


def test_unknown_beta_schedule_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="does_not_exist"):
        getBetaScheduleClass("does_not_exist")


def test_duplicate_registration_raises_value_error() -> None:
    @registerBetaSchedule("dummy_beta_schedule_duplicate_check")
    class _First(AbstractBetaSchedule):
        def __call__(self, step: int) -> float:
            return 1.0

    with pytest.raises(ValueError, match="already registered"):

        @registerBetaSchedule("dummy_beta_schedule_duplicate_check")
        class _Second(AbstractBetaSchedule):
            def __call__(self, step: int) -> float:
                return 1.0


class TestConstantBetaSchedule:
    def test_default_value_is_one(self) -> None:
        schedule = ConstantBetaSchedule()
        assert schedule(0) == 1.0
        assert schedule(1000) == 1.0

    def test_custom_value_is_returned_at_every_step(self) -> None:
        schedule = ConstantBetaSchedule(value=0.25)
        for step in (0, 1, 500, 10_000):
            assert schedule(step) == 0.25


class TestLinearWarmupBetaSchedule:
    def test_starts_at_start_value(self) -> None:
        schedule = LinearWarmupBetaSchedule(warmup_steps=100, start_value=0.0, end_value=1.0)
        assert schedule(0) == pytest.approx(0.0)

    def test_reaches_end_value_at_warmup_steps(self) -> None:
        schedule = LinearWarmupBetaSchedule(warmup_steps=100, start_value=0.0, end_value=1.0)
        assert schedule(100) == pytest.approx(1.0)

    def test_holds_end_value_past_warmup_steps(self) -> None:
        schedule = LinearWarmupBetaSchedule(warmup_steps=100, start_value=0.0, end_value=1.0)
        assert schedule(500) == pytest.approx(1.0)

    def test_midpoint_is_linearly_interpolated(self) -> None:
        schedule = LinearWarmupBetaSchedule(warmup_steps=100, start_value=0.0, end_value=1.0)
        assert schedule(50) == pytest.approx(0.5)

    def test_negative_step_is_clamped_to_start_value(self) -> None:
        schedule = LinearWarmupBetaSchedule(warmup_steps=100, start_value=0.2, end_value=1.0)
        assert schedule(-10) == pytest.approx(0.2)

    def test_non_zero_start_and_end_values(self) -> None:
        schedule = LinearWarmupBetaSchedule(warmup_steps=10, start_value=0.1, end_value=0.9)
        assert schedule(0) == pytest.approx(0.1)
        assert schedule(5) == pytest.approx(0.5)
        assert schedule(10) == pytest.approx(0.9)

    def test_non_positive_warmup_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_steps"):
            LinearWarmupBetaSchedule(warmup_steps=0)
        with pytest.raises(ValueError, match="warmup_steps"):
            LinearWarmupBetaSchedule(warmup_steps=-5)


class TestCyclicalAnnealingBetaSchedule:
    def test_starts_each_cycle_at_start_value(self) -> None:
        schedule = CyclicalAnnealingBetaSchedule(period=100, start_value=0.0, end_value=1.0)
        assert schedule(0) == pytest.approx(0.0)
        assert schedule(100) == pytest.approx(0.0)  # start of the second cycle
        assert schedule(200) == pytest.approx(0.0)  # start of the third cycle

    def test_ramp_reaches_end_value_at_ramp_proportion(self) -> None:
        """ramp_proportion=0.5 over a period of 100 means the ramp completes at step 50."""
        schedule = CyclicalAnnealingBetaSchedule(
            period=100, ramp_proportion=0.5, start_value=0.0, end_value=1.0
        )
        assert schedule(50) == pytest.approx(1.0)

    def test_holds_end_value_for_the_rest_of_the_cycle(self) -> None:
        schedule = CyclicalAnnealingBetaSchedule(
            period=100, ramp_proportion=0.5, start_value=0.0, end_value=1.0
        )
        assert schedule(75) == pytest.approx(1.0)
        assert schedule(99) == pytest.approx(1.0)

    def test_midpoint_of_the_ramp_is_linearly_interpolated(self) -> None:
        schedule = CyclicalAnnealingBetaSchedule(
            period=100, ramp_proportion=0.5, start_value=0.0, end_value=1.0
        )
        assert schedule(25) == pytest.approx(0.5)  # halfway through a 50-step ramp

    def test_full_ramp_proportion_is_a_plain_sawtooth(self) -> None:
        schedule = CyclicalAnnealingBetaSchedule(
            period=100, ramp_proportion=1.0, start_value=0.0, end_value=1.0
        )
        assert schedule(99) == pytest.approx(0.99)
        assert schedule(100) == pytest.approx(0.0)  # new cycle restarts at start_value

    def test_num_cycles_holds_end_value_after_the_last_cycle(self) -> None:
        schedule = CyclicalAnnealingBetaSchedule(
            period=100, ramp_proportion=0.5, start_value=0.0, end_value=1.0, num_cycles=2
        )
        assert schedule(150) == pytest.approx(1.0)  # still inside cycle 2, past its own ramp
        assert schedule(200) == pytest.approx(1.0)  # would restart cycle 3 without num_cycles
        assert schedule(250) == pytest.approx(1.0)

    def test_negative_step_is_clamped_to_start_value(self) -> None:
        schedule = CyclicalAnnealingBetaSchedule(period=100, start_value=0.2, end_value=1.0)
        assert schedule(-10) == pytest.approx(0.2)

    def test_non_positive_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            CyclicalAnnealingBetaSchedule(period=0)
        with pytest.raises(ValueError, match="period"):
            CyclicalAnnealingBetaSchedule(period=-5)

    def test_ramp_proportion_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="ramp_proportion"):
            CyclicalAnnealingBetaSchedule(period=100, ramp_proportion=0.0)
        with pytest.raises(ValueError, match="ramp_proportion"):
            CyclicalAnnealingBetaSchedule(period=100, ramp_proportion=1.5)

    def test_non_positive_num_cycles_raises(self) -> None:
        with pytest.raises(ValueError, match="num_cycles"):
            CyclicalAnnealingBetaSchedule(period=100, num_cycles=0)


def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)).mean()


class TestResolveBetaSchedules:
    def test_resolves_every_schedule_at_a_given_step(self) -> None:
        schedules: dict[str, AbstractBetaSchedule] = {
            "z_shared": LinearWarmupBetaSchedule(warmup_steps=100, start_value=0.0, end_value=1.0),
            "z_private": ConstantBetaSchedule(value=0.1),
        }
        resolved = resolveBetaSchedules(schedules, step=50)
        assert resolved == {"z_shared": pytest.approx(0.5), "z_private": pytest.approx(0.1)}

    def test_resolved_output_feeds_compute_total_regularization_loss_unchanged(self) -> None:
        """The whole point: no changes needed to the aggregation function's `beta` argument."""
        regularizers = nn.ModuleDict(
            {
                "z_shared": KlStandardNormalRegularizer(),
                "z_private": KlStandardNormalRegularizer(),
            }
        )
        mu_shared, logvar_shared = torch.randn(4, 8), torch.randn(4, 8)
        mu_private, logvar_private = torch.randn(4, 8), torch.randn(4, 8)
        schedules: dict[str, AbstractBetaSchedule] = {
            "z_shared": LinearWarmupBetaSchedule(warmup_steps=100, start_value=0.0, end_value=1.0),
            "z_private": ConstantBetaSchedule(value=0.1),
        }
        resolved_beta = resolveBetaSchedules(schedules, step=50)

        loss = computeTotalRegularizationLoss(
            regularizers,
            {"z_shared": (mu_shared, logvar_shared), "z_private": (mu_private, logvar_private)},
            beta=resolved_beta,
        )
        expected = 0.5 * _kl(mu_shared, logvar_shared) + 0.1 * _kl(mu_private, logvar_private)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_empty_schedules_resolves_to_empty_dict(self) -> None:
        assert resolveBetaSchedules({}, step=0) == {}
