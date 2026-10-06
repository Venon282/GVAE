"""Unit tests for `training.callbacks._plateau.PlateauTracker` (ADR 0022).

Value-correctness checks for the shared plateau-detection algorithm behind
both `ReduceLrOnPlateau` and `EarlyStopping` (spec §10's own testing
checklist item: "unit tests for ... value correctness", mirrored here from
`test_beta_schedules.py`'s equivalent treatment of the beta-schedule
strategies). Each callback's own integration behavior (reading its monitor
from `Trainer`'s epoch metrics, reacting to a plateau) is covered in its own
file (`test_reduce_lr_on_plateau.py`, `test_early_stopping.py`).
"""

import pytest

from global_vae.training.callbacks._plateau import PlateauTracker


class TestConstructorValidation:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            PlateauTracker(mode="sideways")

    def test_invalid_threshold_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_mode"):
            PlateauTracker(threshold_mode="sideways")

    def test_negative_patience_raises(self) -> None:
        with pytest.raises(ValueError, match="patience"):
            PlateauTracker(patience=-1)

    def test_negative_cooldown_raises(self) -> None:
        with pytest.raises(ValueError, match="cooldown"):
            PlateauTracker(cooldown=-1)

    def test_initial_best_is_infinite_in_the_direction_that_needs_improving(self) -> None:
        assert PlateauTracker(mode="min").best == float("inf")
        assert PlateauTracker(mode="max").best == float("-inf")


class TestPatienceZero:
    """`patience=0`: a plateau fires on the very first non-improving epoch."""

    def test_min_mode_plateaus_immediately_on_no_improvement(self) -> None:
        tracker = PlateauTracker(mode="min", threshold=1e-4, patience=0)
        assert tracker.step(10.0) is False  # first value: always an "improvement"
        assert tracker.step(9.0) is False  # improves
        assert tracker.step(9.0) is True  # no improvement: plateaus immediately
        assert tracker.step(8.0) is False  # improves again after the reset


class TestPatienceGreaterThanZero:
    def test_plateau_fires_once_bad_epochs_exceed_patience(self) -> None:
        tracker = PlateauTracker(mode="min", threshold=0.0, patience=2)
        results = [tracker.step(value) for value in [5.0, 5.0, 5.0, 5.0, 4.0]]
        assert results == [False, False, False, True, False]

    def test_max_mode_tracks_improvement_in_the_opposite_direction(self) -> None:
        tracker = PlateauTracker(mode="max", threshold=1e-4, patience=1)
        results = [tracker.step(value) for value in [1.0, 2.0, 2.0, 2.0]]
        assert results == [False, False, False, True]


class TestThresholdMode:
    def test_abs_mode_requires_an_absolute_margin(self) -> None:
        tracker = PlateauTracker(mode="min", threshold=0.5, threshold_mode="abs", patience=0)
        assert tracker.step(10.0) is False
        assert tracker.step(9.6) is True  # only 0.4 better than 10: not enough, plateaus
        assert tracker.best == 10.0  # non-improving step never updates best
        assert tracker.step(9.4) is False  # 0.6 better than 10: enough

    def test_rel_mode_requires_a_relative_margin(self) -> None:
        tracker = PlateauTracker(mode="min", threshold=0.1, threshold_mode="rel", patience=0)
        assert tracker.step(100.0) is False
        assert tracker.step(91.0) is True  # only 9% better: not enough (needs > 10%)
        assert tracker.step(89.0) is False  # 11% better: enough


class TestCooldown:
    def test_cooldown_suppresses_bad_epoch_counting_after_a_plateau(self) -> None:
        tracker = PlateauTracker(mode="min", threshold=0.0, patience=0, cooldown=2)
        # Every value is identical, so every step after the first is "non-improving";
        # the plateau at step 2 opens a 2-epoch cooldown that swallows steps 3 and 4,
        # so the next plateau does not fire again until step 5.
        results = [tracker.step(5.0) for _ in range(5)]
        assert results == [False, True, False, False, True]
