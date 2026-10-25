"""Unit tests for `evaluation.streaming_metrics` (roadmap P3-6).

A streaming metric must give the value of the pooled metric of `evaluation.metrics`, whatever the
way the data is cut into batches. The references are computed in `float64`, since the pooled
functions sum in the dtype of their input and would be the less accurate side of a comparison.
"""

import math

import pytest
import torch

from global_vae.evaluation.metrics import (
    DEFAULT_RECONSTRUCTION_METRICS,
    computeMae,
    computeMse,
    computePearsonR,
    computeR2,
    computeRmse,
)
from global_vae.evaluation.streaming_metrics import (
    STREAMING_RECONSTRUCTION_METRICS,
    StreamingMae,
    StreamingMetric,
    StreamingMse,
    StreamingPearsonR,
    StreamingR2,
    StreamingRmse,
    buildStreamingMetrics,
)

_POOLED = {
    "mse": computeMse,
    "rmse": computeRmse,
    "mae": computeMae,
    "r2": computeR2,
    "pearson_r": computePearsonR,
}


def _pair(
    rows: int = 240, width: int = 6, offset: float = 0.0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """A target and a noisy reconstruction of it, in `float64`."""
    generator = torch.Generator().manual_seed(seed)
    target = (torch.randn(rows, width, generator=generator) * 3.0 + offset).double()
    reconstruction = target + 0.7 * torch.randn(rows, width, generator=generator).double()
    return reconstruction, target


def _stream(
    metric: StreamingMetric, reconstruction: torch.Tensor, target: torch.Tensor, sizes: list[int]
) -> float:
    """Feed `metric` the rows in chunks of `sizes` and return its value."""
    start = 0
    for size in sizes:
        metric.update(reconstruction[start : start + size], target[start : start + size])
        start += size
    assert start == reconstruction.shape[0]
    return metric.compute()


class TestRegistry:
    def test_names_and_order_are_those_of_the_pooled_metrics(self) -> None:
        assert list(STREAMING_RECONSTRUCTION_METRICS) == list(DEFAULT_RECONSTRUCTION_METRICS)

    def test_every_entry_builds_a_streaming_metric(self) -> None:
        for name, factory in STREAMING_RECONSTRUCTION_METRICS.items():
            assert isinstance(factory(), StreamingMetric), name

    def test_build_gives_fresh_instances_each_time(self) -> None:
        first, second = buildStreamingMetrics(), buildStreamingMetrics()
        assert list(first) == list(STREAMING_RECONSTRUCTION_METRICS)
        reconstruction, target = _pair(rows=10)
        first["mse"].update(reconstruction, target)
        assert first["mse"].count == reconstruction.numel()
        assert second["mse"].count == 0


@pytest.mark.parametrize("name", list(_POOLED))
class TestAgreesWithThePooledMetric:
    def test_one_batch(self, name: str) -> None:
        reconstruction, target = _pair()
        value = _stream(STREAMING_RECONSTRUCTION_METRICS[name](), reconstruction, target, [240])
        assert value == pytest.approx(_POOLED[name](reconstruction, target), rel=1e-12)

    @pytest.mark.parametrize("sizes", [[120, 120], [1] * 240, [7] * 34 + [2], [200, 30, 10]])
    def test_any_way_of_cutting_the_data_into_batches(self, name: str, sizes: list[int]) -> None:
        reconstruction, target = _pair()
        value = _stream(STREAMING_RECONSTRUCTION_METRICS[name](), reconstruction, target, sizes)
        assert value == pytest.approx(_POOLED[name](reconstruction, target), rel=1e-10)

    def test_a_large_offset_does_not_lose_the_digits(self, name: str) -> None:
        """A mean of 1e8 against a spread of 3: `sum(x ** 2) - n * mean ** 2` is off by 0.2%."""
        reconstruction, target = _pair(offset=1e8)
        value = _stream(
            STREAMING_RECONSTRUCTION_METRICS[name](), reconstruction, target, [50, 90, 100]
        )
        assert value == pytest.approx(_POOLED[name](reconstruction, target), rel=1e-8)

    def test_a_float32_pair_agrees_up_to_the_rounding_of_float32(self, name: str) -> None:
        reconstruction, target = _pair()
        value = _stream(
            STREAMING_RECONSTRUCTION_METRICS[name](),
            reconstruction.float(),
            target.float(),
            [100, 140],
        )
        assert value == pytest.approx(_POOLED[name](reconstruction, target), rel=1e-5)

    def test_images_are_flattened(self, name: str) -> None:
        generator = torch.Generator().manual_seed(1)
        target = torch.randn(9, 5, 4, generator=generator).double()
        reconstruction = target + torch.randn(9, 5, 4, generator=generator).double()
        value = _stream(STREAMING_RECONSTRUCTION_METRICS[name](), reconstruction, target, [4, 5])
        assert value == pytest.approx(_POOLED[name](reconstruction, target), rel=1e-10)

    def test_compute_does_not_consume_the_state(self, name: str) -> None:
        reconstruction, target = _pair(rows=40)
        metric = STREAMING_RECONSTRUCTION_METRICS[name]()
        metric.update(reconstruction[:20], target[:20])
        metric.compute()
        metric.update(reconstruction[20:], target[20:])
        assert metric.compute() == pytest.approx(_POOLED[name](reconstruction, target), rel=1e-10)


@pytest.mark.parametrize("name", list(_POOLED))
class TestUpdateContract:
    def test_compute_before_any_value_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match="before any value"):
            STREAMING_RECONSTRUCTION_METRICS[name]().compute()

    def test_an_empty_batch_adds_nothing(self, name: str) -> None:
        metric = STREAMING_RECONSTRUCTION_METRICS[name]()
        metric.update(torch.empty(0, 3), torch.empty(0, 3))
        assert metric.count == 0
        with pytest.raises(ValueError, match="before any value"):
            metric.compute()

    def test_a_shape_mismatch_raises_instead_of_broadcasting(self, name: str) -> None:
        metric = STREAMING_RECONSTRUCTION_METRICS[name]()
        with pytest.raises(ValueError, match="same shape"):
            metric.update(torch.zeros(4, 1, 3), torch.zeros(4, 3))
        assert metric.count == 0

    def test_a_tensor_that_requires_grad_is_accepted(self, name: str) -> None:
        reconstruction, target = _pair(rows=12)
        metric = STREAMING_RECONSTRUCTION_METRICS[name]()
        metric.update(reconstruction.clone().requires_grad_(), target)
        assert math.isfinite(metric.compute())

    def test_count_is_the_number_of_values_not_samples(self, name: str) -> None:
        metric = STREAMING_RECONSTRUCTION_METRICS[name]()
        reconstruction, target = _pair(rows=5, width=3)
        metric.update(reconstruction, target)
        assert metric.count == 15


class TestKnownValues:
    def test_error_metrics_on_a_hand_computed_pair(self) -> None:
        reconstruction = torch.tensor([1.0, 2.0, 4.0])
        target = torch.tensor([1.0, 0.0, 1.0])
        # errors 0, 2, 3
        assert _stream(StreamingMse(), reconstruction, target, [1, 2]) == pytest.approx(13 / 3)
        assert _stream(StreamingRmse(), reconstruction, target, [3]) == pytest.approx(
            math.sqrt(13 / 3)
        )
        assert _stream(StreamingMae(), reconstruction, target, [2, 1]) == pytest.approx(5 / 3)

    def test_a_perfect_reconstruction(self) -> None:
        _, target = _pair(rows=30)
        assert _stream(StreamingMse(), target, target, [10, 20]) == 0.0
        assert _stream(StreamingR2(), target, target, [10, 20]) == pytest.approx(1.0)
        assert _stream(StreamingPearsonR(), target, target, [10, 20]) == pytest.approx(1.0)

    def test_predicting_the_mean_gives_an_r2_of_zero(self) -> None:
        _, target = _pair(rows=60)
        mean = torch.full_like(target, target.mean().item())
        assert _stream(StreamingR2(), mean, target, [25, 35]) == pytest.approx(0.0, abs=1e-12)

    def test_an_inverted_signal_has_a_correlation_of_minus_one(self) -> None:
        _, target = _pair(rows=30)
        assert _stream(StreamingPearsonR(), -target, target, [12, 18]) == pytest.approx(-1.0)

    def test_the_correlation_ignores_a_scale_and_an_offset(self) -> None:
        reconstruction, target = _pair(rows=50)
        scaled = 5.0 * reconstruction + 11.0
        assert _stream(StreamingPearsonR(), scaled, target, [20, 30]) == pytest.approx(
            _stream(StreamingPearsonR(), reconstruction, target, [50]), rel=1e-10
        )

    def test_r2_penalizes_a_scale_that_the_correlation_ignores(self) -> None:
        reconstruction, target = _pair(rows=50)
        assert _stream(StreamingR2(), 3.0 * reconstruction, target, [50]) < 0.0


class TestUndefinedValues:
    def test_r2_of_a_constant_target_is_nan(self) -> None:
        target = torch.full((20,), 3.0)
        assert math.isnan(_stream(StreamingR2(), torch.randn(20), target, [8, 12]))

    def test_the_target_is_constant_only_if_it_is_in_every_batch(self) -> None:
        """Each batch is constant but the two constants differ, so the target does vary."""
        target = torch.cat([torch.full((10,), 1.0), torch.full((10,), 2.0)])
        value = _stream(StreamingR2(), torch.randn(20), target, [10, 10])
        assert not math.isnan(value)

    def test_pearson_is_nan_if_either_side_is_constant(self) -> None:
        varying = torch.randn(20)
        constant = torch.full((20,), 3.0)
        assert math.isnan(_stream(StreamingPearsonR(), constant, varying, [20]))
        assert math.isnan(_stream(StreamingPearsonR(), varying, constant, [5, 15]))

    def test_the_pooled_metrics_agree_on_which_cases_are_undefined(self) -> None:
        varying, constant = torch.randn(20), torch.full((20,), 3.0)
        assert math.isnan(computeR2(varying, constant))
        assert math.isnan(computePearsonR(constant, varying))
