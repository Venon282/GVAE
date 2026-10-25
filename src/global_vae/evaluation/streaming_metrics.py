"""Streaming reconstruction metrics: the same five numbers without pooling the test set.

`evaluation.metrics` computes each metric once over the whole pooled dataset, which holds every
original and every reconstruction of the test set in memory at once (`evaluate`'s own docstring
explains why: `r2` and `pearson_r` are ratios of sums, not averages of per-batch ratios). For a
very large test set that pool is the memory bound, and `max_samples` is the only way to shrink it.

The five metrics do not need the pool, only a few running sums. A `StreamingMetric` is fed one
batch at a time (`update`) and gives the value over everything it has seen (`compute`):

- `mse`, `rmse` and `mae` keep the sum of the squared or absolute errors and the number of values.
- `r2` and `pearson_r` keep the count, the mean and the sum of squared deviations of the
  quantities they compare (and, for `pearson_r`, their co-moment), merged batch by batch with the
  parallel algorithm of Chan, Golub and LeVeque (1983). It is the numerically stable form: the
  naive `sum(x ** 2) - n * mean ** 2` loses every digit when the mean is large against the spread.

Everything accumulates in `float64`, so the value agrees with the pooled one to the rounding of the
`float32` pool (a relative `1e-6` or so on well-scaled data), and a batch size or a split of the
dataset into batches changes nothing. `STREAMING_RECONSTRUCTION_METRICS` has the same names as
`evaluation.metrics.DEFAULT_RECONSTRUCTION_METRICS`, which `evaluate(..., streaming=True)` relies
on (`docs/adr/0038-paired-data-evaluation.md`).
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch


class StreamingMetric(ABC):
    """A reconstruction metric computed from batches, never from the pooled dataset.

    Subclasses implement `_accumulate`, which receives the flattened `float64` values of one
    batch, and `compute`. `update` does the shape check and the flattening, so a subclass never
    sees a pair that disagrees.
    """

    def __init__(self) -> None:
        """Start with nothing seen."""
        self._count = 0

    @property
    def count(self) -> int:
        """The number of values (elements, not samples) seen so far."""
        return self._count

    def update(self, reconstruction: torch.Tensor, target: torch.Tensor) -> None:
        """Add one batch.

        Args:
            reconstruction: Model output for this batch.
            target: Ground truth, the same shape as `reconstruction`.

        Raises:
            ValueError: If the two shapes differ. `torch.nn.functional`'s own losses only warn
                and then broadcast, which would compare the wrong elements.
        """
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"The reconstruction has shape {tuple(reconstruction.shape)} but its target has "
                f"shape {tuple(target.shape)}: a streaming metric needs the same shape."
            )
        if reconstruction.numel() == 0:
            return
        self._accumulate(
            reconstruction.detach().flatten().double(), target.detach().flatten().double()
        )
        self._count += reconstruction.numel()

    @abstractmethod
    def _accumulate(self, reconstruction: torch.Tensor, target: torch.Tensor) -> None:
        """Fold one non-empty batch of flattened `float64` values into the running state."""

    @abstractmethod
    def compute(self) -> float:
        """Return the metric over everything seen.

        Raises:
            ValueError: If nothing has been seen (an empty evaluation has no metric).
        """

    def _requireValues(self) -> None:
        """Raise `ValueError` if `update` has not seen a single value."""
        if self._count == 0:
            raise ValueError(f"{type(self).__name__}.compute() was called before any value.")


class StreamingMse(StreamingMetric):
    """Mean squared error: `sum((reconstruction - target) ** 2) / count`."""

    def __init__(self) -> None:
        """Start with a zero sum."""
        super().__init__()
        self._sum_of_squares = 0.0

    def _accumulate(self, reconstruction: torch.Tensor, target: torch.Tensor) -> None:
        self._sum_of_squares += (reconstruction - target).pow(2).sum().item()

    def compute(self) -> float:
        """Return the mean squared error over every value seen."""
        self._requireValues()
        return self._sum_of_squares / self._count


class StreamingRmse(StreamingMse):
    """Root mean squared error: the square root of `StreamingMse`."""

    def compute(self) -> float:
        """Return the root mean squared error over every value seen."""
        return math.sqrt(super().compute())


class StreamingMae(StreamingMetric):
    """Mean absolute error: `sum(abs(reconstruction - target)) / count`."""

    def __init__(self) -> None:
        """Start with a zero sum."""
        super().__init__()
        self._sum_of_absolutes = 0.0

    def _accumulate(self, reconstruction: torch.Tensor, target: torch.Tensor) -> None:
        self._sum_of_absolutes += (reconstruction - target).abs().sum().item()

    def compute(self) -> float:
        """Return the mean absolute error over every value seen."""
        self._requireValues()
        return self._sum_of_absolutes / self._count


class _RunningMoments:
    """Count, mean and sum of squared deviations of a stream, merged batch by batch."""

    def __init__(self) -> None:
        """Start empty."""
        self.count = 0
        self.mean = 0.0
        self.sum_of_squares = 0.0

    def merge(self, values: torch.Tensor) -> None:
        """Fold in one non-empty batch of `float64` values."""
        batch_count = values.numel()
        batch_mean = values.mean().item()
        batch_sum_of_squares = (values - batch_mean).pow(2).sum().item()
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.sum_of_squares += (
            batch_sum_of_squares + delta * delta * self.count * batch_count / total
        )
        self.mean += delta * batch_count / total
        self.count = total


class StreamingR2(StreamingMetric):
    """Coefficient of determination, `1 - SS_res / SS_tot`, as `evaluation.metrics.computeR2`.

    The baseline is the single scalar mean of every target element pooled together, so the
    running state is the moments of the target and the sum of squared residuals.
    """

    def __init__(self) -> None:
        """Start with nothing seen."""
        super().__init__()
        self._target = _RunningMoments()
        self._sum_of_squared_residuals = 0.0

    def _accumulate(self, reconstruction: torch.Tensor, target: torch.Tensor) -> None:
        self._target.merge(target)
        self._sum_of_squared_residuals += (target - reconstruction).pow(2).sum().item()

    def compute(self) -> float:
        """Return R^2 over every value seen, or `nan` if the target is exactly constant."""
        self._requireValues()
        if self._target.sum_of_squares == 0.0:
            return float("nan")
        return 1.0 - self._sum_of_squared_residuals / self._target.sum_of_squares


class StreamingPearsonR(StreamingMetric):
    """Pearson correlation of the flattened reconstruction and target, as `computePearsonR`."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        super().__init__()
        self._reconstruction = _RunningMoments()
        self._target = _RunningMoments()
        self._co_moment = 0.0

    def _accumulate(self, reconstruction: torch.Tensor, target: torch.Tensor) -> None:
        # The co-moment merge needs the means before this batch, so it comes first.
        batch_count = reconstruction.numel()
        batch_reconstruction_mean = reconstruction.mean().item()
        batch_target_mean = target.mean().item()
        batch_co_moment = (
            ((reconstruction - batch_reconstruction_mean) * (target - batch_target_mean))
            .sum()
            .item()
        )
        total = self._count + batch_count
        self._co_moment += (
            batch_co_moment
            + (batch_reconstruction_mean - self._reconstruction.mean)
            * (batch_target_mean - self._target.mean)
            * self._count
            * batch_count
            / total
        )
        self._reconstruction.merge(reconstruction)
        self._target.merge(target)

    def compute(self) -> float:
        """Return Pearson r over every value seen, or `nan` if either side is exactly constant."""
        self._requireValues()
        if self._reconstruction.sum_of_squares == 0.0 or self._target.sum_of_squares == 0.0:
            return float("nan")
        denominator = math.sqrt(self._reconstruction.sum_of_squares * self._target.sum_of_squares)
        return max(-1.0, min(1.0, self._co_moment / denominator))


STREAMING_RECONSTRUCTION_METRICS: dict[str, Callable[[], StreamingMetric]] = {
    "mse": StreamingMse,
    "rmse": StreamingRmse,
    "mae": StreamingMae,
    "r2": StreamingR2,
    "pearson_r": StreamingPearsonR,
}
"""Metric name -> a class that builds a fresh `StreamingMetric`, in the order of
`DEFAULT_RECONSTRUCTION_METRICS`. A factory and not an instance, since a metric holds state: one
per modality per pass."""


def buildStreamingMetrics() -> dict[str, StreamingMetric]:
    """Build one fresh instance of every streaming metric.

    Returns:
        Metric name -> a `StreamingMetric` that has seen nothing, in the order of
        `STREAMING_RECONSTRUCTION_METRICS`.
    """
    return {name: factory() for name, factory in STREAMING_RECONSTRUCTION_METRICS.items()}
