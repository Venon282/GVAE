"""Abstract interface for latent distances (roadmap P6-3, ADR 0053).

Stage 3 of a staged recipe minimizes "the distance between the latent spaces" of two modalities
(roadmap Phase 6), and the distance the paper uses is not known (D-13). Several are known, cheap
and different in what they ignore, so a distance is a pluggable strategy, registered like the
latent regularizers and chosen by name in YAML.

A distance compares the posteriors `(mu, logvar)` that the latent spaces of two modalities give to
the same batch of samples, the convention of encoders, fusion and regularizers. It returns one
value per sample, shape `(batch,)`, and does not average over the batch: the loss term that calls
it does, as `computeTotalRegularizationLoss` does for the regularizers. For a distance that is a
sum over the latent dimensions the value is that sum, the footing of the KL term (ADR 0029), so
that the weight of the distance reads next to `beta`.

`forward` is written here, once. It checks the inputs, casts half precision to `float32`, switches
autocast off around the computation (a matrix product would otherwise run in half precision), and
checks the shape of the result. A subclass writes only the mathematics, `computeDistance`.

Concrete subclasses must self-register via `@registerLatentDistance(name)` (see `registry.py`).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import ClassVar

import torch
from torch import nn

LatentParams = tuple[torch.Tensor, torch.Tensor]
"""The `(mu, logvar)` of one latent space for one batch, each of shape `(batch, dim)`."""

_HALF_PRECISION = (torch.float16, torch.bfloat16)
_AUTOCAST_DEVICE_TYPES = ("cpu", "cuda", "xpu", "mps")


class AbstractLatentDistance(nn.Module, ABC):
    """Base class for every distance between the latent spaces of two modalities.

    Attributes:
        is_symmetric: Class-level flag, `True` by default. Whether swapping the two spaces leaves
            the value unchanged. A directional distance (a KL to a frozen reference) sets it to
            `False`.
        is_zero_on_identical: Class-level flag, `True` by default. Whether two equal posteriors
            give 0. `info_nce` sets it to `False`: its minimum is above 0.
        is_stochastic: Class-level flag, `False` by default. Whether the value is a Monte Carlo
            estimate that draws random numbers (`mixture_jsd`).
        max_spaces: Class-level, `2` by default. The number of latent spaces one call compares;
            `None` means any number of at least 2.
        min_batch_size: Class-level, `1` by default. The fewest samples one call needs (`info_nce`
            compares each sample with the others, so it needs 2).
    """

    is_symmetric: ClassVar[bool] = True
    is_zero_on_identical: ClassVar[bool] = True
    is_stochastic: ClassVar[bool] = False
    max_spaces: ClassVar[int | None] = 2
    min_batch_size: ClassVar[int] = 1

    def forward(
        self, first: LatentParams, second: LatentParams, *others: LatentParams
    ) -> torch.Tensor:
        """Compute the distance, one value per sample.

        Args:
            first: `(mu, logvar)` of the first latent space, each `(batch, dim)`.
            second: `(mu, logvar)` of the second latent space, same shapes (the two spaces of a
                pair have equal widths, roadmap D-16).
            *others: More spaces, for a distance whose `max_spaces` is `None` (`mixture_jsd`).
                A distance that compares exactly two refuses them.

        Returns:
            The value of each sample, shape `(batch,)`, `float32` (or `float64` when the inputs
            are). It is still attached to the autograd graph. It is the same inside and outside
            a `torch.autocast` region.

        Raises:
            ValueError: If an input is not a `(mu, logvar)` pair of floating point tensors of
                shape `(batch, dim)`, `logvar` and `mu` differ in shape, the spaces differ in
                shape, there are fewer samples than `min_batch_size`, there are more spaces than
                `max_spaces`, or `computeDistance` returned something that is not `(batch,)`.
        """
        posteriors = self._checkedPosteriors([first, second, *others])
        with self._autocastDisabled(posteriors[0][0]):
            value = self.computeDistance(posteriors)
        batch = posteriors[0][0].shape[0]
        if value.shape != (batch,):
            raise ValueError(
                f"{type(self).__name__}.computeDistance returned shape {tuple(value.shape)}, "
                f"but a distance returns one value per sample, ({batch},)."
            )
        return value

    @abstractmethod
    def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
        """Compute the per-sample distance on checked inputs.

        Args:
            posteriors: One `(mu, logvar)` per latent space, at least two and at most
                `max_spaces`, every tensor `(batch, dim)`, all of one floating point dtype that is
                at least `float32`, all with the same shape. Autocast is off.

        Returns:
            The value of each sample, shape `(batch,)`.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    def _checkedPosteriors(self, posteriors: list[LatentParams]) -> list[LatentParams]:
        """Validate the inputs of `forward` and cast half precision to `float32`."""
        name = type(self).__name__
        count = len(posteriors)
        if self.max_spaces is not None and count > self.max_spaces:
            raise ValueError(
                f"{name} compares {self.max_spaces} latent spaces, got {count}. Only a "
                f"distance whose max_spaces is None takes more than two."
            )
        checked: list[LatentParams] = []
        for index, posterior in enumerate(posteriors):
            if not isinstance(posterior, tuple | list) or len(posterior) != 2:
                raise ValueError(
                    f"{name}: latent space {index} must be a (mu, logvar) pair, "
                    f"got {type(posterior).__name__}."
                )
            mu, logvar = posterior
            for label, tensor in (("mu", mu), ("logvar", logvar)):
                if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                    raise ValueError(
                        f"{name}: {label} of latent space {index} must be a floating point "
                        f"tensor, got {tensor!r}."
                    )
                if tensor.ndim != 2:
                    raise ValueError(
                        f"{name}: {label} of latent space {index} must have shape "
                        f"(batch, dim), got {tuple(tensor.shape)}."
                    )
            if logvar.shape != mu.shape:
                raise ValueError(
                    f"{name}: logvar of latent space {index} has shape {tuple(logvar.shape)}, "
                    f"but its mu has {tuple(mu.shape)}."
                )
            checked.append((_atLeastFloat32(mu), _atLeastFloat32(logvar)))
        reference = checked[0][0].shape
        for index, (mu, _) in enumerate(checked[1:], start=1):
            if mu.shape != reference:
                raise ValueError(
                    f"{name}: latent space {index} has shape {tuple(mu.shape)}, but latent "
                    f"space 0 has {tuple(reference)}. The spaces of a pair must have the same "
                    f"number of samples and the same width."
                )
        if reference[0] < self.min_batch_size:
            raise ValueError(
                f"{name} needs at least {self.min_batch_size} samples per batch, got "
                f"{reference[0]}."
            )
        return checked

    @staticmethod
    def _autocastDisabled(reference: torch.Tensor) -> AbstractContextManager[None]:
        """Return a context that switches autocast off on the device of `reference`."""
        device_type = reference.device.type
        if device_type not in _AUTOCAST_DEVICE_TYPES:
            return nullcontext()
        return torch.autocast(device_type=device_type, enabled=False)


def _atLeastFloat32(tensor: torch.Tensor) -> torch.Tensor:
    """Cast `float16` and `bfloat16` to `float32`; keep `float32` and `float64`."""
    return tensor.float() if tensor.dtype in _HALF_PRECISION else tensor
