"""Elementwise standardization transform (spec §6.2).

A generic `(x - mean) / std` / `y * std + mean` pair. `mean`/`std` are
supplied explicitly by the caller (a plain float, applied uniformly, a
tensor broadcastable against the data, e.g. one value per channel, or a list
of numbers, which is how a YAML file writes one value per column and which is
turned into a tensor) rather
than computed from whatever tensor happens to be passed in: this mirrors
this framework's general "the user supplies the value, never a silently
guessed one" convention (spec §10, §12), and specifically avoids coupling
this transform to a specific dataset or split. Computing statistics over a
training set is a data-pipeline concern (which samples belong to that set,
how they are loaded) squarely outside this framework's scope (`data/NOTE.md`);
this class only ever applies already-known statistics. Works on a tensor of
any shape: `mean`/`std` broadcast against `x` exactly the way any PyTorch
elementwise arithmetic broadcasts, so a scalar standardizes every element
uniformly and a shaped tensor (e.g. one value per leading channel dimension)
standardizes per that broadcastable axis, for a 1D series, a 2D image, a 3D
volume, or anything else, with no per-dimensionality branching.
"""

from collections.abc import Sequence
from typing import Any

import torch

from global_vae.data.transforms.base import AbstractTransform
from global_vae.data.transforms.registry import registerTransform

Statistic = float | torch.Tensor | Sequence[Any]


def _isPositive(value: float | torch.Tensor) -> bool:
    """Check that a scalar-or-tensor value is strictly positive everywhere.

    Args:
        value: A plain float or a tensor of any shape.

    Returns:
        `True` if every element is strictly positive.
    """
    if isinstance(value, torch.Tensor):
        return bool(torch.all(value > 0))
    return value > 0


def _asStatistic(value: object, field_name: str) -> float | torch.Tensor:
    """Turn one statistic of the constructor into a float or a tensor.

    A number and a tensor are kept as they are. A list or a tuple of numbers (nested lists make a
    matrix) becomes a `float32` tensor, which is what a YAML file gives for a statistic per column.

    Args:
        value: What the caller passed for `mean` or `std`.
        field_name: `"mean"` or `"std"`, named in every error.

    Returns:
        `value` for a number or a tensor, else the tensor built from the sequence.

    Raises:
        TypeError: If `value` is none of a number, a tensor, or a list or tuple of numbers.
        ValueError: If the sequence is empty, ragged, or holds something that is not a number.
    """
    if isinstance(value, torch.Tensor | int | float):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) == 0:
            raise ValueError(f"{field_name} must not be an empty list.")
        try:
            return torch.tensor(list(value), dtype=torch.get_default_dtype())
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"{field_name} must be a list of numbers (nested lists of one length make a "
                f"matrix), got {value!r}."
            ) from error
    raise TypeError(
        f"{field_name} must be a float, a tensor or a list of numbers, got "
        f"{type(value).__name__} {value!r}."
    )


@registerTransform("standardize")
class StandardizeTransform(AbstractTransform):
    """`(x - mean) / std`, inverted by `y * std + mean`.

    Attributes:
        mean: Value(s) subtracted before dividing by `std`. A plain
            `float` or a `torch.Tensor` broadcastable against the data
            this transform is applied to. A list of numbers given to the
            constructor is stored as a `float32` tensor.
        std: Value(s) divided by. Same type convention as `mean`. Must
            be strictly positive everywhere.
    """

    def __init__(self, mean: Statistic, std: Statistic) -> None:
        """Initialize the transform.

        Args:
            mean: A float, a tensor, or a list of numbers (one per column or channel, as a
                YAML file writes it). A list is turned into a `float32` tensor.
            std: Same convention as `mean`.

        Raises:
            TypeError: If `mean` or `std` is none of a number, a tensor or a list of numbers.
                The message starts with the name of the argument.
            ValueError: If a list is empty, ragged or holds something that is not a number
                (the message names the argument), or if any element of `std` is not strictly
                positive.
        """
        mean_value = _asStatistic(mean, "mean")
        std_value = _asStatistic(std, "std")
        if not _isPositive(std_value):
            raise ValueError(f"std must be positive everywhere, got {std!r}.")
        self.mean = mean_value
        self.std = std_value

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply `(x - mean) / std` elementwise (broadcasting as needed).

        Args:
            x: Any-shaped tensor, broadcastable against `self.mean`/
                `self.std`.

        Returns:
            The standardized tensor.
        """
        return (x - self.mean) / self.std

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply `y * std + mean` elementwise, undoing `apply`.

        Args:
            y: Any-shaped tensor, typically the output of `apply`.

        Returns:
            The tensor mapped back toward `apply`'s input space.
        """
        return y * self.std + self.mean
