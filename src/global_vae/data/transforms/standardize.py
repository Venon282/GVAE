"""Elementwise standardization transform (spec §6.2).

A generic `(x - mean) / std` / `y * std + mean` pair. `mean`/`std` are
supplied explicitly by the caller (a plain float, applied uniformly, or a
tensor broadcastable against the data, e.g. one value per channel) rather
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

import torch

from global_vae.data.transforms.base import AbstractTransform
from global_vae.data.transforms.registry import registerTransform


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


@registerTransform("standardize")
class StandardizeTransform(AbstractTransform):
    """`(x - mean) / std`, inverted by `y * std + mean`.

    Attributes:
        mean: Value(s) subtracted before dividing by `std`. A plain
            `float` or a `torch.Tensor` broadcastable against the data
            this transform is applied to.
        std: Value(s) divided by. Same type convention as `mean`. Must
            be strictly positive everywhere.
    """

    def __init__(self, mean: float | torch.Tensor, std: float | torch.Tensor) -> None:
        """Initialize the transform.

        Args:
            mean: See the class docstring.
            std: See the class docstring.

        Raises:
            ValueError: If any element of `std` is not strictly
                positive.
        """
        if not _isPositive(std):
            raise ValueError(f"std must be positive everywhere, got {std!r}.")
        self.mean = mean
        self.std = std

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
