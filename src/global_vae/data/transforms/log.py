"""Elementwise log transform (spec §6.2).

A generic `log(x + eps)` / `exp(y) - eps` pair: exactly invertible (up to
floating-point error) for any tensor whose values stay above `-eps`
everywhere, regardless of shape or dimensionality. Useful wherever a
long-tailed, strictly-positive-ish quantity benefits from log-scaling before
being fed to a model with a Gaussian likelihood assumption (e.g. `mse_loss`,
this framework's default reconstruction loss, `losses/reconstruction.py`) —
a common preprocessing step across many modalities (spectroscopy or sensor
intensities, pixel counts, ...), not specific to any one of them. This class
knows nothing about what the values represent; it is applied elementwise
with no assumption on the tensor's shape.
"""

import torch

from global_vae.data.transforms.base import AbstractTransform
from global_vae.data.transforms.registry import registerTransform


@registerTransform("log")
class LogTransform(AbstractTransform):
    """`log(x + eps)`, inverted by `exp(y) - eps`.

    Attributes:
        eps: Additive offset keeping the argument of `log` strictly
            positive. Must be positive itself, and the caller is
            responsible for choosing it so that `x + eps > 0` holds
            for every value the transform is ever applied to (spec
            §12: no attempt is made to guess or silently clamp a
            domain-appropriate value here, since what counts as a
            valid input range is a data-specific decision the
            framework has no way to know).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the transform.

        Args:
            eps: See the class docstring. Defaults to a small
                strictly-positive constant.

        Raises:
            ValueError: If `eps` is not positive.
        """
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply `log(x + eps)` elementwise.

        Args:
            x: Any-shaped tensor. Every element must satisfy
                `x + eps > 0`.

        Returns:
            The log-transformed tensor, same shape as `x`.

        Raises:
            ValueError: If any element of `x + eps` is not strictly
                positive (the transform would otherwise silently
                return `-inf`/`nan`).
        """
        shifted = x + self.eps
        if torch.any(shifted <= 0):
            raise ValueError(
                "LogTransform requires x + eps > 0 everywhere, but the smallest value of "
                f"x + eps in this input is {shifted.min().item()!r}. Increase eps, or "
                "make sure this transform is not applied to data outside its valid range."
            )
        return torch.log(shifted)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply `exp(y) - eps` elementwise, undoing `apply`.

        Args:
            y: Any-shaped tensor, typically the output of `apply`.

        Returns:
            The tensor mapped back toward `apply`'s input space, same
            shape as `y`.
        """
        return torch.exp(y) - self.eps
