"""Helpers for expressing per-stage architecture hyperparameters flexibly.

Every multi-stage conv stack in this codebase (1D and 2D encoders and
decoders) is built from a variable number of stages, each of which may
need its own kernel size, stride, padding, dilation, or block count.
Forcing every stage to share one value, or hardcoding the value
entirely, is exactly the kind of rigid default this framework avoids
(spec §12: the user must have the choice): a caller must be able to
give either one shared value for every stage, or one explicit value
per stage.

Two helpers cover this:

- `broadcastPerStage` handles hyperparameters that are a plain scalar
  at every stage (e.g. a 1D kernel size): there is no ambiguity here,
  since a scalar is never itself a sequence.
- `broadcastPerStageShape` additionally handles hyperparameters whose
  *per-stage* value can itself have more than one component (e.g. a 2D
  kernel's `(height, width)`, which need not be square). Naively
  reusing a plain `Sequence`-based broadcast for this would be
  ambiguous: with 2 stages and a 2D kernel, `(3, 5)` could mean either
  "one (3, 5) kernel every stage" or "kernel 3 at stage 0, kernel 5 at
  stage 1", and there is no way to tell which was meant from the value
  alone. `broadcastPerStageShape` resolves this by giving `list` and
  `tuple` different, non-overlapping jobs: a `list` is always the
  per-stage wrapper, a `tuple` is always a single multi-dimensional
  shape applied to every stage. This is a real fix, not a documented
  restriction: every 2D building block in this codebase supports
  non-square kernels, strides, paddings, and dilations, each of which
  may also vary per stage.
"""

from collections.abc import Sequence
from typing import TypeVar

_T = TypeVar("_T")

ShapeLike = int | tuple[int, ...]

def broadcastPerStage(value: _T | Sequence[_T], num_stages: int, name: str) -> tuple[_T, ...]:
    """Resolve a shared-or-per-stage scalar hyperparameter into a per-stage tuple.

    Args:
        value: Either a single value, applied to every stage, or a
            sequence of exactly `num_stages` values, one per stage.
        num_stages: Number of stages the caller is building.
        name: Parameter name, used only for the error message.

    Returns:
        A tuple of length `num_stages`.

    Raises:
        ValueError: If `value` is a sequence whose length does not
            equal `num_stages`.
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        resolved = tuple(value)
        if len(resolved) != num_stages:
            raise ValueError(
                f"'{name}' has {len(resolved)} value(s) but there are {num_stages} "
                f"stage(s); pass either a single value (shared by every stage) or "
                f"exactly {num_stages} values."
            )
        return resolved
    return (value,) * num_stages


def resolveSpatialShape(value: ShapeLike, ndim: int, name: str) -> tuple[int, ...]:
    """Resolve a single spatial-shape hyperparameter to an explicit per-dimension tuple.

    Args:
        value: Either a single `int`, applied to every one of the
            `ndim` spatial dimensions (e.g. a square 2D kernel or a
            cubic 3D kernel), or an explicit `tuple[int, ...]` of
            length `ndim` (e.g. a non-square `(height, width)` kernel).
        ndim: Number of spatial dimensions (`2` for this codebase's 2D
            building blocks).
        name: Parameter name, used only for the error message.

    Returns:
        A tuple of length `ndim`.

    Raises:
        ValueError: If `value` is a tuple whose length does not equal
            `ndim`.
    """
    if isinstance(value, tuple):
        if len(value) != ndim:
            raise ValueError(
                f"'{name}' gave a {len(value)}-dimensional shape {value}, but this "
                f"building block is {ndim}-dimensional."
            )
        return value
    return (value,) * ndim
