"""Helpers for expressing per-stage architecture hyperparameters flexibly.

Every multi-stage conv stack in this codebase (1D and 2D encoders and
decoders) is built from a variable number of stages, each of which may
need its own kernel size, stride, padding, dilation, or block count.
Forcing every stage to share one value, or hardcoding the value
entirely, is exactly the kind of rigid default this framework avoids
(spec §12: the user must have the choice): a caller must be able to
give either one shared value for every stage, or one explicit value
per stage.

Four helpers cover this:

- `broadcastPerStage` handles hyperparameters that are a plain scalar
  at every stage (e.g. a 1D kernel size): there is no ambiguity here,
  since a scalar is never itself a sequence.
- `resolveSpatialShape` resolves a *single* shape-like hyperparameter
  (not yet per-stage) to an explicit per-dimension tuple: a plain
  `int` broadcasts to every spatial dimension (e.g. a square 2D
  kernel), an explicit `tuple[int, ...]` of length `ndim` is used
  as-is (e.g. a non-square `(height, width)` kernel).
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
- `broadcastPerStageOptionalShape` is `broadcastPerStageShape`'s
  counterpart for hyperparameters where a given stage may have no
  shape at all (e.g. `pool_kernel_sizes`/`pool_strides`, where a stage
  may skip pooling entirely). Kept as a separate function rather than
  folding `None` into `broadcastPerStageShape` itself: Python's
  `list[...]` typing is invariant, so a caller statically known to
  never pass `None` (`kernel_sizes`, `paddings`, ...) could not
  otherwise satisfy `mypy --strict` when handing its own `list` to a
  parameter typed to *possibly* contain `None`.
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


def broadcastPerStageShape(
    value: ShapeLike | list[ShapeLike],
    num_stages: int,
    ndim: int,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    """Resolve a shared-or-per-stage, possibly multi-dimensional shape hyperparameter.

    Generalizes `broadcastPerStage` to shape-like hyperparameters (a 2D
    building block's kernel, stride, padding, or dilation, each an
    `(height, width)` pair) whose *per-stage* value can itself have
    more than one component. A plain `Sequence`-based broadcast
    (`broadcastPerStage`'s own approach) would be ambiguous here: with
    2 stages and a 2D kernel, `(3, 5)` could mean either "one (3, 5)
    kernel every stage" or "kernel 3 at stage 0, kernel 5 at stage 1",
    and there is no way to tell which was meant from the value alone.

    This ambiguity is resolved by giving `list` and `tuple` different,
    non-overlapping jobs, exactly as `resolveSpatialShape` already does
    for a single (not-yet-per-stage) shape:
      - a `list` is always the per-stage wrapper: exactly `num_stages`
        entries, one per stage, each independently resolved via
        `resolveSpatialShape`;
      - a `tuple` is always a single, explicit, multi-dimensional shape
        (`resolveSpatialShape`'s own single-value behavior) applied to
        every stage;
      - a plain `int` is likewise applied to every stage, broadcast to
        a square/cubic shape of `ndim` equal components.

    See `broadcastPerStageOptionalShape` for hyperparameters where a
    given stage may have no shape at all (e.g. `pool_kernel_sizes`/
    `pool_strides`, where a stage may skip pooling entirely, or defer
    its stride to its own kernel size): a separate function, not an
    overload of this one, because `None`-vs-not is a genuine
    difference in what every caller and every element of the returned
    tuple can be, and Python's invariant `list[...]` typing cannot
    otherwise let a caller pass a `list` that is statically known to
    never contain `None` wherever a `list` that *may* contain `None`
    is expected.

    Args:
        value: A single `int`/`tuple[int, ...]`, applied to every
            stage, or a `list` of exactly `num_stages` such values, one
            per stage.
        num_stages: Number of stages being built.
        ndim: Number of spatial dimensions each shape must resolve to
            (`2` for this codebase's 2D building blocks).
        name: Parameter name, used only for error messages.

    Returns:
        A tuple of length `num_stages`, each entry a tuple of length
        `ndim`.

    Raises:
        ValueError: If `value` is a `list` whose length is not
            `num_stages`, or if any resolved shape does not have
            exactly `ndim` components (delegated to
            `resolveSpatialShape`).
    """
    if isinstance(value, list):
        if len(value) != num_stages:
            raise ValueError(
                f"'{name}' is a list with {len(value)} entry(ies) but there are {num_stages} "
                f"stage(s); pass either a single value (shared by every stage) or exactly "
                f"{num_stages} entries wrapped in a list, one per stage."
            )
        return tuple(
            resolveSpatialShape(entry, ndim, f"{name}[{index}]")
            for index, entry in enumerate(value)
        )
    shared_shape = resolveSpatialShape(value, ndim, name)
    return (shared_shape,) * num_stages


def broadcastPerStageOptionalShape(
    value: ShapeLike | None | list[ShapeLike | None],
    num_stages: int,
    ndim: int,
    name: str,
) -> tuple[tuple[int, ...] | None, ...]:
    """Like `broadcastPerStageShape`, but a per-stage or shared `None` is passed through
    unchanged instead of being resolved by `resolveSpatialShape`.

    For hyperparameters where a given stage may have no shape at all,
    e.g. `pool_kernel_sizes`/`pool_strides` (a stage may skip pooling
    entirely, or defer its stride to its own kernel size, mirroring
    `OneDCnnEncoder`'s identical 1D convention). See
    `broadcastPerStageShape`'s own docstring for why this is a
    separate function rather than a single one covering both cases.

    Args:
        value: A single `int`/`tuple[int, ...]`/`None`, applied to
            every stage, or a `list` of exactly `num_stages` such
            values, one per stage (each entry independently an
            `int`/`tuple[int, ...]`/`None`).
        num_stages: Number of stages being built.
        ndim: Number of spatial dimensions each non-`None` shape must
            resolve to.
        name: Parameter name, used only for error messages.

    Returns:
        A tuple of length `num_stages`, each entry either `None` or a
        tuple of length `ndim`.

    Raises:
        ValueError: If `value` is a `list` whose length is not
            `num_stages`, or if any resolved shape does not have
            exactly `ndim` components.
    """
    if isinstance(value, list):
        if len(value) != num_stages:
            raise ValueError(
                f"'{name}' is a list with {len(value)} entry(ies) but there are {num_stages} "
                f"stage(s); pass either a single value (shared by every stage) or exactly "
                f"{num_stages} entries wrapped in a list, one per stage."
            )
        return tuple(
            resolveSpatialShape(entry, ndim, f"{name}[{index}]") if entry is not None else None
            for index, entry in enumerate(value)
        )
    if value is None:
        return (None,) * num_stages
    shared_shape = resolveSpatialShape(value, ndim, name)
    return (shared_shape,) * num_stages
