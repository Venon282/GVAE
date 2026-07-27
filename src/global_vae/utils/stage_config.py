"""Helper for expressing per-stage architecture hyperparameters flexibly.

Every multi-stage conv stack in this codebase (1D and 2D encoders and
decoders) is built from a variable number of stages, each of which may
need its own kernel size, stride, padding, dilation, or block count.
Forcing every stage to share one value, or hardcoding the value
entirely, is exactly the kind of rigid default this framework avoids
(spec §12: the user must have the choice): a caller must be able to
give either one shared value for every stage, or one explicit value
per stage. `broadcastPerStage` is the single place that resolves
either form into an explicit per-stage tuple, so every conv stack
validates and reports mismatches the same way instead of
re-implementing this by hand.

Scope note: this only ever broadcasts a single scalar per stage (e.g.
one int per stage for a kernel size), never a per-stage shape tuple.
2D building blocks in this codebase therefore only support square
kernels (`kernel_size` as a plain int, applied to both spatial
dimensions), not independently-sized `(height, width)` kernels: with
scalar-only stage values, a value and a per-stage sequence of values
are unambiguous to tell apart. Supporting non-square kernels as well
would need a different, shape-aware broadcasting rule; that is a
documented limitation, not a silent one.
"""

from collections.abc import Sequence
from typing import TypeVar

_T = TypeVar("_T")

def boradcastPerStage(value: _T | Sequence[_T], num_stages: int, name: str) -> tuple[_T, ...]:
    """Resolve a shared-or-per-stage hyperparameter into a per-stage tuple.

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
