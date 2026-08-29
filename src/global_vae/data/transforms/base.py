"""Abstract interface for generic, invertible data transforms (spec §6.2).

A transform maps a raw tensor to a preprocessed tensor (`apply`) and back
(`inverse`). This is the mechanism spec §6 already anticipated but never
implemented: `data/NOTE.md` used to say "SAXS-specific preprocessing (e.g.
log-scale intensity) belongs in `transforms/`, not in the encoder", and §6
itself says other signal sources "should slot into the same... family later,
with only preprocessing differing, not the architecture." The reusable part
of "preprocessing" (an elementwise log, a normalization, a resampling) is
exactly what this subpackage provides; the dataset-specific part (which
transforms to apply, in what order, with what parameters, on which files)
stays the caller's own `loader_factory` responsibility, unchanged from
before (see `data/NOTE.md`).

Deliberately **not** an `nn.Module`, mirroring `AbstractBetaSchedule`
(`training/beta_schedules/base.py`): a transform is a deterministic function
of a tensor, not a computation with learnable parameters that needs to
participate in the autograd graph or be moved across devices as a model
submodule would. Transforms typically run once, in the data pipeline,
before a tensor ever reaches a `GlobalVae` encoder.

**Genericity is a hard requirement, not a style preference.** Every
transform in this subpackage must work on a tensor of *any* shape or
dimensionality (a 1D signal, a 2D image, a 3D volume, a batch of any of
these, or anything else): behaviour that depends on shape/dimensionality is
expressed through an explicit, caller-supplied parameter (e.g. an elementwise
transform needs no such parameter at all; a transform that must know which
axes are "spatial" takes `num_spatial_dims` as a constructor argument), never
by branching internally on `x.dim()` into per-modality code paths, and never
by hardcoding a value, a shape, or a name tied to one concrete dataset (e.g.
SAXS) or one specific modality. A transform whose logic cannot be written
this way does not belong in this subpackage; it belongs in the caller's own
data pipeline, exactly like dataset loading and pairing already do (see
`data/NOTE.md`).

Concrete subclasses must self-register via `@registerTransform(name)` (see
`registry.py`), e.g. `log`, `standardize`, `resample`.
"""

from abc import ABC, abstractmethod

import torch


class AbstractTransform(ABC):
    """Base class for every generic, invertible data transform."""

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this transform to a raw tensor.

        Args:
            x: Input tensor, any shape. Concrete subclasses document
                any further shape requirement (e.g. a minimum number
                of dimensions) explicitly; none may assume a fixed
                number of dimensions outright.

        Returns:
            The transformed tensor.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Undo this transform.

        Exact up to floating-point error for transforms whose
        `apply`/`inverse` pair is a genuine bijection (e.g. `log`,
        `standardize`); documented as approximate on any subclass
        where the forward direction is inherently lossy (e.g.
        `resample`, which discards information when it downsamples).

        Args:
            y: A tensor previously produced by `apply` (or a tensor in
                that same space), any shape.

        Returns:
            The tensor mapped back toward `apply`'s input space.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for `apply`, so an instance can be used directly as a callable.

        Args:
            x: As in `apply`.

        Returns:
            As in `apply`.
        """
        return self.apply(x)
