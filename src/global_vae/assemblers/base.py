"""Abstract interface for assembler strategies (spec §2.2).

An Assembler combines several already-realized latent vectors into one
decoder input. This is distinct from Fusion (`fusion/base.py`): Fusion
combines distribution parameters before sampling; an Assembler combines
vectors that have already been sampled (or, at evaluation time, are
already the posterior mean). No probabilistic machinery is needed here,
only tensor merging, which may be parameter-free (`concat`, `sum`,
`average`) or learned (`weighted_sum`, `attention`).

Concrete subclasses must self-register via `@registerAssembler(name)`
(see `registry.py`), e.g. `concat`, `sum`, `average`, `weighted_sum`,
`attention`. The model builds one from its registry name with
`getAssemblerClass(name)(input_dims, **kwargs)`, so every concrete class
must accept `input_dims` as its first constructor argument; any further
argument must have a default. Knowing `input_dims` at construction is what
lets an assembler size its parameters and announce its `output_dim`, the
width of the tensor the decoder receives.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch
from torch import nn


class AbstractAssembler(nn.Module, ABC):
    """Base class for every assembler strategy.

    Attributes:
        requires_matching_dims: Class-level flag, `False` by default. A
            subclass sets it to `True` when all of its input latent spaces
            must share one dimensionality (`sum`, `average`,
            `weighted_sum`). `validateRoutingGraph` reads it through the
            assembler registry, and `__init__` enforces it, so a new
            strategy gets both checks without any change to the model or to
            the validator (spec §10, §12).
    """

    requires_matching_dims: ClassVar[bool] = False

    def __init__(self, input_dims: Sequence[int]) -> None:
        """Store and check the widths of the latent vectors this assembler receives.

        Args:
            input_dims: Width of each latent vector, in the order `forward`
                receives them: the `dim` of each latent space the decoder
                consumes, in `RoutingGraph.decoderInputs` order. Must not be
                empty.

        Raises:
            ValueError: If `input_dims` is empty, holds anything but positive
                integers, or has differing widths while the class sets
                `requires_matching_dims`.
        """
        super().__init__()
        dims = tuple(input_dims)
        if not dims:
            raise ValueError("`input_dims` must list the width of at least one latent vector.")
        for index, dim in enumerate(dims):
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"`input_dims[{index}]` must be a positive integer, got {dim!r}.")
        if self.requires_matching_dims and len(set(dims)) > 1:
            raise ValueError(
                f"{type(self).__name__} requires matching dimensionality, but its input latent "
                f"vectors have differing dims {sorted(set(dims))} (input_dims={list(dims)})."
            )
        self._input_dims = dims

    @property
    def input_dims(self) -> tuple[int, ...]:
        """Width of each latent vector this assembler combines.

        Returns:
            The widths given at construction, in order.
        """
        return self._input_dims

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Width of the tensor `forward` returns, which is the decoder's `latent_dim`.

        `GlobalVae` reads it to check, or to fill in, the `latent_dim` of the
        decoder that consumes this assembler's output. It depends on
        `input_dims` and on the strategy's own arguments only, never on which
        latent spaces are present in a given pass.

        Returns:
            The last dimension of the output of `forward`.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, latents: list[torch.Tensor], present: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Combine several realized latent vectors into one tensor.

        Args:
            latents: List of already-sampled latent tensors, each of
                shape `(batch, dim_i)`.
            present: Position, in `input_dims`, of each tensor of `latents`,
                or `None` (the default) when `latents` holds every latent
                space, in order. `GlobalVae` passes it when a modality was
                left out of the pass and the decoder runs on a subset
                (`handlesMissingLatents`); a strategy with a parameter per
                latent space needs it to tell which spaces it was given.

        Returns:
            The combined tensor to feed the decoder, shape
            `(batch, output_dim)`.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    @property
    def handlesMissingLatents(self) -> bool:
        """Whether `forward` accepts only some of the latent spaces its decoder consumes.

        A decoder can have fewer latent spaces than its graph lists: the
        private latent of a modality that was left out of a pass (modality
        dropout, a cross-modal query, spec §5) has no encoder to feed it. This
        is the assembler-side counterpart of
        `AbstractFusion.handlesMissingModalities`. `GlobalVae.forward` runs a
        decoder with an incomplete input only if its assembler says `True`
        here, passing `present` to `forward`; otherwise the decoder is left out
        of that pass's `"reconstructions"`, as a decoder with no input at all
        is.

        Returns:
            `False` by default: an assembler is not assumed to accept a subset
            (`concat` produces a tensor of the wrong width). Strategies
            defined for any non-empty subset, such as `sum`, `average`,
            `weighted_sum` and `attention`, override it to `True`.
        """
        return False

    def _resolvePresent(
        self, latents: Sequence[torch.Tensor], present: Sequence[int] | None
    ) -> tuple[int, ...]:
        """Check the inputs of `forward` against `input_dims` and name their positions.

        Concrete strategies call this first in `forward`, so a wiring mistake
        (a latent of the wrong width, a subset given to an assembler that
        cannot take one) fails with a message naming the widths instead of a
        shape error deep inside a matrix product.

        Args:
            latents: The tensors given to `forward`.
            present: The `present` argument of `forward`.

        Returns:
            The position in `input_dims` of each tensor of `latents`.

        Raises:
            ValueError: If `present` is given with a different length than
                `latents`, is empty, repeats a position or names one outside
                `input_dims`; if `latents` is not a complete list while
                `handlesMissingLatents` is `False`; or if a tensor is not
                2D with the width `input_dims` gives for its position.
        """
        if present is None:
            if len(latents) != len(self._input_dims):
                raise ValueError(
                    f"{type(self).__name__} was built for {len(self._input_dims)} latent "
                    f"vectors (input_dims={list(self._input_dims)}) but got {len(latents)}. "
                    f"To assemble a subset, pass `present` (the position of each tensor)."
                )
            positions = tuple(range(len(latents)))
        else:
            positions = tuple(present)
            if len(positions) != len(latents) or not positions:
                raise ValueError(
                    f"`present` must give one position per latent tensor, got {len(positions)} "
                    f"position(s) for {len(latents)} tensor(s)."
                )
            if len(set(positions)) != len(positions) or not all(
                0 <= position < len(self._input_dims) for position in positions
            ):
                raise ValueError(
                    f"`present` must hold distinct positions in range({len(self._input_dims)}), "
                    f"got {list(positions)}."
                )
        if len(positions) < len(self._input_dims) and not self.handlesMissingLatents:
            raise ValueError(
                f"{type(self).__name__} needs all {len(self._input_dims)} latent vectors "
                f"(handlesMissingLatents is False), got {len(positions)}."
            )
        for position, latent in zip(positions, latents, strict=True):
            expected = self._input_dims[position]
            if latent.dim() != 2 or latent.shape[-1] != expected:
                raise ValueError(
                    f"{type(self).__name__} expects latent vector {position} of shape (batch, "
                    f"{expected}), got {tuple(latent.shape)}."
                )
        return positions
