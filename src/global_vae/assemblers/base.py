"""Abstract interface for assembler strategies (spec §2.2).

An Assembler combines several already-realized latent vectors into one
decoder input. This is distinct from Fusion (`fusion/base.py`): Fusion
combines distribution parameters before sampling; an Assembler combines
vectors that have already been sampled (or, at evaluation time, are
already the posterior mean), so no probabilistic machinery is needed
here, only tensor merging.

Concrete subclasses must self-register via `@registerAssembler(name)`
(see `registry.py`), e.g. `concat`, `sum`, `average`.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractAssembler(nn.Module, ABC):
    """Base class for every assembler strategy."""

    @abstractmethod
    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Combine several realized latent vectors into one tensor.

        Args:
            latents: List of already-sampled latent tensors, each of
                shape `(batch, dim_i)`.

        Returns:
            The combined tensor to feed the decoder.

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
        here; otherwise the decoder is left out of that pass's
        `"reconstructions"`, as a decoder with no input at all is.

        Returns:
            `False` by default: an assembler is not assumed to accept a subset
            (`concat` produces a tensor of the wrong width). Strategies
            defined for any non-empty list, such as `sum` and `average`,
            override it to `True`.
        """
        return False
