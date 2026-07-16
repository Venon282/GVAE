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
