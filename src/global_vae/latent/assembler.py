"""Assembler strategies: combine several realized latent vectors into one
decoder input (spec §2.2).

Unlike Fusion (`fusion/base.py`), which combines distribution
*parameters* before sampling, an Assembler combines already-sampled
latent *vectors* — no probabilistic machinery needed, this is just
merging tensors for a decoder's input layer. Registered the same way
as encoders/decoders/fusion strategies.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import nn

_ASSEMBLER_REGISTRY: dict[str, type["AbstractAssembler"]] = {}


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


def registerAssembler(name: str) -> Callable[[type[AbstractAssembler]], type[AbstractAssembler]]:
    """Class decorator registering an assembler implementation under `name`.

    Args:
        name: Unique registry key (e.g. `"concat"`, `"sum"`, `"average"`).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorator(cls: type[AbstractAssembler]) -> type[AbstractAssembler]:
        if name in _ASSEMBLER_REGISTRY:
            raise ValueError(f"Assembler '{name}' is already registered.")
        _ASSEMBLER_REGISTRY[name] = cls
        return cls

    return decorator


def getAssemblerClass(name: str) -> type[AbstractAssembler]:
    """Look up a registered assembler class by name.

    Args:
        name: Registry key used at registration time.

    Returns:
        The assembler class registered under `name`.

    Raises:
        KeyError: If no assembler is registered under `name`.
    """
    if name not in _ASSEMBLER_REGISTRY:
        available = ", ".join(sorted(_ASSEMBLER_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown assembler '{name}'. Available: {available}")
    return _ASSEMBLER_REGISTRY[name]


def listRegisteredAssemblers() -> list[str]:
    """Return all currently registered assembler names.

    Returns:
        Sorted list of registered assembler names.
    """
    return sorted(_ASSEMBLER_REGISTRY)

@registerAssembler("concat")
class ConcatAssembler(AbstractAssembler):
    """Concatenates latent vectors along the feature dimension.

    No dimensionality restriction across inputs (spec §2.2).
    """
    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(latents, dim=-1)

@registerAssembler("sum")
class SumAssembler(AbstractAssembler):
    """Sums latent vectors elementwise.

    Requires all input latent spaces to share the same dimensionality
    (enforced at construction time by `validateRoutingGraph`).
    """

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(latents, dim=0).sum(dim=0)

@registerAssembler("average")
class AverageAssembler(AbstractAssembler):
    """Averages latent vectors elementwise.

    Requires all input latent spaces to share the same dimensionality
    (enforced at construction time by `validateRoutingGraph`).
    """

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(latents, dim=0).mean(dim=0)
