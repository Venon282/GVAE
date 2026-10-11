"""Abstract interface for latent heads (spec §2.2, §3).

A latent head is an optional module on one encoder -> latent edge. It adapts an
encoder's `(mu, logvar)` output to the dimensionality (or purpose) of the latent space
that edge feeds. It is what makes encoder fan-out possible: when one encoder feeds
several latent spaces, each edge gets its own head, so the spaces can differ, instead
of giving `AbstractEncoder` several output heads or duplicating the encoder (spec §12).

Concrete subclasses must self-register via `@registerHead(name)` (see `registry.py`),
e.g. `identity`, `linear`. The model builds a head from its registry name with
`getHeadClass(name)(in_dim=..., out_dim=...)`, so every concrete class must accept
`in_dim` and `out_dim` as its first two constructor arguments. Any further argument must
have a default.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractLatentHead(nn.Module, ABC):
    """Base class for every latent head.

    A head maps the distribution parameters of an encoder output, `(mu, logvar)` of
    shape `(batch, in_dim)`, to the distribution parameters of a latent space,
    `(mu, logvar)` of shape `(batch, out_dim)`. It acts on distribution parameters,
    before fusion and before sampling, the same convention as encoders, fusion
    strategies and regularizers (spec §2.2). Combining realized (sampled) vectors from
    several latent spaces is the job of an assembler (`assemblers/base.py`), not of a
    head.

    Declared as an `nn.Module` so that heads with learnable parameters (`linear`) and
    stateless ones (`identity`) go through the same registry and the same `nn.ModuleDict`.

    Attributes:
        in_dim: Width of the `mu` and `logvar` the head receives.
        out_dim: Width of the `mu` and `logvar` the head returns.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Store the input and output widths.

        Args:
            in_dim: Width of the encoder output the head adapts. Must be positive.
            out_dim: Width of the latent space the head feeds. Must be positive.

        Raises:
            ValueError: If `in_dim` or `out_dim` is not a positive integer.
        """
        super().__init__()
        for label, value in (("in_dim", in_dim), ("out_dim", out_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{label}` must be a positive integer, got {value!r}.")
        self._in_dim = in_dim
        self._out_dim = out_dim

    @property
    def in_dim(self) -> int:
        """Width of the `mu` and `logvar` the head receives.

        Returns:
            The input width given at construction.
        """
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """Width of the `mu` and `logvar` the head returns.

        Returns:
            The output width given at construction.
        """
        return self._out_dim

    @abstractmethod
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapt an encoder's distribution parameters to a latent space.

        Args:
            mu: Posterior mean, shape `(batch, in_dim)`.
            logvar: Posterior log-variance, shape `(batch, in_dim)`.

        Returns:
            The adapted `(mu, logvar)` pair, each of shape `(batch, out_dim)`.

        Raises:
            NotImplementedError: If called on the abstract base class.
            ValueError: If `mu` or `logvar` does not have shape `(batch, in_dim)`, or
                the two shapes differ.
        """
        raise NotImplementedError

    def _checkParams(self, mu: torch.Tensor, logvar: torch.Tensor) -> None:
        """Check that `mu` and `logvar` are two `(batch, in_dim)` tensors.

        Concrete heads call this first in `forward`, so a wiring mistake (an encoder
        whose output width is not the head's `in_dim`) fails with a message naming both
        widths instead of a shape error deep inside a matrix product.

        Args:
            mu: Posterior mean given to `forward`.
            logvar: Posterior log-variance given to `forward`.

        Raises:
            ValueError: If either tensor is not 2D with last dimension `in_dim`, or the
                two shapes differ.
        """
        for label, tensor in (("mu", mu), ("logvar", logvar)):
            if tensor.dim() != 2 or tensor.shape[-1] != self._in_dim:
                raise ValueError(
                    f"{type(self).__name__} expects `{label}` of shape (batch, "
                    f"{self._in_dim}), got {tuple(tensor.shape)}."
                )
        if mu.shape != logvar.shape:
            raise ValueError(
                f"{type(self).__name__} expects `mu` and `logvar` of the same shape, got "
                f"{tuple(mu.shape)} and {tuple(logvar.shape)}."
            )
