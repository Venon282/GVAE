"""Chains several transforms into one invertible pipeline (spec §6.2).

Not registered under `@registerTransform`: this is a combinator built from
already-resolved `AbstractTransform` instances (typically via
`config.data.buildTransformPipeline`), not itself a named strategy a caller
selects by string. Giving `DataConfig.transforms` (a list of steps) and
`visualization.reconstruction_plot`'s own `inverse_transform` hook (a single
`Callable[[Tensor], Tensor]`) a common, correctly-ordered inverse is the
whole point of this class: hand-chaining several transforms' own
`.apply`/`.inverse` calls in the right (and easy to get backwards) order is
exactly the kind of boilerplate this removes.
"""

from collections.abc import Sequence

import torch

from global_vae.data.transforms.base import AbstractTransform


class ComposeTransform(AbstractTransform):
    """Applies a sequence of transforms in order; inverts them in reverse order.

    Attributes:
        transforms: The steps this pipeline chains, in application
            order.
    """

    def __init__(self, transforms: Sequence[AbstractTransform]) -> None:
        """Initialize the pipeline.

        Args:
            transforms: Steps to chain, in the order `apply` runs them
                (`inverse` runs them in the exact reverse order). May
                be empty, in which case both `apply` and `inverse` are
                the identity.
        """
        self.transforms = list(transforms)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Run every step's `apply`, in order.

        Args:
            x: Input tensor, any shape (whatever the first step, if
                any, accepts).

        Returns:
            The tensor after every step has been applied.
        """
        for transform in self.transforms:
            x = transform.apply(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Run every step's `inverse`, in reverse order.

        Args:
            y: A tensor previously produced by `apply` (or a tensor in
                that same space).

        Returns:
            The tensor mapped back toward `apply`'s input space.

        Raises:
            ValueError: If any step's own `inverse` cannot be run
                (e.g. a `ResampleTransform` step built without a
                `source_size`); propagated unchanged from that step.
        """
        for transform in reversed(self.transforms):
            y = transform.inverse(y)
        return y
