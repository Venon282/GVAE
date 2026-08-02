"""Abstract interface for beta-weighting schedules (spec §2.3).

A beta schedule resolves a training step to the scalar weight applied
to one latent space's regularization term. It is a separate,
orthogonal concern from *which* regularization strategy computes that
latent space's raw penalty (`AbstractLatentRegularizer`,
`losses/regularizers/base.py`): choosing a schedule never depends on,
and never changes, the regularizer in use. Spec §2.3 requires exactly
this: "the schedule ... is a separate, orthogonal config concern from
*which* regularizer is used for a given latent space."

Deliberately not an `nn.Module`: a schedule is a plain function of an
integer step, not a tensor computation that participates in the
autograd graph, so a plain Python object is enough and no parameters
ever need gradients here.

Concrete subclasses must self-register via `@registerBetaSchedule(name)`
(see `registry.py`), e.g. `constant`, `linear_warmup`.
"""

from abc import ABC, abstractmethod


class AbstractBetaSchedule(ABC):
    """Base class for every beta-weighting schedule."""

    @abstractmethod
    def __call__(self, step: int) -> float:
        """Resolve this schedule's beta value at a given training step.

        Args:
            step: Current training step. Concrete implementations
                document their own handling of negative or
                out-of-range values (e.g. `LinearWarmupBetaSchedule`
                clamps them instead of raising).

        Returns:
            The beta weight to apply at this step.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError
