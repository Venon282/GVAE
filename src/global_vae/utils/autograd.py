"""Typed wrapper around `torch.Tensor.backward` (spec §10).

`torch`'s type stubs leave `Tensor.backward` unannotated. Calling it
directly from a fully-typed context (i.e. wherever `mypy` has not
already lost track of the tensor's type through an untyped chain, such
as `nn.Module.__call__`) trips `mypy --strict`'s
`disallow_untyped_calls` check with a `no-untyped-call` error, even
though the call is correct at runtime. Every future training step
(`training/trainer.py`, still deferred) will need to call
`.backward()`, so this centralizes the one necessary `type: ignore`
here instead of scattering it at each call site.
"""

import torch


def backward(loss: torch.Tensor, retain_graph: bool = False) -> None:
    """Run backpropagation from a scalar loss tensor.

    Args:
        loss: Scalar tensor to backpropagate from.
        retain_graph: Forwarded to `torch.Tensor.backward`; set to
            `True` to keep the autograd graph around for a further
            `backward()` call against the same graph.
    """
    loss.backward(retain_graph=retain_graph)  # type: ignore[no-untyped-call]
