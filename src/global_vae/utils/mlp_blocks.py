"""Building blocks of the MLP encoder and decoder of the tabular modality (roadmap P3-8).

A fully connected stack is the one thing a tabular encoder and a tabular decoder have in common,
so it lives here once and both call it. The activation and the normalization are chosen by name,
which a YAML file can express (a `Callable` factory, as the CNN encoders take, cannot be written
in one).
"""

from collections.abc import Sequence

from torch import nn

from global_vae.utils.activations import getActivationFactory

NORMALIZATIONS: tuple[str | None, ...] = (None, "layer", "batch")
"""Every value `buildMlp` accepts as `normalization`: none, `LayerNorm` or `BatchNorm1d`."""


def buildMlp(
    in_features: int,
    hidden_dims: Sequence[int],
    activation: str = "relu",
    normalization: str | None = None,
    dropout: float = 0.0,
) -> tuple[nn.Module, int]:
    """Build a stack of `Linear` layers, each followed by a normalization, an activation, a dropout.

    The stack ends after the last hidden layer's activation: the layer that follows (the
    `(mu, logvar)` heads of an encoder, the output layer of a decoder) is the caller's, since it
    is the part that differs.

    Args:
        in_features: Width of the input.
        hidden_dims: Width of each hidden layer. Empty gives an identity stack, so the caller's
            output layer reads the input directly.
        activation: One of `utils.activations.listSupportedActivations()`.
        normalization: `None` (default), `"layer"` (`LayerNorm`, independent of the batch) or
            `"batch"` (`BatchNorm1d`, which needs a training batch of at least two samples).
        dropout: Dropout probability after each hidden layer, in `[0, 1)`. `0` (default) adds no
            layer.

    Returns:
        `(stack, out_features)`: the module, and the width it outputs (the last hidden width, or
        `in_features` for an empty `hidden_dims`).

    Raises:
        ValueError: If `in_features` or a hidden width is below `1`, `activation` or
            `normalization` is not a known choice, or `dropout` is outside `[0, 1)`. The message
            names the argument.
    """
    if in_features < 1:
        raise ValueError(f"in_features must be at least 1, got {in_features}.")
    if any(width < 1 for width in hidden_dims):
        raise ValueError(f"hidden_dims must all be at least 1, got {list(hidden_dims)}.")
    activation_factory = getActivationFactory(activation)
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {normalization!r}. Expected one of {list(NORMALIZATIONS)}."
        )
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    layers: list[nn.Module] = []
    width = in_features
    for hidden in hidden_dims:
        layers.append(nn.Linear(width, hidden))
        if normalization == "layer":
            layers.append(nn.LayerNorm(hidden))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(activation_factory())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        width = hidden
    stack: nn.Module = nn.Sequential(*layers) if layers else nn.Identity()
    return stack, width
