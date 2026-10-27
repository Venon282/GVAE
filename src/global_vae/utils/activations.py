"""Activation functions chosen by name, for the encoders and decoders a YAML file configures.

A `Callable` factory, as the CNN encoders take, cannot be written in YAML, so the MLP and waveform
backbones (roadmap P3-8) take a name and resolve it here.
"""

from collections.abc import Callable

from torch import nn

ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}
"""Activation name -> zero-argument factory returning a fresh module."""


def listSupportedActivations() -> list[str]:
    """Return every activation name `getActivationFactory` can resolve.

    Returns:
        The sorted activation names.
    """
    return sorted(ACTIVATIONS)


def getActivationFactory(name: str) -> Callable[[], nn.Module]:
    """Look an activation up by name.

    Args:
        name: One of `listSupportedActivations()`.

    Returns:
        A zero-argument factory returning a fresh activation module, which is the form the
        residual blocks of `utils.conv_blocks` take.

    Raises:
        ValueError: If `name` is not a known activation. The message lists the known ones.
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Expected one of {listSupportedActivations()}."
        )
    return ACTIVATIONS[name]
