"""Building blocks of the waveform encoder and decoder of the audio modality (roadmap P3-8).

A raw waveform is a long series (16,000 values per second at 16 kHz), so the stack that reads it
downsamples in a few large strides, and a stack of dilated residual units at each scale gives it
a wide receptive field without a large kernel. Every length rule below is exact and needs no
case analysis: a downsampling convolution of stride `s` has kernel `2 * s + 1` and padding `s`,
which maps a length `L` to `ceil(L / s)`, so no input collapses, and an upsampling stage
multiplies the length by exactly `s`. A decoder can therefore reach any output length by cropping
a slightly longer waveform, the way the transformer decoder crops its last patch.
"""

import math
from collections.abc import Callable, Sequence

from torch import nn

from global_vae.utils.activations import getActivationFactory
from global_vae.utils.conv_blocks import Residual1DBlock
from global_vae.utils.stage_config import broadcastPerStage

NORMALIZATIONS: tuple[str | None, ...] = (None, "group", "batch")
"""Every value `getNormalizationFactory` accepts: none, one-group `GroupNorm`, `BatchNorm1d`."""


def getNormalizationFactory(name: str | None) -> Callable[[int], nn.Module] | None:
    """Resolve a normalization name into the factory the residual blocks take.

    Args:
        name: `None` (no normalization), `"group"` (`GroupNorm` with a single group, which
            normalizes each sample over its channels and time, so it does not depend on the
            batch and does not care about the length) or `"batch"` (`BatchNorm1d`, whose
            statistics come from the batch in training, so training and evaluation differ).

    Returns:
        A one-argument factory taking a channel count, or `None` for no normalization.

    Raises:
        ValueError: If `name` is not one of `NORMALIZATIONS`.
    """
    if name not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization {name!r}. Expected one of {list(NORMALIZATIONS)}.")
    if name == "group":
        return lambda channels: nn.GroupNorm(1, channels)
    if name == "batch":
        return nn.BatchNorm1d
    return None


def resolveWaveformStages(
    hidden_channels: Sequence[int], strides: int | Sequence[int], owner: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Check the channel widths and strides of a waveform stack and give each stage its stride.

    Args:
        hidden_channels: Channel width of each stage. Its length is the number of stages.
        strides: One stride shared by every stage, or one per stage.
        owner: Name of the calling class, for the error messages.

    Returns:
        `(hidden_channels, strides)` as tuples of the same length, one entry per stage.

    Raises:
        ValueError: If there is no stage, a width or a stride is below `1`, or `strides` is a
            sequence whose length is not the number of stages. The message names the argument.
    """
    if len(hidden_channels) < 1:
        raise ValueError(f"{owner}: hidden_channels needs at least one stage, got none.")
    if any(width < 1 for width in hidden_channels):
        raise ValueError(
            f"{owner}: hidden_channels must all be at least 1, got {list(hidden_channels)}."
        )
    per_stage = broadcastPerStage(strides, len(hidden_channels), "strides")
    if any(stride < 1 for stride in per_stage):
        raise ValueError(f"{owner}: strides must all be at least 1, got {list(per_stage)}.")
    return tuple(hidden_channels), per_stage


def buildDilatedResidualUnits(
    channels: int,
    dilations: Sequence[int],
    kernel_size: int,
    activation: str,
    normalization: str | None,
) -> nn.Sequential:
    """Build a stack of residual units, one per dilation, at one channel width.

    Each unit is a `Residual1DBlock` of two convolutions with that dilation and "same" padding,
    so it keeps the length. Dilations such as `(1, 3, 9)` multiply the receptive field of a stage
    by about 13 for the price of three units.

    Args:
        channels: Channel width of every unit (input and output).
        dilations: One dilation per unit, each at least `1`. At least one.
        kernel_size: Kernel size of the convolutions, odd.
        activation: An activation name, see `utils.activations`.
        normalization: A name of `NORMALIZATIONS`.

    Returns:
        An `nn.Sequential` of the units.

    Raises:
        ValueError: If `dilations` is empty or has a value below `1`, `kernel_size` is even, or
            `activation` or `normalization` is unknown.
    """
    if len(dilations) < 1 or any(dilation < 1 for dilation in dilations):
        raise ValueError(
            f"dilations must be a non-empty list of integers of at least 1, got {list(dilations)}."
        )
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
    activation_factory = getActivationFactory(activation)
    normalization_factory = getNormalizationFactory(normalization)
    return nn.Sequential(
        *(
            Residual1DBlock(
                channels,
                channels,
                depth=2,
                kernel_size=kernel_size,
                dilation=dilation,
                activation=activation_factory,
                normalization=normalization_factory,
            )
            for dilation in dilations
        )
    )


def buildDownsampleConv(in_channels: int, out_channels: int, stride: int) -> nn.Conv1d:
    """Build the strided convolution that ends an encoder stage: `L` becomes `ceil(L / stride)`.

    Args:
        in_channels: Input channel width.
        out_channels: Output channel width.
        stride: Downsampling factor. `1` keeps the length (a kernel of 3).

    Returns:
        A `Conv1d` with kernel `2 * stride + 1` and padding `stride`.
    """
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=2 * stride + 1, stride=stride, padding=stride
    )


def buildUpsampleStage(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    """Build the layers that begin a decoder stage: `L` becomes exactly `L * stride`.

    Nearest-neighbour upsampling followed by a convolution, which does not leave the
    checkerboard of a transposed convolution on a waveform.

    Args:
        in_channels: Input channel width.
        out_channels: Output channel width.
        stride: Upsampling factor. `1` keeps the length.

    Returns:
        An `nn.Sequential` of `nn.Upsample` and a `Conv1d` with kernel `2 * stride + 1`.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode="nearest"),
        nn.Conv1d(in_channels, out_channels, kernel_size=2 * stride + 1, padding=stride),
    )


def computeDownsampledLength(length: int, strides: Sequence[int]) -> int:
    """Return the number of frames an encoder stack leaves from `length` samples.

    Args:
        length: Number of input samples, at least `1`.
        strides: The stride of each stage.

    Returns:
        `length` divided by each stride in turn, rounded up each time. Never below `1`.
    """
    frames = length
    for stride in strides:
        frames = math.ceil(frames / stride)
    return frames


def computeSeedLength(output_length: int, strides: Sequence[int]) -> int:
    """Return the length of the seed a decoder upsamples to reach `output_length` samples.

    Args:
        output_length: Number of samples to reconstruct, at least `1`.
        strides: The upsampling factor of each stage.

    Returns:
        `ceil(output_length / prod(strides))`: the smallest seed whose upsampled length is at
        least `output_length`, so that the decoder crops fewer than `prod(strides)` samples.
    """
    return math.ceil(output_length / math.prod(strides))
