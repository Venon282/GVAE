def computeConv1dOutputLength(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """Compute a `Conv1d`'s exact output length (PyTorch's own formula).

    Args:
        input_length: Length of the input sequence.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding (applied to both sides).
        dilation: Convolution dilation.

    Returns:
        The resulting output length.
    """
    return (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def computeConvTranspose1dOutputLength(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    """Compute a `ConvTranspose1d`'s exact output length (PyTorch's own formula).

    Args:
        input_length: Length of the input sequence.
        kernel_size: Convolution kernel size.
        stride: Convolution stride (the upsampling factor).
        padding: Convolution padding (applied to both sides).
        output_padding: Extra length added to one side, resolving the
            stride-induced output-size ambiguity that `ConvTranspose1d`
            otherwise has. Must satisfy `0 <= output_padding < stride`
            (or `< dilation`, whichever is larger) for PyTorch itself
            to accept it.
        dilation: Convolution dilation.

    Returns:
        The resulting output length.
    """
    return (
        (input_length - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )

def computeUpsampleThenConv1dOutputLength(
    input_length: int,
    scale_factor: int,
    kernel_size: int,
    padding: int,
    dilation: int,
) -> int:
    """Compute the output length of `nn.Upsample` followed by a stride-1 `Conv1d`.

    This is the `"interpolate_conv"` upsampling mode's length formula:
    `nn.Upsample(scale_factor=..., mode="nearest")` exactly multiplies
    the length by `scale_factor` (nearest-neighbor repetition, not
    blurring), then the following `Conv1d` (implicitly stride 1)
    changes it by a fixed, computable amount.

    Args:
        input_length: Length of the input sequence, before upsampling.
        scale_factor: `nn.Upsample`'s integer scale factor.
        kernel_size: The following `Conv1d`'s kernel size.
        padding: The following `Conv1d`'s padding.
        dilation: The following `Conv1d`'s dilation.

    Returns:
        The resulting output length.
    """
    upsampled_length = input_length * scale_factor
    return computeConv1dOutputLength(
        upsampled_length, kernel_size, stride=1, padding=padding, dilation=dilation
    )

def solveConvTranspose1dOutputPadding(
    input_length: int,
    target_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """Solve for the `output_padding` that makes a `ConvTranspose1d` hit `target_length` exactly.

    Every other parameter fixed, `ConvTranspose1d`'s output length is
    an increasing, one-to-one function of `output_padding` alone (see
    `computeConvTranspose1dOutputLength`), so there is at most one
    value that closes the gap exactly; this solves for it directly
    rather than searching.

    Args:
        input_length: Length of the input sequence.
        target_length: Desired output length.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.

    Returns:
        The `output_padding` value that makes
        `computeConvTranspose1dOutputLength` return `target_length`
        exactly, for the same `input_length`/`kernel_size`/`stride`/
        `padding`/`dilation`. Not guaranteed to be a value PyTorch
        actually accepts (it must additionally satisfy
        `0 <= output_padding < max(stride, dilation)`); callers must
        check that themselves, since what counts as "acceptable" is a
        modeling decision, not something this function should silently
        decide.
    """
    length_with_zero_output_padding = computeConvTranspose1dOutputLength(
        input_length, kernel_size, stride, padding, output_padding=0, dilation=dilation
    )
    return target_length - length_with_zero_output_padding
