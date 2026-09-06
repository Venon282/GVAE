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

def solveMinimumInputLengthForConv1d(
    min_output_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """Solve for the minimum input length whose `Conv1d` output length reaches `min_output_length`.

    `computeConv1dOutputLength` is a nondecreasing function of `input_length` (stride,
    padding, dilation held fixed), so there is a unique minimum input length that reaches
    any given output length target. Also valid for pooling layers (`MaxPool1d`/`AvgPool1d`),
    which share the same length formula: call with `dilation=1` for `AvgPool1d`, which has
    no dilation parameter of its own.

    Args:
        min_output_length: Smallest acceptable output length for this layer (typically `1`,
            to keep the layer from collapsing its input to an empty or invalid feature map).
        kernel_size: Layer kernel size.
        stride: Layer stride.
        padding: Layer padding (applied to both sides).
        dilation: Layer dilation.

    Returns:
        The minimum input length whose `computeConv1dOutputLength` result is at least
        `min_output_length`. Always at least `1`.
    """
    minimum = stride * (min_output_length - 1) - 2 * padding + dilation * (kernel_size - 1) + 1
    return max(1, minimum)

def computeConv1dLengthOffset(kernel_size: int, padding: int, dilation: int) -> int:
    """Compute a stride-1 `Conv1d`'s length-changing "offset" (`utils/conv_blocks.py`).

    At `stride=1`, `computeConv1dOutputLength` reduces to exactly
    `input_length + offset`, where `offset = 2 * padding - dilation *
    (kernel_size - 1)`. This is the one number that determines whether
    two differently-parameterized stride-1 `Conv1d` layers produce the
    *same* output length for every input length (they do iff their
    offsets are equal), which is exactly the guarantee a residual
    block's shortcut path needs against its main path
    (`utils.conv_blocks.Residual1DBlock`/`Residual1DUpBlock`): checking
    one example length is not enough, since floor-division-based
    formulas can agree at one length and disagree at another.

    Args:
        kernel_size: Convolution kernel size.
        padding: Convolution padding (applied to both sides).
        dilation: Convolution dilation.

    Returns:
        The length offset. `0` means this stride-1 configuration
        preserves input length exactly (see `isLengthPreservingConv1d`).
    """
    return 2 * padding - dilation * (kernel_size - 1)

def isLengthPreservingConv1d(kernel_size: int, padding: int, dilation: int) -> bool:
    """Whether a stride-1 `Conv1d` with these hyperparameters preserves input length exactly.

    Args:
        kernel_size: Convolution kernel size.
        padding: Convolution padding (applied to both sides).
        dilation: Convolution dilation.

    Returns:
        `True` if `computeConv1dOutputLength(L, kernel_size, 1, padding, dilation) == L`
        for every `L`. Note this can never be `True` for an even
        `kernel_size` when `dilation` is odd (no integer `padding`
        solves `2 * padding == dilation * (kernel_size - 1)` when the
        right-hand side is odd), which is why every "same"-padded,
        length-preserving layer in this codebase (this one included)
        needs an odd effective kernel span.
    """
    return computeConv1dLengthOffset(kernel_size, padding, dilation) == 0

def computeConvTranspose1dLengthOffset(
    kernel_size: int, padding: int, output_padding: int, dilation: int
) -> int:
    """Compute a `ConvTranspose1d`'s length-changing "offset" (`utils/conv_blocks.py`).

    `computeConvTranspose1dOutputLength` reduces to exactly
    `(input_length - 1) * stride + offset + 1`, where `offset =
    -2 * padding + dilation * (kernel_size - 1) + output_padding`. Two
    `ConvTranspose1d` layers sharing the same `stride` produce the
    *same* output length for every input length iff their offsets
    match, the `"conv_transpose"`-mode analogue of
    `computeConv1dLengthOffset`, used the same way to verify a
    residual up-block's shortcut against its main path
    (`utils.conv_blocks.Residual1DUpBlock`).

    Args:
        kernel_size: Convolution kernel size.
        padding: Convolution padding (applied to both sides).
        output_padding: `ConvTranspose1d`'s `output_padding`.
        dilation: Convolution dilation.

    Returns:
        The length offset.
    """
    return -2 * padding + dilation * (kernel_size - 1) + output_padding

def computeUpsampleStackOutputLength(
    seed_length: int,
    kernel_sizes: tuple[int, ...],
    strides: tuple[int, ...],
    paddings: tuple[int, ...],
    output_paddings: tuple[int, ...],
    dilations: tuple[int, ...],
    upsample_modes: tuple[str, ...],
) -> int:
    """Chain the per-transition upsampling length formula across an already-resolved stack.

    Shared by `OneDCnnDecoder` and `OneDResidualDecoder` (spec §6, §12:
    the two decoders' transitions differ in what happens *between* each
    length change, plain layers versus a residual block, but the length
    arithmetic that gets a series from `seed_length` to the final
    `output_length` is identical either way, since only a transition's
    first layer ever changes length; see `Residual1DUpBlock`'s own
    docstring). Originally private to `OneDCnnDecoder`
    (`_computeLengthFromResolved`); relocated here, unchanged, so both
    decoders share one implementation instead of a second copy drifting
    out of sync with the first.

    Args:
        seed_length: Length before any transition is applied.
        kernel_sizes: Per-transition kernel sizes.
        strides: Per-transition strides.
        paddings: Per-transition paddings.
        output_paddings: Per-transition `ConvTranspose1d` output
            paddings. Ignored (but must still be a same-length tuple)
            if `upsample_mode` is `"interpolate_conv"`.
        dilations: Per-transition dilations.
        upsample_modes: `"conv_transpose"` or `"interpolate_conv"`.

    Returns:
        The length after every transition.

    Raises:
        ValueError: If `upsample_mode` is not recognized.
    """
    length = seed_length
    for stage in range(len(kernel_sizes)):
        if upsample_modes[stage] == "conv_transpose":
            length = computeConvTranspose1dOutputLength(
                length,
                kernel_sizes[stage],
                strides[stage],
                paddings[stage],
                output_paddings[stage],
                dilations[stage],
            )
        elif upsample_modes[stage] == "interpolate_conv":
            length = computeUpsampleThenConv1dOutputLength(
                length,
                strides[stage],
                kernel_sizes[stage],
                paddings[stage],
                dilations[stage],
            )
        else:
            raise ValueError(
                f"Unknown upsample_mode '{upsample_modes[stage]}'. Expected 'conv_transpose' or "
                f"'interpolate_conv'."
            )
    return length
