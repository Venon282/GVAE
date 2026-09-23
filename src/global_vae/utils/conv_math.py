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

    Shared by `OneDCnnDecoder` and `OneDCnnResidualDecoder` (spec §6, §12:
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


# --- 2D counterparts (spec §6's image modality; TwoDCnnEncoder/TwoDCnnDecoder) ---
#
# Both spatial axes of a Conv2d/ConvTranspose2d/MaxPool2d/AvgPool2d are independent
# under PyTorch's own formulas: each axis's output length only ever depends on that
# same axis's own kernel_size/stride/padding/dilation, never on the other axis's.
# Every function below is therefore exactly the corresponding 1D function above,
# applied once per axis to an already-resolved `(height, width)` shape (see
# `utils.stage_config.broadcastPerStageShape`, which is what produces these
# per-axis tuples from a caller's shared-or-per-stage, square-or-non-square
# hyperparameters in the first place). This keeps one single implementation of the
# actual arithmetic (the 1D functions above); nothing below re-derives it.


def computeConv2dOutputShape(
    input_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Compute a `Conv2d`'s exact output shape (PyTorch's own formula, per axis).

    Args:
        input_shape: `(height, width)` of the input feature map.
        kernel_size: `(kernel_height, kernel_width)`.
        stride: `(stride_height, stride_width)`.
        padding: `(padding_height, padding_width)`, applied to both
            sides of each axis.
        dilation: `(dilation_height, dilation_width)`.

    Returns:
        `(output_height, output_width)`.
    """
    return (
        computeConv1dOutputLength(
            input_shape[0], kernel_size[0], stride[0], padding[0], dilation[0]
        ),
        computeConv1dOutputLength(
            input_shape[1], kernel_size[1], stride[1], padding[1], dilation[1]
        ),
    )


def computeConvTranspose2dOutputShape(
    input_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Compute a `ConvTranspose2d`'s exact output shape (PyTorch's own formula, per axis).

    Args:
        input_shape: `(height, width)` of the input feature map.
        kernel_size: `(kernel_height, kernel_width)`.
        stride: `(stride_height, stride_width)`, the upsampling factor
            along each axis.
        padding: `(padding_height, padding_width)`.
        output_padding: `(output_padding_height, output_padding_width)`,
            resolving each axis's own stride-induced output-size
            ambiguity independently. Each component must satisfy
            `0 <= output_padding[axis] < max(stride[axis],
            dilation[axis])` for PyTorch itself to accept it.
        dilation: `(dilation_height, dilation_width)`.

    Returns:
        `(output_height, output_width)`.
    """
    return (
        computeConvTranspose1dOutputLength(
            input_shape[0], kernel_size[0], stride[0], padding[0], output_padding[0], dilation[0]
        ),
        computeConvTranspose1dOutputLength(
            input_shape[1], kernel_size[1], stride[1], padding[1], output_padding[1], dilation[1]
        ),
    )


def computeUpsampleThenConv2dOutputShape(
    input_shape: tuple[int, int],
    scale_factor: tuple[int, int],
    kernel_size: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Compute the output shape of `nn.Upsample` followed by a stride-1 `Conv2d`.

    The `"interpolate_conv"` upsampling mode's shape formula (see
    `computeUpsampleThenConv1dOutputLength`, applied per axis):
    `nn.Upsample(scale_factor=..., mode="nearest")` multiplies each
    axis's length by that axis's own scale factor, then the following
    `Conv2d` (implicitly stride 1) changes each axis by a fixed,
    independently computable amount.

    Args:
        input_shape: `(height, width)`, before upsampling.
        scale_factor: `(scale_height, scale_width)`, `nn.Upsample`'s
            own per-axis integer scale factors.
        kernel_size: The following `Conv2d`'s `(kernel_height,
            kernel_width)`.
        padding: The following `Conv2d`'s `(padding_height,
            padding_width)`.
        dilation: The following `Conv2d`'s `(dilation_height,
            dilation_width)`.

    Returns:
        `(output_height, output_width)`.
    """
    upsampled_shape = (input_shape[0] * scale_factor[0], input_shape[1] * scale_factor[1])
    return computeConv2dOutputShape(
        upsampled_shape, kernel_size, stride=(1, 1), padding=padding, dilation=dilation
    )


def solveConvTranspose2dOutputPadding(
    input_shape: tuple[int, int],
    target_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Solve, independently per axis, the `output_padding` that makes a `ConvTranspose2d`
    hit `target_shape` exactly.

    Each axis of a `ConvTranspose2d` is an independent 1D
    `ConvTranspose1d`-equivalent computation (see the module-level
    note above), so this is `solveConvTranspose1dOutputPadding`
    applied once per axis; the two axes' solved values are entirely
    unrelated to one another (e.g. reconstructing a non-square image
    routinely needs a different `output_padding` on each axis).

    Args:
        input_shape: `(height, width)` of the input feature map.
        target_shape: Desired `(height, width)`.
        kernel_size: `(kernel_height, kernel_width)`.
        stride: `(stride_height, stride_width)`.
        padding: `(padding_height, padding_width)`.
        dilation: `(dilation_height, dilation_width)`.

    Returns:
        `(output_padding_height, output_padding_width)`. Not
        guaranteed to be a value PyTorch actually accepts on either
        axis (each component must additionally satisfy `0 <=
        output_padding[axis] < max(stride[axis], dilation[axis])`);
        callers must check that themselves, exactly as in the 1D case.
    """
    return (
        solveConvTranspose1dOutputPadding(
            input_shape[0], target_shape[0], kernel_size[0], stride[0], padding[0], dilation[0]
        ),
        solveConvTranspose1dOutputPadding(
            input_shape[1], target_shape[1], kernel_size[1], stride[1], padding[1], dilation[1]
        ),
    )


def solveMinimumInputShapeForConv2d(
    min_output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Solve, independently per axis, the minimum input shape whose `Conv2d` output
    shape reaches `min_output_shape`.

    Also valid for 2D pooling layers (`MaxPool2d`/`AvgPool2d`), exactly
    as `solveMinimumInputLengthForConv1d` is for their 1D counterparts:
    call with `dilation=(1, 1)` for `AvgPool2d`, which has no dilation
    parameter of its own.

    Args:
        min_output_shape: Smallest acceptable `(height, width)` for
            this layer's output (typically `(1, 1)`, to keep the layer
            from collapsing either axis of its input to an empty or
            invalid feature map).
        kernel_size: `(kernel_height, kernel_width)`.
        stride: `(stride_height, stride_width)`.
        padding: `(padding_height, padding_width)`.
        dilation: `(dilation_height, dilation_width)`.

    Returns:
        The minimum `(height, width)` whose `computeConv2dOutputShape`
        result is at least `min_output_shape` on both axes. Each
        component is always at least `1`.
    """
    return (
        solveMinimumInputLengthForConv1d(
            min_output_shape[0], kernel_size[0], stride[0], padding[0], dilation[0]
        ),
        solveMinimumInputLengthForConv1d(
            min_output_shape[1], kernel_size[1], stride[1], padding[1], dilation[1]
        ),
    )


def computeConv2dLengthOffset(
    kernel_size: tuple[int, int], padding: tuple[int, int], dilation: tuple[int, int]
) -> tuple[int, int]:
    """Compute a stride-1 `Conv2d`'s length-changing "offset", independently per axis.

    The 2D counterpart of `computeConv1dLengthOffset`, applied once per
    axis for the identical reason `computeConv2dOutputShape` is (see
    the module-level note above): each axis of a `Conv2d` is an
    independent 1D computation under PyTorch's own formula, so there is
    nothing new to derive here, only the existing 1D offset applied
    twice. Used by `utils.conv_blocks.Residual2DBlock`/
    `Residual2DUpBlock` to verify a residual shortcut reaches the exact
    same output shape as its main path, for every input shape,
    independently on both axes (checking one example shape is not
    enough, for the same floor-division reason
    `computeConv1dLengthOffset`'s own docstring explains).

    Args:
        kernel_size: `(kernel_height, kernel_width)`.
        padding: `(padding_height, padding_width)`, applied to both
            sides of each axis.
        dilation: `(dilation_height, dilation_width)`.

    Returns:
        `(offset_height, offset_width)`. `(0, 0)` means this stride-1
        configuration preserves input shape exactly on both axes.
    """
    return (
        computeConv1dLengthOffset(kernel_size[0], padding[0], dilation[0]),
        computeConv1dLengthOffset(kernel_size[1], padding[1], dilation[1]),
    )


def computeConvTranspose2dLengthOffset(
    kernel_size: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Compute a `ConvTranspose2d`'s length-changing "offset", independently per axis.

    The 2D counterpart of `computeConvTranspose1dLengthOffset`, applied
    once per axis for the identical reason `computeConv2dLengthOffset`
    is. Used by `utils.conv_blocks.Residual2DUpBlock` to verify a
    `"conv_transpose"`-mode shortcut reaches the exact same output
    shape as its main path, for every input shape, on both axes.

    Args:
        kernel_size: `(kernel_height, kernel_width)`.
        padding: `(padding_height, padding_width)`.
        output_padding: `(output_padding_height, output_padding_width)`.
        dilation: `(dilation_height, dilation_width)`.

    Returns:
        `(offset_height, offset_width)`.
    """
    return (
        computeConvTranspose1dLengthOffset(
            kernel_size[0], padding[0], output_padding[0], dilation[0]
        ),
        computeConvTranspose1dLengthOffset(
            kernel_size[1], padding[1], output_padding[1], dilation[1]
        ),
    )


def computeUpsampleStack2dOutputShape(
    seed_shape: tuple[int, int],
    kernel_sizes: tuple[tuple[int, int], ...],
    strides: tuple[tuple[int, int], ...],
    paddings: tuple[tuple[int, int], ...],
    output_paddings: tuple[tuple[int, int], ...],
    dilations: tuple[tuple[int, int], ...],
    upsample_modes: tuple[str, ...],
) -> tuple[int, int]:
    """Chain the per-transition 2D upsampling shape formula across an already-resolved stack.

    The `TwoDCnnDecoder` counterpart of `computeUpsampleStackOutputLength`,
    generalized to `(height, width)` shapes throughout (both axes are
    independent at every transition, so this is not a new formula, only
    the same chaining logic driven by 2D shapes instead of 1D lengths).

    Args:
        seed_shape: `(height, width)` before any transition is applied.
        kernel_sizes: Per-transition `(kernel_height, kernel_width)`.
        strides: Per-transition `(stride_height, stride_width)`.
        paddings: Per-transition `(padding_height, padding_width)`.
        output_paddings: Per-transition `ConvTranspose2d` output
            paddings. Ignored (but must still be a same-length tuple)
            for any transition whose `upsample_mode` is
            `"interpolate_conv"`.
        dilations: Per-transition `(dilation_height, dilation_width)`.
        upsample_modes: `"conv_transpose"` or `"interpolate_conv"`,
            per transition.

    Returns:
        The `(height, width)` after every transition.

    Raises:
        ValueError: If any transition's `upsample_mode` is not
            recognized.
    """
    shape = seed_shape
    for stage in range(len(kernel_sizes)):
        if upsample_modes[stage] == "conv_transpose":
            shape = computeConvTranspose2dOutputShape(
                shape,
                kernel_sizes[stage],
                strides[stage],
                paddings[stage],
                output_paddings[stage],
                dilations[stage],
            )
        elif upsample_modes[stage] == "interpolate_conv":
            shape = computeUpsampleThenConv2dOutputShape(
                shape,
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
    return shape
