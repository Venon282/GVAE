from typing import Any

from torch import nn


def build1DPoolLayer(
    pooling: str | None,
    kernel_size: int | None,
    stride: int | None,
    padding: int,
    **kwargs: Any,
) -> nn.Module | None:
    """Build one stage's 1D pooling layer, or `None` if pooling is disabled.

    Args:
        pooling: `"max"`, `"avg"`, or `None` to skip pooling for this
            stage entirely (e.g. if strides alone handle
            downsampling).
        kernel_size: Pooling window size. Required (not `None`) unless
            `pooling` is `None`.
        stride: Pooling stride. `None` resolves to `kernel_size`, i.e.
            non-overlapping windows.
        padding: Pooling padding.
        **kwargs: Forwarded to the underlying `nn.MaxPool1d` /
            `nn.AvgPool1d` constructor (e.g. `ceil_mode`,
            `count_include_pad`, `return_indices`).

    Returns:
        The pooling module, or `None`.

    Raises:
        ValueError: If `pooling` is not one of `"max"`, `"avg"`, `None`,
            or if `pooling` is set but `kernel_size` is `None`.
    """
    if pooling is None:
        return None
    if kernel_size is None:
        raise ValueError(f"pooling='{pooling}' requires a kernel_size, but kernel_size is None.")
    resolved_stride = stride if stride is not None else kernel_size

    if pooling == "max":
        return nn.MaxPool1d(
            kernel_size=kernel_size, stride=resolved_stride, padding=padding, **kwargs
        )
    if pooling == "avg":
        return nn.AvgPool1d(
            kernel_size=kernel_size, stride=resolved_stride, padding=padding, **kwargs
        )

    raise ValueError(f"Unknown pooling '{pooling}'. Expected 'max', 'avg', or None.")


def build1DUpSampleStage(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    upsample_mode: str = "interpolate_conv",
    groups: int = 1,
    interp_mode: str = "nearest",
) -> nn.Module:
    """Build one transition's upsampling layer.

    Args:
        in_channels: Input channel width.
        out_channels: Output channel width.
        kernel_size: Convolution kernel size.
        stride: Upsampling factor for this transition.
        padding: Convolution padding.
        output_padding: Extra size added to one side of the
            `ConvTranspose1d` output, resolving stride-induced output
            size ambiguity. Ignored if `upsample_mode` is
            `"interpolate_conv"`.
        dilation: Convolution dilation.
        upsample_mode: `"conv_transpose"` (a single learned
            `ConvTranspose1d`) or `"interpolate_conv"`
            (nearest-neighbor upsampling followed by a stride-1
            `Conv1d`), which avoids the checkerboard artifacts
            transposed convolutions are prone to, at the cost of a
            fixed (non-learned) upsampling step.

    Returns:
        The upsampling module.

    Raises:
        ValueError: If `upsample_mode` is not recognized.
    """
    if upsample_mode == "conv_transpose":
        return nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
    if upsample_mode == "interpolate_conv":
        return nn.Sequential(
            nn.Upsample(scale_factor=stride, mode=interp_mode),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
        )

    raise ValueError(
        f"Unknown upsample_mode '{upsample_mode}'. Expected 'conv_transpose' or 'interpolate_conv'."
    )


# --- 2D counterparts (spec §6's image modality; TwoDCnnEncoder/TwoDCnnDecoder) ---
#
# Every hyperparameter below is an already-resolved `(height, width)` shape (see
# `utils.stage_config.broadcastPerStageShape`), not a plain scalar: a 2D building
# block may legitimately need a non-square kernel/stride/padding/dilation, and
# `nn.Conv2d`/`nn.ConvTranspose2d`/`nn.MaxPool2d`/`nn.AvgPool2d` all natively accept
# such a tuple directly, so no per-axis unpacking is needed here beyond what
# `broadcastPerStageShape` has already resolved upstream.


def build2DPoolLayer(
    pooling: str | None,
    kernel_size: tuple[int, int] | None,
    stride: tuple[int, int] | None,
    padding: tuple[int, int],
    **kwargs: Any,
) -> nn.Module | None:
    """Build one stage's 2D pooling layer, or `None` if pooling is disabled.

    Mirrors `build1DPoolLayer`, generalized to 2D: every shape
    hyperparameter is a resolved `(height, width)` pair instead of a
    plain scalar.

    Args:
        pooling: `"max"`, `"avg"`, or `None` to skip pooling for this
            stage entirely (e.g. if strides alone handle
            downsampling).
        kernel_size: Pooling window shape. Required (not `None`)
            unless `pooling` is `None`.
        stride: Pooling stride shape. `None` resolves to `kernel_size`,
            i.e. non-overlapping windows.
        padding: Pooling padding shape.
        **kwargs: Forwarded to the underlying `nn.MaxPool2d` /
            `nn.AvgPool2d` constructor (e.g. `ceil_mode`,
            `count_include_pad`, `return_indices`).

    Returns:
        The pooling module, or `None`.

    Raises:
        ValueError: If `pooling` is not one of `"max"`, `"avg"`, `None`,
            or if `pooling` is set but `kernel_size` is `None`.
    """
    if pooling is None:
        return None
    if kernel_size is None:
        raise ValueError(f"pooling='{pooling}' requires a kernel_size, but kernel_size is None.")
    resolved_stride = stride if stride is not None else kernel_size

    if pooling == "max":
        return nn.MaxPool2d(
            kernel_size=kernel_size, stride=resolved_stride, padding=padding, **kwargs
        )
    if pooling == "avg":
        return nn.AvgPool2d(
            kernel_size=kernel_size, stride=resolved_stride, padding=padding, **kwargs
        )

    raise ValueError(f"Unknown pooling '{pooling}'. Expected 'max', 'avg', or None.")


def build2DUpSampleStage(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
    upsample_mode: str = "interpolate_conv",
    groups: int = 1,
    interp_mode: str = "nearest",
) -> nn.Module:
    """Build one transition's 2D upsampling layer. Mirrors `build1DUpSampleStage`.

    Args:
        in_channels: Input channel width.
        out_channels: Output channel width.
        kernel_size: `(kernel_height, kernel_width)`.
        stride: `(stride_height, stride_width)`, the upsampling factor
            for this transition along each axis.
        padding: `(padding_height, padding_width)`.
        output_padding: `(output_padding_height, output_padding_width)`,
            resolving each axis's own stride-induced output-size
            ambiguity of `ConvTranspose2d`, independently. Ignored if
            `upsample_mode` is `"interpolate_conv"`.
        dilation: `(dilation_height, dilation_width)`.
        upsample_mode: `"conv_transpose"` (a single learned
            `ConvTranspose2d`) or `"interpolate_conv"`
            (nearest-neighbor upsampling followed by a stride-1
            `Conv2d`), which avoids the checkerboard artifacts
            transposed convolutions are prone to, at the cost of a
            fixed (non-learned) upsampling step. Exactly the same
            trade-off `build1DUpSampleStage` already documents, now
            per spatial axis.
        groups: Forwarded to the underlying `Conv2d`/`ConvTranspose2d`.
        interp_mode: Forwarded to `nn.Upsample`'s `mode` (only used by
            `"interpolate_conv"`); `"nearest"` (default) needs no
            `align_corners` and matches `build1DUpSampleStage`'s own
            convention of a fixed, non-learned, artifact-free upsample
            step.

    Returns:
        The upsampling module.

    Raises:
        ValueError: If `upsample_mode` is not recognized.
    """
    if upsample_mode == "conv_transpose":
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
    if upsample_mode == "interpolate_conv":
        return nn.Sequential(
            nn.Upsample(scale_factor=stride, mode=interp_mode),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
        )

    raise ValueError(
        f"Unknown upsample_mode '{upsample_mode}'. Expected 'conv_transpose' or 'interpolate_conv'."
    )
