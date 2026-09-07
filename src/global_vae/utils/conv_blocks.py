"""Reusable 1D residual building blocks (spec §6, §7, §12).

Shared by `encoders.OneDCnnResidualEncoder` and `decoders.OneDCnnResidualDecoder`:
a residual block stacks `depth` convolutional layers (`depth >= 1`,
independently configurable per stage, e.g. "3 layers before the connection,
then 4" for a deeper second stage) and adds a shortcut connection from the
block's input to its output,

    y = activation(F(x) + shortcut(x))

the standard ResNet "BasicBlock" pattern (He et al., 2015), generalized to an
arbitrary, per-block depth. Only the block's *first* layer may change the
number of channels and/or the spatial length (a stride, for the
encoder-side `Residual1DBlock`; an upsampling transition, for the
decoder-side `Residual1DUpBlock`); every layer after it keeps both channel
width and length fixed. This is what makes a single shortcut per block
well-defined (rather than one per internal layer) and, just as importantly,
is what makes the block's contribution to the surrounding encoder's/
decoder's overall length bookkeeping identical to a plain (non-residual)
stage in `OneDCnnEncoder`/`OneDCnnDecoder`: the *whole* block changes length
exactly the way its first layer alone would, so `OneDCnnResidualEncoder`/
`OneDCnnResidualDecoder` can reuse the exact same length-solving machinery
(`utils/conv_math.py`) that the non-residual classes already use, per
stage, unmodified.

The shortcut path is a single projection layer (kernel size configurable,
default `1`, the standard ResNet choice) whenever the block changes channel
count and/or length; otherwise the input is passed through unchanged
(identity shortcut, no extra parameters). Its hyperparameters are resolved
so its output length is *provably* identical to the main path's, for every
input length, not merely verified against one example length: two
stride-1 `Conv1d` layers (or two `ConvTranspose1d` layers at the same
stride) produce the same output length for every input length if and only
if their length "offsets" match (`computeConv1dLengthOffset`/
`computeConvTranspose1dLengthOffset`, `utils/conv_math.py`); this module
checks that exactly, at construction time, and raises `ValueError`
immediately if a chosen configuration cannot guarantee it (spec §12:
verify a configuration instead of silently producing a shape mismatch a
later `+` would fail on, or worse, silently broadcast/crop around).

A block's internal (non-first) layers must exactly preserve length (they
run at stride 1, feeding directly into the next internal layer or into the
final addition), which requires an odd `kernel_size` whenever `depth > 1`:
no integer `padding` makes a stride-1, dilation-`d` layer with an even
kernel span exactly length-preserving (`2 * padding == dilation *
(kernel_size - 1)` has no integer solution when the right-hand side is
odd). This is a real, unavoidable constraint of convolution arithmetic, not
a limitation specific to this codebase; construction raises a clear
`ValueError` naming it if violated, rather than a stray runtime shape
mismatch several layers downstream. `depth=1` (a single, possibly-strided
projection layer plus a shortcut) has no internal layer at all and is
unaffected by this constraint.
"""

from collections.abc import Callable

import torch
from torch import nn

from global_vae.utils.builders import build1DUpSampleStage
from global_vae.utils.conv_math import (
    computeConv1dLengthOffset,
    computeConvTranspose1dLengthOffset,
)


def _requireOddKernelForDepth(depth: int, kernel_size: int, block_name: str) -> None:
    """Raise a clear error if `depth > 1` cannot possibly preserve length internally.

    Args:
        depth: Number of conv layers in the block.
        kernel_size: Kernel size shared by every layer in the block.
        block_name: Which class is raising, used only for the error message.

    Raises:
        ValueError: If `depth > 1` and `kernel_size` is even (see the
            module docstring: no integer padding can make a stride-1,
            even-kernel layer exactly length-preserving).
    """
    if depth > 1 and kernel_size % 2 == 0:
        raise ValueError(
            f"{block_name} with depth={depth} > 1 requires an odd kernel_size for its "
            f"internal, length-preserving layers (a stride-1 layer with an even kernel_size "
            f"cannot preserve length exactly with any integer padding), got "
            f"kernel_size={kernel_size}. Use an odd kernel_size, or depth=1 if an even "
            f"kernel_size is required."
        )


class Residual1DBlock(nn.Module):
    """One 1D residual block for an encoder: `depth` stacked `Conv1d` layers plus a shortcut.

    Only the first layer may downsample (`stride > 1`) and/or change
    channel width (`in_channels -> out_channels`); every subsequent layer
    keeps both fixed (stride 1, `out_channels -> out_channels`, "same"
    padding), which is what keeps the block's shortcut and its
    contribution to the surrounding encoder's overall output length
    identical to a plain, single-layer stage of the same
    stride/kernel_size/padding/dilation (see the module docstring).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        shortcut_kernel_size: int = 1,
        activation: Callable[[], nn.Module] | None = nn.ReLU,
        normalization: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        """Build the block.

        Args:
            in_channels: Input channel width.
            out_channels: Output channel width of every layer in this
                block (the first layer projects `in_channels ->
                out_channels`; every later layer is `out_channels ->
                out_channels`).
            depth: Number of `Conv1d` layers in this block. Must be at
                least `1`. `2` (the default) is the classic ResNet
                "BasicBlock"; pass a larger value for a deeper block
                (e.g. `3`), or `1` for a single projection layer plus a
                shortcut (no internal layers at all).
            kernel_size: Kernel size shared by every layer in this
                block. Must be odd if `depth > 1` (see the module
                docstring).
            stride: Stride of the first layer only (the block's only
                length-changing layer, aside from `padding`/`dilation`
                choices); every later layer always uses stride `1`.
            padding: Padding of the first layer. `None` (default)
                resolves to `dilation * (kernel_size // 2)`, matching
                every other "same"-style default in this codebase
                (`OneDCnnEncoder`'s own default padding).
            dilation: Dilation shared by every layer in this block.
            shortcut_kernel_size: Kernel size of the shortcut's
                projection conv, only built when a projection is
                actually needed (`in_channels != out_channels` or
                `stride != 1`). Defaults to `1`, the standard ResNet
                choice. Its own padding is always the same "same"-style
                default as above (with `dilation=1`, since the
                shortcut is a single, un-dilated projection); if that
                does not reach the exact same output length as the
                main path's first layer for every input length,
                construction raises `ValueError` naming both offsets
                (see `utils.conv_math.computeConv1dLengthOffset`)
                rather than silently building a mismatched shortcut.
            activation: Zero-argument factory returning a fresh
                activation module, applied after every internal layer
                and, once more, after the residual addition itself.
                Pass `None` to disable activation entirely (including
                after the addition).
            normalization: One-argument factory taking a channel count
                and returning a fresh normalization module, applied
                after every layer in the main path and, if a
                projection shortcut is built, after the shortcut's own
                conv too (standard ResNet convention: the shortcut is
                normalized, never activated, before the addition). Pass
                `None` to disable normalization entirely.

        Raises:
            ValueError: If `depth` is not at least `1`; if `depth > 1`
                and `kernel_size` is even; or if a projection shortcut
                is needed but cannot be built to match the main path's
                output length for every input length (adjust `padding`
                for this block, or `shortcut_kernel_size`).
        """
        super().__init__()
        if depth < 1:
            raise ValueError(f"Residual1DBlock requires depth >= 1, got {depth}.")
        _requireOddKernelForDepth(depth, kernel_size, "Residual1DBlock")

        resolved_padding = padding if padding is not None else dilation * (kernel_size // 2)
        same_padding = dilation * (kernel_size // 2)  # used by every internal (non-first) layer

        layers: list[nn.Module] = []
        current_in = in_channels
        for index in range(depth):
            is_first = index == 0
            is_last = index == depth - 1
            layer_stride = stride if is_first else 1
            layer_padding = resolved_padding if is_first else same_padding
            layers.append(
                nn.Conv1d(
                    current_in,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=layer_stride,
                    padding=layer_padding,
                    dilation=dilation,
                )
            )
            if normalization is not None:
                layers.append(normalization(out_channels))
            if not is_last and activation is not None:
                layers.append(activation())
            current_in = out_channels
        self.main = nn.Sequential(*layers)

        self.needs_projection = in_channels != out_channels or stride != 1
        if self.needs_projection:
            main_offset = computeConv1dLengthOffset(kernel_size, resolved_padding, dilation)
            # The shortcut's own dilation is always fixed at 1 (a single un-dilated
            # projection layer, the standard ResNet choice), so its "same"-style padding
            # only ever depends on its own kernel_size, not on this block's `dilation`.
            shortcut_padding = shortcut_kernel_size // 2
            shortcut_offset = computeConv1dLengthOffset(shortcut_kernel_size, shortcut_padding, 1)
            if main_offset != shortcut_offset:
                raise ValueError(
                    f"Residual1DBlock cannot build a shortcut that reaches the main path's "
                    f"output length for every input length: the main path's first layer "
                    f"(kernel_size={kernel_size}, padding={resolved_padding}, "
                    f"dilation={dilation}) has length offset {main_offset}, but the shortcut "
                    f"(kernel_size={shortcut_kernel_size}, padding={shortcut_padding}, "
                    f"dilation=1) has offset {shortcut_offset}. Adjust this block's padding, "
                    f"or shortcut_kernel_size, so both offsets match (the defaults, an odd "
                    f"kernel_size with padding=None, always satisfy this)."
                )
            shortcut_layers: list[nn.Module] = [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=shortcut_kernel_size,
                    stride=stride,
                    padding=shortcut_padding,
                )
            ]
            if normalization is not None:
                shortcut_layers.append(normalization(out_channels))
            self.shortcut: nn.Module = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

        self.output_activation: nn.Module = (
            activation() if activation is not None else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block: `activation(main(x) + shortcut(x))`.

        Args:
            x: Input tensor, shape `(batch, in_channels, length)`.

        Returns:
            Output tensor, shape `(batch, out_channels, out_length)`,
            `out_length` given by this block's first layer's own
            stride/kernel_size/padding/dilation
            (`utils.conv_math.computeConv1dOutputLength`).
        """
        main_output: torch.Tensor = self.main(x)
        shortcut_output: torch.Tensor = self.shortcut(x)
        result: torch.Tensor = self.output_activation(main_output + shortcut_output)
        return result


class Residual1DUpBlock(nn.Module):
    """One 1D residual block for a decoder: `depth` stacked upsampling/conv layers plus a shortcut.

    Mirrors `Residual1DBlock` on the decoder side: only the first layer
    upsamples (via `build1DUpSampleStage`, the exact same
    `"conv_transpose"`/`"interpolate_conv"` choice `OneDCnnDecoder`
    offers); every subsequent layer is a stride-1, length-preserving
    `Conv1d` at the block's `out_channels` width. The shortcut always
    uses the same `upsample_mode` and the same `stride` as the main
    path's first layer, so both paths' length formulas share the same
    shape; it is verified, at construction time, to reach the exact
    same output length as the main path for every input length (see the
    module docstring), never merely for one example length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        dilation: int = 1,
        upsample_mode: str = "interpolate_conv",
        shortcut_kernel_size: int = 1,
        activation: Callable[[], nn.Module] | None = nn.ReLU,
        normalization: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
        apply_output_normalization: bool = True,
        apply_output_activation: bool = True,
    ) -> None:
        """Build the block.

        Args:
            in_channels: Input channel width.
            out_channels: Output channel width of every layer in this
                block.
            depth: Number of layers in this block (the first upsamples,
                every later one is a stride-1 `Conv1d`). Must be at
                least `1`. See `Residual1DBlock.__init__` for the same
                parameter's full explanation; identical here.
            kernel_size: Kernel size shared by every layer in this
                block. Must be odd if `depth > 1` (see the module
                docstring).
            stride: Upsampling factor of the first layer, and of the
                shortcut (kept identical on purpose, see the class
                docstring).
            padding: Padding of the first (upsampling) layer. Unlike
                `Residual1DBlock`, this has no `None`-resolves-to-"same"
                default: `OneDCnnDecoder`'s own upsampling stages have
                no natural "same" padding either (an upsampling
                transition is expected to change length), so this
                mirrors that class's own explicit-`padding` convention
                instead.
            output_padding: `ConvTranspose1d`'s `output_padding` for
                the first layer. Ignored if `upsample_mode` is
                `"interpolate_conv"` (matching `OneDCnnDecoder`).
            dilation: Dilation shared by every layer in this block.
            upsample_mode: `"conv_transpose"` or `"interpolate_conv"`,
                exactly as in `OneDCnnDecoder`/`build1DUpSampleStage`.
            shortcut_kernel_size: Kernel size of the shortcut's own
                upsampling layer (built via the same
                `build1DUpSampleStage`, same `upsample_mode`, same
                `stride`, same `output_padding` for `"conv_transpose"`
                mode). Defaults to `1`. Its own padding is always
                resolved so the shortcut reaches the exact same output
                length as the main path's first layer for every input
                length; if that resolved padding would need to be
                negative, or (for `"conv_transpose"` mode) the implied
                `output_padding` combination is infeasible,
                construction raises `ValueError` explaining why (adjust
                `padding`/`output_padding` for this block, or
                `shortcut_kernel_size`).
            activation: As in `Residual1DBlock.__init__`.
            normalization: As in `Residual1DBlock.__init__`.
            apply_output_normalization: If `False`, this block's very
                last layer (the one immediately feeding the residual
                addition, on both the main and shortcut paths) skips
                normalization even if `normalization` is given.
                `OneDCnnResidualDecoder` passes `False` for its final
                transition only, matching `OneDCnnDecoder`'s own
                convention that the last transition must be free to
                produce values at any scale; every earlier layer inside
                this same block (when `depth > 1`) is unaffected, so a
                deep last-stage block still gets normalization on its
                internal layers, only not on the one right before the
                reconstruction is read out.
            apply_output_activation: As `apply_output_normalization`,
                but for the activation applied after the residual
                addition itself (`OneDCnnResidualDecoder`'s final
                transition must produce unconstrained reconstruction
                values, exactly like `OneDCnnDecoder`'s own last
                transition).

        Raises:
            ValueError: If `depth` is not at least `1`; if `depth > 1`
                and `kernel_size` is even; if `upsample_mode` is not
                recognized; or if a projection shortcut cannot be built
                to match the main path's output length for every input
                length.
        """
        super().__init__()
        if depth < 1:
            raise ValueError(f"Residual1DUpBlock requires depth >= 1, got {depth}.")
        _requireOddKernelForDepth(depth, kernel_size, "Residual1DUpBlock")
        if upsample_mode not in ("conv_transpose", "interpolate_conv"):
            raise ValueError(
                f"Unknown upsample_mode '{upsample_mode}'. Expected 'conv_transpose' or "
                f"'interpolate_conv'."
            )

        same_padding = dilation * (kernel_size // 2)  # used by every internal (non-first) layer

        layers: list[nn.Module] = []
        current_in = in_channels
        for index in range(depth):
            is_first = index == 0
            is_last = index == depth - 1
            if is_first:
                layers.append(
                    build1DUpSampleStage(
                        current_in,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        upsample_mode=upsample_mode,
                    )
                )
            else:
                layers.append(
                    nn.Conv1d(
                        current_in,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=same_padding,
                        dilation=dilation,
                    )
                )
            if normalization is not None and (not is_last or apply_output_normalization):
                layers.append(normalization(out_channels))
            if not is_last and activation is not None:
                layers.append(activation())
            current_in = out_channels
        self.main = nn.Sequential(*layers)

        self.needs_projection = in_channels != out_channels or stride != 1
        if self.needs_projection:
            if upsample_mode == "conv_transpose":
                main_offset = computeConvTranspose1dLengthOffset(
                    kernel_size, padding, output_padding, dilation
                )
                # The shortcut's own dilation is always fixed at 1, so its "same"-style
                # padding only depends on its own kernel_size (see Residual1DBlock's
                # identical comment).
                shortcut_padding = shortcut_kernel_size // 2
                shortcut_output_padding = (
                    main_offset + 2 * shortcut_padding - (shortcut_kernel_size - 1)
                )
                max_valid = max(stride, 1)
                if not (0 <= shortcut_output_padding < max_valid):
                    raise ValueError(
                        f"Residual1DUpBlock cannot build a conv_transpose shortcut that "
                        f"reaches the main path's output length for every input length: "
                        f"doing so would need output_padding={shortcut_output_padding} for "
                        f"the shortcut (kernel_size={shortcut_kernel_size}), but it must "
                        f"satisfy 0 <= output_padding < {max_valid} given stride={stride}. "
                        f"Adjust this block's padding/output_padding, or shortcut_kernel_size."
                    )
                shortcut_module = build1DUpSampleStage(
                    in_channels,
                    out_channels,
                    shortcut_kernel_size,
                    stride,
                    shortcut_padding,
                    shortcut_output_padding,
                    1,
                    upsample_mode="conv_transpose",
                )
            else:
                main_offset = computeConv1dLengthOffset(kernel_size, padding, dilation)
                shortcut_padding = shortcut_kernel_size // 2  # shortcut dilation fixed at 1
                shortcut_offset = computeConv1dLengthOffset(
                    shortcut_kernel_size, shortcut_padding, 1
                )
                if main_offset != shortcut_offset:
                    raise ValueError(
                        f"Residual1DUpBlock cannot build an interpolate_conv shortcut that "
                        f"reaches the main path's output length for every input length: the "
                        f"main path's post-upsample conv (kernel_size={kernel_size}, "
                        f"padding={padding}, dilation={dilation}) has length offset "
                        f"{main_offset}, but the shortcut (kernel_size={shortcut_kernel_size}, "
                        f"padding={shortcut_padding}, dilation=1) has offset {shortcut_offset}. "
                        f"Adjust this block's padding, or shortcut_kernel_size, so both "
                        f"offsets match (the defaults, an odd kernel_size, always satisfy "
                        f"this)."
                    )
                shortcut_module = build1DUpSampleStage(
                    in_channels,
                    out_channels,
                    shortcut_kernel_size,
                    stride,
                    shortcut_padding,
                    0,
                    1,
                    upsample_mode="interpolate_conv",
                )
            shortcut_layers: list[nn.Module] = [shortcut_module]
            if normalization is not None and apply_output_normalization:
                shortcut_layers.append(normalization(out_channels))
            self.shortcut: nn.Module = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

        self.output_activation: nn.Module = (
            activation() if activation is not None and apply_output_activation else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block: `activation(main(x) + shortcut(x))`.

        Args:
            x: Input tensor, shape `(batch, in_channels, length)`.

        Returns:
            Output tensor, shape `(batch, out_channels, out_length)`.
        """
        main_output: torch.Tensor = self.main(x)
        shortcut_output: torch.Tensor = self.shortcut(x)
        result: torch.Tensor = self.output_activation(main_output + shortcut_output)
        return result
