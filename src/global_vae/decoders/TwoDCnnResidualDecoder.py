"""2D residual (ResNet-style) CNN decoder (spec §6, §7, §12)."""

from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.conv_blocks import Residual2DUpBlock
from global_vae.utils.conv_math import (
    computeUpsampleStack2dOutputShape,
    solveConvTranspose2dOutputPadding,
)
from global_vae.utils.stage_config import ShapeLike, broadcastPerStage, broadcastPerStageShape


@registerDecoder("2d_cnn_resnet_decoder_v1")
class TwoDCnnResidualDecoder(AbstractDecoder):
    """2D residual convolutional decoder reconstructing a fixed-size image.

    Direct 2D generalization of `OneDCnnResidualDecoder` (spec §6's image
    modality), the decoder-side counterpart of `TwoDCnnResidualEncoder`. A stack
    of residual up-blocks (`utils.conv_blocks.Residual2DUpBlock`, spec §7's
    "scaling toward larger backbones"), mirroring `TwoDCnnDecoder`'s own central
    guarantee: the *exact* output shape a given configuration produces, on both
    axes, is verified at construction time, and a mismatch raises `ValueError`
    instead of ever resizing its way to `output_shape`.

    Deliberately as permissive as `TwoDCnnDecoder` along every axis that class
    already varies per transition (`kernel_sizes`, `strides`, `paddings`,
    `output_paddings`, `dilations`, `upsample_modes`, `activations`,
    `normalizations`, all via `utils.stage_config.broadcastPerStage`/
    `broadcastPerStageShape`, with the same `int`/`tuple[int, int]`/`list`
    convention), plus the same two additions `OneDCnnResidualDecoder` gets over
    the plain 1D decoder: `block_depths` (per transition, e.g. `(3, 4)` for a
    shallower first transition and a deeper second one) and
    `shortcut_kernel_sizes`.

    One default deliberately differs from `TwoDCnnDecoder`, mirroring the
    identical difference `OneDCnnResidualDecoder` already makes relative to
    `OneDCnnDecoder`: `kernel_sizes` defaults to `3` here, not `4`.
    `TwoDCnnDecoder`'s own docstring already notes that its `kernel_sizes=4`
    default is tuned specifically for `upsample_modes="conv_transpose"`, and
    does *not* reach an exact shape doubling under this class's own
    `upsample_modes` default, `"interpolate_conv"` (`kernel_sizes=3, paddings=1`
    is the pairing that does, on both axes independently). This class's residual
    blocks additionally *require* an odd `kernel_size` on both axes whenever a
    transition's `block_depths` entry is greater than `1` (the default, `2`; see
    `utils.conv_blocks.Residual2DUpBlock`), so `kernel_sizes=3` is both the
    self-consistent choice for this class's own default `upsample_mode` and the
    only one its own default `block_depths` would even accept. With every other
    default left as `TwoDCnnDecoder`'s own (`strides=2`, `paddings=1`), this
    class's bare defaults reach an exact shape doubling per transition, on both
    axes, out of the box.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int],
        out_channels: int = 1,
        hidden_channels: tuple[int, ...] = (128, 64, 32),
        block_depths: int | Sequence[int] = 2,
        seed_shape: tuple[int, int] = (8, 8),
        kernel_sizes: ShapeLike | list[ShapeLike] = 3,
        strides: ShapeLike | list[ShapeLike] = 2,
        paddings: ShapeLike | list[ShapeLike] = 1,
        output_paddings: ShapeLike | list[ShapeLike] | None = None,
        dilations: ShapeLike | list[ShapeLike] = 1,
        upsample_modes: str | Sequence[str] = "interpolate_conv",
        shortcut_kernel_sizes: ShapeLike | list[ShapeLike] = 1,
        activations: (
            Callable[[], nn.Module] | Sequence[Callable[[], nn.Module] | None] | None
        ) = nn.ReLU,
        normalizations: (
            Callable[[int], nn.Module] | Sequence[Callable[[int], nn.Module] | None] | None
        ) = nn.BatchNorm2d,
        head_hidden_dims: tuple[int, ...] = (),
        head_activation: Callable[[], nn.Module] | None = nn.ReLU,
        modality_name: str = "image",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the input latent vector.
            output_shape: `(height, width)` of the reconstructed image. The
                chosen configuration must reach this exactly, on both axes (see
                `output_paddings`); this class never resizes its way to it.
            out_channels: Number of output channels (`1` for a plain grayscale
                image, `3` for RGB).
            hidden_channels: Channel width of the projected seed
                (`hidden_channels[0]`) and of every subsequent transition. There
                are `len(hidden_channels)` upsampling transitions in total: from
                the seed, through each remaining value in `hidden_channels`, to
                `out_channels`.
            block_depths: Number of internal layers in each transition's
                residual block, per transition or shared (spec's own request:
                e.g. `(3, 4)` for a first transition with `3` layers before its
                shortcut and a deeper second transition with `4`). `2` (default)
                is the classic ResNet "BasicBlock"; any transition whose own
                value is greater than `1` requires an odd `kernel_sizes` entry
                on both axes (see `utils.conv_blocks.Residual2DUpBlock`).
            seed_shape: `(seed_height, seed_width)` of the projected seed
                feature map, upsampled by the transition stack.
            kernel_sizes: Upsampling kernel shape, per transition or shared,
                applied to every layer within that transition's block. Both
                components must be odd for any transition whose `block_depths`
                entry is greater than `1`.
            strides: Upsampling factor shape of each transition's first
                (upsampling) layer, per transition or shared: each axis's own
                upsampling factor for that transition; every later layer within
                the same block always runs at stride `(1, 1)`.
            paddings: Convolution padding shape of each transition's first
                layer, per transition or shared.
            output_paddings: `ConvTranspose2d`'s `output_padding` shape for each
                transition's first layer, per transition or shared. Ignored for
                any transition whose `upsample_mode` is `"interpolate_conv"`.
                Defaults to `None`: if the last transition's `upsample_mode` is
                `"conv_transpose"`, every transition except the last gets
                `output_padding=(0, 0)`, and the last transition's
                `output_padding` is solved automatically, independently per
                axis, so the stack reaches `output_shape` exactly (raising
                `ValueError` if no valid value would work on some axis, or if
                the last transition's own residual shortcut cannot then reach
                that same shape; in either case, adjust `seed_shape`,
                `kernel_sizes`, `strides`, `paddings`, `dilations`, or
                `shortcut_kernel_sizes` instead). If the last transition's
                `upsample_mode` is `"interpolate_conv"`, `None` defaults to
                `(0, 0)` for every transition, since there is no equivalent
                lever to auto-solve there.
            dilations: Convolution dilation shape, per transition or shared.
            upsample_modes: `"conv_transpose"` or `"interpolate_conv"` per
                transition or shared, exactly as in `TwoDCnnDecoder`. See this
                class's own docstring for why its bare defaults (`kernel_sizes=3`,
                `upsample_modes="interpolate_conv"`) already reach an exact
                shape doubling on both axes, unlike `TwoDCnnDecoder`'s own bare
                defaults.
            shortcut_kernel_sizes: Kernel shape of each transition's shortcut
                projection, per transition or shared. Defaults to `1` (square
                `(1, 1)`), the standard ResNet choice.
            activations: Zero-argument factory returning a fresh activation
                module, per transition or shared, used inside and after every
                transition's residual block except the very last transition's
                own final output activation, which is always suppressed
                (matching `TwoDCnnDecoder`'s own convention: the last transition
                must be able to produce unconstrained reconstruction values).
                Earlier internal layers of that same last transition's block, if
                its own `block_depths` entry is greater than `1`, are unaffected
                (see `utils.conv_blocks.Residual2DUpBlock`). Pass `None` to
                disable activation for a transition entirely.
            normalizations: One-argument factory taking a channel count and
                returning a fresh normalization module, per transition or
                shared, used the same places as `activations` (including the
                same "not on the very last transition's final layer" exception).
            head_hidden_dims: Hidden layer sizes for an optional small MLP
                inserted between the latent vector and the seed projection.
                Empty tuple (default) keeps a single linear layer straight from
                `z` to the seed.
            head_activation: Optional activation layer to use for the head.
            modality_name: Name of the modality this decoder reconstructs.

        Raises:
            ValueError: If any per-transition shape argument does not resolve
                cleanly (see `utils.stage_config.broadcastPerStageShape`), if
                any transition's `upsample_mode` is not recognized, if any
                transition's `block_depths` is not at least `1` or is greater
                than `1` with an even `kernel_sizes` component on some axis, if
                no valid `output_padding` for the last transition would reach
                `output_shape` on some axis (`output_paddings=None`,
                `upsample_mode="conv_transpose"` case), if any transition
                needing a projection shortcut cannot reach its own main path's
                output shape for every input shape on both axes, or if the
                resulting configuration's computed shape does not equal
                `output_shape` (every other case).
        """
        super().__init__()
        self._output_shape = output_shape
        self._modality_name = modality_name
        num_transitions = len(hidden_channels)

        block_depths_ = broadcastPerStage(block_depths, num_transitions, "block_depths")
        kernel_sizes_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(kernel_sizes, num_transitions, 2, "kernel_sizes"),
        )
        strides_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(strides, num_transitions, 2, "strides"),
        )
        paddings_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(paddings, num_transitions, 2, "paddings"),
        )
        dilations_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(dilations, num_transitions, 2, "dilations"),
        )
        activations_ = broadcastPerStage(activations, num_transitions, "activations")
        normalizations_ = broadcastPerStage(normalizations, num_transitions, "normalizations")
        upsample_modes_: tuple[str, ...] = broadcastPerStage(
            upsample_modes, num_transitions, "upsample_modes"
        )
        shortcut_kernel_sizes_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(
                shortcut_kernel_sizes, num_transitions, 2, "shortcut_kernel_sizes"
            ),
        )

        for stage in range(num_transitions):
            if block_depths_[stage] < 1:
                raise ValueError(
                    f"TwoDCnnResidualDecoder: block_depths[{stage}]={block_depths_[stage]} "
                    f"must be at least 1."
                )
            if block_depths_[stage] > 1 and (
                kernel_sizes_[stage][0] % 2 == 0 or kernel_sizes_[stage][1] % 2 == 0
            ):
                raise ValueError(
                    f"TwoDCnnResidualDecoder: transition {stage} has block_depths="
                    f"{block_depths_[stage]} > 1, which requires an odd kernel_size on both "
                    f"axes for its internal, length-preserving layers (a stride-1 layer with "
                    f"an even kernel_size cannot preserve length exactly with any integer "
                    f"padding, independently per axis), got "
                    f"kernel_sizes[{stage}]={kernel_sizes_[stage]}. Use an odd kernel_size on "
                    f"both axes for this transition, or block_depths=1 if an even kernel_size "
                    f"is required."
                )

        if output_paddings is not None:
            output_paddings_: tuple[tuple[int, int], ...] = cast(
                "tuple[tuple[int, int], ...]",
                broadcastPerStageShape(output_paddings, num_transitions, 2, "output_paddings"),
            )
        elif upsample_modes_[-1] == "conv_transpose":
            shape_before_last = computeUpsampleStack2dOutputShape(
                seed_shape,
                kernel_sizes_[:-1],
                strides_[:-1],
                paddings_[:-1],
                tuple((0, 0) for _ in range(num_transitions - 1)),
                dilations_[:-1],
                upsample_modes_[:-1],
            )
            solved_output_padding = solveConvTranspose2dOutputPadding(
                shape_before_last,
                output_shape,
                kernel_sizes_[-1],
                strides_[-1],
                paddings_[-1],
                dilations_[-1],
            )
            max_valid = (
                max(strides_[-1][0], dilations_[-1][0]),
                max(strides_[-1][1], dilations_[-1][1]),
            )
            if not (
                0 <= solved_output_padding[0] < max_valid[0]
                and 0 <= solved_output_padding[1] < max_valid[1]
            ):
                raise ValueError(
                    f"TwoDCnnResidualDecoder cannot reach output_shape={output_shape} by "
                    f"solving only the last transition's output_padding (would need "
                    f"output_padding={solved_output_padding}, but it must satisfy "
                    f"0 <= output_padding[axis] < {max_valid}[axis] given the last "
                    f"transition's stride={strides_[-1]}/dilation={dilations_[-1]}). Adjust "
                    f"seed_shape, kernel_sizes, strides, paddings, or dilations, or pass "
                    f"output_paddings explicitly. Use "
                    f"TwoDCnnResidualDecoder.computeOutputShape(...) to check a configuration "
                    f"before constructing."
                )
            output_paddings_ = (
                *((0, 0) for _ in range(num_transitions - 1)),
                solved_output_padding,
            )
        else:
            output_paddings_ = tuple((0, 0) for _ in range(num_transitions))

        computed_shape = computeUpsampleStack2dOutputShape(
            seed_shape,
            kernel_sizes_,
            strides_,
            paddings_,
            output_paddings_,
            dilations_,
            upsample_modes_,
        )
        if computed_shape != output_shape:
            raise ValueError(
                f"TwoDCnnResidualDecoder's configuration produces shape {computed_shape}, not "
                f"the requested output_shape={output_shape}. This class verifies the exact "
                f"shape instead of resizing a mismatch away. Adjust seed_shape, "
                f"kernel_sizes, strides, paddings, output_paddings, or dilations, or call "
                f"TwoDCnnResidualDecoder.computeOutputShape(...) with the same arguments to "
                f"explore configurations before constructing."
            )

        self._seed_channels = hidden_channels[0]
        self._seed_height, self._seed_width = seed_shape

        head_layers: list[nn.Module] = []
        head_in = latent_dim
        for hidden_dim in head_hidden_dims:
            head_layers.append(nn.Linear(head_in, hidden_dim))
            if head_activation is not None:
                head_layers.append(head_activation())
            head_in = hidden_dim
        self.head: nn.Module = nn.Sequential(*head_layers) if head_layers else nn.Identity()

        self.project = nn.Linear(head_in, hidden_channels[0] * seed_shape[0] * seed_shape[1])

        widths = (*hidden_channels, out_channels)
        layers: list[nn.Module] = []
        for stage in range(num_transitions):
            is_last = stage == num_transitions - 1
            layers.append(
                Residual2DUpBlock(
                    widths[stage],
                    widths[stage + 1],
                    depth=block_depths_[stage],
                    kernel_size=kernel_sizes_[stage],
                    stride=strides_[stage],
                    padding=paddings_[stage],
                    output_padding=output_paddings_[stage],
                    dilation=dilations_[stage],
                    upsample_mode=upsample_modes_[stage],
                    shortcut_kernel_size=shortcut_kernel_sizes_[stage],
                    activation=activations_[stage],
                    normalization=normalizations_[stage],
                    apply_output_normalization=not is_last,
                    apply_output_activation=not is_last,
                )
            )
        self.deconv = nn.Sequential(*layers)

    @staticmethod
    def computeOutputShape(
        seed_shape: tuple[int, int],
        hidden_channels: tuple[int, ...],
        kernel_sizes: ShapeLike | list[ShapeLike] = 3,
        strides: ShapeLike | list[ShapeLike] = 2,
        paddings: ShapeLike | list[ShapeLike] = 1,
        output_paddings: ShapeLike | list[ShapeLike] = 0,
        dilations: ShapeLike | list[ShapeLike] = 1,
        upsample_modes: str | Sequence[str] = "interpolate_conv",
    ) -> tuple[int, int]:
        """Compute the output `(height, width)` a given configuration produces.

        Lets a caller check, before constructing a full `TwoDCnnResidualDecoder`
        (or after, to understand why construction raised), exactly what shape a
        chosen architecture reaches from `seed_shape` (spec §12). Mirrors
        `TwoDCnnDecoder.computeOutputShape`/`OneDCnnResidualDecoder.
        computeOutputLength` exactly (all three delegate to the same shape/
        length-chaining logic): only the first layer of each transition's block
        ever changes shape, so `block_depths`/`shortcut_kernel_sizes` do not
        affect this computation at all and are not parameters here.

        Unlike `__init__`, this always takes an explicit `output_paddings`
        (defaulting to `0`, i.e. `(0, 0)` on both axes, for every transition):
        it does not perform `__init__`'s `output_paddings=None` auto-solving,
        since the point of calling this method is to see what a
        fully-specified configuration produces.

        Args:
            seed_shape: `(seed_height, seed_width)` of the projected seed.
            hidden_channels: Channel width of the seed and every subsequent
                transition, as in `__init__`. Only its length (the number of
                transitions) affects the result.
            kernel_sizes: As in `__init__`.
            strides: As in `__init__`.
            paddings: As in `__init__`.
            output_paddings: As in `__init__`, except always an explicit
                shared-or-per-transition value.
            dilations: As in `__init__`.
            upsample_modes: As in `__init__`.

        Returns:
            The resulting `(height, width)`.

        Raises:
            ValueError: If any per-transition shape argument does not resolve
                cleanly, or if `upsample_mode` is not recognized.
        """
        num_transitions = len(hidden_channels)
        kernel_sizes_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(kernel_sizes, num_transitions, 2, "kernel_sizes"),
        )
        strides_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(strides, num_transitions, 2, "strides"),
        )
        paddings_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(paddings, num_transitions, 2, "paddings"),
        )
        output_paddings_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(output_paddings, num_transitions, 2, "output_paddings"),
        )
        dilations_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(dilations, num_transitions, 2, "dilations"),
        )
        upsample_modes_: tuple[str, ...] = broadcastPerStage(
            upsample_modes, num_transitions, "upsample_modes"
        )

        return computeUpsampleStack2dOutputShape(
            seed_shape,
            kernel_sizes_,
            strides_,
            paddings_,
            output_paddings_,
            dilations_,
            upsample_modes_,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of fixed-size images from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Reconstructed image. Shape `(batch, height, width)` for the default
            single-channel case (matching the encoder's plain-image input
            convention); `(batch, out_channels, height, width)` if
            `out_channels` was set above `1`.
        """
        batch_size = z.shape[0]
        seed = self.project(self.head(z)).view(
            batch_size, self._seed_channels, self._seed_height, self._seed_width
        )
        reconstruction: torch.Tensor = self.deconv(seed)
        return reconstruction.squeeze(1) if reconstruction.shape[1] == 1 else reconstruction

    @property
    def modality_name(self) -> str:
        return self._modality_name
