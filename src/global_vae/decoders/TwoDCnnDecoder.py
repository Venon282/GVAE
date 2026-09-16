"""2D CNN decoder (spec §6, §12).

The central behavior under test/design, mirrored exactly from
`OneDCnnDecoder`: this decoder verifies the *exact* output shape its
configuration produces at construction time, instead of forcing a
mismatch to fit with a blurring final resize.
"""

from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.builders import build2DUpSampleStage
from global_vae.utils.conv_math import (
    computeUpsampleStack2dOutputShape as _computeShapeFromResolved,
)
from global_vae.utils.conv_math import (
    solveConvTranspose2dOutputPadding,
)
from global_vae.utils.stage_config import ShapeLike, broadcastPerStage, broadcastPerStageShape


@registerDecoder("2d_cnn_decoder_v1")
class TwoDCnnDecoder(AbstractDecoder):
    """2D convolutional decoder reconstructing a fixed-size image.

    Direct 2D generalization of `OneDCnnDecoder` (spec §6's image
    modality). A stack of upsampling transitions (`build2DUpSampleStage`:
    a learned `ConvTranspose2d`, or nearest-neighbor upsampling
    followed by a stride-1 `Conv2d`), from a small, learned
    `(seed_height, seed_width)` feature map up to the requested
    `output_shape`. Exactly like `OneDCnnDecoder`, this class verifies
    the *exact* output shape a given configuration produces, on both
    axes, at construction time, and raises `ValueError` on any
    mismatch instead of ever resizing its way to `output_shape`.

    Every per-transition hyperparameter that is shape-like
    (`kernel_sizes`, `strides`, `paddings`, `output_paddings`,
    `dilations`) accepts either one value shared by every transition or
    a sequence of exactly `len(hidden_channels)` per-transition values,
    generalized to 2D exactly like `TwoDCnnEncoder`:
      - a single `int` (square), a single `tuple[int, int]` (explicit
        non-square, shared across every transition), or a `list` of
        exactly `len(hidden_channels)` per-transition entries, each
        itself an `int` or a `tuple[int, int]`.
    See `utils.stage_config.broadcastPerStageShape` for why `list` and
    `tuple` are given these different, non-overlapping jobs.
    `upsample_modes`, `activations`, and `normalizations` are not
    shape-like and keep `OneDCnnDecoder`'s own `broadcastPerStage`
    convention unchanged.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int],
        out_channels: int = 1,
        hidden_channels: tuple[int, ...] = (128, 64, 32),
        seed_shape: tuple[int, int] = (8, 8),
        kernel_sizes: ShapeLike | list[ShapeLike] = 4,
        strides: ShapeLike | list[ShapeLike] = 2,
        paddings: ShapeLike | list[ShapeLike] = 1,
        output_paddings: ShapeLike | list[ShapeLike] | None = None,
        dilations: ShapeLike | list[ShapeLike] = 1,
        upsample_modes: str | Sequence[str] = "interpolate_conv",
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
            output_shape: `(height, width)` of the reconstructed
                image. The chosen configuration must reach this
                exactly, on both axes (see `output_paddings`); this
                class never resizes its way to it.
            out_channels: Number of output channels (`1` for a plain
                grayscale image, `3` for RGB).
            hidden_channels: Channel width of the projected seed
                (`hidden_channels[0]`) and of every subsequent
                transition. There are `len(hidden_channels)` upsampling
                transitions in total: from the seed, through each
                remaining value in `hidden_channels`, to `out_channels`.
            seed_shape: `(seed_height, seed_width)` of the projected
                seed feature map, upsampled by the transition stack.
            kernel_sizes: Upsampling kernel shape, per transition or
                shared.
            strides: Upsampling factor shape, per transition or
                shared: each axis's own upsampling factor for that
                transition.
            paddings: Convolution padding shape, per transition or
                shared.
            output_paddings: `ConvTranspose2d`'s `output_padding`
                shape, per transition or shared. Ignored for any
                transition whose `upsample_mode` is `"interpolate_conv"`.
                Defaults to `None`: if the last transition's
                `upsample_mode` is `"conv_transpose"`, every transition
                except the last gets `output_padding=(0, 0)`, and the
                last transition's `output_padding` is solved
                automatically, independently per axis, so the stack
                reaches `output_shape` exactly (raising `ValueError` if
                no valid value would work on some axis, or if the last
                transition's own residual shortcut cannot then reach
                that same shape; in either case, adjust `seed_shape`,
                `kernel_sizes`, `strides`, `paddings`, or `dilations`
                instead). If the last transition's `upsample_mode` is
                `"interpolate_conv"`, `None` defaults to `(0, 0)` for
                every transition, since there is no equivalent lever to
                auto-solve there.
            dilations: Convolution dilation shape, per transition or
                shared.
            upsample_modes: `"conv_transpose"` or `"interpolate_conv"`
                per transition or shared, exactly as in
                `OneDCnnDecoder`. Note: this class's default
                `kernel_sizes=4, strides=2, paddings=1` reach an exact
                shape doubling per transition, on both axes, under
                `"conv_transpose"` (with the last transition's
                `output_padding` auto-solved to `(0, 0)`); reaching the
                same exact doubling under `"interpolate_conv"` instead
                needs `kernel_sizes=3, paddings=1` (the "same"-padded,
                stride-1 conv that leaves a nearest-neighbor-doubled
                shape unchanged on both axes), exactly the same pairing
                `OneDCnnDecoder`'s own docstring documents for the 1D
                case. Construction always verifies the actual result,
                so picking the wrong pairing here fails immediately
                with a clear error instead of silently producing the
                wrong shape.
            activations: Zero-argument factory returning a fresh
                activation module, per transition or shared, applied
                after every transition except the last (which must be
                able to produce unconstrained reconstruction values).
                Pass `None` to disable activation for a transition.
            normalizations: One-argument factory taking a channel count
                and returning a fresh normalization module, per
                transition or shared, applied the same places as
                `activations`.
            head_hidden_dims: Hidden layer sizes for an optional small
                MLP inserted between the latent vector and the seed
                projection. Empty tuple (default) keeps a single
                linear layer straight from `z` to the seed.
            head_activation: Optional activation layer to use for the
                head.
            modality_name: Name of the modality this decoder
                reconstructs.

        Raises:
            ValueError: If any per-transition shape argument does not
                resolve cleanly (see
                `utils.stage_config.broadcastPerStageShape`), if any
                transition's `upsample_mode` is not recognized, if no
                valid `output_padding` for the last transition would
                reach `output_shape` on some axis
                (`output_paddings=None`, `upsample_mode="conv_transpose"`
                case), or if the resulting configuration's computed
                shape does not equal `output_shape` (every other case).
        """
        super().__init__()
        self._output_shape = output_shape
        self._modality_name = modality_name
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
        dilations_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(dilations, num_transitions, 2, "dilations"),
        )
        activations_ = broadcastPerStage(activations, num_transitions, "activations")
        normalizations_ = broadcastPerStage(normalizations, num_transitions, "normalizations")
        upsample_modes_: tuple[str, ...] = broadcastPerStage(
            upsample_modes, num_transitions, "upsample_modes"
        )

        if output_paddings is not None:
            output_paddings_: tuple[tuple[int, int], ...] = cast(
                "tuple[tuple[int, int], ...]",
                broadcastPerStageShape(output_paddings, num_transitions, 2, "output_paddings"),
            )
        elif upsample_modes_[-1] == "conv_transpose":
            shape_before_last = _computeShapeFromResolved(
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
                    f"TwoDCnnDecoder cannot reach output_shape={output_shape} by solving "
                    f"only the last transition's output_padding (would need "
                    f"output_padding={solved_output_padding}, but it must satisfy "
                    f"0 <= output_padding[axis] < {max_valid}[axis] given the last "
                    f"transition's stride={strides_[-1]}/dilation={dilations_[-1]}). Adjust "
                    f"seed_shape, kernel_sizes, strides, paddings, or dilations, or pass "
                    f"output_paddings explicitly. Use "
                    f"TwoDCnnDecoder.computeOutputShape(...) to check a configuration "
                    f"before constructing."
                )
            output_paddings_ = (
                *((0, 0) for _ in range(num_transitions - 1)),
                solved_output_padding,
            )
        else:
            output_paddings_ = tuple((0, 0) for _ in range(num_transitions))

        computed_shape = _computeShapeFromResolved(
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
                f"TwoDCnnDecoder's configuration produces shape {computed_shape}, not the "
                f"requested output_shape={output_shape}. This class verifies the exact "
                f"shape instead of resizing a mismatch away. Adjust seed_shape, "
                f"kernel_sizes, strides, paddings, output_paddings, or dilations, or call "
                f"TwoDCnnDecoder.computeOutputShape(...) with the same arguments to "
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
                build2DUpSampleStage(
                    widths[stage],
                    widths[stage + 1],
                    kernel_sizes_[stage],
                    strides_[stage],
                    paddings_[stage],
                    output_paddings_[stage],
                    dilations_[stage],
                    upsample_mode=upsample_modes_[stage],
                )
            )
            if not is_last:
                normalization = normalizations_[stage]
                if normalization is not None:
                    layers.append(normalization(widths[stage + 1]))
                activation = activations_[stage]
                if activation is not None:
                    layers.append(activation())
        self.deconv = nn.Sequential(*layers)

    @staticmethod
    def computeOutputShape(
        seed_shape: tuple[int, int],
        hidden_channels: tuple[int, ...],
        kernel_sizes: ShapeLike | list[ShapeLike] = 4,
        strides: ShapeLike | list[ShapeLike] = 2,
        paddings: ShapeLike | list[ShapeLike] = 1,
        output_paddings: ShapeLike | list[ShapeLike] = 0,
        dilations: ShapeLike | list[ShapeLike] = 1,
        upsample_modes: str | Sequence[str] = "interpolate_conv",
    ) -> tuple[int, int]:
        """Compute the output `(height, width)` a given configuration produces.

        Lets a caller check, before constructing a full
        `TwoDCnnDecoder` (or after, to understand why construction
        raised), exactly what shape a chosen architecture reaches from
        `seed_shape` (spec §12). Mirrors `OneDCnnDecoder.computeOutputLength`
        exactly (both delegate to the same per-axis chaining logic,
        `utils.conv_math.computeUpsampleStack2dOutputShape`): only the
        first layer of each transition ever changes shape, so
        `block_depths`-style residual concerns do not apply here at all
        (this is the plain, non-residual decoder).

        Unlike `__init__`, this always takes an explicit
        `output_paddings` (defaulting to `0`, i.e. `(0, 0)` on both
        axes, for every transition): it does not perform `__init__`'s
        `output_paddings=None` auto-solving, since the point of calling
        this method is to see what a fully-specified configuration
        produces.

        Args:
            seed_shape: `(seed_height, seed_width)` of the projected
                seed.
            hidden_channels: Channel width of the seed and every
                subsequent transition, as in `__init__`. Only its
                length (the number of transitions) affects the result.
            kernel_sizes: As in `__init__`.
            strides: As in `__init__`.
            paddings: As in `__init__`.
            output_paddings: As in `__init__`, except always an
                explicit shared-or-per-transition value.
            dilations: As in `__init__`.
            upsample_modes: As in `__init__`.

        Returns:
            The resulting `(height, width)`.

        Raises:
            ValueError: If any per-transition shape argument does not
                resolve cleanly, or if `upsample_mode` is not
                recognized.
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

        return _computeShapeFromResolved(
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
            Reconstructed image. Shape `(batch, height, width)` for
            the default single-channel case (matching the encoder's
            plain-image input convention); `(batch, out_channels,
            height, width)` if `out_channels` was set above `1`.
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
