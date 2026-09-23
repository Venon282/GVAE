"""2D residual (ResNet-style) CNN encoder (spec §6, §7, §12)."""

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.builders import build2DPoolLayer
from global_vae.utils.conv_blocks import Residual2DBlock
from global_vae.utils.conv_math import solveMinimumInputShapeForConv2d
from global_vae.utils.stage_config import (
    ShapeLike,
    broadcastPerStage,
    broadcastPerStageOptionalShape,
    broadcastPerStageShape,
)


def _computeMinimumInputShapeFromResolved(
    num_stages: int,
    block_depths: tuple[int, ...],
    kernel_sizes: tuple[tuple[int, int], ...],
    strides: tuple[tuple[int, int], ...],
    paddings: tuple[tuple[int, int], ...],
    dilations: tuple[tuple[int, int], ...],
    poolings: tuple[str | None, ...],
    pool_kernel_sizes: tuple[tuple[int, int] | None, ...],
    pool_strides: tuple[tuple[int, int] | None, ...],
    pool_paddings: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """Core of `computeMinimumInputShape`, operating on already-resolved per-stage tuples.

    Mirrors `TwoDCnnEncoder.py`'s own module-level helper of the same name, and for
    the identical reason: `__init__` has already resolved every per-stage tuple by
    the time it needs this computation, and re-broadcasting an already-resolved
    `tuple[tuple[int, int], ...]` through `broadcastPerStageShape` would misread it
    as a single, wrongly-sized shared shape (a `tuple` always means "one shape
    shared by every stage" there, a `list` always means "one entry per stage").
    `computeMinimumInputShape` broadcasts its raw, possibly-shared arguments once,
    then delegates here; `__init__` delegates directly.

    The one addition over `TwoDCnnEncoder`'s non-residual helper, generalizing
    `OneDCnnResidualEncoder.computeMinimumInputLength`'s identical inner loop to
    2D: each stage now contributes `block_depths[stage]` conv layers (a
    strided/kernel-sized first layer, then `block_depths[stage] - 1`
    length-preserving internal layers, each solved independently per axis) to the
    requirement, instead of a single layer.

    Args:
        num_stages: Number of stages.
        block_depths: Per-stage number of `Conv2d` layers in that stage's
            residual block.
        kernel_sizes: Per-stage `(kernel_height, kernel_width)`.
        strides: Per-stage `(stride_height, stride_width)` of that stage's first
            layer only.
        paddings: Per-stage `(padding_height, padding_width)` of that stage's
            first layer only.
        dilations: Per-stage `(dilation_height, dilation_width)`.
        poolings: Per-stage pooling kind, or `None`.
        pool_kernel_sizes: Per-stage pooling `(height, width)`, or `None`.
        pool_strides: Per-stage pooling stride `(height, width)`, or `None`.
        pool_paddings: Per-stage pooling `(padding_height, padding_width)`.

    Returns:
        The minimum `(height, width)` this configuration can accept.

    Raises:
        ValueError: If some stage requests pooling but has no
            `pool_kernel_sizes` entry.
    """
    required_min_shape = (1, 1)
    for stage in reversed(range(num_stages)):
        if poolings[stage] is not None and pool_kernel_sizes[stage] is None:
            raise ValueError(
                f"pooling='{poolings[stage]}' requires a kernel_size, but kernel_size is None."
            )
        if poolings[stage] is not None:
            kernel_size_for_stage = pool_kernel_sizes[stage]
            # Guaranteed non-None by the ValueError check above: reaching this branch
            # means poolings[stage] is not None, so the compound condition there would
            # already have raised if this were None.
            assert kernel_size_for_stage is not None
            stride_for_stage = pool_strides[stage]
            pool_stride = (
                stride_for_stage if stride_for_stage is not None else kernel_size_for_stage
            )
            required_min_shape = solveMinimumInputShapeForConv2d(
                required_min_shape,
                kernel_size_for_stage,
                pool_stride,
                pool_paddings[stage],
                (1, 1),
            )

        same_padding = (
            dilations[stage][0] * (kernel_sizes[stage][0] // 2),
            dilations[stage][1] * (kernel_sizes[stage][1] // 2),
        )
        for layer_index in reversed(range(block_depths[stage])):
            is_first = layer_index == 0
            layer_stride = strides[stage] if is_first else (1, 1)
            layer_padding = paddings[stage] if is_first else same_padding
            required_min_shape = solveMinimumInputShapeForConv2d(
                required_min_shape,
                kernel_sizes[stage],
                layer_stride,
                layer_padding,
                dilations[stage],
            )

    return required_min_shape


@registerEncoder("2d_cnn_resnet_encoder_v1")
class TwoDCnnResidualEncoder(AbstractEncoder):
    """2D residual convolutional encoder, robust to variable input spatial size.

    A stack of residual blocks (`utils.conv_blocks.Residual2DBlock`, spec §7's
    "scaling toward larger backbones", the 2D counterpart of
    `OneDCnnResidualEncoder`'s own `Residual1DBlock`), each optionally followed by
    a pooling layer, followed by global adaptive pooling. As in `TwoDCnnEncoder`,
    the adaptive pool is what makes the encoder resolution-agnostic: two images of
    different height/width produce feature maps of different spatial size after
    the residual stack, but the same fixed-size vector after pooling, so a single
    set of weights covers images of any size (subject to the architecture's own
    minimum, see `computeMinimumInputShape`).

    Deliberately as permissive as `TwoDCnnEncoder` along every axis that class
    already varies per stage (`kernel_sizes`, `strides`, `paddings`, `dilations`,
    `poolings` and its own sub-parameters, `activations`, `normalizations`, all via
    `utils.stage_config.broadcastPerStage`/`broadcastPerStageShape`, with the same
    `int` (square, shared) / `tuple[int, int]` (non-square, shared) / `list`
    (per-stage) convention `TwoDCnnEncoder` already established for every
    shape-like hyperparameter), plus the same two additions
    `OneDCnnResidualEncoder` adds over the plain 1D encoder: `block_depths`,
    letting each stage's residual block have its own number of internal layers
    (e.g. `3` layers before the first stage's shortcut, then `4` for a deeper
    second stage), and `shortcut_kernel_sizes`, the one further per-stage knob a
    residual architecture specifically needs (`Residual2DBlock`'s own shortcut
    projection kernel), itself possibly non-square and broadcastable the same way.

    One deliberate, documented scope narrowing relative to `TwoDCnnEncoder`,
    mirroring the identical narrowing `OneDCnnResidualEncoder` already makes
    relative to `OneDCnnEncoder`: `hidden_channels` here is always a plain
    `tuple[int, ...]` (channel width per stage), not `TwoDCnnEncoder`'s
    `tuple[int | nn.Module, ...]` escape hatch for dropping in an arbitrary custom
    stage. A residual block is a coupled main-path/shortcut pair (see
    `Residual2DBlock`'s own docstring for why the shortcut's shape must be
    provably tied to the main path's, on both axes independently), which does not
    compose with an opaque, arbitrary `nn.Module` standing in for one stage the way
    a single plain conv stage does. A caller wanting a fully custom stage inside an
    otherwise-residual encoder can still compose one directly with `torch.nn.Module`,
    outside this class, exactly as any other custom architecture would.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
        block_depths: int | Sequence[int] = 2,
        kernel_sizes: ShapeLike | list[ShapeLike] = 3,
        strides: ShapeLike | list[ShapeLike] = 1,
        paddings: ShapeLike | list[ShapeLike] | None = None,
        dilations: ShapeLike | list[ShapeLike] = 1,
        shortcut_kernel_sizes: ShapeLike | list[ShapeLike] = 1,
        poolings: str | None | Sequence[str | None] = "max",
        pool_kernel_sizes: ShapeLike | None | list[ShapeLike | None] = 2,
        pool_strides: ShapeLike | None | list[ShapeLike | None] = None,
        pool_paddings: ShapeLike | list[ShapeLike] = 0,
        pool_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        activations: (
            Callable[[], nn.Module] | Sequence[Callable[[], nn.Module] | None] | None
        ) = nn.ReLU,
        normalizations: (
            Callable[[int], nn.Module] | Sequence[Callable[[int], nn.Module] | None] | None
        ) = nn.BatchNorm2d,
        global_pool: str = "avg",
        head_hidden_dims: tuple[int, ...] = (),
        head_activation: Callable[[], nn.Module] | None = nn.ReLU,
        modality_name: str = "image",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            in_channels: Number of input channels (`1` for a plain grayscale
                image; `3` for RGB; more if further co-registered channels are
                stacked).
            hidden_channels: Output channel width of each stage's residual block,
                applied in order. Its length fixes the number of stages. Always a
                plain `tuple[int, ...]` here (see the class docstring: no
                custom-`nn.Module`-stage escape hatch, unlike `TwoDCnnEncoder`).
            block_depths: Number of internal `Conv2d` layers in each stage's
                residual block, per stage or shared (spec's own request: e.g.
                `(3, 4)` for a first stage with `3` layers before its shortcut and
                a second, deeper stage with `4`). `2` (default) is the classic
                ResNet "BasicBlock"; must be odd-`kernel_size`-compatible on both
                axes whenever greater than `1` (see
                `utils.conv_blocks.Residual2DBlock`).
            kernel_sizes: Convolution kernel shape, per stage or shared (see the
                class docstring for the `int`/`tuple[int, int]`/`list`
                convention), applied to every layer within that stage's block.
                Both components must be odd for any stage whose `block_depths`
                entry is greater than `1`.
            strides: Downsampling stride shape of each stage's *first* layer
                only, per stage or shared; every later layer within the same
                block always uses stride `(1, 1)` (see `Residual2DBlock`). Use
                this (with `poolings=None`) to downsample via strided
                convolutions instead of a separate pooling layer, exactly as in
                `TwoDCnnEncoder`.
            paddings: Convolution padding shape of each stage's first layer, per
                stage or shared. Defaults (`None`) to `dilation[axis] *
                (kernel_size[axis] // 2)` per axis for each stage, matching
                `TwoDCnnEncoder`'s own default and, not coincidentally, the one
                choice that always keeps a projection shortcut's output shape
                matching the main path's on both axes (see `Residual2DBlock`).
            dilations: Convolution dilation shape, per stage or shared.
            shortcut_kernel_sizes: Kernel shape of each stage's shortcut
                projection (only built when that stage changes channel width
                and/or downsamples), per stage or shared. Defaults to `1`
                (square `(1, 1)`), the standard ResNet choice.
            poolings: `"max"`, `"avg"`, or `None` to disable pooling, per stage
                or shared, applied *after* that stage's residual block (not
                inside it). Different stages may use different pooling
                strategies.
            pool_kernel_sizes: Pooling window shape, per stage or shared.
                Required (not `None`) for any stage whose `poolings` entry is
                not `None`; ignored otherwise.
            pool_strides: Pooling stride shape, per stage or shared. Defaults
                (`None`) to that stage's pooling kernel shape (non-overlapping
                windows). Ignored for stages whose `poolings` entry is `None`.
            pool_paddings: Pooling padding shape, per stage or shared. Ignored
                for stages whose `poolings` entry is `None`.
            pool_kwargs: Additional keyword arguments forwarded to the pooling
                layer's constructor (e.g. `ceil_mode`, `count_include_pad`), per
                stage or shared. `None` (default) forwards no extra arguments
                for any stage.
            activations: Zero-argument factory returning a fresh activation
                module, per stage or shared, used inside and after every stage's
                residual block (see `Residual2DBlock`). Pass `None` (either
                overall, or as one stage's entry in a sequence) to disable
                activation for that stage.
            normalizations: One-argument factory taking a channel count and
                returning a fresh normalization module, per stage or shared,
                used the same places as `activations`.
            global_pool: `"avg"` or `"max"`: which adaptive pooling reduces the
                final feature map to a single fixed-size vector, regardless of
                input spatial size.
            head_hidden_dims: Hidden layer sizes for an optional small MLP
                inserted between the pooled features and the `to_mu`/`to_logvar`
                heads. Empty tuple (default) keeps a single linear layer
                straight from pooled features to each head.
            head_activation: Optional activation layer to use for the head.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If any per-stage sequence argument does not have exactly
                `len(hidden_channels)` values, if a stage requests pooling
                without a `pool_kernel_sizes` entry, if `global_pool` is not a
                recognized choice, if any stage's `block_depths` is not at least
                `1`, if any stage's `block_depths` is greater than `1` and its
                `kernel_sizes` is even on some axis, or if any stage needing a
                projection shortcut cannot reach the main path's output shape
                for every input shape on both axes (see `Residual2DBlock`).
        """
        super().__init__()
        self._latent_dim = latent_dim
        self._modality_name = modality_name
        num_stages = len(hidden_channels)

        block_depths_ = broadcastPerStage(block_depths, num_stages, "block_depths")
        kernel_sizes_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(kernel_sizes, num_stages, 2, "kernel_sizes"),
        )
        strides_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(strides, num_stages, 2, "strides"),
        )
        dilations_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(dilations, num_stages, 2, "dilations"),
        )
        if paddings is None:
            paddings_: tuple[tuple[int, int], ...] = tuple(
                (
                    dilations_[stage][0] * (kernel_sizes_[stage][0] // 2),
                    dilations_[stage][1] * (kernel_sizes_[stage][1] // 2),
                )
                for stage in range(num_stages)
            )
        else:
            paddings_ = cast(
                "tuple[tuple[int, int], ...]",
                broadcastPerStageShape(paddings, num_stages, 2, "paddings"),
            )
        shortcut_kernel_sizes_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(
                shortcut_kernel_sizes, num_stages, 2, "shortcut_kernel_sizes"
            ),
        )
        poolings_: tuple[str | None, ...] = broadcastPerStage(poolings, num_stages, "poolings")
        pool_kernel_sizes_: tuple[tuple[int, int] | None, ...] = cast(
            "tuple[tuple[int, int] | None, ...]",
            broadcastPerStageOptionalShape(pool_kernel_sizes, num_stages, 2, "pool_kernel_sizes"),
        )
        pool_strides_: tuple[tuple[int, int] | None, ...] = (
            cast(
                "tuple[tuple[int, int] | None, ...]",
                broadcastPerStageOptionalShape(pool_strides, num_stages, 2, "pool_strides"),
            )
            if pool_strides is not None
            else (None,) * num_stages
        )
        pool_paddings_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(pool_paddings, num_stages, 2, "pool_paddings"),
        )
        resolved_pool_kwargs: dict[str, Any] | Sequence[dict[str, Any]] = (
            pool_kwargs if pool_kwargs is not None else {}
        )
        pool_kwargs_ = broadcastPerStage(resolved_pool_kwargs, num_stages, "pool_kwargs")
        activations_ = broadcastPerStage(activations, num_stages, "activations")
        normalizations_: tuple[Callable[[int], nn.Module] | None, ...] = broadcastPerStage(
            normalizations, num_stages, "normalizations"
        )

        for stage in range(num_stages):
            if block_depths_[stage] < 1:
                raise ValueError(
                    f"TwoDCnnResidualEncoder: block_depths[{stage}]={block_depths_[stage]} must "
                    f"be at least 1."
                )
            if block_depths_[stage] > 1 and (
                kernel_sizes_[stage][0] % 2 == 0 or kernel_sizes_[stage][1] % 2 == 0
            ):
                raise ValueError(
                    f"TwoDCnnResidualEncoder: stage {stage} has block_depths="
                    f"{block_depths_[stage]} > 1, which requires an odd kernel_size on both "
                    f"axes for its internal, length-preserving layers (a stride-1 layer with "
                    f"an even kernel_size cannot preserve length exactly with any integer "
                    f"padding, independently per axis), got "
                    f"kernel_sizes[{stage}]={kernel_sizes_[stage]}. Use an odd kernel_size on "
                    f"both axes for this stage, or block_depths=1 if an even kernel_size is "
                    f"required."
                )

        resolved_pool_strides = tuple(
            pool_strides_[stage] if pool_strides_[stage] is not None else pool_kernel_sizes_[stage]
            for stage in range(num_stages)
        )
        self._min_input_shape = _computeMinimumInputShapeFromResolved(
            num_stages,
            block_depths_,
            kernel_sizes_,
            strides_,
            paddings_,
            dilations_,
            poolings_,
            pool_kernel_sizes_,
            resolved_pool_strides,
            pool_paddings_,
        )

        layers: list[nn.Module] = []
        channels = in_channels
        for stage in range(num_stages):
            out_channels = hidden_channels[stage]
            layers.append(
                Residual2DBlock(
                    channels,
                    out_channels,
                    depth=block_depths_[stage],
                    kernel_size=kernel_sizes_[stage],
                    stride=strides_[stage],
                    padding=paddings_[stage],
                    dilation=dilations_[stage],
                    shortcut_kernel_size=shortcut_kernel_sizes_[stage],
                    activation=activations_[stage],
                    normalization=normalizations_[stage],
                )
            )
            pool_layer = build2DPoolLayer(
                poolings_[stage],
                pool_kernel_sizes_[stage],
                pool_strides_[stage],
                pool_paddings_[stage],
                **pool_kwargs_[stage],
            )
            if pool_layer is not None:
                layers.append(pool_layer)
            channels = out_channels
        self.conv = nn.Sequential(*layers)

        if global_pool == "avg":
            self.pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unknown global_pool '{global_pool}'. Expected 'avg' or 'max'.")

        head_layers: list[nn.Module] = []
        head_in = channels
        for hidden_dim in head_hidden_dims:
            head_layers.append(nn.Linear(head_in, hidden_dim))
            if head_activation is not None:
                head_layers.append(head_activation())
            head_in = hidden_dim
        self.head: nn.Module = nn.Sequential(*head_layers) if head_layers else nn.Identity()

        self.to_mu = nn.Linear(head_in, self._latent_dim)
        self.to_logvar = nn.Linear(head_in, self._latent_dim)

    @staticmethod
    def computeMinimumInputShape(
        hidden_channels: Sequence[int],
        block_depths: int | Sequence[int] = 2,
        kernel_sizes: ShapeLike | list[ShapeLike] = 3,
        strides: ShapeLike | list[ShapeLike] = 1,
        paddings: ShapeLike | list[ShapeLike] | None = None,
        dilations: ShapeLike | list[ShapeLike] = 1,
        poolings: str | None | Sequence[str | None] = "max",
        pool_kernel_sizes: ShapeLike | None | list[ShapeLike | None] = 2,
        pool_strides: ShapeLike | None | list[ShapeLike | None] = None,
        pool_paddings: ShapeLike | list[ShapeLike] = 0,
    ) -> tuple[int, int]:
        """Compute the minimum input `(height, width)` a given configuration can accept.

        Lets a caller check, before constructing a full `TwoDCnnResidualEncoder`
        (or after, to understand why `forward()` raised), the smallest
        `(height, width)` that keeps every intermediate feature map at a length
        `>= 1` on *both* axes for this architecture (spec §12: verify a
        configuration before committing to it). Generalizes
        `OneDCnnResidualEncoder.computeMinimumInputLength` to 2D, solved
        independently per axis via `utils.conv_math.solveMinimumInputShapeForConv2d`
        (each axis of every layer in this stack only ever depends on that same
        axis's own hyperparameters, so the two axes never interact in this
        computation), and accounts for `block_depths` exactly as that method does:
        each stage contributes `block_depths[stage]` conv layers, not a single one.

        Solves the requirement backward, from the last stage to the first,
        inverting each stage's residual block (and pooling layer, if present).

        Args:
            hidden_channels: As in `__init__`. Only its length (the number of
                stages) affects the result.
            block_depths: As in `__init__`.
            kernel_sizes: As in `__init__`.
            strides: As in `__init__`.
            paddings: As in `__init__`. `None` resolves the same way `__init__`
                does: `dilation[axis] * (kernel_size[axis] // 2)` per axis, per
                stage.
            dilations: As in `__init__`.
            poolings: As in `__init__`.
            pool_kernel_sizes: As in `__init__`.
            pool_strides: As in `__init__`. `None` resolves to that stage's
                `pool_kernel_sizes` entry, matching `build2DPoolLayer`'s own
                default.
            pool_paddings: As in `__init__`.

        Returns:
            The minimum `(height, width)` this configuration can accept without
            any intermediate feature map collapsing to a length `<= 0` on either
            axis.

        Raises:
            ValueError: If any per-stage shape argument does not resolve cleanly
                (see `utils.stage_config.broadcastPerStageShape`).
        """
        num_stages = len(hidden_channels)
        block_depths_ = broadcastPerStage(block_depths, num_stages, "block_depths")
        kernel_sizes_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(kernel_sizes, num_stages, 2, "kernel_sizes"),
        )
        strides_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(strides, num_stages, 2, "strides"),
        )
        dilations_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(dilations, num_stages, 2, "dilations"),
        )
        if paddings is None:
            paddings_: tuple[tuple[int, int], ...] = tuple(
                (
                    dilations_[stage][0] * (kernel_sizes_[stage][0] // 2),
                    dilations_[stage][1] * (kernel_sizes_[stage][1] // 2),
                )
                for stage in range(num_stages)
            )
        else:
            paddings_ = cast(
                "tuple[tuple[int, int], ...]",
                broadcastPerStageShape(paddings, num_stages, 2, "paddings"),
            )
        poolings_: tuple[str | None, ...] = broadcastPerStage(poolings, num_stages, "poolings")
        pool_kernel_sizes_: tuple[tuple[int, int] | None, ...] = cast(
            "tuple[tuple[int, int] | None, ...]",
            broadcastPerStageOptionalShape(pool_kernel_sizes, num_stages, 2, "pool_kernel_sizes"),
        )
        pool_strides_: tuple[tuple[int, int] | None, ...] = (
            cast(
                "tuple[tuple[int, int] | None, ...]",
                broadcastPerStageOptionalShape(pool_strides, num_stages, 2, "pool_strides"),
            )
            if pool_strides is not None
            else (None,) * num_stages
        )
        pool_paddings_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(pool_paddings, num_stages, 2, "pool_paddings"),
        )

        return _computeMinimumInputShapeFromResolved(
            num_stages,
            block_depths_,
            kernel_sizes_,
            strides_,
            paddings_,
            dilations_,
            poolings_,
            pool_kernel_sizes_,
            pool_strides_,
            pool_paddings_,
        )

    def _validateInputShape(self, height: int, width: int) -> None:
        """Verify that `(height, width)` meets this architecture's precomputed minimum.

        Args:
            height: Height of the raw input image for this call.
            width: Width of the raw input image for this call.

        Raises:
            ValueError: If `height` or `width` is below the architecture's
                minimum on that axis, i.e. some intermediate feature map would
                collapse to a length `<= 0` along that axis.
        """
        min_height, min_width = self._min_input_shape
        if height < min_height or width < min_width:
            raise ValueError(
                f"TwoDCnnResidualEncoder: input shape (height={height}, width={width}) is "
                f"below the minimum shape (height={min_height}, width={min_width}) this "
                f"architecture can accept without an intermediate feature map collapsing to "
                f"length <= 0 along some axis. Use "
                f"TwoDCnnResidualEncoder.computeMinimumInputShape(...) with the same "
                f"architecture arguments to see which stage/axis is responsible."
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images.

        Args:
            x: Raw images, shape `(batch, height, width)` or
                `(batch, in_channels, height, width)`. A 3D input is treated as
                `(batch, height, width)` and given an explicit channel dimension
                of `1`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.
        """
        images = x.unsqueeze(1) if x.dim() == 3 else x
        self._validateInputShape(images.shape[-2], images.shape[-1])
        features = self.conv(images)
        pooled: torch.Tensor = self.pool(features).flatten(1)
        pooled = self.head(pooled)
        mu: torch.Tensor = self.to_mu(pooled)
        logvar: torch.Tensor = self.to_logvar(pooled)
        return mu, logvar

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return self._modality_name

    @property
    def minimal_input_length(self) -> int:
        """Conservative scalar summary of `minimal_input_shape` (spec of `AbstractEncoder`).

        Mirrors `TwoDCnnEncoder.minimal_input_length`: `AbstractEncoder.
        minimal_input_length` is a single `int`, a contract written with 1D
        encoders in mind; this encoder's true minimum is a `(height, width)`
        pair, which does not collapse into one number without losing
        information (see `minimal_input_shape` for the exact, independent
        per-axis requirement). This property still implements the abstract
        contract, returning the larger of the two per-axis minimums: the
        smallest side length a *square* input can have and still be guaranteed
        acceptable on both axes.

        Returns:
            `max(minimal_input_shape)`.
        """
        return max(self._min_input_shape)

    @property
    def minimal_input_shape(self) -> tuple[int, int]:
        """Minimal `(height, width)` this encoder can accept without collapsing.

        The precise, per-axis counterpart of `minimal_input_length` (see that
        property's own docstring for why a single `int` cannot fully express a
        2D minimum): a real, non-square input must satisfy both bounds
        independently, not merely have its larger side above
        `minimal_input_length`.

        Returns:
            The minimal `(height, width)`.
        """
        return self._min_input_shape
