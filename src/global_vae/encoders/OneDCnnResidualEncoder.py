"""1D residual (ResNet-style) CNN encoder (spec §6, §7, §12)."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.builders import build1DPoolLayer
from global_vae.utils.conv_blocks import Residual1DBlock
from global_vae.utils.conv_math import solveMinimumInputLengthForConv1d
from global_vae.utils.stage_config import broadcastPerStage


@registerEncoder("1d_cnn_resnet_encoder_v1")
class OneDCnnResidualEncoder(AbstractEncoder):
    """1D residual convolutional encoder, robust to variable input length.

    A stack of residual blocks (`utils.conv_blocks.Residual1DBlock`, spec
    §7's "scaling toward larger backbones"), each optionally followed by a
    pooling layer, followed by global adaptive pooling. As in
    `OneDCnnEncoder`, the adaptive pool is what makes the encoder
    length-agnostic: two series of different lengths produce feature maps
    of different spatial size after the residual stack, but the same
    fixed-size vector after pooling, so a single set of weights covers
    series of any length.

    Deliberately as permissive as `OneDCnnEncoder` along every axis that
    class already varies per stage (`kernel_sizes`, `strides`, `paddings`,
    `dilations`, `poolings` and its own sub-parameters, `activations`,
    `normalizations`, all via `utils.stage_config.broadcastPerStage`), plus
    one new axis this class adds: `block_depths`, letting each stage's
    residual block have its own number of internal layers (e.g. `3` layers
    before the first stage's shortcut, then `4` for a deeper second stage).
    `shortcut_kernel_sizes` is the one further per-stage knob a residual
    architecture specifically needs (`Residual1DBlock`'s own shortcut
    projection kernel), also broadcastable the same way.

    One deliberate, documented scope narrowing relative to
    `OneDCnnEncoder`: `hidden_channels` here is always a plain
    `tuple[int, ...]` (channel width per stage), not `OneDCnnEncoder`'s
    `tuple[int | nn.Module, ...]` escape hatch for dropping in an
    arbitrary custom stage. A residual block is a coupled main-path/
    shortcut pair (see `Residual1DBlock`'s own docstring for why the
    shortcut's shape must be provably tied to the main path's), which does
    not compose with an opaque, arbitrary `nn.Module` standing in for one
    stage the way a single plain conv stage does; supporting that escape
    hatch here would either silently skip the residual connection for that
    one stage, or require the caller's own module to expose enough
    structure for a shortcut to be derived from it, neither of which is a
    small addition. A caller wanting a fully custom stage inside an
    otherwise-residual encoder can still compose one directly with
    `torch.nn.Module`, outside this class, exactly as any other custom
    architecture would.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
        block_depths: int | Sequence[int] = 2,
        kernel_sizes: int | Sequence[int] = 3,
        strides: int | Sequence[int] = 1,
        paddings: int | Sequence[int] | None = None,
        dilations: int | Sequence[int] = 1,
        shortcut_kernel_sizes: int | Sequence[int] = 1,
        poolings: str | None | Sequence[str | None] = "max",
        pool_kernel_sizes: int | None | Sequence[int | None] = 2,
        pool_strides: int | None | Sequence[int | None] = None,
        pool_paddings: int | Sequence[int] = 0,
        pool_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        activations: (
            Callable[[], nn.Module] | Sequence[Callable[[], nn.Module] | None] | None
        ) = nn.ReLU,
        normalizations: (
            Callable[[int], nn.Module] | Sequence[Callable[[int], nn.Module] | None] | None
        ) = nn.BatchNorm1d,
        global_pool: str = "avg",
        head_hidden_dims: tuple[int, ...] = (),
        head_activation: Callable[[], nn.Module] | None = nn.ReLU,
        modality_name: str = "vector",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            in_channels: Number of input channels (`1` for a plain
                scalar series; more if several co-registered channels
                are stacked, e.g. multiple detectors).
            hidden_channels: Output channel width of each residual
                block's stage, applied in order. Its length fixes the
                number of stages.
            block_depths: Number of internal `Conv1d` layers in each
                stage's residual block, per stage or shared (spec's own
                request: e.g. `(3, 4)` for a first stage with `3`
                layers before its shortcut and a second, deeper stage
                with `4`). `2` (default) is the classic ResNet
                "BasicBlock"; must be odd-`kernel_size`-compatible
                whenever greater than `1` (see
                `utils.conv_blocks.Residual1DBlock`).
            kernel_sizes: Convolution kernel size, per stage or shared,
                applied to every layer within that stage's block. Must
                be odd for any stage whose `block_depths` entry is
                greater than `1`.
            strides: Downsampling stride of each stage's *first* layer
                only, per stage or shared; every later layer within the
                same block always uses stride `1` (see
                `Residual1DBlock`). Use this (with `poolings=None`) to
                downsample via strided convolutions instead of a
                separate pooling layer, exactly as in `OneDCnnEncoder`.
            paddings: Convolution padding of each stage's first layer,
                per stage or shared. Defaults (`None`) to `dilation *
                (kernel_size // 2)` for each stage, matching
                `OneDCnnEncoder`'s own default and, not coincidentally,
                the one choice that always keeps a projection
                shortcut's output length matching the main path's (see
                `Residual1DBlock`).
            dilations: Convolution dilation, per stage or shared.
            shortcut_kernel_sizes: Kernel size of each stage's shortcut
                projection (only built when that stage changes channel
                width and/or downsamples), per stage or shared. Defaults
                to `1`, the standard ResNet choice.
            poolings: `"max"`, `"avg"`, or `None` to disable pooling,
                per stage or shared, applied *after* that stage's
                residual block (not inside it). Different stages may
                use different pooling strategies.
            pool_kernel_sizes: Pooling window size, per stage or
                shared. Required (not `None`) for any stage whose
                `poolings` entry is not `None`; ignored otherwise.
            pool_strides: Pooling stride, per stage or shared. Defaults
                (`None`) to that stage's pooling kernel size
                (non-overlapping windows). Ignored for stages whose
                `poolings` entry is `None`.
            pool_paddings: Pooling padding, per stage or shared.
                Ignored for stages whose `poolings` entry is `None`.
            pool_kwargs: Additional keyword arguments forwarded to the
                pooling layer's constructor (e.g. `ceil_mode`,
                `count_include_pad`), per stage or shared. `None`
                (default) forwards no extra arguments for any stage.
            activations: Zero-argument factory returning a fresh
                activation module, per stage or shared, used inside
                and after every stage's residual block (see
                `Residual1DBlock`). Pass `None` (either overall, or as
                one stage's entry in a sequence) to disable activation
                for that stage.
            normalizations: One-argument factory taking a channel count
                and returning a fresh normalization module, per stage
                or shared, used the same places as `activations`.
            global_pool: `"avg"` or `"max"`: which adaptive pooling
                reduces the final feature map to a single fixed-size
                vector, regardless of input length.
            head_hidden_dims: Hidden layer sizes for an optional small
                MLP inserted between the pooled features and the
                `to_mu`/`to_logvar` heads. Empty tuple (default) keeps
                a single linear layer straight from pooled features to
                each head.
            head_activation: Optional activation layer to use for the
                head.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If any per-stage sequence argument does not
                have exactly `len(hidden_channels)` values, if a stage
                requests pooling without a `pool_kernel_sizes` entry,
                if `global_pool` is not a recognized choice, if any
                stage's `block_depths` is not at least `1`, if any
                stage's `block_depths` is greater than `1` and its
                `kernel_sizes` is even, or if any stage needing a
                projection shortcut cannot reach the main path's output
                length for every input length (see `Residual1DBlock`).
        """
        super().__init__()
        self._latent_dim = latent_dim
        self._modality_name = modality_name
        num_stages = len(hidden_channels)

        block_depths_ = broadcastPerStage(block_depths, num_stages, "block_depths")
        kernel_sizes_ = broadcastPerStage(kernel_sizes, num_stages, "kernel_sizes")
        strides_ = broadcastPerStage(strides, num_stages, "strides")
        dilations_ = broadcastPerStage(dilations, num_stages, "dilations")
        if paddings is None:
            paddings_: tuple[int, ...] = tuple(
                dilation * (kernel_size // 2)
                for kernel_size, dilation in zip(kernel_sizes_, dilations_, strict=True)
            )
        else:
            paddings_ = broadcastPerStage(paddings, num_stages, "paddings")
        shortcut_kernel_sizes_ = broadcastPerStage(
            shortcut_kernel_sizes, num_stages, "shortcut_kernel_sizes"
        )
        poolings_: tuple[str | None, ...] = broadcastPerStage(poolings, num_stages, "poolings")
        pool_kernel_sizes_: tuple[int | None, ...] = broadcastPerStage(
            pool_kernel_sizes, num_stages, "pool_kernel_sizes"
        )
        pool_strides_: tuple[int | None, ...] = (
            broadcastPerStage(pool_strides, num_stages, "pool_strides")
            if pool_strides is not None
            else (None,) * num_stages
        )
        pool_paddings_ = broadcastPerStage(pool_paddings, num_stages, "pool_paddings")
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
                    f"OneDCnnResidualEncoder: block_depths[{stage}]={block_depths_[stage]} must "
                    f"be at least 1."
                )
            if block_depths_[stage] > 1 and kernel_sizes_[stage] % 2 == 0:
                raise ValueError(
                    f"OneDCnnResidualEncoder: stage {stage} has block_depths="
                    f"{block_depths_[stage]} > 1, which requires an odd kernel_size for its "
                    f"internal, length-preserving layers (a stride-1 layer with an even "
                    f"kernel_size cannot preserve length exactly with any integer padding), "
                    f"got kernel_sizes[{stage}]={kernel_sizes_[stage]}. Use an odd kernel_size "
                    f"for this stage, or block_depths=1 if an even kernel_size is required."
                )

        resolved_pool_strides = tuple(
            pool_strides_[stage] if pool_strides_[stage] is not None else pool_kernel_sizes_[stage]
            for stage in range(num_stages)
        )
        self._min_input_length = OneDCnnResidualEncoder.computeMinimumInputLength(
            hidden_channels=hidden_channels,
            block_depths=block_depths_,
            kernel_sizes=kernel_sizes_,
            strides=strides_,
            paddings=paddings_,
            dilations=dilations_,
            poolings=poolings_,
            pool_kernel_sizes=pool_kernel_sizes_,
            pool_strides=resolved_pool_strides,
            pool_paddings=pool_paddings_,
        )

        layers: list[nn.Module] = []
        channels = in_channels
        for stage in range(num_stages):
            out_channels = hidden_channels[stage]
            layers.append(
                Residual1DBlock(
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
            pool_layer = build1DPoolLayer(
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
            self.pool: nn.Module = nn.AdaptiveAvgPool1d(1)
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
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
    def computeMinimumInputLength(
        hidden_channels: Sequence[int],
        block_depths: int | Sequence[int] = 2,
        kernel_sizes: int | Sequence[int] = 3,
        strides: int | Sequence[int] = 1,
        paddings: int | Sequence[int] | None = None,
        dilations: int | Sequence[int] = 1,
        poolings: str | None | Sequence[str | None] = "max",
        pool_kernel_sizes: int | None | Sequence[int | None] = 2,
        pool_strides: int | None | Sequence[int | None] = None,
        pool_paddings: int | Sequence[int] = 0,
    ) -> int:
        """Compute the minimum input length a given configuration can accept.

        Generalizes `OneDCnnEncoder.computeMinimumInputLength` (see its
        own docstring for the underlying reasoning: solving each layer's
        requirement backward, from the last stage to the first, via
        `utils.conv_math.solveMinimumInputLengthForConv1d`) with one
        added inner loop: each stage now contributes `block_depths[stage]`
        conv layers (a strided/kernel-sized first layer, then
        length-preserving internal layers) instead of a single one.

        Args:
            hidden_channels: As in `__init__`. Only its length (the
                number of stages) affects the result.
            block_depths: As in `__init__`.
            kernel_sizes: As in `__init__`.
            strides: As in `__init__`.
            paddings: As in `__init__`. `None` resolves the same way
                `__init__` does: `dilation * (kernel_size // 2)` per
                stage.
            dilations: As in `__init__`.
            poolings: As in `__init__`.
            pool_kernel_sizes: As in `__init__`.
            pool_strides: As in `__init__`. `None` resolves to that
                stage's `pool_kernel_sizes` entry, matching
                `build1DPoolLayer`'s own default.
            pool_paddings: As in `__init__`.

        Returns:
            The minimum `input_length` this configuration can accept
            without any intermediate feature map collapsing to a length
            `<= 0`.

        Raises:
            ValueError: If any per-stage sequence argument does not
                have exactly `len(hidden_channels)` values.
        """
        num_stages = len(hidden_channels)
        block_depths_ = broadcastPerStage(block_depths, num_stages, "block_depths")
        kernel_sizes_ = broadcastPerStage(kernel_sizes, num_stages, "kernel_sizes")
        strides_ = broadcastPerStage(strides, num_stages, "strides")
        dilations_ = broadcastPerStage(dilations, num_stages, "dilations")
        if paddings is None:
            paddings_ = tuple(
                dilation * (kernel_size // 2)
                for kernel_size, dilation in zip(kernel_sizes_, dilations_, strict=True)
            )
        else:
            paddings_ = broadcastPerStage(paddings, num_stages, "paddings")
        poolings_: tuple[str | None, ...] = broadcastPerStage(poolings, num_stages, "poolings")
        pool_kernel_sizes_: tuple[int | None, ...] = broadcastPerStage(
            pool_kernel_sizes, num_stages, "pool_kernel_sizes"
        )
        pool_strides_: tuple[int | None, ...] = (
            broadcastPerStage(pool_strides, num_stages, "pool_strides")
            if pool_strides is not None
            else (None,) * num_stages
        )
        pool_paddings_ = broadcastPerStage(pool_paddings, num_stages, "pool_paddings")

        required_min_length = 1
        for stage in reversed(range(num_stages)):
            if poolings_[stage] is not None and pool_kernel_sizes_[stage] is None:
                raise ValueError(
                    f"pooling='{poolings_[stage]}' requires a kernel_size, but kernel_size is None."
                )
            if poolings_[stage] is not None:
                kernel_size_for_stage = pool_kernel_sizes_[stage]
                assert kernel_size_for_stage is not None
                stride_for_stage = pool_strides_[stage]
                pool_stride = (
                    stride_for_stage if stride_for_stage is not None else kernel_size_for_stage
                )
                required_min_length = solveMinimumInputLengthForConv1d(
                    required_min_length,
                    kernel_size_for_stage,
                    pool_stride,
                    pool_paddings_[stage],
                    dilation=1,
                )

            same_padding = dilations_[stage] * (kernel_sizes_[stage] // 2)
            for layer_index in reversed(range(block_depths_[stage])):
                is_first = layer_index == 0
                layer_stride = strides_[stage] if is_first else 1
                layer_padding = paddings_[stage] if is_first else same_padding
                required_min_length = solveMinimumInputLengthForConv1d(
                    required_min_length,
                    kernel_sizes_[stage],
                    layer_stride,
                    layer_padding,
                    dilations_[stage],
                )

        return required_min_length

    def _validateInputLength(self, input_length: int) -> None:
        """Verify that `input_length` meets this architecture's precomputed minimum.

        Args:
            input_length: Length of the raw input series for this call.

        Raises:
            ValueError: If `input_length` is below the architecture's
                minimum, i.e. some intermediate feature map would
                collapse to a length `<= 0`.
        """
        if input_length < self._min_input_length:
            raise ValueError(
                f"OneDCnnResidualEncoder: input_length={input_length} is below the minimum "
                f"input_length={self._min_input_length} this architecture can accept "
                f"without an intermediate feature map collapsing to length <= 0. Use "
                f"OneDCnnResidualEncoder.computeMinimumInputLength(...) with the same "
                f"architecture arguments to see which stage is responsible."
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of 1D series.

        Args:
            x: Raw series, shape `(batch, length)` or
                `(batch, in_channels, length)`. A 2D input is treated
                as `(batch, length)` and given an explicit channel
                dimension of `1`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.
        """
        series = x.unsqueeze(1) if x.dim() == 2 else x
        self._validateInputLength(series.shape[-1])
        features = self.conv(series)
        pooled: torch.Tensor = self.pool(features).squeeze(-1)
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
        return self._min_input_length
