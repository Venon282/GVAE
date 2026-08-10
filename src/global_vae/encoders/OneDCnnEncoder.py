"""1D CNN encoder
"""

from collections.abc import Callable, Sequence, Sized
from typing import Any, cast

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.stage_config import broadcastPerStage
from global_vae.utils.builders import build1DPoolLayer
from global_vae.utils.conv_math import solveMinimumInputLengthForConv1d

@registerEncoder("1d_cnn_encoder_v1")
class OneDCnnEncoder(AbstractEncoder):
    """1D convolutional encoder, robust to variable input length.

    A stack of conv (+ optional normalization, optional activation,
    optional pooling) stages, followed by global adaptive pooling. The
    adaptive pool is what makes the encoder length-agnostic: two series
    of different lengths produce feature maps of different spatial
    size after the conv stack, but the same fixed-size vector after
    pooling, so a single set of weights covers series of any length.

    Every per-stage hyperparameter (`kernel_sizes`, `strides`,
    `paddings`, `dilations`, `poolings` and its own sub-parameters,
    `paddings`, `dilations`) accepts either one value shared by every
    `activations`, `normalizations`) accepts either one value shared by every
    stage or a sequence of exactly `len(hidden_channels)` values, one
    per stage (see `utils.stage_config.broadcastPerStage`).

    Note: if a stage's `poolings` entry is not `None`, that stage's
    sequence length; if `strides` greater than 1 are also used, both
    pooling step reduces sequence length; if that stage's `strides` is
    reductions compound. The input length must stay large enough,
    also greater than 1, both reductions compound. The input length
    after however much downsampling the chosen configuration performs,
    must stay large enough, after however much downsampling the chosen
    that no intermediate feature map collapses to zero length. This is
    configuration performs, that no intermediate feature map collapses
    a real constraint of any strided/pooled conv stack, not something
    to zero length. This is a real constraint of any strided/pooled
    silently handled.
    conv stack, not something silently handled.
    """
    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        hidden_channels: tuple[int | nn.Module, ...] = (32, 64, 128),
        kernel_sizes: int | Sequence[int] = 5,
        strides: int | Sequence[int] = 1,
        paddings: int | Sequence[int] | None = None,
        dilations: int | Sequence[int] = 1,
        poolings: str | None | Sequence[str | None] = "max",
        pool_kernel_sizes: int | None | Sequence[int | None] = 2,
        pool_strides: int | None | Sequence[int | None] = None,
        pool_paddings: int | Sequence[int] = 0,
        pool_kwargs: dict[str, Any] | Sequence[dict[str, Any]] = {},
        activations: Callable[[], nn.Module] | Sequence[Callable[[], nn.Module] | None] | None = nn.ReLU,
        normalizations: Callable[[int], nn.Module] | Sequence[Callable[[int], nn.Module] | None] | None = nn.BatchNorm1d,
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
            hidden_channels: Output channel width of each conv stage,
                applied in order. Its length fixes the number of
                stages. Or directly a layer.
            kernel_sizes: Convolution kernel size, per stage or shared.
            strides: Convolution stride, per stage or shared. Use this
                (with `poolings=None`) to downsample via strided
                convolutions instead of a separate pooling layer.
            paddings: Convolution padding, per stage or shared.
                Defaults (`None`) to `dilation * (kernel_size // 2)`
                for each stage, which keeps a stride-1 stage's output
                length equal to its input length.
            dilations: Convolution dilation, per stage or shared.
            poolings: `"max"`, `"avg"`, or `None` to disable pooling,
                per stage or shared. Different stages may use
                different pooling strategies.
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
                `count_include_pad`), per stage or shared. Defaults
                (`None`) to no extra arguments for any stage.
            activations: Zero-argument factory returning a fresh
                activation module, per stage or shared. Pass `None`
                (either overall, or as one stage's entry in a sequence)
                to disable activation for that stage.
            normalizations: One-argument factory taking a channel count
                and returning a fresh normalization module, per stage
                or shared. Pass `None` the same way to disable
                normalization for a stage.
            global_pool: `"avg"` or `"max"`: which adaptive pooling
                            reduces the final feature map to a single fixed-size
                            vector, regardless of input length.
            head_hidden_dims: Hidden layer sizes for an optional small MLP
                inserted between the pooled features and the `to_mu`/
                `to_logvar` heads. Empty tuple (default) keeps today's
                behavior: a single linear layer straight from pooled
                features to each head.
            head_activation: Optional head activation layer


        Raises:
            ValueError: If any per-stage sequence argument does not
                have exactly `len(hidden_channels)` values, if a stage
                requests pooling without a `pool_kernel_sizes` entry, or
                if `global_pool` is not a recognized choice.
        """
        super().__init__()
        self._latent_dim = latent_dim
        self._modality_name = modality_name
        num_stages = len(hidden_channels)

        kernel_sizes_ = broadcastPerStage(kernel_sizes, num_stages, "kernel_sizes")
        strides_ = broadcastPerStage(strides, num_stages, "strides")
        dilations_ = broadcastPerStage(dilations, num_stages, "dilations")
        if paddings is None:
            paddings_ = tuple(dilation * (kernel_size // 2) for kernel_size, dilation in zip(kernel_sizes_, dilations_, strict=True))
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
        pool_kwargs_ = broadcastPerStage(pool_kwargs, num_stages, "pool_kwargs")
        activations_ = broadcastPerStage(activations, num_stages, "activations")
        normalizations_: tuple[Callable[[int], nn.Module] | None, ...] = broadcastPerStage(
            normalizations, num_stages, "normalizations"
        )

        # Get the minimal input len need for this configuation
        resolved_pool_strides = tuple(
            pool_strides_[stage] if pool_strides_[stage] is not None else pool_kernel_sizes_[stage]
            for stage in range(num_stages)
        )
        self._min_input_length = OneDCnnEncoder.computeMinimumInputLength(
            hidden_channels=hidden_channels,
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
            stage_channels = hidden_channels[stage]
            if isinstance(stage_channels, nn.Module):
                if hasattr(stage_channels, "out_channels"):
                    # nn.Module does not statically declare `out_channels`; every
                    # standard PyTorch layer that has one sets it as a plain int
                    # (e.g. Conv1d.out_channels), a convention mypy's generic
                    # Module.__getattr__ stub (-> Tensor | Module) cannot express.
                    out_channels = cast(int, stage_channels.out_channels)
                elif hasattr(stage_channels, "out_features"):
                    out_channels = cast(int, stage_channels.out_features)
                else:
                    raise AttributeError(
                        f"The {stage_channels.__class__.__name__} passed as stage {stage} of "
                        f"hidden_channels has no out_channels or out_features attribute, which "
                        f"is required so later stages know this stage's output width."
                    )
                layer: nn.Module = stage_channels
            else:
                out_channels = stage_channels
                layer = nn.Conv1d(
                    channels,
                    out_channels,
                    kernel_size=kernel_sizes_[stage],
                    stride=strides_[stage],
                    padding=paddings_[stage],
                    dilation=dilations_[stage],
                )

            layers.append(layer)

            normalization = normalizations_[stage]
            if normalization is not None:
                layers.append(normalization(out_channels))

            activation = activations_[stage]
            if activation is not None:
                layers.append(activation())

            pool_layer = build1DPoolLayer(
                                          poolings_[stage],
                                          pool_kernel_sizes_[stage],
                                          pool_strides_[stage],
                                          pool_paddings_[stage],
                                          **pool_kwargs_[stage]
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

        # add an optional mlp before the mu and log var
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
        hidden_channels: Sized,
        kernel_sizes: int | Sequence[int] = 5,
        strides: int | Sequence[int] = 1,
        paddings: int | Sequence[int] | None = None,
        dilations: int | Sequence[int] = 1,
        poolings: str | None | Sequence[str | None] = "max",
        pool_kernel_sizes: int | None | Sequence[int | None] = 2,
        pool_strides: int | None | Sequence[int | None] = None,
        pool_paddings: int | Sequence[int] = 0,
    ) -> int:
        """Compute the minimum input length a given configuration can accept.

        Lets a caller check, before constructing a full `OneDCnnEncoder` (or after, to
        understand why `forward()` raised), the shortest `input_length` that keeps every
        intermediate feature map at a length `>= 1` for this architecture (spec §12: verify
        a configuration before committing to it, rather than discovering a mismatch only
        once a strided/pooled conv stack has already collapsed).

        Solves the requirement backward, from the last stage to the first, inverting each
        stage's convolution (and pooling layer, if present) via
        `solveMinimumInputLengthForConv1d`.

        Args:
            hidden_channels: As in `__init__`. Only its length (the number of stages)
                affects the result.
            kernel_sizes: As in `__init__`.
            strides: As in `__init__`.
            paddings: As in `__init__`. `None` resolves the same way `__init__` does:
                `dilation * (kernel_size // 2)` per stage.
            dilations: As in `__init__`.
            poolings: As in `__init__`.
            pool_kernel_sizes: As in `__init__`.
            pool_strides: As in `__init__`. `None` resolves to that stage's
                `pool_kernel_sizes` entry, matching `build1DPoolLayer`'s own default.
            pool_paddings: As in `__init__`.

        Returns:
            The minimum `input_length` this configuration can accept without any
            intermediate feature map collapsing to a length `<= 0`.

        Raises:
            ValueError: If any per-stage sequence argument does not have exactly
                `len(hidden_channels)` values.
        """
        num_stages = len(hidden_channels)
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
                    f"pooling='{poolings_[stage]}' requires a kernel_size, but "
                    f"kernel_size is None."
                )
            if poolings_[stage] is not None:
                kernel_size_for_stage = pool_kernel_sizes_[stage]
                # Guaranteed non-None by the ValueError check above: reaching this
                # branch means poolings_[stage] is not None, so the compound
                # condition there would already have raised if this were None.
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
            required_min_length = solveMinimumInputLengthForConv1d(
                required_min_length,
                kernel_sizes_[stage],
                strides_[stage],
                paddings_[stage],
                dilations_[stage],
            )

        return required_min_length

    def _validateInputLength(self, input_length: int) -> None:
        """Verify that `input_length` meets this architecture's precomputed minimum.

        The minimum itself is computed once, at construction time, by
        `computeMinimumInputLength` (architecture-only, independent of any actual input);
        this check is therefore a single integer comparison per `forward()` call, not a
        replay of the stage-by-stage computation, so it adds no meaningful cost next to the
        convolutions that follow.

        Args:
            input_length: Length of the raw input series for this call.

        Raises:
            ValueError: If `input_length` is below the architecture's minimum, i.e. some
                intermediate feature map would collapse to a length `<= 0`.
        """
        if input_length < self._min_input_length:
            raise ValueError(
                f"OneDCnnEncoder: input_length={input_length} is below the minimum "
                f"input_length={self._min_input_length} this architecture can accept "
                f"without an intermediate feature map collapsing to length <= 0. Use "
                f"OneDCnnEncoder.computeMinimumInputLength(...) with the same "
                f"architecture arguments to see which stage is responsible."
            )

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
