"""1D CNN encoder
"""

from collections.abc import Callable, Sequence

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.stage_config import broadcastPerStage
from global_vae.utils.builders import build1DPoolLayer

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
        pool_kwargs: dict | Sequence[dict] = {},
        activations: Callable[[], nn.Module] | Sequence[Callable[[], nn.Module] | None] | None = nn.ReLU,
        normalizations: Callable[[int], nn.Module] | Sequence[Callable[[int], nn.Module | None]] | None = nn.BatchNorm1d,
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
        poolings_ = broadcastPerStage(poolings, num_stages, "poolings")
        pool_kernel_sizes_ = broadcastPerStage(pool_kernel_sizes, num_stages, "pool_kernel_sizes")
        pool_strides_: tuple[int | None, ...] = (
            broadcastPerStage(pool_strides, num_stages, "pool_strides")
            if pool_strides is not None
            else (None,) * num_stages
        )
        pool_paddings_ = broadcastPerStage(pool_paddings, num_stages, "pool_paddings")
        pool_kwargs_ = broadcastPerStage(pool_kwargs, num_stages, "pool_kwargs")
        activations_ = broadcastPerStage(activations, num_stages, "activations")
        normalizations_ = broadcastPerStage(normalizations, num_stages, "normalizations")

        layers: list[nn.Module] = []
        channels = in_channels
        for stage in range(num_stages):



            if isinstance(hidden_channels, nn.Module):
                out_channels = hidden_channels.out_channels if hasattr(hidden_channels, "out_channels") else \
                                hidden_channels.out_features if hasattr(hidden_channels, "out_features") else \
                                (_ for _ in ()).throw(AttributeError(f"The {hidden_channels.__class__.__name__} in hidden_channels do not have an out_channels or out_features which is required."))
                layer = hidden_channels
            else:
                out_channels = hidden_channels[stage]
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

        self.to_mu = nn.Linear(head_in, latent_dim)
        self.to_logvar = nn.Linear(head_in, latent_dim)

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
        features = self.conv(series)
        pooled: torch.Tensor = self.pool(features).squeeze(-1)
        pooled = self.head(pooled)
        mu: torch.Tensor = self.to_mu(pooled)
        logvar: torch.Tensor = self.to_logvar(pooled)
        return mu, logvar

    @property
    def latentDim(self) -> int:
        return self._latent_dim

    @property
    def modalityName(self) -> str:
        return self._modality_name
