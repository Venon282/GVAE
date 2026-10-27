"""Audio waveform encoder (spec §6, roadmap P3-8)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.activations import getActivationFactory
from global_vae.utils.mlp_blocks import buildMlp
from global_vae.utils.waveform_blocks import (
    buildDilatedResidualUnits,
    buildDownsampleConv,
    resolveWaveformStages,
)

POOLING_CHOICES = ("avg", "max")


@registerEncoder("audio_waveform_encoder_v1")
class AudioWaveformEncoder(AbstractEncoder):
    """Strided, dilated-residual convolutional encoder for a raw audio waveform.

    A convolution (`stem_kernel_size`) lifts the `in_channels` of the waveform to
    `hidden_channels[0]`. Each stage then runs a stack of residual units, one per value of
    `dilations`, at its width, and ends with a strided convolution that divides the length by its
    stride and moves to the next width. The result is reduced over time (the mean, or the max with
    `global_pool="max"`) and goes through an optional MLP and two linear layers to give
    `(mu, logvar)`.

    The reduction over time makes the encoder length-agnostic: clips of any length from `1` sample
    up give the same output shape (a downsampling convolution maps a length `L` to
    `ceil(L / stride)`, so no length collapses). It also means one latent vector describes the whole
    clip, and the order of the frames is not in it.

    The input is a float waveform, normally within `[-1, 1]`, at a sample rate chosen by the
    caller's `loader_factory` (spec §6.2): the model knows a number of samples, not a rate.
    Decoding, resampling and normalizing audio are not done here.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        hidden_channels: Sequence[int] = (16, 32, 64, 128),
        strides: int | Sequence[int] = 4,
        dilations: Sequence[int] = (1, 3, 9),
        kernel_size: int = 3,
        stem_kernel_size: int = 7,
        activation: str = "elu",
        normalization: str | None = "group",
        global_pool: str = "avg",
        head_hidden_dims: Sequence[int] = (),
        modality_name: str = "audio",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            in_channels: Number of audio channels (`1` for mono, `2` for stereo).
            hidden_channels: Channel width of each stage, whose length is the number of stages.
            strides: Downsampling factor of each stage, or one value for every stage. The product
                is the number of samples one frame covers (`4` for four stages is 256).
            dilations: Dilation of each residual unit in a stage, the same in every stage.
            kernel_size: Odd kernel size of the convolutions of the residual units.
            stem_kernel_size: Odd kernel size of the first convolution.
            activation: An activation name, see `global_vae.utils.activations`.
            normalization: `"group"` (default, independent of the batch and of the length),
                `"batch"` or `None`.
            global_pool: `"avg"` (default) or `"max"`, the reduction over time.
            head_hidden_dims: Hidden widths of an optional MLP between the pooled features and
                the `mu` and `logvar` layers. Empty (default) goes straight to them.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If `latent_dim` or `in_channels` is below `1`, `global_pool` is not a
                known choice, or an argument of the stack is out of range (the message names it).
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if in_channels < 1:
            raise ValueError(f"in_channels must be at least 1, got {in_channels}.")
        if global_pool not in POOLING_CHOICES:
            raise ValueError(
                f"Unknown global_pool '{global_pool}'. Expected one of {POOLING_CHOICES}."
            )
        if stem_kernel_size < 1 or stem_kernel_size % 2 == 0:
            raise ValueError(f"stem_kernel_size must be odd, got {stem_kernel_size}.")
        channels, per_stage_strides = resolveWaveformStages(
            hidden_channels, strides, "AudioWaveformEncoder"
        )
        self._latent_dim = latent_dim
        self._in_channels = in_channels
        self._global_pool = global_pool
        self._modality_name = modality_name

        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, channels[0], stem_kernel_size, padding=stem_kernel_size // 2)
        ]
        for index, (width, stride) in enumerate(zip(channels, per_stage_strides, strict=True)):
            layers.append(
                buildDilatedResidualUnits(width, dilations, kernel_size, activation, normalization)
            )
            next_width = channels[index + 1] if index + 1 < len(channels) else width
            layers.append(buildDownsampleConv(width, next_width, stride))
            layers.append(getActivationFactory(activation)())
        self.body = nn.Sequential(*layers)
        self.head, head_width = buildMlp(channels[-1], head_hidden_dims, activation="relu")
        self.to_mu = nn.Linear(head_width, latent_dim)
        self.to_logvar = nn.Linear(head_width, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of waveforms.

        Args:
            x: Float waveform, shape `(batch, samples)` for `in_channels=1`, or `(batch,
                in_channels, samples)`. A 2D input is given an explicit channel dimension of `1`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.

        Raises:
            ValueError: If `x` is not 2D or 3D, has no sample, or has a channel count other than
                `in_channels`.
        """
        waveform = x.unsqueeze(1) if x.dim() == 2 else x
        if waveform.dim() != 3 or waveform.shape[2] < 1:
            raise ValueError(
                f"AudioWaveformEncoder expects (batch, samples) or (batch, channels, samples) "
                f"with at least one sample, got shape {tuple(x.shape)}."
            )
        if waveform.shape[1] != self._in_channels:
            raise ValueError(
                f"AudioWaveformEncoder was built for in_channels={self._in_channels}, got "
                f"{waveform.shape[1]} channel(s) in shape {tuple(x.shape)}."
            )
        features = self.body(waveform)
        pooled = features.mean(dim=2) if self._global_pool == "avg" else features.amax(dim=2)
        pooled = self.head(pooled)
        mu: torch.Tensor = self.to_mu(pooled)
        logvar: torch.Tensor = self.to_logvar(pooled)
        return mu, logvar

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the `(mu, logvar)` output."""
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        """Name of the modality this encoder handles."""
        return self._modality_name

    @property
    def minimal_input_length(self) -> int:
        """Shortest clip this encoder accepts: `1` sample.

        A downsampling convolution maps a length `L` to `ceil(L / stride)`, which is never below
        `1`, so no length collapses a feature map the way it can in a plain strided stack.

        Returns:
            `1`.
        """
        return 1
