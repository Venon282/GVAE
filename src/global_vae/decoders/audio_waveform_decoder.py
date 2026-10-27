"""Audio waveform decoder (spec §6, roadmap P3-8)."""

from collections.abc import Sequence

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.activations import getActivationFactory
from global_vae.utils.waveform_blocks import (
    buildDilatedResidualUnits,
    buildUpsampleStage,
    computeSeedLength,
    getNormalizationFactory,
    resolveWaveformStages,
)


@registerDecoder("audio_waveform_decoder_v1")
class AudioWaveformDecoder(AbstractDecoder):
    """Upsampling, dilated-residual convolutional decoder reconstructing a raw audio waveform.

    The latent vector is projected to a seed of `hidden_channels[0]` channels and
    `ceil(output_length / prod(strides))` frames. Each stage upsamples by its stride (nearest
    neighbour followed by a convolution), moves to its width and runs a stack of residual units,
    one per value of `dilations`. A last convolution gives `out_channels`, an optional output
    activation bounds the waveform (`tanh` by default, for audio within `[-1, 1]`), and the result
    is cropped to `output_length`.

    The seed length is rounded up and the surplus is cropped, so any `output_length` is reachable
    whatever the strides (16,000 samples with strides of 4, which no stack of exact doublings
    reaches). The crop removes fewer than `prod(strides)` samples and is the only length
    adjustment: like every decoder here it reconstructs one fixed length, never a resized one.
    """

    def __init__(
        self,
        latent_dim: int,
        output_length: int,
        out_channels: int = 1,
        hidden_channels: Sequence[int] = (128, 64, 32, 16),
        strides: int | Sequence[int] = 4,
        dilations: Sequence[int] = (1, 3, 9),
        kernel_size: int = 3,
        output_kernel_size: int = 7,
        activation: str = "elu",
        normalization: str | None = "group",
        output_activation: str | None = "tanh",
        modality_name: str = "audio",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the latent vector it receives.
            output_length: Number of samples to reconstruct. Data-dependent, so the config layer
                never fills it in: it is the number of samples of a clip, set to
                `data.sequence_length` of the same modality.
            out_channels: Number of audio channels (`1` for mono, `2` for stereo).
            hidden_channels: Channel width of each stage, from the seed outwards, whose length is
                the number of stages.
            strides: Upsampling factor of each stage, or one value for every stage.
            dilations: Dilation of each residual unit in a stage, the same in every stage.
            kernel_size: Odd kernel size of the convolutions of the residual units.
            output_kernel_size: Odd kernel size of the last convolution.
            activation: An activation name, see `global_vae.utils.activations`.
            normalization: `"group"` (default, independent of the batch and of the length),
                `"batch"` or `None`.
            output_activation: Activation applied to the waveform, `"tanh"` by default, or `None`
                for an unbounded output (data that is not within `[-1, 1]`, for example
                standardized).
            modality_name: Name of the modality this decoder reconstructs.

        Raises:
            ValueError: If `latent_dim`, `output_length` or `out_channels` is below `1`, or an
                argument of the stack is out of range (the message names it).
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")
        if output_length < 1:
            raise ValueError(f"output_length must be at least 1, got {output_length}.")
        if out_channels < 1:
            raise ValueError(f"out_channels must be at least 1, got {out_channels}.")
        if output_kernel_size < 1 or output_kernel_size % 2 == 0:
            raise ValueError(f"output_kernel_size must be odd, got {output_kernel_size}.")
        channels, per_stage_strides = resolveWaveformStages(
            hidden_channels, strides, "AudioWaveformDecoder"
        )
        getNormalizationFactory(normalization)  # fail on an unknown name before building
        self._modality_name = modality_name
        self._output_length = output_length
        self._latent_dim = latent_dim
        self._out_channels = out_channels
        self._seed_length = computeSeedLength(output_length, per_stage_strides)

        self.seed = nn.Sequential(
            nn.Linear(latent_dim, channels[0] * self._seed_length),
            nn.Unflatten(1, (channels[0], self._seed_length)),
            getActivationFactory(activation)(),
        )
        layers: list[nn.Module] = []
        previous = channels[0]
        for width, stride in zip(channels, per_stage_strides, strict=True):
            layers.append(buildUpsampleStage(previous, width, stride))
            layers.append(getActivationFactory(activation)())
            layers.append(
                buildDilatedResidualUnits(width, dilations, kernel_size, activation, normalization)
            )
            previous = width
        layers.append(
            nn.Conv1d(previous, out_channels, output_kernel_size, padding=output_kernel_size // 2)
        )
        self.body = nn.Sequential(*layers)
        self.output_activation: nn.Module = (
            getActivationFactory(output_activation)() if output_activation else nn.Identity()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of fixed-length waveforms from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Reconstructed waveform. Shape `(batch, output_length)` for the default single-channel
            case (matching the encoder's plain-waveform input convention); `(batch, out_channels,
            output_length)` if `out_channels` was set above `1`.

        Raises:
            ValueError: If `z` is not of shape `(batch, latent_dim)`.
        """
        if z.dim() != 2 or z.shape[1] != self._latent_dim:
            raise ValueError(
                f"AudioWaveformDecoder expects (batch, {self._latent_dim}) latent vectors, got "
                f"shape {tuple(z.shape)}."
            )
        waveform: torch.Tensor = self.body(self.seed(z))
        waveform = self.output_activation(waveform[:, :, : self._output_length])
        return waveform.squeeze(1) if self._out_channels == 1 else waveform

    @property
    def modality_name(self) -> str:
        """Name of the modality this decoder reconstructs."""
        return self._modality_name
