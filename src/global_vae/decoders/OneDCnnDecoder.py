"""1D CNN decoder (spec §6, §12).
"""

from collections.abc import Callable, Sequence

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.utils.stage_config import broadcastPerStage
from global_vae.utils.builders import build1DUpSampleStage
from global_vae.utils.conv_math import (
    computeConvTranspose1dOutputLength,
    computeUpsampleThenConv1dOutputLength,
    solveConvTranspose1dOutputPadding,
)

def _computeLengthFromResolved(
    seed_length: int,
    kernel_sizes: tuple[int, ...],
    strides: tuple[int, ...],
    paddings: tuple[int, ...],
    output_paddings: tuple[int, ...],
    dilations: tuple[int, ...],
    upsample_modes: tuple[str, ...],
) -> int:
    """Chain the per-transition length formula across an already-resolved stack.

    Args:
        seed_length: Length before any transition is applied.
        kernel_sizes: Per-transition kernel sizes.
        strides: Per-transition strides.
        paddings: Per-transition paddings.
        output_paddings: Per-transition `ConvTranspose1d` output
            paddings. Ignored (but must still be a same-length tuple)
            if `upsample_mode` is `"interpolate_conv"`.
        dilations: Per-transition dilations.
        upsample_modes: `"conv_transpose"` or `"interpolate_conv"`.

    Returns:
        The length after every transition.

    Raises:
        ValueError: If `upsample_mode` is not recognized.
    """
    length = seed_length
    for stage in range(len(kernel_sizes)):
        if upsample_modes[stage] == "conv_transpose":
            length = computeConvTranspose1dOutputLength(
                length,
                kernel_sizes[stage],
                strides[stage],
                paddings[stage],
                output_paddings[stage],
                dilations[stage],
            )
        elif upsample_modes[stage] == "interpolate_conv":
            length = computeUpsampleThenConv1dOutputLength(
                length,
                strides[stage],
                kernel_sizes[stage],
                paddings[stage],
                dilations[stage],
            )
        else:
            raise ValueError(
                f"Unknown upsample_mode '{upsample_modes[stage]}'. Expected 'conv_transpose' or "
                f"'interpolate_conv'."
            )
    return length

@registerDecoder("1d_cnn_decoder_v1")
class OneDCnnDecoder(AbstractDecoder):
    """1D convolutional decoder reconstructing a fixed-length series.
    """
    def __init__(
        self,
        latent_dim: int,
        output_length: int,
        out_channels: int = 1,
        hidden_channels: tuple[int, ...] = (128, 64, 32),
        seed_length: int = 8,
        kernel_sizes: int | Sequence[int] = 4,
        strides: int | Sequence[int] = 2,
        paddings: int | Sequence[int] = 1,
        output_paddings: int | Sequence[int] | None = None,
        dilations: int | Sequence[int] = 1,
        upsample_modes: str | Sequence[str] = "interpolate_conv",
        activations: (
            Callable[[], nn.Module] | Sequence[Callable[[], nn.Module] | None] | None
        ) = nn.ReLU,
        normalizations: (
            Callable[[int], nn.Module] | Sequence[Callable[[int], nn.Module] | None] | None
        ) = nn.BatchNorm1d,
        head_hidden_dims: tuple[int, ...] = (),
        head_activation: Callable[[], nn.Module] | None = nn.ReLU,
        modality_name: str = "vector",
    ) -> None:
        """Build the decoder.

        Args:
            latent_dim: Dimensionality of the input latent vector.
            output_length: Length of the reconstructed series. The
                chosen configuration must reach this exactly (see
                `output_paddings`); this class never resizes its way
                to it.
            out_channels: Number of output channels (`1` for a plain
                scalar series).
            hidden_channels: Channel width of the projected seed
                (`hidden_channels[0]`) and of every subsequent
                transition. There are `len(hidden_channels)` upsampling
                transitions in total: from the seed, through each
                remaining value in `hidden_channels`, to `out_channels`.
            seed_length: Spatial length of the projected seed,
                upsampled by the transition stack.
            kernel_sizes: Upsampling kernel size, per transition or
                shared.
            strides: Upsampling factor, per transition or shared.
            paddings: Convolution padding, per transition or shared.
            output_paddings: `ConvTranspose1d`'s `output_padding`, per
                transition or shared. Ignored if `upsample_mode` is
                `"interpolate_conv"`. Defaults to `None`: if
                `upsample_mode` is `"conv_transpose"`, every transition
                except the last gets `output_padding=0`, and the last
                transition's `output_padding` is solved automatically
                so the stack reaches `output_length` exactly (raising
                `ValueError` if no valid value, i.e. one satisfying
                `0 <= output_padding < max(stride, dilation)` for that
                transition, would work; in that case, adjust
                `seed_length`, `kernel_sizes`, `strides`, `paddings`,
                or `dilations` instead). If `upsample_mode` is
                `"interpolate_conv"`, `None` defaults to `0` for every
                transition, since there is no equivalent lever to
                auto-solve there.
            dilations: Convolution dilation, per transition or shared.
            upsample_modes: `"conv_transpose"` (a single learned
                `ConvTranspose1d` per transition) or `"interpolate_conv"`
                (nearest-neighbor upsampling followed by a stride-1
                `Conv1d`), which avoids the checkerboard artifacts
                transposed convolutions are prone to, at the cost of a
                fixed (non-learned) upsampling step and no
                `output_padding`-style auto-solving. Note: this class's
                default `kernel_sizes=4, strides=2, paddings=1` reach an
                exact length doubling per transition under
                `"conv_transpose"` (with the last transition's
                `output_padding` auto-solved to `0`); reaching the same
                exact doubling under `"interpolate_conv"` instead needs
                `kernel_sizes=3, paddings=1` (the "same"-padded,
                stride-1 conv that leaves a nearest-neighbor-doubled
                length unchanged). Construction always verifies the
                actual result, so picking the wrong pairing here fails
                immediately with a clear error instead of silently
                producing the wrong length.
            activations: Zero-argument factory returning a fresh
                activation module, per transition or shared, applied
                after every transition except the last (which must be
                able to produce unconstrained reconstruction values).
                Pass `None` to disable activation for a transition.
            normalizations: One-argument factory taking a channel count
                and returning a fresh normalization module, per
                transition or shared, applied the same places as
                `activations`.
            head_hidden_dims: Hidden layer sizes for an optional small MLP
                inserted between the latent vector and the seed projection.
                Empty tuple (default) keeps today's behavior: a single
                linear layer straight from `z` to the seed.
            head_activation: Optional activation layer to use for the head
            modality_name:

        Raises:
            ValueError: If any per-transition sequence argument does
                not have exactly `len(hidden_channels)` values, if
                `upsample_mode` is not recognized, if no valid
                `output_padding` for the last transition would reach
                `output_length` (`output_paddings=None`,
                `upsample_mode="conv_transpose"` case), or if the
                resulting configuration's computed length does not
                equal `output_length` (every other case).
        """
        super().__init__()
        self._output_length = output_length
        self._modality_name = modality_name
        num_transitions = len(hidden_channels)

        kernel_sizes_ = broadcastPerStage(kernel_sizes, num_transitions, "kernel_sizes")
        strides_ = broadcastPerStage(strides, num_transitions, "strides")
        paddings_ = broadcastPerStage(paddings, num_transitions, "paddings")
        dilations_ = broadcastPerStage(dilations, num_transitions, "dilations")
        activations_ = broadcastPerStage(activations, num_transitions, "activations")
        normalizations_ = broadcastPerStage(normalizations, num_transitions, "normalizations")
        upsample_modes_ = broadcastPerStage(upsample_modes, num_transitions, "upsample_modes")

        if output_paddings is not None:
            output_paddings_ = broadcastPerStage(
                output_paddings, num_transitions, "output_paddings"
            )
        elif upsample_modes_[-1] == "conv_transpose":
            length_before_last = _computeLengthFromResolved(
                seed_length,
                kernel_sizes_[:-1],
                strides_[:-1],
                paddings_[:-1],
                tuple(0 for _ in range(num_transitions - 1)),
                dilations_[:-1],
                upsample_modes_[:-1],
            )
            solved_output_padding = solveConvTranspose1dOutputPadding(
                length_before_last,
                output_length,
                kernel_sizes_[-1],
                strides_[-1],
                paddings_[-1],
                dilations_[-1],
            )
            max_valid = max(strides_[-1], dilations_[-1])
            if not (0 <= solved_output_padding < max_valid):
                raise ValueError(
                    f"OneDCnnDecoder cannot reach output_length={output_length} by solving "
                    f"only the last transition's output_padding (would need "
                    f"output_padding={solved_output_padding}, but it must satisfy "
                    f"0 <= output_padding < {max_valid} given the last transition's "
                    f"stride={strides_[-1]}/dilation={dilations_[-1]}). Adjust seed_length, "
                    f"kernel_sizes, strides, paddings, or dilations, or pass "
                    f"output_paddings explicitly. Use OneDCnnDecoder.computeOutputLength(...) "
                    f"to check a configuration before constructing."
                )
            output_paddings_ = (*(0 for _ in range(num_transitions - 1)), solved_output_padding)
        else:
            output_paddings_ = tuple(0 for _ in range(num_transitions))

        computed_length = _computeLengthFromResolved(
            seed_length,
            kernel_sizes_,
            strides_,
            paddings_,
            output_paddings_,
            dilations_,
            upsample_modes_,
        )
        if computed_length != output_length:
            raise ValueError(
                f"OneDCnnDecoder's configuration produces length {computed_length}, not the "
                f"requested output_length={output_length}. This class verifies the exact "
                f"shape instead of resizing a mismatch away. Adjust seed_length, "
                f"kernel_sizes, strides, paddings, output_paddings, or dilations, or call "
                f"OneDCnnDecoder.computeOutputLength(...) with the same arguments to explore "
                f"configurations before constructing."
            )

        self._seed_channels = hidden_channels[0]
        self._seed_length = seed_length

        head_layers: list[nn.Module] = []
        head_in = latent_dim
        for hidden_dim in head_hidden_dims:
            head_layers.append(nn.Linear(head_in, hidden_dim))
            if head_activation is not None:
                head_layers.append(head_activation())
            head_in = hidden_dim
        self.head: nn.Module = nn.Sequential(*head_layers) if head_layers else nn.Identity()

        self.project = nn.Linear(head_in, hidden_channels[0] * seed_length)

        widths = (*hidden_channels, out_channels)
        layers: list[nn.Module] = []
        for stage in range(num_transitions):
            is_last = stage == num_transitions - 1
            layers.append(
                build1DUpSampleStage(
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
    def computeOutputLength(
        seed_length: int,
        hidden_channels: tuple[int, ...],
        kernel_sizes: int | Sequence[int] = 4,
        strides: int | Sequence[int] = 2,
        paddings: int | Sequence[int] = 1,
        output_paddings: int | Sequence[int] = 0,
        dilations: int | Sequence[int] = 1,
        upsample_modes: str = "interpolate_conv",
    ) -> int:
        """Compute the output length a given configuration produces.

        Lets a caller check, before constructing a full
        `OneDCnnDecoder` (or after, to understand why construction
        raised), exactly what length a chosen architecture reaches
        from `seed_length` (spec §12: verify a configuration is valid
        before committing to it, rather than discovering a mismatch
        only once forward-time resizing has already blurred it away).

        Unlike `__init__`, this always takes an explicit
        `output_paddings` (defaulting to `0` for every transition): it
        does not perform `__init__`'s `output_paddings=None`
        auto-solving, since the point of calling this method is to see
        what a fully-specified configuration produces.

        Args:
            seed_length: Spatial length of the projected seed.
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
            The resulting output length.

        Raises:
            ValueError: If any per-transition sequence argument does
                not have exactly `len(hidden_channels)` values, or if
                `upsample_mode` is not recognized.
        """
        num_transitions = len(hidden_channels)
        kernel_sizes_ = broadcastPerStage(kernel_sizes, num_transitions, "kernel_sizes")
        strides_ = broadcastPerStage(strides, num_transitions, "strides")
        paddings_ = broadcastPerStage(paddings, num_transitions, "paddings")
        output_paddings_ = broadcastPerStage(output_paddings, num_transitions, "output_paddings")
        dilations_ = broadcastPerStage(dilations, num_transitions, "dilations")
        upsample_modes_ = broadcastPerStage(upsample_modes, num_transitions, "upsample_modes")

        return _computeLengthFromResolved(
            seed_length,
            kernel_sizes_,
            strides_,
            paddings_,
            output_paddings_,
            dilations_,
            upsample_modes_,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of fixed-length series from latent vectors.

        Args:
            z: Latent tensor, shape `(batch, latent_dim)`.

        Returns:
            Reconstructed series. Shape `(batch, output_length)` for
            the default single-channel case (matching the encoder's
            plain-series input convention); `(batch, out_channels,
            output_length)` if `out_channels` was set above `1`.
        """
        batch_size = z.shape[0]
        seed = self.project(self.head(z)).view(batch_size, self._seed_channels, self._seed_length)
        reconstruction: torch.Tensor = self.deconv(seed)
        return reconstruction.squeeze(1) if reconstruction.shape[1] == 1 else reconstruction

    @property
    def modalityName(self) -> str:
        return self._modality_name
