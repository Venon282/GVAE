"""Unit tests for `TwoDCnnResidualDecoder` (spec §6, §7, §12).

Mirrors `test_image_decoder.py`'s coverage of `TwoDCnnDecoder` (exact,
per-axis output-shape verification instead of resizing, both upsample modes,
non-square shapes and the independent-per-axis auto-solve, gradient flow,
registration, per-transition configurability, and error paths) combined with
`test_residual_signal_decoder.py`'s coverage of `OneDCnnResidualDecoder`
(flexible `block_depths` per transition, `shortcut_kernel_sizes`, and the
fact that its bare defaults already reach an exact shape doubling under
`upsample_modes="interpolate_conv"`), since this class is exactly that
intersection.
"""

import pytest
import torch
from torch import nn

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
from global_vae.decoders.registry import getDecoderClass
from global_vae.decoders.TwoDCnnResidualDecoder import TwoDCnnResidualDecoder
from global_vae.utils.autograd import backward

# Defaults (seed_shape=(8, 8), hidden_channels=(128, 64, 32), kernel_sizes=3, strides=2,
# paddings=1, upsample_modes="interpolate_conv") exactly double each axis at every
# transition, independently on height and width: 8 -> 16 -> 32 -> 64. Unlike
# TwoDCnnDecoder, this holds for this class's *bare* defaults (no need to override
# upsample_modes to "conv_transpose" first): see the class's own docstring for why
# kernel_sizes=3 (not TwoDCnnDecoder's 4) is the self-consistent default here.
_DEFAULT_NATURAL_SHAPE = (64, 64)


def test_output_shape_matches_the_natural_default_shape() -> None:
    decoder = TwoDCnnResidualDecoder(latent_dim=16, output_shape=_DEFAULT_NATURAL_SHAPE)
    reconstruction = decoder(torch.randn(4, 16))
    assert reconstruction.shape == (4, *_DEFAULT_NATURAL_SHAPE)


def test_compute_output_shape_matches_default_construction() -> None:
    """The whole point: a caller can verify a config before ever constructing the class."""
    computed = TwoDCnnResidualDecoder.computeOutputShape(
        seed_shape=(8, 8), hidden_channels=(128, 64, 32)
    )
    assert computed == _DEFAULT_NATURAL_SHAPE


def test_unreachable_output_shape_raises_instead_of_resizing() -> None:
    with pytest.raises(ValueError, match="output_shape"):
        TwoDCnnResidualDecoder(latent_dim=16, output_shape=(200, 200))


def test_output_shape_smaller_than_natural_also_raises() -> None:
    with pytest.raises(ValueError, match="output_shape"):
        TwoDCnnResidualDecoder(
            latent_dim=16, output_shape=(_DEFAULT_NATURAL_SHAPE[0] - 10, _DEFAULT_NATURAL_SHAPE[1])
        )


def test_registered_under_2d_cnn_resnet_decoder_v1() -> None:
    assert getDecoderClass("2d_cnn_resnet_decoder_v1") is TwoDCnnResidualDecoder


class TestFlexibleBlockDepths:
    def test_three_then_two_layers(self) -> None:
        """Spec's own example, decoder side: a first transition with 3 internal layers, a
        second with 2."""
        output_shape = TwoDCnnResidualDecoder.computeOutputShape(
            seed_shape=(16, 16), hidden_channels=(32, 16), kernel_sizes=3, strides=2, paddings=1
        )
        decoder = TwoDCnnResidualDecoder(
            latent_dim=8,
            output_shape=output_shape,
            hidden_channels=(32, 16),
            block_depths=(3, 2),
            seed_shape=(16, 16),
        )
        reconstruction = decoder(torch.randn(2, 8))
        assert reconstruction.shape == (2, *output_shape)

    def test_shared_block_depth_applies_to_every_transition(self) -> None:
        output_shape = TwoDCnnResidualDecoder.computeOutputShape(
            seed_shape=(16, 16), hidden_channels=(32, 16), kernel_sizes=3, strides=2, paddings=1
        )
        decoder = TwoDCnnResidualDecoder(
            latent_dim=8,
            output_shape=output_shape,
            hidden_channels=(32, 16),
            block_depths=3,
            seed_shape=(16, 16),
        )
        reconstruction = decoder(torch.randn(2, 8))
        assert reconstruction.shape == (2, *output_shape)

    def test_block_depth_one_is_supported(self) -> None:
        output_shape = TwoDCnnResidualDecoder.computeOutputShape(
            seed_shape=(16, 16), hidden_channels=(32, 16), kernel_sizes=3, strides=2, paddings=1
        )
        decoder = TwoDCnnResidualDecoder(
            latent_dim=8,
            output_shape=output_shape,
            hidden_channels=(32, 16),
            block_depths=1,
            seed_shape=(16, 16),
        )
        reconstruction = decoder(torch.randn(2, 8))
        assert reconstruction.shape == (2, *output_shape)

    def test_mismatched_block_depths_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="block_depths"):
            TwoDCnnResidualDecoder(
                latent_dim=8,
                output_shape=(64, 64),
                hidden_channels=(32, 16, 8),
                block_depths=(2, 3),
            )

    def test_even_kernel_size_with_depth_greater_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="odd kernel_size"):
            TwoDCnnResidualDecoder(
                latent_dim=8, output_shape=(64, 64), hidden_channels=(32,), kernel_sizes=4, block_depths=2
            )


def test_auto_solves_output_padding_for_a_small_square_gap() -> None:
    """A gap of 1 on both axes, within the last transition's stride, is closeable
    without any blur."""
    natural = TwoDCnnResidualDecoder.computeOutputShape(
        seed_shape=(4, 4), hidden_channels=(16, 32), upsample_modes="conv_transpose"
    )
    target = (natural[0] + 1, natural[1] + 1)
    decoder = TwoDCnnResidualDecoder(
        latent_dim=8,
        output_shape=target,
        hidden_channels=(16, 32),
        seed_shape=(4, 4),
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, *target)


def test_auto_solves_output_padding_independently_per_axis_for_a_non_square_gap() -> None:
    """A gap on only ONE axis must not perturb the other."""
    natural = TwoDCnnResidualDecoder.computeOutputShape(
        seed_shape=(4, 6), hidden_channels=(16, 32), upsample_modes="conv_transpose"
    )
    target = (natural[0] + 1, natural[1])  # height-only gap
    decoder = TwoDCnnResidualDecoder(
        latent_dim=8,
        output_shape=target,
        hidden_channels=(16, 32),
        seed_shape=(4, 6),
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, *target)


def test_explicit_output_paddings_bypasses_auto_solve() -> None:
    """Uses this class's own default `kernel_sizes=3` (odd, block_depths-compatible),
    unlike `TwoDCnnDecoder`'s own `kernel_sizes=4` default, which this residual class's
    internal, length-preserving layers cannot accept at `block_depths=2` (the default)."""
    decoder = TwoDCnnResidualDecoder(
        latent_dim=16,
        output_shape=_DEFAULT_NATURAL_SHAPE,
        output_paddings=1,
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 16))
    assert reconstruction.shape == (2, *_DEFAULT_NATURAL_SHAPE)


def test_explicit_wrong_output_paddings_still_gets_verified() -> None:
    """Manual control does not bypass verification, only auto-solving."""
    with pytest.raises(ValueError, match="output_shape"):
        TwoDCnnResidualDecoder(
            latent_dim=16,
            output_shape=_DEFAULT_NATURAL_SHAPE,
            output_paddings=0,
            upsample_modes="conv_transpose",
        )


def test_interpolate_conv_mode_is_the_default_and_reaches_the_natural_shape() -> None:
    decoder = TwoDCnnResidualDecoder(latent_dim=8, output_shape=(64, 64))
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, 64, 64)


def test_conv_transpose_mode_with_matching_parameters() -> None:
    decoder = TwoDCnnResidualDecoder(
        latent_dim=8,
        output_shape=(64, 64),
        upsample_modes="conv_transpose",
        kernel_sizes=3,
        strides=2,
        paddings=1,
        output_paddings=1,
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, 64, 64)


def test_unknown_upsample_mode_raises() -> None:
    with pytest.raises(ValueError, match="upsample_mode"):
        TwoDCnnResidualDecoder(latent_dim=8, output_shape=(64, 64), upsample_modes="magic")


def test_multi_channel_output_keeps_channel_dimension() -> None:
    decoder = TwoDCnnResidualDecoder(
        latent_dim=16, output_shape=_DEFAULT_NATURAL_SHAPE, out_channels=3
    )
    reconstruction = decoder(torch.randn(2, 16))
    assert reconstruction.shape == (2, 3, *_DEFAULT_NATURAL_SHAPE)


def test_modality_name_defaults_to_image_but_is_configurable() -> None:
    default_decoder = TwoDCnnResidualDecoder(latent_dim=16, output_shape=_DEFAULT_NATURAL_SHAPE)
    assert default_decoder.modality_name == "image"

    named_decoder = TwoDCnnResidualDecoder(
        latent_dim=16, output_shape=_DEFAULT_NATURAL_SHAPE, modality_name="xray"
    )
    assert named_decoder.modality_name == "xray"


class TestNonSquareShapes:
    """Rectangular (non-square) images are not a special case: every axis is solved
    and verified independently throughout."""

    def test_non_square_seed_gives_a_non_square_natural_output(self) -> None:
        natural = TwoDCnnResidualDecoder.computeOutputShape(
            seed_shape=(6, 8), hidden_channels=(128, 64, 32)
        )
        assert natural == (48, 64)

    def test_decoder_actually_reconstructs_the_rectangular_shape(self) -> None:
        decoder = TwoDCnnResidualDecoder(latent_dim=8, output_shape=(48, 64), seed_shape=(6, 8))
        reconstruction = decoder(torch.randn(3, 8))
        assert reconstruction.shape == (3, 48, 64)


def test_per_transition_activation_and_normalization_can_differ() -> None:
    output_shape = TwoDCnnResidualDecoder.computeOutputShape(
        seed_shape=(8, 8), hidden_channels=(16, 32), kernel_sizes=3, strides=2, paddings=1
    )
    decoder = TwoDCnnResidualDecoder(
        latent_dim=8,
        output_shape=output_shape,
        hidden_channels=(16, 32),
        seed_shape=(8, 8),
        activations=(nn.GELU, nn.ReLU),
        normalizations=(None, nn.BatchNorm2d),
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, *output_shape)
    assert any(isinstance(module, nn.GELU) for module in decoder.modules())


def test_the_very_last_transition_produces_unconstrained_values() -> None:
    """Matches `TwoDCnnDecoder`'s own convention: negative values must survive."""
    decoder = TwoDCnnResidualDecoder(latent_dim=16, output_shape=_DEFAULT_NATURAL_SHAPE)
    reconstruction = decoder(torch.randn(8, 16))
    assert bool((reconstruction < 0).any())


def test_internal_layers_of_the_last_transition_still_get_normalization() -> None:
    """Only the very last layer of the very last transition skips norm/activation; a
    deeper last-stage block must still benefit from it on its earlier internal layers."""
    output_shape = TwoDCnnResidualDecoder.computeOutputShape(
        seed_shape=(16, 16), hidden_channels=(32, 16), kernel_sizes=3, strides=2, paddings=1
    )
    decoder = TwoDCnnResidualDecoder(
        latent_dim=8,
        output_shape=output_shape,
        hidden_channels=(32, 16),
        block_depths=(2, 3),
        seed_shape=(16, 16),
    )
    last_block = decoder.deconv[-1]
    assert any(isinstance(module, nn.BatchNorm2d) for module in last_block.main.modules())


def test_shortcut_kernel_sizes_are_configurable_per_transition() -> None:
    output_shape = TwoDCnnResidualDecoder.computeOutputShape(
        seed_shape=(8, 8), hidden_channels=(16, 32), kernel_sizes=3, strides=2, paddings=1
    )
    decoder = TwoDCnnResidualDecoder(
        latent_dim=8,
        output_shape=output_shape,
        hidden_channels=(16, 32),
        seed_shape=(8, 8),
        shortcut_kernel_sizes=(1, 3),
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, *output_shape)


def test_gradients_reach_every_parameter() -> None:
    decoder = TwoDCnnResidualDecoder(latent_dim=8, output_shape=_DEFAULT_NATURAL_SHAPE)
    reconstruction = decoder(torch.randn(3, 8))
    backward(reconstruction.sum())
    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"parameter '{name}' got no gradient"


def test_mismatched_per_transition_list_length_raises() -> None:
    with pytest.raises(ValueError, match="kernel_sizes"):
        TwoDCnnResidualDecoder(
            latent_dim=8,
            output_shape=(64, 64),
            hidden_channels=(16, 32, 64),
            kernel_sizes=[3, 5],
        )


def test_per_transition_differing_numeric_values_require_a_list_not_a_tuple() -> None:
    """Same 1D->2D API convention as the encoder: a bare tuple is always a *shared*
    shape (never per-transition)."""
    with pytest.raises(ValueError, match="2-dimensional"):
        TwoDCnnResidualDecoder(
            latent_dim=8,
            output_shape=(64, 64),
            hidden_channels=(16, 32, 64),
            strides=(2, 2, 2),
        )


class TestEncoderDecoderRoundTrip:
    """End-to-end sanity check, mirroring how `GlobalVae` wires an encoder and a
    decoder of the same modality together (`image -> z -> image`)."""

    def test_square_rgb_round_trip(self) -> None:
        from global_vae.encoders.TwoDCnnResidualEncoder import TwoDCnnResidualEncoder

        encoder = TwoDCnnResidualEncoder(
            latent_dim=12, in_channels=3, hidden_channels=(16, 32, 64)
        )
        decoder = TwoDCnnResidualDecoder(
            latent_dim=12,
            output_shape=(64, 64),
            out_channels=3,
            hidden_channels=(64, 32, 16),
            seed_shape=(8, 8),
        )
        batch = torch.randn(5, 3, 64, 64)
        mu, _ = encoder(batch)
        reconstruction = decoder(mu)
        assert reconstruction.shape == batch.shape

    def test_rectangular_grayscale_round_trip(self) -> None:
        from global_vae.encoders.TwoDCnnResidualEncoder import TwoDCnnResidualEncoder

        encoder = TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32, 64), strides=1)
        decoder = TwoDCnnResidualDecoder(
            latent_dim=8,
            output_shape=(48, 80),
            hidden_channels=(64, 32, 16),
            seed_shape=(6, 10),
        )
        batch = torch.randn(3, 48, 80)
        mu, _ = encoder(batch)
        reconstruction = decoder(mu)
        assert reconstruction.shape == batch.shape
