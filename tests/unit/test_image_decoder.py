"""Unit tests for `TwoDCnnDecoder` (spec §6, §12).

The central behavior under test, mirrored from `test_signal_decoder.py`: this
decoder verifies the *exact* output shape its configuration produces at
construction time, on both axes independently, instead of forcing a mismatch to
fit with a blurring final resize.
"""

import pytest
import torch
from torch import nn

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
from global_vae.decoders.registry import getDecoderClass
from global_vae.decoders.TwoDCnnDecoder import TwoDCnnDecoder
from global_vae.utils.autograd import backward

# Defaults (seed_shape=(8, 8), hidden_channels=(128, 64, 32), kernel_sizes=4,
# strides=2, paddings=1) exactly double each axis at every transition under
# conv_transpose mode: 8 -> 16 -> 32 -> 64, independently on height and width.
_DEFAULT_NATURAL_SHAPE = (64, 64)


def test_output_shape_matches_the_natural_default_shape() -> None:
    decoder = TwoDCnnDecoder(
        latent_dim=16, output_shape=_DEFAULT_NATURAL_SHAPE, upsample_modes="conv_transpose"
    )
    reconstruction = decoder(torch.randn(4, 16))
    assert reconstruction.shape == (4, *_DEFAULT_NATURAL_SHAPE)


def test_compute_output_shape_matches_default_construction() -> None:
    """The whole point: a caller can verify a config before ever constructing the class."""
    computed = TwoDCnnDecoder.computeOutputShape(
        seed_shape=(8, 8), hidden_channels=(128, 64, 32), upsample_modes="conv_transpose"
    )
    assert computed == _DEFAULT_NATURAL_SHAPE


def test_unreachable_output_shape_raises_instead_of_resizing() -> None:
    """An old version of this class would have silently blurred its way to shape (200, 200)."""
    with pytest.raises(ValueError, match="cannot reach output_shape"):
        TwoDCnnDecoder(latent_dim=16, output_shape=(200, 200), upsample_modes="conv_transpose")


def test_output_shape_smaller_than_natural_also_raises() -> None:
    with pytest.raises(ValueError, match="output_shape"):
        TwoDCnnDecoder(latent_dim=16, output_shape=(54, 54), upsample_modes="conv_transpose")


def test_auto_solves_output_padding_for_a_small_square_gap() -> None:
    """A gap of 1 on both axes, within the last transition's stride, is closeable
    without any blur."""
    natural = TwoDCnnDecoder.computeOutputShape(
        seed_shape=(4, 4), hidden_channels=(16, 32), upsample_modes="conv_transpose"
    )
    target = (natural[0] + 1, natural[1] + 1)
    decoder = TwoDCnnDecoder(
        latent_dim=8,
        output_shape=target,
        hidden_channels=(16, 32),
        seed_shape=(4, 4),
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, *target)


def test_auto_solves_output_padding_independently_per_axis_for_a_non_square_gap() -> None:
    """The key new 2D behavior: a gap on only ONE axis must not perturb the other."""
    natural = TwoDCnnDecoder.computeOutputShape(
        seed_shape=(4, 6), hidden_channels=(16, 32), upsample_modes="conv_transpose"
    )
    target = (natural[0] + 1, natural[1])  # height-only gap
    decoder = TwoDCnnDecoder(
        latent_dim=8,
        output_shape=target,
        hidden_channels=(16, 32),
        seed_shape=(4, 6),
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, *target)


def test_explicit_output_paddings_bypasses_auto_solve() -> None:
    decoder = TwoDCnnDecoder(
        latent_dim=16,
        output_shape=_DEFAULT_NATURAL_SHAPE,
        output_paddings=0,
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 16))
    assert reconstruction.shape == (2, *_DEFAULT_NATURAL_SHAPE)


def test_explicit_wrong_output_paddings_still_gets_verified() -> None:
    """Manual control does not bypass verification, only auto-solving."""
    with pytest.raises(ValueError, match="output_shape"):
        TwoDCnnDecoder(
            latent_dim=16,
            output_shape=_DEFAULT_NATURAL_SHAPE,
            output_paddings=1,
            upsample_modes="conv_transpose",
        )


def test_interpolate_conv_mode_with_matching_parameters() -> None:
    """kernel_size=3/padding=1 is interpolate_conv's exact-doubling pairing, not the
    default 4/1, on both axes."""
    decoder = TwoDCnnDecoder(
        latent_dim=8,
        output_shape=(64, 64),
        upsample_modes="interpolate_conv",
        kernel_sizes=3,
        paddings=1,
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, 64, 64)


def test_interpolate_conv_mode_has_no_auto_solve_and_raises_on_mismatch() -> None:
    """The default kernel_sizes=4/paddings=1 pairing is tuned for conv_transpose, not
    this mode."""
    with pytest.raises(ValueError, match="output_shape"):
        TwoDCnnDecoder(latent_dim=8, output_shape=(64, 64), upsample_modes="interpolate_conv")


def test_unknown_upsample_mode_raises() -> None:
    with pytest.raises(ValueError, match="upsample_mode"):
        TwoDCnnDecoder(latent_dim=8, output_shape=(64, 64), upsample_modes="magic")


def test_multi_channel_output_keeps_channel_dimension() -> None:
    decoder = TwoDCnnDecoder(
        latent_dim=16,
        output_shape=_DEFAULT_NATURAL_SHAPE,
        out_channels=3,
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 16))
    assert reconstruction.shape == (2, 3, *_DEFAULT_NATURAL_SHAPE)


def test_modality_name_defaults_to_image_but_is_configurable() -> None:
    default_decoder = TwoDCnnDecoder(
        latent_dim=16, output_shape=_DEFAULT_NATURAL_SHAPE, upsample_modes="conv_transpose"
    )
    assert default_decoder.modality_name == "image"

    named_decoder = TwoDCnnDecoder(
        latent_dim=16,
        output_shape=_DEFAULT_NATURAL_SHAPE,
        modality_name="xray",
        upsample_modes="conv_transpose",
    )
    assert named_decoder.modality_name == "xray"


class TestNonSquareShapes:
    """Rectangular (non-square) images are not a special case: every axis is solved
    and verified independently throughout."""

    def test_non_square_seed_gives_a_non_square_natural_output(self) -> None:
        natural = TwoDCnnDecoder.computeOutputShape(
            seed_shape=(6, 8), hidden_channels=(128, 64, 32), upsample_modes="conv_transpose"
        )
        assert natural == (48, 64)

    def test_decoder_actually_reconstructs_the_rectangular_shape(self) -> None:
        decoder = TwoDCnnDecoder(
            latent_dim=8,
            output_shape=(48, 64),
            seed_shape=(6, 8),
            upsample_modes="conv_transpose",
        )
        reconstruction = decoder(torch.randn(3, 8))
        assert reconstruction.shape == (3, 48, 64)


def test_per_transition_activation_and_normalization_can_differ() -> None:
    decoder = TwoDCnnDecoder(
        latent_dim=8,
        output_shape=(32, 32),
        hidden_channels=(16, 32),
        seed_shape=(8, 8),
        activations=(nn.GELU, nn.ReLU),
        normalizations=(None, nn.BatchNorm2d),
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, 32, 32)
    assert any(isinstance(module, nn.GELU) for module in decoder.modules())


def test_activation_and_normalization_can_be_disabled() -> None:
    decoder = TwoDCnnDecoder(
        latent_dim=8,
        output_shape=_DEFAULT_NATURAL_SHAPE,
        activations=None,
        normalizations=None,
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, *_DEFAULT_NATURAL_SHAPE)
    assert not any(isinstance(module, nn.BatchNorm2d) for module in decoder.modules())


def test_the_very_last_transition_produces_unconstrained_values() -> None:
    """Matches OneDCnnDecoder's own convention: negative values must survive."""
    decoder = TwoDCnnDecoder(
        latent_dim=16, output_shape=_DEFAULT_NATURAL_SHAPE, upsample_modes="conv_transpose"
    )
    reconstruction = decoder(torch.randn(8, 16))
    assert bool((reconstruction < 0).any())


def test_gradients_reach_every_parameter() -> None:
    decoder = TwoDCnnDecoder(
        latent_dim=8, output_shape=_DEFAULT_NATURAL_SHAPE, upsample_modes="conv_transpose"
    )
    reconstruction = decoder(torch.randn(3, 8))
    backward(reconstruction.sum())
    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"parameter '{name}' got no gradient"


def test_registered_under_2d_cnn_decoder_v1() -> None:
    assert getDecoderClass("2d_cnn_decoder_v1") is TwoDCnnDecoder


def test_mismatched_per_transition_list_length_raises() -> None:
    with pytest.raises(ValueError, match="kernel_sizes"):
        TwoDCnnDecoder(
            latent_dim=8,
            output_shape=(64, 64),
            hidden_channels=(16, 32, 64),
            kernel_sizes=[3, 5],
            upsample_modes="conv_transpose",
        )


def test_per_transition_differing_numeric_values_require_a_list_not_a_tuple() -> None:
    """Same 1D->2D API convention as the encoder: a bare tuple is always a *shared*
    shape (never per-transition)."""
    with pytest.raises(ValueError, match="2-dimensional"):
        TwoDCnnDecoder(
            latent_dim=8,
            output_shape=(64, 64),
            hidden_channels=(16, 32, 64),
            strides=(2, 2, 2),
            upsample_modes="conv_transpose",
        )


class TestEncoderDecoderRoundTrip:
    """End-to-end sanity check, mirroring how `GlobalVae` wires an encoder and a
    decoder of the same modality together (spec §6.1's `signal -> z -> signal`
    case, here `image -> z -> image`)."""

    def test_square_rgb_round_trip(self) -> None:
        from global_vae.encoders.TwoDCnnEncoder import TwoDCnnEncoder

        encoder = TwoDCnnEncoder(latent_dim=12, in_channels=3, hidden_channels=(16, 32, 64))
        decoder = TwoDCnnDecoder(
            latent_dim=12,
            output_shape=(64, 64),
            out_channels=3,
            hidden_channels=(64, 32, 16),
            seed_shape=(8, 8),
            upsample_modes="conv_transpose",
        )
        batch = torch.randn(5, 3, 64, 64)
        mu, _ = encoder(batch)
        reconstruction = decoder(mu)
        assert reconstruction.shape == batch.shape

    def test_rectangular_grayscale_round_trip(self) -> None:
        from global_vae.encoders.TwoDCnnEncoder import TwoDCnnEncoder

        encoder = TwoDCnnEncoder(latent_dim=8, hidden_channels=(16, 32, 64), strides=1)
        decoder = TwoDCnnDecoder(
            latent_dim=8,
            output_shape=(48, 80),
            hidden_channels=(64, 32, 16),
            seed_shape=(6, 10),
            upsample_modes="conv_transpose",
        )
        batch = torch.randn(3, 48, 80)
        mu, _ = encoder(batch)
        reconstruction = decoder(mu)
        assert reconstruction.shape == batch.shape
