"""Unit tests for `OneDCnnDecoder` (spec §6, §12).

The central behavior under test: this decoder verifies the *exact*
output length its configuration produces at construction time, instead
of forcing a mismatch to fit with a blurring final resize.
"""

import pytest
import torch
from torch import nn

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
from global_vae.decoders.registry import getDecoderClass
from global_vae.decoders.OneDCnnDecoder import OneDCnnDecoder
from global_vae.utils.autograd import backward

# Defaults (seed_length=8, hidden_channels=(128, 64, 32), kernel_sizes=4,
# strides=2, paddings=1) exactly double the length at every transition
# under conv_transpose mode: 8 -> 16 -> 32 -> 64.
_DEFAULT_NATURAL_LENGTH = 64


def test_output_shape_matches_the_natural_default_length() -> None:
    decoder = OneDCnnDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH, upsample_modes="conv_transpose")
    reconstruction = decoder(torch.randn(4, 16))
    assert reconstruction.shape == (4, _DEFAULT_NATURAL_LENGTH)


def test_compute_output_length_matches_default_construction() -> None:
    """The whole point: a caller can verify a config before ever constructing the class."""
    computed = OneDCnnDecoder.computeOutputLength(seed_length=8, hidden_channels=(128, 64, 32), upsample_modes="conv_transpose")
    assert computed == _DEFAULT_NATURAL_LENGTH


def test_unreachable_output_length_raises_instead_of_resizing() -> None:
    """An old version of this class would have silently blurred its way to length 200."""
    with pytest.raises(ValueError, match="cannot reach output_length"):
        OneDCnnDecoder(latent_dim=16, output_length=200, upsample_modes="conv_transpose")


def test_output_length_smaller_than_natural_also_raises() -> None:
    with pytest.raises(ValueError, match="output_length"):
        OneDCnnDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH - 10, upsample_modes="conv_transpose")


def test_auto_solves_output_padding_for_a_small_reachable_gap() -> None:
    """A gap of 1, within the last transition's stride, is closeable without any blur."""
    natural = OneDCnnDecoder.computeOutputLength(seed_length=4, hidden_channels=(16, 32), upsample_modes="conv_transpose")
    decoder = OneDCnnDecoder(
        latent_dim=8, output_length=natural + 1, hidden_channels=(16, 32), seed_length=4, upsample_modes="conv_transpose"
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, natural + 1)


def test_explicit_output_paddings_bypasses_auto_solve() -> None:
    decoder = OneDCnnDecoder(
        latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH, output_paddings=0, upsample_modes="conv_transpose"
    )
    reconstruction = decoder(torch.randn(2, 16))
    assert reconstruction.shape == (2, _DEFAULT_NATURAL_LENGTH)


def test_explicit_wrong_output_paddings_still_gets_verified() -> None:
    """Manual control does not bypass verification, only auto-solving."""
    with pytest.raises(ValueError, match="output_length"):
        OneDCnnDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH, output_paddings=1, upsample_modes="conv_transpose")


def test_interpolate_conv_mode_with_matching_parameters() -> None:
    """kernel_size=3/padding=1 is interpolate_conv's exact-doubling pairing, not the default 4/1."""
    decoder = OneDCnnDecoder(
        latent_dim=8,
        output_length=64,
        upsample_modes="interpolate_conv",
        kernel_sizes=3,
        paddings=1,
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, 64)


def test_interpolate_conv_mode_has_no_auto_solve_and_raises_on_mismatch() -> None:
    """The default kernel_sizes=4/paddings=1 pairing is tuned for conv_transpose, not this mode."""
    with pytest.raises(ValueError, match="output_length"):
        OneDCnnDecoder(latent_dim=8, output_length=64, upsample_modes="interpolate_conv")


def test_unknown_upsample_mode_raises() -> None:
    with pytest.raises(ValueError, match="upsample_mode"):
        OneDCnnDecoder(latent_dim=8, output_length=64, upsample_modes="magic")


def test_multi_channel_output_keeps_channel_dimension() -> None:
    decoder = OneDCnnDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH, out_channels=3, upsample_modes="conv_transpose")
    reconstruction = decoder(torch.randn(2, 16))
    assert reconstruction.shape == (2, 3, _DEFAULT_NATURAL_LENGTH)


def test_modality_name_defaults_to_signal_but_is_configurable() -> None:
    default_decoder = OneDCnnDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH, upsample_modes="conv_transpose")
    assert default_decoder.modality_name == "vector"

    audio_decoder = OneDCnnDecoder(
        latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH, modality_name="audio", upsample_modes="conv_transpose"
    )
    assert audio_decoder.modality_name == "audio"


def test_per_transition_activation_and_normalization_can_differ() -> None:
    decoder = OneDCnnDecoder(
        latent_dim=8,
        output_length=32,
        hidden_channels=(16, 32),
        seed_length=8,
        activations=(nn.GELU, nn.ReLU),
        normalizations=(None, nn.BatchNorm1d),
        upsample_modes="conv_transpose"
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, 32)
    assert any(isinstance(module, nn.GELU) for module in decoder.modules())


def test_activation_and_normalization_can_be_disabled() -> None:
    decoder = OneDCnnDecoder(
        latent_dim=8,
        output_length=_DEFAULT_NATURAL_LENGTH,
        activations=None,
        normalizations=None,
        upsample_modes="conv_transpose"
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, _DEFAULT_NATURAL_LENGTH)
    assert not any(isinstance(module, nn.BatchNorm1d) for module in decoder.modules())


def test_gradients_reach_every_parameter() -> None:
    decoder = OneDCnnDecoder(latent_dim=8, output_length=_DEFAULT_NATURAL_LENGTH, upsample_modes="conv_transpose")
    reconstruction = decoder(torch.randn(3, 8))
    backward(reconstruction.sum())
    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"parameter '{name}' got no gradient"


def test_registered_under_1d_cnn_decoder_v1() -> None:
    assert getDecoderClass("1d_cnn_decoder_v1") is OneDCnnDecoder


def test_mismatched_per_transition_sequence_raises() -> None:
    with pytest.raises(ValueError, match="kernel_sizes"):
        OneDCnnDecoder(
            latent_dim=8,
            output_length=64,
            hidden_channels=(16, 32, 64),
            kernel_sizes=(3, 5),
            upsample_modes="conv_transpose"
        )
