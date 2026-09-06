"""Unit tests for `OneDCnnResidualDecoder` (spec §6, §7, §12).

Mirrors `test_signal_decoder.py`'s coverage of `OneDCnnDecoder` (exact
output-length verification instead of resizing, both upsample modes,
gradient flow, registration, per-transition configurability, and error
paths), plus the tests specific to this class's own additions: flexible
`block_depths` per transition, `shortcut_kernel_sizes`, and the fact that
its bare defaults (unlike `OneDCnnDecoder`'s own) already reach an exact
length doubling under its default `upsample_modes="interpolate_conv"`.
"""

import pytest
import torch
from torch import nn

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
from global_vae.decoders.OneDCnnResidualDecoder import OneDCnnResidualDecoder
from global_vae.decoders.registry import getDecoderClass
from global_vae.utils.autograd import backward

# Defaults (seed_length=8, hidden_channels=(128, 64, 32), kernel_sizes=3, strides=2,
# paddings=1, upsample_modes="interpolate_conv") exactly double the length at every
# transition: 8 -> 16 -> 32 -> 64. Unlike OneDCnnDecoder, this holds for this class's
# *bare* defaults (no need to override upsample_modes to "conv_transpose" first): see
# the class's own docstring for why kernel_sizes=3 (not OneDCnnDecoder's 4) is the
# self-consistent default here.
_DEFAULT_NATURAL_LENGTH = 64


def test_output_shape_matches_the_natural_default_length() -> None:
    decoder = OneDCnnResidualDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH)
    reconstruction = decoder(torch.randn(4, 16))
    assert reconstruction.shape == (4, _DEFAULT_NATURAL_LENGTH)


def test_compute_output_length_matches_default_construction() -> None:
    """The whole point: a caller can verify a config before ever constructing the class."""
    computed = OneDCnnResidualDecoder.computeOutputLength(seed_length=8, hidden_channels=(128, 64, 32))
    assert computed == _DEFAULT_NATURAL_LENGTH


def test_unreachable_output_length_raises_instead_of_resizing() -> None:
    with pytest.raises(ValueError, match="OneDCnnResidualDecoder's configuration produces length"):
        OneDCnnResidualDecoder(latent_dim=16, output_length=200)


def test_output_length_smaller_than_natural_also_raises() -> None:
    with pytest.raises(ValueError, match="output_length"):
        OneDCnnResidualDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH - 10)


def test_registered_under_1d_cnn_resnet_decoder_v1() -> None:
    assert getDecoderClass("1d_cnn_resnet_decoder_v1") is OneDCnnResidualDecoder


def test_flexible_block_depths_three_then_two_layers() -> None:
    """Spec's own example, decoder side: a first transition with 3 internal layers, a
    second with 2."""
    output_length = OneDCnnResidualDecoder.computeOutputLength(
        seed_length=16, hidden_channels=(32, 16), kernel_sizes=3, strides=2, paddings=1
    )
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=output_length,
        hidden_channels=(32, 16),
        block_depths=(3, 2),
        seed_length=16,
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, output_length)


def test_shared_block_depth_applies_to_every_transition() -> None:
    output_length = OneDCnnResidualDecoder.computeOutputLength(
        seed_length=16, hidden_channels=(32, 16), kernel_sizes=3, strides=2, paddings=1
    )
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=output_length,
        hidden_channels=(32, 16),
        block_depths=3,
        seed_length=16,
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, output_length)


def test_block_depth_one_is_supported() -> None:
    output_length = OneDCnnResidualDecoder.computeOutputLength(
        seed_length=16, hidden_channels=(32, 16), kernel_sizes=3, strides=2, paddings=1
    )
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=output_length,
        hidden_channels=(32, 16),
        block_depths=1,
        seed_length=16,
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, output_length)


def test_mismatched_block_depths_sequence_raises() -> None:
    with pytest.raises(ValueError, match="block_depths"):
        OneDCnnResidualDecoder(
            latent_dim=8, output_length=64, hidden_channels=(32, 16, 8), block_depths=(2, 3)
        )


def test_auto_solves_output_padding_for_a_small_reachable_gap() -> None:
    """A gap of 1, within the last transition's stride, is closeable without any blur."""
    natural = OneDCnnResidualDecoder.computeOutputLength(
        seed_length=4,
        hidden_channels=(16, 32),
        kernel_sizes=3,
        strides=2,
        paddings=1,
        upsample_modes="conv_transpose",
    )
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=natural + 1,
        hidden_channels=(16, 32),
        seed_length=4,
        kernel_sizes=3,
        strides=2,
        paddings=1,
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, natural + 1)


def test_explicit_output_paddings_bypasses_auto_solve() -> None:
    decoder = OneDCnnResidualDecoder(
        latent_dim=16,
        output_length=_DEFAULT_NATURAL_LENGTH,
        output_paddings=1,
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(2, 16))
    assert reconstruction.shape == (2, _DEFAULT_NATURAL_LENGTH)


def test_interpolate_conv_mode_is_the_default_and_reaches_the_natural_length() -> None:
    decoder = OneDCnnResidualDecoder(latent_dim=8, output_length=64)
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, 64)


def test_conv_transpose_mode_with_matching_parameters() -> None:
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=64,
        upsample_modes="conv_transpose",
        kernel_sizes=3,
        strides=2,
        paddings=1,
        output_paddings=1,
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, 64)


def test_unknown_upsample_mode_raises() -> None:
    with pytest.raises(ValueError, match="upsample_mode"):
        OneDCnnResidualDecoder(latent_dim=8, output_length=64, upsample_modes="magic")


def test_multi_channel_output_keeps_channel_dimension() -> None:
    decoder = OneDCnnResidualDecoder(
        latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH, out_channels=3
    )
    reconstruction = decoder(torch.randn(2, 16))
    assert reconstruction.shape == (2, 3, _DEFAULT_NATURAL_LENGTH)


def test_modality_name_defaults_to_vector_but_is_configurable() -> None:
    default_decoder = OneDCnnResidualDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH)
    assert default_decoder.modality_name == "vector"

    audio_decoder = OneDCnnResidualDecoder(
        latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH, modality_name="audio"
    )
    assert audio_decoder.modality_name == "audio"


def test_per_transition_activation_and_normalization_can_differ() -> None:
    output_length = OneDCnnResidualDecoder.computeOutputLength(
        seed_length=8, hidden_channels=(16, 32), kernel_sizes=3, strides=2, paddings=1
    )
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=output_length,
        hidden_channels=(16, 32),
        seed_length=8,
        activations=(nn.GELU, nn.ReLU),
        normalizations=(None, nn.BatchNorm1d),
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, output_length)
    assert any(isinstance(module, nn.GELU) for module in decoder.modules())


def test_the_very_last_transition_produces_unconstrained_values() -> None:
    """Matches OneDCnnDecoder's own convention: negative values must survive."""
    decoder = OneDCnnResidualDecoder(latent_dim=16, output_length=_DEFAULT_NATURAL_LENGTH)
    reconstruction = decoder(torch.randn(8, 16))
    assert bool((reconstruction < 0).any())


def test_internal_layers_of_the_last_transition_still_get_normalization() -> None:
    """Only the very last layer of the very last transition skips norm/activation; a
    deeper last-stage block must still benefit from it on its earlier internal layers."""
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=64,
        hidden_channels=(32, 16),
        block_depths=(2, 3),
        seed_length=16,
    )
    last_block = decoder.deconv[-1]
    assert any(isinstance(module, nn.BatchNorm1d) for module in last_block.main.modules())


def test_shortcut_kernel_sizes_are_configurable_per_transition() -> None:
    output_length = OneDCnnResidualDecoder.computeOutputLength(
        seed_length=8, hidden_channels=(16, 32), kernel_sizes=3, strides=2, paddings=1
    )
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=output_length,
        hidden_channels=(16, 32),
        seed_length=8,
        shortcut_kernel_sizes=(1, 3),
    )
    reconstruction = decoder(torch.randn(2, 8))
    assert reconstruction.shape == (2, output_length)


def test_gradients_reach_every_parameter() -> None:
    decoder = OneDCnnResidualDecoder(latent_dim=8, output_length=_DEFAULT_NATURAL_LENGTH)
    reconstruction = decoder(torch.randn(3, 8))
    backward(reconstruction.sum())
    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"parameter '{name}' got no gradient"


def test_gradients_reach_every_parameter_with_flexible_depths_and_conv_transpose() -> None:
    natural = OneDCnnResidualDecoder.computeOutputLength(
        seed_length=4,
        hidden_channels=(16, 32),
        kernel_sizes=3,
        strides=2,
        paddings=1,
        upsample_modes="conv_transpose",
    )
    decoder = OneDCnnResidualDecoder(
        latent_dim=8,
        output_length=natural,
        hidden_channels=(16, 32),
        block_depths=(3, 2),
        seed_length=4,
        kernel_sizes=3,
        strides=2,
        paddings=1,
        upsample_modes="conv_transpose",
    )
    reconstruction = decoder(torch.randn(3, 8))
    backward(reconstruction.sum())
    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"parameter '{name}' got no gradient"


def test_even_kernel_size_with_depth_greater_than_one_raises() -> None:
    with pytest.raises(ValueError, match="odd kernel_size"):
        OneDCnnResidualDecoder(
            latent_dim=8, output_length=64, hidden_channels=(32,), kernel_sizes=4, block_depths=2
        )


def test_mismatched_per_transition_sequence_raises() -> None:
    with pytest.raises(ValueError, match="kernel_sizes"):
        OneDCnnResidualDecoder(
            latent_dim=8,
            output_length=64,
            hidden_channels=(16, 32, 64),
            kernel_sizes=(3, 5),
        )
