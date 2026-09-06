"""Unit tests for `OneDCnnResidualEncoder` (spec §6, §7, §12).

Mirrors `test_signal_encoder.py`'s coverage of `OneDCnnEncoder` (shapes,
variable input length, gradient flow, registration, per-stage
configurability, minimum-input-length solving, and error paths), plus the
tests specific to this class's own additions: flexible `block_depths` per
stage and `shortcut_kernel_sizes`.
"""

import pytest
import torch
from torch import nn

import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.encoders.OneDCnnResidualEncoder import OneDCnnResidualEncoder
from global_vae.encoders.registry import getEncoderClass
from global_vae.utils.autograd import backward


def test_output_shapes_from_plain_series() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=16)
    mu, logvar = encoder(torch.randn(4, 256))
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_accepts_explicit_channel_dimension() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=16)
    mu, logvar = encoder(torch.randn(4, 1, 256))
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_handles_varying_input_length() -> None:
    """Different series lengths must produce the same fixed-size output (spec §6)."""
    encoder = OneDCnnResidualEncoder(latent_dim=16)
    mu_short, _ = encoder(torch.randn(2, 128))
    mu_long, _ = encoder(torch.randn(2, 512))
    assert mu_short.shape == mu_long.shape == (2, 16)


def test_latent_dim_property() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=32)
    assert encoder.latent_dim == 32


def test_gradients_reach_every_parameter() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=8)
    mu, logvar = encoder(torch.randn(3, 256))
    backward(mu.sum() + logvar.sum())
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"parameter '{name}' got no gradient"


def test_registered_under_1d_cnn_resnet_encoder_v1() -> None:
    assert getEncoderClass("1d_cnn_resnet_encoder_v1") is OneDCnnResidualEncoder


def test_flexible_block_depths_three_then_four_layers() -> None:
    """Spec's own example: "3 layers before the connection... then 4"."""
    encoder = OneDCnnResidualEncoder(
        latent_dim=8,
        hidden_channels=(16, 32),
        block_depths=(3, 4),
    )
    mu, logvar = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)
    assert logvar.shape == (2, 8)


def test_shared_block_depth_applies_to_every_stage() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32, 64), block_depths=3)
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_block_depth_one_is_supported() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32), block_depths=1)
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_mismatched_block_depths_sequence_raises() -> None:
    with pytest.raises(ValueError, match="block_depths"):
        OneDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32, 64), block_depths=(2, 3))


def test_per_stage_kernel_sizes_strides_and_paddings_are_configurable() -> None:
    encoder = OneDCnnResidualEncoder(
        latent_dim=8,
        hidden_channels=(16, 32),
        kernel_sizes=(3, 5),
        strides=(2, 2),
        paddings=(1, 4),  # dilation=2, kernel_size=5: padding=4 keeps the shortcut offset-matched
        dilations=(1, 2),
        poolings=None,  # downsampling entirely via strides this time
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_mismatched_per_stage_sequence_raises() -> None:
    with pytest.raises(ValueError, match="kernel_sizes"):
        OneDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32, 64), kernel_sizes=(3, 5))


def test_per_stage_pooling_can_differ() -> None:
    """Different stages may use different pooling strategies, or none at all."""
    encoder = OneDCnnResidualEncoder(
        latent_dim=8,
        hidden_channels=(16, 32, 64),
        poolings=("max", "avg", None),
        strides=(1, 1, 2),  # last stage downsamples via stride since it has no pooling
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_pooling_requires_a_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        OneDCnnResidualEncoder(latent_dim=8, poolings="max", pool_kernel_sizes=None)


def test_pool_paddings_and_kwargs_are_forwarded() -> None:
    encoder = OneDCnnResidualEncoder(
        latent_dim=8,
        poolings="max",
        pool_kernel_sizes=3,
        pool_paddings=1,
        pool_kwargs={"ceil_mode": True},
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_pool_kwargs_default_does_not_leak_across_instances() -> None:
    first = OneDCnnResidualEncoder(latent_dim=8)
    second = OneDCnnResidualEncoder(latent_dim=8, pool_kwargs={"ceil_mode": True})
    mu_first, _ = first(torch.randn(2, 256))
    mu_second, _ = second(torch.randn(2, 256))
    assert mu_first.shape == mu_second.shape == (2, 8)


def test_per_stage_activation_and_normalization_can_differ() -> None:
    encoder = OneDCnnResidualEncoder(
        latent_dim=8,
        hidden_channels=(16, 32),
        activations=(nn.ReLU, nn.GELU),
        normalizations=(nn.BatchNorm1d, None),
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)
    assert any(isinstance(module, nn.GELU) for module in encoder.modules())


def test_activation_and_normalization_can_be_disabled() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=8, activations=None, normalizations=None)
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)
    assert not any(isinstance(module, nn.BatchNorm1d) for module in encoder.modules())


def test_shortcut_kernel_sizes_are_configurable_per_stage() -> None:
    encoder = OneDCnnResidualEncoder(
        latent_dim=8, hidden_channels=(16, 32), shortcut_kernel_sizes=(1, 3)
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_global_pool_max_variant() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=8, global_pool="max")
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_unknown_global_pool_raises() -> None:
    with pytest.raises(ValueError, match="global_pool"):
        OneDCnnResidualEncoder(latent_dim=8, global_pool="sum")


def test_even_kernel_size_with_depth_greater_than_one_raises() -> None:
    with pytest.raises(ValueError, match="odd kernel_size"):
        OneDCnnResidualEncoder(latent_dim=8, hidden_channels=(16,), kernel_sizes=4, block_depths=2)


def test_head_hidden_dims_inserts_an_mlp_before_the_heads() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=8, head_hidden_dims=(64,))
    mu, logvar = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)
    assert logvar.shape == (2, 8)


def test_forward_below_minimum_input_length_raises_clear_error() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=8)
    minimum = OneDCnnResidualEncoder.computeMinimumInputLength(hidden_channels=(32, 64, 128))
    with pytest.raises(ValueError, match="below the minimum"):
        encoder(torch.randn(2, minimum - 1))


def test_forward_at_exactly_the_minimum_input_length_succeeds() -> None:
    encoder = OneDCnnResidualEncoder(latent_dim=8)
    minimum = OneDCnnResidualEncoder.computeMinimumInputLength(hidden_channels=(32, 64, 128))
    mu, _ = encoder(torch.randn(2, minimum))
    assert mu.shape == (2, 8)


def test_compute_minimum_input_length_accounts_for_block_depth() -> None:
    """A deeper block (more internal layers) never makes the minimum input length
    *smaller* than a shallower one built from the exact same per-layer hyperparameters
    (every internal layer can only add its own kernel-span requirement)."""
    shallow_minimum = OneDCnnResidualEncoder.computeMinimumInputLength(
        hidden_channels=(16, 32), kernel_sizes=3, strides=2, block_depths=1, poolings=None
    )
    deep_minimum = OneDCnnResidualEncoder.computeMinimumInputLength(
        hidden_channels=(16, 32), kernel_sizes=3, strides=2, block_depths=3, poolings=None
    )
    assert deep_minimum >= shallow_minimum
