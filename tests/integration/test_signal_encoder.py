"""Unit tests for `OneDCnnEncoder` (spec §6, §12)."""

import pytest
import torch
from torch import nn

import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.encoders.registry import getEncoderClass
from global_vae.encoders.OneDCnnEncoder import OneDCnnEncoder
from global_vae.utils.autograd import backward


def test_output_shapes_from_plain_series() -> None:
    encoder = OneDCnnEncoder(latent_dim=16)
    mu, logvar = encoder(torch.randn(4, 256))
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_accepts_explicit_channel_dimension() -> None:
    encoder = OneDCnnEncoder(latent_dim=16)
    mu, logvar = encoder(torch.randn(4, 1, 256))
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_handles_varying_input_length() -> None:
    """Different series lengths must produce the same fixed-size output (spec §6)."""
    encoder = OneDCnnEncoder(latent_dim=16)
    mu_short, _ = encoder(torch.randn(2, 128))
    mu_long, _ = encoder(torch.randn(2, 512))
    assert mu_short.shape == mu_long.shape == (2, 16)


def test_latent_dim_property() -> None:
    """`latentDim` (camelCase) is the property `AbstractEncoder` requires; a `latent_dim`
    property alone would leave the class abstract and unable to be instantiated."""
    encoder = OneDCnnEncoder(latent_dim=32)
    assert encoder.latentDim == 32


def test_gradients_reach_every_parameter() -> None:
    encoder = OneDCnnEncoder(latent_dim=8)
    mu, logvar = encoder(torch.randn(3, 256))
    backward(mu.sum() + logvar.sum())
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"parameter '{name}' got no gradient"


def test_registered_under_1d_cnn_encoder_v1() -> None:
    assert getEncoderClass("1d_cnn_encoder_v1") is OneDCnnEncoder


def test_per_stage_kernel_sizes_strides_and_paddings_are_configurable() -> None:
    encoder = OneDCnnEncoder(
        latent_dim=8,
        hidden_channels=(16, 32),
        kernel_sizes=(3, 7),
        strides=(2, 2),
        paddings=(1, 3),
        dilations=(1, 2),
        poolings=None,  # downsampling entirely via strides this time
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_mismatched_per_stage_sequence_raises() -> None:
    with pytest.raises(ValueError, match="kernel_sizes"):
        OneDCnnEncoder(latent_dim=8, hidden_channels=(16, 32, 64), kernel_sizes=(3, 5))


def test_per_stage_pooling_can_differ() -> None:
    """Different stages may use different pooling strategies, or none at all."""
    encoder = OneDCnnEncoder(
        latent_dim=8,
        hidden_channels=(16, 32, 64),
        poolings=("max", "avg", None),
        strides=(1, 1, 2),  # last stage downsamples via stride since it has no pooling
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_pooling_requires_a_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        OneDCnnEncoder(latent_dim=8, poolings="max", pool_kernel_sizes=None)


def test_pool_paddings_and_kwargs_are_forwarded() -> None:
    encoder = OneDCnnEncoder(
        latent_dim=8,
        poolings="max",
        pool_kernel_sizes=3,
        pool_paddings=1,
        pool_kwargs={"ceil_mode": True},
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_pool_kwargs_default_does_not_leak_across_instances() -> None:
    """The old `pool_kwargs: dict = {}` default was a classic mutable-default-argument bug."""
    first = OneDCnnEncoder(latent_dim=8)
    second = OneDCnnEncoder(latent_dim=8, pool_kwargs={"ceil_mode": True})
    mu_first, _ = first(torch.randn(2, 256))
    mu_second, _ = second(torch.randn(2, 256))
    assert mu_first.shape == mu_second.shape == (2, 8)


def test_per_stage_activation_and_normalization_can_differ() -> None:
    encoder = OneDCnnEncoder(
        latent_dim=8,
        hidden_channels=(16, 32),
        activations=(nn.ReLU, nn.GELU),
        normalizations=(nn.BatchNorm1d, None),
    )
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)
    assert any(isinstance(module, nn.GELU) for module in encoder.modules())


def test_activation_and_normalization_can_be_disabled() -> None:
    encoder = OneDCnnEncoder(latent_dim=8, activations=None, normalizations=None)
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)
    assert not any(isinstance(module, nn.BatchNorm1d) for module in encoder.modules())


def test_global_pool_max_variant() -> None:
    encoder = OneDCnnEncoder(latent_dim=8, global_pool="max")
    mu, _ = encoder(torch.randn(2, 256))
    assert mu.shape == (2, 8)


def test_unknown_global_pool_raises() -> None:
    with pytest.raises(ValueError, match="global_pool"):
        OneDCnnEncoder(latent_dim=8, global_pool="sum")


def test_unknown_pooling_raises() -> None:
    with pytest.raises(ValueError, match="pooling"):
        OneDCnnEncoder(latent_dim=8, poolings="sum")
