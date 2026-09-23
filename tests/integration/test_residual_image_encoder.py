"""Unit tests for `TwoDCnnResidualEncoder` (spec §6, §7, §12).

Mirrors `test_image_encoder.py`'s coverage of `TwoDCnnEncoder` (shapes,
variable/non-square input shape, per-stage shape flexibility including the
tuple-vs-list convention and its error path, pooling/activation/normalization
configurability, minimum-input-shape solving, registration, and error paths)
combined with `test_residual_signal_encoder.py`'s coverage of
`OneDCnnResidualEncoder` (flexible `block_depths` per stage and
`shortcut_kernel_sizes`), since this class is exactly that intersection: the
2D generalization of the residual 1D encoder, generalized the same way
`TwoDCnnEncoder` generalized `OneDCnnEncoder`.
"""

import pytest
import torch
from torch import nn

import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.encoders.registry import getEncoderClass
from global_vae.encoders.TwoDCnnResidualEncoder import TwoDCnnResidualEncoder
from global_vae.utils.autograd import backward


def test_output_shapes_from_plain_images() -> None:
    encoder = TwoDCnnResidualEncoder(latent_dim=16)
    mu, logvar = encoder(torch.randn(4, 64, 64))
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_accepts_explicit_channel_dimension() -> None:
    encoder = TwoDCnnResidualEncoder(latent_dim=16)
    mu, logvar = encoder(torch.randn(4, 1, 64, 64))
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_handles_varying_input_shape() -> None:
    """Different image sizes must produce the same fixed-size output (spec §6)."""
    encoder = TwoDCnnResidualEncoder(latent_dim=16)
    mu_small, _ = encoder(torch.randn(2, 32, 32))
    mu_large, _ = encoder(torch.randn(2, 96, 96))
    assert mu_small.shape == mu_large.shape == (2, 16)


def test_handles_non_square_input() -> None:
    """Height and width are independent axes throughout; a rectangular image is not
    a special case."""
    encoder = TwoDCnnResidualEncoder(latent_dim=16)
    mu, logvar = encoder(torch.randn(2, 48, 96))
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)


def test_multi_channel_input() -> None:
    encoder = TwoDCnnResidualEncoder(latent_dim=8, in_channels=3)
    mu, _ = encoder(torch.randn(4, 3, 64, 64))
    assert mu.shape == (4, 8)


def test_latent_dim_property() -> None:
    encoder = TwoDCnnResidualEncoder(latent_dim=32)
    assert encoder.latent_dim == 32


def test_modality_name_defaults_to_image_but_is_configurable() -> None:
    default_encoder = TwoDCnnResidualEncoder(latent_dim=8)
    assert default_encoder.modality_name == "image"

    named_encoder = TwoDCnnResidualEncoder(latent_dim=8, modality_name="xray")
    assert named_encoder.modality_name == "xray"


def test_gradients_reach_every_parameter() -> None:
    encoder = TwoDCnnResidualEncoder(latent_dim=8)
    mu, logvar = encoder(torch.randn(3, 64, 64))
    backward(mu.sum() + logvar.sum())
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"parameter '{name}' got no gradient"


def test_registered_under_2d_cnn_resnet_encoder_v1() -> None:
    assert getEncoderClass("2d_cnn_resnet_encoder_v1") is TwoDCnnResidualEncoder


class TestFlexibleBlockDepths:
    def test_three_then_four_layers(self) -> None:
        """Spec's own example: "3 layers before the connection... then 4"."""
        encoder = TwoDCnnResidualEncoder(
            latent_dim=8,
            hidden_channels=(16, 32),
            block_depths=(3, 4),
        )
        mu, logvar = encoder(torch.randn(2, 64, 64))
        assert mu.shape == (2, 8)
        assert logvar.shape == (2, 8)

    def test_shared_block_depth_applies_to_every_stage(self) -> None:
        encoder = TwoDCnnResidualEncoder(
            latent_dim=8, hidden_channels=(16, 32, 64), block_depths=3
        )
        mu, _ = encoder(torch.randn(2, 64, 64))
        assert mu.shape == (2, 8)

    def test_block_depth_one_is_supported(self) -> None:
        encoder = TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32), block_depths=1)
        mu, _ = encoder(torch.randn(2, 64, 64))
        assert mu.shape == (2, 8)

    def test_mismatched_block_depths_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="block_depths"):
            TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32, 64), block_depths=(2, 3))

    def test_non_positive_block_depths_raises(self) -> None:
        with pytest.raises(ValueError, match="block_depths"):
            TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(16,), block_depths=0)

    def test_even_kernel_size_with_depth_greater_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="odd kernel_size"):
            TwoDCnnResidualEncoder(
                latent_dim=8, hidden_channels=(16,), kernel_sizes=4, block_depths=2
            )

    def test_accounts_for_block_depth_but_never_shrinks_the_minimum(self) -> None:
        """With odd kernels (this class's default), every internal layer is exactly
        length-preserving, so a deeper block never *raises* the architecture's minimum
        input shape above a shallower one built from the same per-layer hyperparameters."""
        shallow_minimum = TwoDCnnResidualEncoder.computeMinimumInputShape(
            hidden_channels=(16, 32), kernel_sizes=3, strides=2, block_depths=1, poolings=None
        )
        deep_minimum = TwoDCnnResidualEncoder.computeMinimumInputShape(
            hidden_channels=(16, 32), kernel_sizes=3, strides=2, block_depths=3, poolings=None
        )
        assert deep_minimum == shallow_minimum


class TestPerStageShapeFlexibility:
    """`kernel_sizes`/`strides`/`paddings`/`dilations`/`pool_*`/`shortcut_kernel_sizes` each
    accept a shared `int`, a shared non-square `tuple[int, int]`, or a per-stage `list`
    (spec §12); see `utils.stage_config.broadcastPerStageShape` for why `list` and `tuple`
    mean different things here."""

    def test_shared_non_square_kernel_tuple_applies_to_every_stage(self) -> None:
        encoder = TwoDCnnResidualEncoder(
            latent_dim=8,
            hidden_channels=(16, 32),
            kernel_sizes=(3, 5),
            strides=(2, 2),
            paddings=(1, 2),
            poolings=None,
        )
        mu, _ = encoder(torch.randn(2, 64, 64))
        assert mu.shape == (2, 8)

    def test_per_stage_list_mixing_square_and_non_square_shapes(self) -> None:
        encoder = TwoDCnnResidualEncoder(
            latent_dim=8,
            hidden_channels=(16, 32, 64),
            kernel_sizes=[3, (5, 7), 3],
            strides=[1, 2, 2],
            poolings=None,
        )
        mu, _ = encoder(torch.randn(2, 96, 96))
        assert mu.shape == (2, 8)

    def test_per_stage_differing_numeric_values_require_a_list_not_a_tuple(self) -> None:
        """A bare tuple is always a *shared* shape (never per-stage); genuinely differing
        per-stage values must use a list, exactly like `TwoDCnnEncoder`."""
        encoder = TwoDCnnResidualEncoder(
            latent_dim=8, hidden_channels=(16, 32, 64), strides=[1, 1, 2], poolings=None
        )
        mu, _ = encoder(torch.randn(2, 64, 64))
        assert mu.shape == (2, 8)

        with pytest.raises(ValueError, match="2-dimensional"):
            TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32, 64), strides=(1, 1, 2))

    def test_mismatched_per_stage_list_length_raises(self) -> None:
        with pytest.raises(ValueError, match="kernel_sizes"):
            TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(16, 32, 64), kernel_sizes=[3, 5])

    def test_shortcut_kernel_sizes_are_configurable_per_stage(self) -> None:
        encoder = TwoDCnnResidualEncoder(
            latent_dim=8, hidden_channels=(16, 32), shortcut_kernel_sizes=(1, 3)
        )
        mu, _ = encoder(torch.randn(2, 64, 64))
        assert mu.shape == (2, 8)


def test_per_stage_pooling_can_differ() -> None:
    """Different stages may use different pooling strategies, or none at all."""
    encoder = TwoDCnnResidualEncoder(
        latent_dim=8,
        hidden_channels=(16, 32, 64),
        poolings=("max", "avg", None),
        strides=[1, 1, 2],  # last stage downsamples via stride since it has no pooling
    )
    mu, _ = encoder(torch.randn(2, 64, 64))
    assert mu.shape == (2, 8)


def test_pooling_requires_a_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        TwoDCnnResidualEncoder(latent_dim=8, poolings="max", pool_kernel_sizes=None)


def test_pool_paddings_and_kwargs_are_forwarded() -> None:
    encoder = TwoDCnnResidualEncoder(
        latent_dim=8,
        poolings="max",
        pool_kernel_sizes=3,
        pool_paddings=1,
        pool_kwargs={"ceil_mode": True},
    )
    mu, _ = encoder(torch.randn(2, 64, 64))
    assert mu.shape == (2, 8)


def test_non_square_pooling_shapes() -> None:
    encoder = TwoDCnnResidualEncoder(
        latent_dim=8, poolings="max", pool_kernel_sizes=(2, 3), pool_strides=(2, 3)
    )
    mu, _ = encoder(torch.randn(2, 64, 96))
    assert mu.shape == (2, 8)


def test_pool_kwargs_default_does_not_leak_across_instances() -> None:
    first = TwoDCnnResidualEncoder(latent_dim=8)
    second = TwoDCnnResidualEncoder(latent_dim=8, pool_kwargs={"ceil_mode": True})
    mu_first, _ = first(torch.randn(2, 64, 64))
    mu_second, _ = second(torch.randn(2, 64, 64))
    assert mu_first.shape == mu_second.shape == (2, 8)


def test_per_stage_activation_and_normalization_can_differ() -> None:
    encoder = TwoDCnnResidualEncoder(
        latent_dim=8,
        hidden_channels=(16, 32),
        activations=(nn.ReLU, nn.GELU),
        normalizations=(nn.BatchNorm2d, None),
    )
    mu, _ = encoder(torch.randn(2, 64, 64))
    assert mu.shape == (2, 8)
    assert any(isinstance(module, nn.GELU) for module in encoder.modules())


def test_activation_and_normalization_can_be_disabled() -> None:
    encoder = TwoDCnnResidualEncoder(latent_dim=8, activations=None, normalizations=None)
    mu, _ = encoder(torch.randn(2, 64, 64))
    assert mu.shape == (2, 8)
    assert not any(isinstance(module, nn.BatchNorm2d) for module in encoder.modules())


def test_global_pool_max_variant() -> None:
    encoder = TwoDCnnResidualEncoder(latent_dim=8, global_pool="max")
    mu, _ = encoder(torch.randn(2, 64, 64))
    assert mu.shape == (2, 8)


def test_unknown_global_pool_raises() -> None:
    with pytest.raises(ValueError, match="global_pool"):
        TwoDCnnResidualEncoder(latent_dim=8, global_pool="sum")


def test_head_hidden_dims_inserts_an_mlp_before_the_heads() -> None:
    encoder = TwoDCnnResidualEncoder(latent_dim=8, head_hidden_dims=(64,))
    mu, logvar = encoder(torch.randn(2, 64, 64))
    assert mu.shape == (2, 8)
    assert logvar.shape == (2, 8)


class TestMinimumInputShape:
    def test_matches_the_1d_reference_value_on_both_axes_for_a_square_config(self) -> None:
        """Same hyperparameters on both axes must reduce to exactly the same number
        `OneDCnnResidualEncoder.computeMinimumInputLength` gives for the 1D case, and
        the same value `TwoDCnnEncoder.computeMinimumInputShape` gives for the plain
        (non-residual) 2D case, since odd-kernel internal layers add no requirement."""
        min_shape = TwoDCnnResidualEncoder.computeMinimumInputShape(
            hidden_channels=(32, 64, 128)
        )
        assert min_shape == (8, 8)

    def test_property_matches_the_static_method(self) -> None:
        encoder = TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(32, 64, 128))
        assert encoder.minimal_input_shape == (8, 8)

    def test_minimal_input_length_is_the_max_of_both_axes(self) -> None:
        encoder = TwoDCnnResidualEncoder(
            latent_dim=8,
            hidden_channels=(32, 64, 128),
            kernel_sizes=(5, 3),
            pool_kernel_sizes=(2, 3),
        )
        assert encoder.minimal_input_length == max(encoder.minimal_input_shape)

    def test_non_square_hyperparameters_give_a_non_square_minimum(self) -> None:
        min_shape = TwoDCnnResidualEncoder.computeMinimumInputShape(
            hidden_channels=(32, 64, 128),
            kernel_sizes=(5, 3),
            poolings="max",
            pool_kernel_sizes=(2, 3),
        )
        assert min_shape[0] != min_shape[1]

    def test_forward_at_exactly_the_minimum_shape_succeeds(self) -> None:
        encoder = TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(32, 64, 128))
        minimum = encoder.minimal_input_shape
        mu, _ = encoder(torch.randn(2, *minimum))
        assert mu.shape == (2, 8)

    def test_forward_below_minimum_on_either_axis_raises_clear_error(self) -> None:
        encoder = TwoDCnnResidualEncoder(latent_dim=8, hidden_channels=(32, 64, 128))
        min_h, min_w = encoder.minimal_input_shape
        with pytest.raises(ValueError, match="below"):
            encoder(torch.randn(2, min_h - 1, min_w))
        with pytest.raises(ValueError, match="below"):
            encoder(torch.randn(2, min_h, min_w - 1))
