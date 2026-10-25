"""Unit tests for `TwoDVitEncoder` and `TwoDVitDecoder` (spec §6, roadmap P3-5)."""

import math

import pytest
import torch
import torch.nn.functional as functional

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders
from global_vae.decoders.two_d_vit_decoder import TwoDVitDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders
from global_vae.encoders.two_d_vit_encoder import TwoDVitEncoder
from global_vae.utils.autograd import backward

LATENT = 8
SMALL = {"d_model": 16, "num_heads": 2, "num_layers": 2, "patch_size": 4}


def _encoder(**kwargs: object) -> TwoDVitEncoder:
    return TwoDVitEncoder(latent_dim=LATENT, **{**SMALL, **kwargs})  # type: ignore[arg-type]


def _decoder(output_shape: tuple[int, int] = (16, 16), **kwargs: object) -> TwoDVitDecoder:
    return TwoDVitDecoder(
        latent_dim=LATENT,
        output_shape=output_shape,
        **{**SMALL, **kwargs},  # type: ignore[arg-type]
    )


class TestRegistration:
    def test_the_encoder_is_registered_under_its_name(self) -> None:
        assert getEncoderClass("2d_vit_encoder_v1") is TwoDVitEncoder
        assert "2d_vit_encoder_v1" in listRegisteredEncoders()

    def test_the_decoder_is_registered_under_its_name(self) -> None:
        assert getDecoderClass("2d_vit_decoder_v1") is TwoDVitDecoder
        assert "2d_vit_decoder_v1" in listRegisteredDecoders()

    def test_they_implement_the_one_modality_interfaces(self) -> None:
        assert issubclass(TwoDVitEncoder, AbstractEncoder)
        assert issubclass(TwoDVitDecoder, AbstractDecoder)


class TestEncoderShapes:
    def test_output_shapes_from_plain_images(self) -> None:
        mu, logvar = _encoder()(torch.randn(3, 16, 16))
        assert mu.shape == logvar.shape == (3, LATENT)

    def test_accepts_an_explicit_channel_dimension(self) -> None:
        encoder = _encoder()
        images = torch.randn(3, 16, 16)
        assert torch.allclose(encoder(images)[0], encoder(images.unsqueeze(1))[0])

    def test_colour_images(self) -> None:
        mu, logvar = _encoder(in_channels=3)(torch.randn(3, 3, 16, 16))
        assert mu.shape == logvar.shape == (3, LATENT)

    @pytest.mark.parametrize("size", [(1, 1), (3, 5), (16, 16), (17, 9), (33, 50), (64, 8)])
    def test_handles_varying_input_size(self, size: tuple[int, int]) -> None:
        """Images of any height and width give the same output shape (spec §6)."""
        mu, logvar = _encoder()(torch.randn(2, *size))
        assert mu.shape == logvar.shape == (2, LATENT)

    def test_an_image_smaller_than_a_patch_is_accepted(self) -> None:
        encoder = _encoder(patch_size=16)
        assert encoder.minimal_input_length == 1
        assert encoder(torch.randn(2, 3, 3))[0].shape == (2, LATENT)

    def test_a_rectangular_patch_can_be_given_as_a_list(self) -> None:
        encoder = _encoder(patch_size=[2, 8])
        assert encoder.embed.patch_size == (2, 8)
        assert encoder(torch.randn(2, 16, 32))[0].shape == (2, LATENT)

    def test_rows_of_a_batch_do_not_influence_each_other(self) -> None:
        encoder = _encoder().eval()
        images = torch.randn(4, 16, 16)
        mu, _ = encoder(images)
        for row in range(4):
            assert torch.allclose(mu[row], encoder(images[row : row + 1])[0][0], atol=1e-5)


class TestEncoderBehaviour:
    def test_an_edge_that_is_not_a_multiple_is_padded_with_the_edge_pixels(self) -> None:
        encoder = _encoder(patch_size=4).eval()
        image = torch.randn(3, 1, 10, 7)
        explicit = functional.pad(image, (0, 1, 0, 2), mode="replicate")
        assert torch.allclose(encoder(image)[0], encoder(explicit)[0], atol=1e-6)

    @pytest.mark.parametrize("pooling", ["cls", "mean"])
    def test_swapping_two_patches_changes_the_posterior(self, pooling: str) -> None:
        """Without positions the output could not tell where a patch was."""
        encoder = _encoder(pooling=pooling).eval()
        images = torch.randn(2, 16, 16)
        swapped = images.clone()
        swapped[:, :4, :4] = images[:, :4, 4:8]
        swapped[:, :4, 4:8] = images[:, :4, :4]
        assert not torch.allclose(encoder(images)[0], encoder(swapped)[0], atol=1e-4)

    def test_rows_and_columns_are_told_apart(self) -> None:
        """A transposed image is another image: the two position halves are not symmetric."""
        encoder = _encoder().eval()
        images = torch.randn(2, 16, 16)
        assert not torch.allclose(encoder(images)[0], encoder(images.transpose(1, 2))[0], atol=1e-4)

    def test_eval_mode_is_deterministic_even_with_dropout(self) -> None:
        encoder = _encoder(dropout=0.5).eval()
        images = torch.randn(2, 16, 16)
        assert torch.equal(encoder(images)[0], encoder(images)[0])

    def test_train_mode_applies_dropout(self) -> None:
        torch.manual_seed(0)
        encoder = _encoder(dropout=0.5).train()
        images = torch.randn(2, 16, 16)
        assert not torch.equal(encoder(images)[0], encoder(images)[0])

    def test_gradient_reaches_every_parameter(self) -> None:
        encoder = _encoder()
        mu, logvar = encoder(torch.randn(3, 16, 16))
        backward(mu.sum() + logvar.sum())
        assert [name for name, p in encoder.named_parameters() if p.grad is None] == []

    def test_mean_pooling_has_no_cls_token(self) -> None:
        assert _encoder(pooling="mean").cls_token is None
        assert _encoder(pooling="cls").cls_token is not None

    def test_the_head_mlp_changes_the_depth_not_the_output_shape(self) -> None:
        plain, deep = _encoder(), _encoder(head_hidden_dims=(32,))
        assert sum(p.numel() for p in deep.parameters()) > sum(
            p.numel() for p in plain.parameters()
        )
        assert deep(torch.randn(2, 16, 16))[0].shape == (2, LATENT)

    def test_properties(self) -> None:
        encoder = _encoder(modality_name="micrograph")
        assert encoder.latent_dim == LATENT
        assert encoder.modality_name == "micrograph"
        assert _encoder().modality_name == "image"


class TestEncoderPositions:
    def test_sinusoidal_positions_add_no_parameters(self) -> None:
        assert list(_encoder().positions.parameters()) == []

    def test_learned_positions_need_an_image_size(self) -> None:
        for bad in (None, [16]):
            with pytest.raises(ValueError, match="needs image_size"):
                _encoder(positional_encoding="learned", image_size=bad)

    def test_learned_positions_have_one_vector_per_patch_of_the_image_size(self) -> None:
        encoder = _encoder(positional_encoding="learned", image_size=[16, 20])
        assert [tuple(p.shape) for p in encoder.positions.parameters()] == [(20, 16)]

    def test_learned_positions_accept_images_with_the_same_patch_grid(self) -> None:
        encoder = _encoder(positional_encoding="learned", image_size=[16, 20])
        assert encoder(torch.randn(2, 16, 20))[0].shape == (2, LATENT)
        assert encoder(torch.randn(2, 13, 17))[0].shape == (2, LATENT)  # same 4 x 5 grid

    def test_learned_positions_refuse_another_grid(self) -> None:
        encoder = _encoder(positional_encoding="learned", image_size=[16, 20])
        with pytest.raises(ValueError, match="fix the image size"):
            encoder(torch.randn(2, 20, 16))

    def test_image_size_is_ignored_by_sinusoidal_positions(self) -> None:
        assert _encoder(image_size=[8, 8])(torch.randn(2, 40, 40))[0].shape == (2, LATENT)


class TestEncoderValidation:
    def test_rejects_an_unknown_pooling(self) -> None:
        with pytest.raises(ValueError, match="Unknown pooling 'max'"):
            _encoder(pooling="max")

    def test_rejects_an_unknown_positional_encoding(self) -> None:
        with pytest.raises(ValueError, match="Unknown positional encoding"):
            _encoder(positional_encoding="rotary")

    def test_rejects_heads_that_do_not_divide_the_width(self) -> None:
        with pytest.raises(ValueError, match="not evenly divisible"):
            _encoder(d_model=20, num_heads=8)

    def test_sinusoidal_positions_need_a_width_that_is_a_multiple_of_four(self) -> None:
        with pytest.raises(ValueError, match="multiple of 4"):
            _encoder(d_model=18, num_heads=2)

    def test_learned_positions_accept_any_width(self) -> None:
        encoder = _encoder(
            d_model=18, num_heads=2, positional_encoding="learned", image_size=[8, 8]
        )
        assert encoder(torch.randn(2, 8, 8))[0].shape == (2, LATENT)

    @pytest.mark.parametrize("patch_size", [0, [4, 0], [1, 2, 3]])
    def test_rejects_an_invalid_patch(self, patch_size: object) -> None:
        with pytest.raises(ValueError, match="patch_size"):
            _encoder(patch_size=patch_size)

    def test_rejects_a_wrong_channel_count(self) -> None:
        with pytest.raises(ValueError, match="in_channels=1, got 3"):
            _encoder()(torch.randn(2, 3, 16, 16))

    def test_rejects_a_matrix_and_a_five_dimensional_input(self) -> None:
        encoder = _encoder()
        for bad in (torch.randn(16, 16), torch.randn(2, 1, 1, 16, 16)):
            with pytest.raises(ValueError, match="expects"):
                encoder(bad)


class TestDecoder:
    @pytest.mark.parametrize("shape", [(1, 1), (3, 5), (16, 16), (17, 9), (30, 45), (64, 8)])
    def test_output_has_the_requested_shape(self, shape: tuple[int, int]) -> None:
        assert _decoder(shape)(torch.randn(3, LATENT)).shape == (3, *shape)

    def test_colour_images_keep_their_channel_axis(self) -> None:
        assert _decoder(out_channels=3)(torch.randn(2, LATENT)).shape == (2, 3, 16, 16)

    def test_a_single_channel_is_squeezed_like_the_cnn_decoder(self) -> None:
        assert _decoder(out_channels=1)(torch.randn(2, LATENT)).dim() == 3

    def test_the_grid_is_the_number_of_patches_on_each_axis(self) -> None:
        decoder = _decoder((30, 45), patch_size=[8, 16])
        assert decoder.unembed.grid_shape == (4, 3)
        assert decoder.unembed.num_tokens == 12

    def test_the_output_depends_on_the_latent_vector(self) -> None:
        decoder = _decoder().eval()
        assert not torch.allclose(
            decoder(torch.randn(2, LATENT)), decoder(torch.randn(2, LATENT)), atol=1e-4
        )

    def test_the_output_is_not_the_same_in_every_patch(self) -> None:
        """Positions make the tokens differ: one latent vector does not give a tiled patch."""
        decoder = _decoder((8, 8), patch_size=4).eval()
        image = decoder(torch.randn(1, LATENT))[0]
        top_left, top_right = image[:4, :4], image[:4, 4:]
        assert not torch.allclose(top_left, top_right, atol=1e-4)

    def test_rows_of_a_batch_do_not_influence_each_other(self) -> None:
        decoder = _decoder().eval()
        z = torch.randn(3, LATENT)
        batch = decoder(z)
        for row in range(3):
            assert torch.allclose(batch[row], decoder(z[row : row + 1])[0], atol=1e-5)

    def test_gradient_reaches_every_parameter(self) -> None:
        decoder = _decoder()
        backward(decoder(torch.randn(3, LATENT)).sum())
        assert [name for name, p in decoder.named_parameters() if p.grad is None] == []

    def test_learned_positions_have_one_vector_per_patch(self) -> None:
        decoder = _decoder((16, 20), positional_encoding="learned")
        assert [tuple(p.shape) for p in decoder.positions.parameters()] == [(20, 16)]
        assert decoder(torch.randn(2, LATENT)).shape == (2, 16, 20)

    def test_eval_mode_is_deterministic_even_with_dropout(self) -> None:
        decoder = _decoder(dropout=0.5).eval()
        z = torch.randn(2, LATENT)
        assert torch.equal(decoder(z), decoder(z))

    def test_modality_name(self) -> None:
        assert _decoder(modality_name="micrograph").modality_name == "micrograph"
        assert _decoder().modality_name == "image"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            TwoDVitDecoder(latent_dim=0, output_shape=(8, 8))
        with pytest.raises(ValueError, match="height, width"):
            _decoder((16, 16, 16))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="not evenly divisible"):
            _decoder(d_model=20, num_heads=8)
        with pytest.raises(ValueError, match="Unknown positional encoding"):
            _decoder(positional_encoding="rotary")
        with pytest.raises(ValueError, match="multiple of 4"):
            _decoder(d_model=18, num_heads=2)
        with pytest.raises(ValueError, match="at least 1"):
            _decoder((0, 8))
        with pytest.raises(ValueError, match="patch_size"):
            _decoder(patch_size=[4, 0])


class TestEncoderDecoderTogether:
    def test_the_decoder_reconstructs_the_shape_the_encoder_read(self) -> None:
        images = torch.randn(3, 21, 37)
        mu, _ = _encoder()(images)
        assert _decoder((21, 37))(mu).shape == images.shape

    def test_an_overfit_run_lowers_the_reconstruction_error(self) -> None:
        torch.manual_seed(0)
        encoder, decoder = _encoder(), _decoder((16, 16))
        rows = torch.linspace(0, 1, 16).view(16, 1)
        images = (rows * torch.linspace(1, 0, 16).view(1, 16)).repeat(4, 1, 1)
        images = images + 0.05 * torch.randn(4, 16, 16)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)

        def error() -> torch.Tensor:
            return functional.mse_loss(decoder(encoder(images)[0]), images)

        first = error().item()
        for _ in range(60):
            optimizer.zero_grad()
            loss = error()
            backward(loss)
            optimizer.step()
        assert math.isfinite(loss.item())
        assert error().item() < 0.5 * first
