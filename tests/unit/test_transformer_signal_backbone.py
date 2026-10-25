"""Unit tests for `OneDTransformerEncoder` and `OneDTransformerDecoder` (spec §6, roadmap P3-5)."""

import math

import pytest
import torch
import torch.nn.functional as functional

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.one_d_transformer_decoder import OneDTransformerDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.one_d_transformer_encoder import OneDTransformerEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders
from global_vae.utils.autograd import backward

LATENT = 8
SMALL = {"d_model": 16, "num_heads": 2, "num_layers": 2}


def _encoder(**kwargs: object) -> OneDTransformerEncoder:
    return OneDTransformerEncoder(latent_dim=LATENT, **{**SMALL, **kwargs})  # type: ignore[arg-type]


def _decoder(output_length: int = 40, **kwargs: object) -> OneDTransformerDecoder:
    return OneDTransformerDecoder(
        latent_dim=LATENT,
        output_length=output_length,
        **{**SMALL, **kwargs},  # type: ignore[arg-type]
    )


class TestRegistration:
    def test_the_encoder_is_registered_under_its_name(self) -> None:
        assert getEncoderClass("1d_transformer_encoder_v1") is OneDTransformerEncoder
        assert "1d_transformer_encoder_v1" in listRegisteredEncoders()

    def test_the_decoder_is_registered_under_its_name(self) -> None:
        assert getDecoderClass("1d_transformer_decoder_v1") is OneDTransformerDecoder
        assert "1d_transformer_decoder_v1" in listRegisteredDecoders()

    def test_they_implement_the_one_modality_interfaces(self) -> None:
        assert issubclass(OneDTransformerEncoder, AbstractEncoder)
        assert issubclass(OneDTransformerDecoder, AbstractDecoder)


class TestEncoderShapes:
    def test_output_shapes_from_plain_series(self) -> None:
        mu, logvar = _encoder()(torch.randn(4, 64))
        assert mu.shape == logvar.shape == (4, LATENT)

    def test_accepts_an_explicit_channel_dimension(self) -> None:
        plain = _encoder()
        series = torch.randn(4, 64)
        assert torch.allclose(plain(series)[0], plain(series.unsqueeze(1))[0])

    def test_several_channels(self) -> None:
        mu, logvar = _encoder(in_channels=3)(torch.randn(4, 3, 64))
        assert mu.shape == logvar.shape == (4, LATENT)

    @pytest.mark.parametrize("length", [1, 5, 16, 17, 100, 257])
    def test_handles_varying_input_length(self, length: int) -> None:
        """Series of any length give the same output shape (spec §6)."""
        mu, logvar = _encoder()(torch.randn(2, length))
        assert mu.shape == logvar.shape == (2, LATENT)

    def test_a_series_shorter_than_a_patch_is_accepted(self) -> None:
        encoder = _encoder(patch_size=32)
        assert encoder.minimal_input_length == 1
        assert encoder(torch.randn(2, 3))[0].shape == (2, LATENT)

    def test_rows_of_a_batch_do_not_influence_each_other(self) -> None:
        encoder = _encoder().eval()
        series = torch.randn(5, 50)
        mu, logvar = encoder(series)
        for row in range(5):
            single_mu, single_logvar = encoder(series[row : row + 1])
            assert torch.allclose(mu[row], single_mu[0], atol=1e-5)
            assert torch.allclose(logvar[row], single_logvar[0], atol=1e-5)


class TestEncoderBehaviour:
    def test_a_length_that_is_not_a_multiple_is_padded_with_the_last_value(self) -> None:
        encoder = _encoder(patch_size=8).eval()
        series = torch.randn(3, 1, 21)
        explicit = functional.pad(series, (0, 3), mode="replicate")
        assert torch.allclose(encoder(series)[0], encoder(explicit)[0], atol=1e-6)

    def test_the_order_of_the_patches_matters(self) -> None:
        """Without positions a [CLS] transformer cannot tell a swap of two patches."""
        encoder = _encoder(patch_size=8).eval()
        series = torch.randn(2, 32)
        swapped = torch.cat([series[:, 8:16], series[:, :8], series[:, 16:]], dim=1)
        assert not torch.allclose(encoder(series)[0], encoder(swapped)[0], atol=1e-4)

    def test_the_order_of_the_patches_matters_with_mean_pooling(self) -> None:
        encoder = _encoder(patch_size=8, pooling="mean").eval()
        series = torch.randn(2, 32)
        swapped = torch.cat([series[:, 8:16], series[:, :8], series[:, 16:]], dim=1)
        assert not torch.allclose(encoder(series)[0], encoder(swapped)[0], atol=1e-4)

    def test_a_different_series_gives_a_different_posterior(self) -> None:
        encoder = _encoder().eval()
        assert not torch.allclose(encoder(torch.randn(2, 64))[0], encoder(torch.randn(2, 64))[0])

    def test_eval_mode_is_deterministic_even_with_dropout(self) -> None:
        encoder = _encoder(dropout=0.5).eval()
        series = torch.randn(2, 64)
        assert torch.equal(encoder(series)[0], encoder(series)[0])

    def test_train_mode_applies_dropout(self) -> None:
        torch.manual_seed(0)
        encoder = _encoder(dropout=0.5).train()
        series = torch.randn(2, 64)
        assert not torch.equal(encoder(series)[0], encoder(series)[0])

    def test_gradient_reaches_every_parameter(self) -> None:
        encoder = _encoder()
        mu, logvar = encoder(torch.randn(3, 64))
        backward(mu.sum() + logvar.sum())
        assert [name for name, p in encoder.named_parameters() if p.grad is None] == []

    def test_mean_pooling_has_no_cls_token(self) -> None:
        assert _encoder(pooling="mean").cls_token is None
        assert _encoder(pooling="cls").cls_token is not None

    def test_the_head_mlp_changes_the_depth_not_the_output_shape(self) -> None:
        plain, deep = _encoder(), _encoder(head_hidden_dims=(32, 24))
        assert sum(p.numel() for p in deep.parameters()) > sum(
            p.numel() for p in plain.parameters()
        )
        assert deep(torch.randn(2, 64))[0].shape == (2, LATENT)

    def test_activation_relu_is_accepted(self) -> None:
        assert _encoder(activation="relu")(torch.randn(2, 64))[0].shape == (2, LATENT)

    def test_properties(self) -> None:
        encoder = _encoder(modality_name="saxs")
        assert encoder.latent_dim == LATENT
        assert encoder.modality_name == "saxs"
        assert _encoder().modality_name == "vector"


class TestEncoderPositions:
    def test_sinusoidal_positions_add_no_parameters(self) -> None:
        assert list(_encoder().positions.parameters()) == []

    def test_learned_positions_need_max_length(self) -> None:
        with pytest.raises(ValueError, match="needs max_length"):
            _encoder(positional_encoding="learned")

    def test_learned_positions_have_one_vector_per_patch_of_max_length(self) -> None:
        encoder = _encoder(positional_encoding="learned", max_length=50, patch_size=8)
        assert [tuple(p.shape) for p in encoder.positions.parameters()] == [(7, 16)]

    def test_learned_positions_accept_up_to_max_length(self) -> None:
        encoder = _encoder(positional_encoding="learned", max_length=50, patch_size=8)
        for length in (1, 8, 50, 56):
            assert encoder(torch.randn(2, length))[0].shape == (2, LATENT)

    def test_learned_positions_refuse_a_longer_series(self) -> None:
        encoder = _encoder(positional_encoding="learned", max_length=50, patch_size=8)
        with pytest.raises(ValueError, match="learned positions cover 7"):
            encoder(torch.randn(2, 57))

    def test_max_length_is_ignored_by_sinusoidal_positions(self) -> None:
        assert _encoder(max_length=10)(torch.randn(2, 500))[0].shape == (2, LATENT)


class TestEncoderValidation:
    def test_rejects_an_unknown_pooling(self) -> None:
        with pytest.raises(ValueError, match="Unknown pooling 'max'"):
            _encoder(pooling="max")

    def test_rejects_an_unknown_positional_encoding(self) -> None:
        with pytest.raises(ValueError, match="Unknown positional encoding"):
            _encoder(positional_encoding="rotary")

    def test_rejects_heads_that_do_not_divide_the_width(self) -> None:
        with pytest.raises(ValueError, match="not evenly divisible"):
            _encoder(d_model=18, num_heads=4)

    @pytest.mark.parametrize("patch_size", [0, -4])
    def test_rejects_a_patch_below_one(self, patch_size: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _encoder(patch_size=patch_size)

    def test_rejects_a_wrong_channel_count(self) -> None:
        with pytest.raises(ValueError, match="in_channels=1, got 2"):
            _encoder()(torch.randn(2, 2, 32))

    def test_rejects_a_vector_and_a_four_dimensional_input(self) -> None:
        encoder = _encoder()
        for bad in (torch.randn(32), torch.randn(2, 1, 4, 8)):
            with pytest.raises(ValueError, match="expects"):
                encoder(bad)


class TestDecoder:
    @pytest.mark.parametrize("length", [1, 5, 16, 40, 100, 257])
    def test_output_has_the_requested_length(self, length: int) -> None:
        assert _decoder(length)(torch.randn(3, LATENT)).shape == (3, length)

    def test_several_channels_keep_their_axis(self) -> None:
        assert _decoder(40, out_channels=2)(torch.randn(3, LATENT)).shape == (3, 2, 40)

    def test_a_single_channel_is_squeezed_like_the_cnn_decoder(self) -> None:
        assert _decoder(40, out_channels=1)(torch.randn(3, LATENT)).dim() == 2

    def test_the_number_of_tokens_is_the_number_of_patches(self) -> None:
        assert _decoder(40, patch_size=16).unembed.num_tokens == 3
        assert _decoder(48, patch_size=16).unembed.num_tokens == 3
        assert _decoder(49, patch_size=16).unembed.num_tokens == 4

    def test_the_output_depends_on_the_latent_vector(self) -> None:
        decoder = _decoder().eval()
        assert not torch.allclose(
            decoder(torch.randn(2, LATENT)), decoder(torch.randn(2, LATENT)), atol=1e-4
        )

    def test_the_output_is_not_the_same_in_every_patch(self) -> None:
        """Positions make the tokens differ: one latent vector does not give a repeated patch."""
        decoder = _decoder(48, patch_size=16).eval()
        series = decoder(torch.randn(1, LATENT))[0].view(3, 16)
        assert not torch.allclose(series[0], series[1], atol=1e-4)

    def test_rows_of_a_batch_do_not_influence_each_other(self) -> None:
        decoder = _decoder().eval()
        z = torch.randn(4, LATENT)
        batch = decoder(z)
        for row in range(4):
            assert torch.allclose(batch[row], decoder(z[row : row + 1])[0], atol=1e-5)

    def test_gradient_reaches_every_parameter(self) -> None:
        decoder = _decoder()
        backward(decoder(torch.randn(3, LATENT)).sum())
        assert [name for name, p in decoder.named_parameters() if p.grad is None] == []

    def test_learned_positions_have_one_vector_per_token(self) -> None:
        decoder = _decoder(40, patch_size=16, positional_encoding="learned")
        assert [tuple(p.shape) for p in decoder.positions.parameters()] == [(3, 16)]
        assert decoder(torch.randn(2, LATENT)).shape == (2, 40)

    def test_eval_mode_is_deterministic_even_with_dropout(self) -> None:
        decoder = _decoder(dropout=0.5).eval()
        z = torch.randn(2, LATENT)
        assert torch.equal(decoder(z), decoder(z))

    def test_modality_name(self) -> None:
        assert _decoder(modality_name="saxs").modality_name == "saxs"
        assert _decoder().modality_name == "vector"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            OneDTransformerDecoder(latent_dim=0, output_length=16)
        with pytest.raises(ValueError, match="not evenly divisible"):
            _decoder(d_model=18, num_heads=4)
        with pytest.raises(ValueError, match="Unknown positional encoding"):
            _decoder(positional_encoding="rotary")
        with pytest.raises(ValueError, match="at least 1"):
            _decoder(0)
        with pytest.raises(ValueError, match="at least 1"):
            _decoder(40, patch_size=0)


class TestEncoderDecoderTogether:
    def test_the_decoder_reconstructs_the_length_the_encoder_read(self) -> None:
        length = 123
        series = torch.randn(3, length)
        mu, _ = _encoder()(series)
        assert _decoder(length)(mu).shape == series.shape

    def test_an_overfit_step_lowers_the_reconstruction_error(self) -> None:
        torch.manual_seed(0)
        encoder, decoder = _encoder(patch_size=8), _decoder(64, patch_size=8)
        series = torch.sin(torch.linspace(0, 6.0, 64)).repeat(4, 1) + 0.1 * torch.randn(4, 64)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)

        def error() -> torch.Tensor:
            return functional.mse_loss(decoder(encoder(series)[0]), series)

        first = error().item()
        for _ in range(60):
            optimizer.zero_grad()
            loss = error()
            backward(loss)
            optimizer.step()
        assert math.isfinite(loss.item())
        assert error().item() < 0.5 * first
