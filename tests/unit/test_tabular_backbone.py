"""Unit tests for the tabular MLP encoder and decoder and their building blocks (roadmap P3-8)."""

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders
from global_vae.decoders.tabular_mlp_decoder import TabularMlpDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders
from global_vae.encoders.tabular_mlp_encoder import TabularMlpEncoder
from global_vae.utils.activations import ACTIVATIONS, listSupportedActivations
from global_vae.utils.autograd import backward
from global_vae.utils.mlp_blocks import buildMlp

LATENT = 4
FEATURES = 12


def _encoder(**kwargs: object) -> TabularMlpEncoder:
    return TabularMlpEncoder(latent_dim=LATENT, num_features=FEATURES, **kwargs)  # type: ignore[arg-type]


def _decoder(**kwargs: object) -> TabularMlpDecoder:
    return TabularMlpDecoder(latent_dim=LATENT, num_features=FEATURES, **kwargs)  # type: ignore[arg-type]


class TestRegistration:
    def test_the_encoder_is_registered_under_its_name(self) -> None:
        assert getEncoderClass("tabular_mlp_encoder_v1") is TabularMlpEncoder
        assert "tabular_mlp_encoder_v1" in listRegisteredEncoders()

    def test_the_decoder_is_registered_under_its_name(self) -> None:
        assert getDecoderClass("tabular_mlp_decoder_v1") is TabularMlpDecoder
        assert "tabular_mlp_decoder_v1" in listRegisteredDecoders()

    def test_they_implement_the_one_modality_interfaces(self) -> None:
        assert issubclass(TabularMlpEncoder, AbstractEncoder)
        assert issubclass(TabularMlpDecoder, AbstractDecoder)

    def test_the_modality_name_defaults_to_tabular_and_can_be_changed(self) -> None:
        assert _encoder().modality_name == _decoder().modality_name == "tabular"
        assert _encoder(modality_name="orders").modality_name == "orders"
        assert _decoder(modality_name="orders").modality_name == "orders"


class TestBuildMlp:
    def test_it_returns_the_stack_and_its_output_width(self) -> None:
        stack, width = buildMlp(10, [32, 16])
        assert width == 16
        assert stack(torch.randn(5, 10)).shape == (5, 16)

    def test_no_hidden_layer_is_an_identity_that_keeps_the_input_width(self) -> None:
        stack, width = buildMlp(10, [])
        assert width == 10
        assert isinstance(stack, nn.Identity)

    @pytest.mark.parametrize("name", sorted(ACTIVATIONS))
    def test_every_listed_activation_builds(self, name: str) -> None:
        stack, _ = buildMlp(6, [8], activation=name)
        assert isinstance(stack, nn.Sequential)
        assert isinstance(stack[1], type(ACTIVATIONS[name]()))

    def test_the_activations_are_listed_sorted(self) -> None:
        assert listSupportedActivations() == sorted(ACTIVATIONS)

    @pytest.mark.parametrize(
        ("normalization", "layer_type"), [("layer", nn.LayerNorm), ("batch", nn.BatchNorm1d)]
    )
    def test_a_normalization_follows_each_linear_layer(
        self, normalization: str, layer_type: type[nn.Module]
    ) -> None:
        stack, _ = buildMlp(6, [8, 8], normalization=normalization)
        assert isinstance(stack, nn.Sequential)
        assert sum(isinstance(layer, layer_type) for layer in stack) == 2

    def test_no_normalization_and_no_dropout_add_no_layer_by_default(self) -> None:
        stack, _ = buildMlp(6, [8, 8])
        assert isinstance(stack, nn.Sequential)
        assert [type(layer) for layer in stack] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU]

    def test_dropout_adds_a_layer_after_each_hidden_layer(self) -> None:
        stack, _ = buildMlp(6, [8, 8], dropout=0.25)
        assert isinstance(stack, nn.Sequential)
        assert sum(isinstance(layer, nn.Dropout) for layer in stack) == 2

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"in_features": 0, "hidden_dims": [4]}, "in_features"),
            ({"in_features": 4, "hidden_dims": [4, 0]}, "hidden_dims"),
            ({"in_features": 4, "hidden_dims": [4], "activation": "swish"}, "activation"),
            ({"in_features": 4, "hidden_dims": [4], "normalization": "group"}, "normalization"),
            ({"in_features": 4, "hidden_dims": [4], "dropout": 1.0}, "dropout"),
            ({"in_features": 4, "hidden_dims": [4], "dropout": -0.1}, "dropout"),
        ],
    )
    def test_an_argument_out_of_range_is_refused_and_named(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            buildMlp(**kwargs)  # type: ignore[arg-type]


class TestEncoder:
    def test_output_shapes(self) -> None:
        mu, logvar = _encoder()(torch.randn(7, FEATURES))
        assert mu.shape == logvar.shape == (7, LATENT)

    def test_a_batch_of_one_row_is_encoded(self) -> None:
        mu, _ = _encoder()(torch.randn(1, FEATURES))
        assert mu.shape == (1, LATENT)

    def test_no_hidden_layer_is_two_linear_maps(self) -> None:
        encoder = _encoder(hidden_dims=[])
        assert sum(p.numel() for p in encoder.parameters()) == 2 * (FEATURES * LATENT + LATENT)

    def test_it_reports_its_latent_width_and_row_width(self) -> None:
        encoder = _encoder()
        assert encoder.latent_dim == LATENT
        assert encoder.minimal_input_length == FEATURES

    @pytest.mark.parametrize(
        "shape", [(FEATURES,), (4, FEATURES - 1), (4, FEATURES + 1), (4, 1, FEATURES)]
    )
    def test_a_tensor_that_is_not_a_batch_of_rows_is_refused(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=rf"\(batch, {FEATURES}\).*{shape}"):
            _encoder()(torch.randn(*shape))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"latent_dim": 0, "num_features": 4}, "latent_dim"),
            ({"latent_dim": 2, "num_features": 0}, "num_features"),
            ({"latent_dim": 2, "num_features": 4, "hidden_dims": [0]}, "hidden_dims"),
            ({"latent_dim": 2, "num_features": 4, "activation": "swish"}, "activation"),
        ],
    )
    def test_a_bad_constructor_argument_is_refused_and_named(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            TabularMlpEncoder(**kwargs)  # type: ignore[arg-type]

    def test_every_parameter_receives_a_gradient(self) -> None:
        encoder = _encoder(normalization="layer", dropout=0.1)
        mu, logvar = encoder(torch.randn(8, FEATURES))
        backward(mu.sum() + logvar.sum())
        assert [n for n, p in encoder.named_parameters() if p.grad is None] == []

    def test_dropout_is_off_in_eval_mode(self) -> None:
        encoder = _encoder(dropout=0.5).eval()
        x = torch.randn(4, FEATURES)
        assert torch.equal(encoder(x)[0], encoder(x)[0])

    def test_batch_normalization_needs_two_rows_in_training_only(self) -> None:
        encoder = _encoder(normalization="batch")
        with pytest.raises(ValueError):
            encoder(torch.randn(1, FEATURES))
        assert encoder.eval()(torch.randn(1, FEATURES))[0].shape == (1, LATENT)

    def test_a_missing_value_is_not_hidden_it_makes_the_row_nan(self) -> None:
        """The caller imputes: the encoder does not guess a value for a NaN."""
        x = torch.randn(3, FEATURES)
        x[1, 2] = float("nan")
        mu, _ = _encoder()(x)
        assert torch.isnan(mu[1]).all()
        assert torch.isfinite(mu[[0, 2]]).all()


class TestDecoder:
    def test_output_shape_is_the_row_width(self) -> None:
        assert _decoder()(torch.randn(7, LATENT)).shape == (7, FEATURES)

    def test_without_categorical_columns_every_output_is_linear(self) -> None:
        out = _decoder()(torch.randn(64, LATENT))
        assert (out < 0).any()
        assert not torch.allclose(out.sum(dim=1), torch.ones(64))

    def test_each_one_hot_block_is_a_probability_vector(self) -> None:
        out = _decoder(categorical_sizes=[3, 4])(torch.randn(16, LATENT))
        numeric = FEATURES - 7
        first, second = out[:, numeric : numeric + 3], out[:, numeric + 3 :]
        for block in (first, second):
            assert (block >= 0).all() and (block <= 1).all()
            assert torch.allclose(block.sum(dim=1), torch.ones(16), atol=1e-6)

    def test_the_numeric_columns_stay_linear_next_to_the_blocks(self) -> None:
        torch.manual_seed(0)
        plain = _decoder()
        mixed = _decoder(categorical_sizes=[3, 4])
        mixed.load_state_dict(plain.state_dict())
        z = torch.randn(16, LATENT)
        assert torch.equal(mixed(z)[:, : FEATURES - 7], plain(z)[:, : FEATURES - 7])

    def test_a_table_of_one_hot_columns_only_is_reconstructed(self) -> None:
        decoder = TabularMlpDecoder(latent_dim=LATENT, num_features=7, categorical_sizes=[3, 4])
        out = decoder(torch.randn(5, LATENT))
        assert torch.allclose(out[:, :3].sum(dim=1), torch.ones(5), atol=1e-6)
        assert torch.allclose(out[:, 3:].sum(dim=1), torch.ones(5), atol=1e-6)

    def test_no_hidden_layer_is_one_linear_map(self) -> None:
        decoder = _decoder(hidden_dims=[])
        assert sum(p.numel() for p in decoder.parameters()) == LATENT * FEATURES + FEATURES

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"latent_dim": 0, "num_features": 4}, "latent_dim"),
            ({"latent_dim": 2, "num_features": 0}, "num_features"),
            ({"latent_dim": 2, "num_features": 4, "categorical_sizes": [1, 2]}, "at least 2"),
            ({"latent_dim": 2, "num_features": 4, "categorical_sizes": [3, 3]}, "num_features=4"),
            ({"latent_dim": 2, "num_features": 4, "dropout": 1.5}, "dropout"),
        ],
    )
    def test_a_bad_constructor_argument_is_refused_and_named(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            TabularMlpDecoder(**kwargs)  # type: ignore[arg-type]

    def test_every_parameter_receives_a_gradient(self) -> None:
        decoder = _decoder(categorical_sizes=[3], normalization="layer")
        backward(decoder(torch.randn(8, LATENT)).sum())
        assert [n for n, p in decoder.named_parameters() if p.grad is None] == []

    def test_the_mse_against_a_one_hot_target_is_the_brier_score_and_falls_in_training(
        self,
    ) -> None:
        torch.manual_seed(0)
        decoder = TabularMlpDecoder(latent_dim=2, num_features=3, categorical_sizes=[3])
        z = torch.randn(32, 2)
        target = functional.one_hot((z[:, 0] > 0).long() + (z[:, 1] > 0).long(), 3).float()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)
        first = functional.mse_loss(decoder(z), target).item()
        for _ in range(100):
            optimizer.zero_grad()
            backward(functional.mse_loss(decoder(z), target))
            optimizer.step()
        assert functional.mse_loss(decoder(z), target).item() < first


class TestEncoderAndDecoderTogether:
    def test_a_row_goes_through_both_and_comes_back_at_its_width(self) -> None:
        x = torch.randn(6, FEATURES)
        mu, _ = _encoder()(x)
        assert _decoder()(mu).shape == x.shape
