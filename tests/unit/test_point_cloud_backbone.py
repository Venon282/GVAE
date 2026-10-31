"""Unit tests for the point cloud encoder, decoder and blocks (roadmap P3-8, ADR 0044)."""

import pytest
import torch

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.point_cloud_decoder import PointCloudDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.point_cloud_encoder import PointCloudEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders
from global_vae.utils.autograd import backward
from global_vae.utils.point_cloud_blocks import POOLING_CHOICES, poolPoints, splitPoints

NAN = float("nan")
LATENT = 4
POINTS = 20
SMALL = {"point_dims": (8, 16), "head_hidden_dims": (12,)}


def _encoder(**kwargs: object) -> PointCloudEncoder:
    settings = {**SMALL, "latent_dim": LATENT, **kwargs}
    return PointCloudEncoder(**settings).eval()  # type: ignore[arg-type]


def _decoder(**kwargs: object) -> PointCloudDecoder:
    settings = {"latent_dim": LATENT, "num_points": POINTS, "hidden_dims": (16,), **kwargs}
    return PointCloudDecoder(**settings).eval()  # type: ignore[arg-type]


def _clouds(rows: int = 3, points: int = POINTS, dim: int = 3, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, points, dim, generator=generator)


def _padded(x: torch.Tensor, extra: int) -> torch.Tensor:
    """`x` followed by `extra` rows of `NaN`, the padding of a cloud with fewer points."""
    return torch.cat([x, torch.full((x.shape[0], extra, x.shape[2]), NAN)], dim=1)


def _shuffled(x: torch.Tensor, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return x[:, torch.randperm(x.shape[1], generator=generator)]


class TestRegistration:
    def test_the_encoder_is_registered_under_its_name(self) -> None:
        assert getEncoderClass("point_cloud_encoder_v1") is PointCloudEncoder
        assert "point_cloud_encoder_v1" in listRegisteredEncoders()

    def test_the_decoder_is_registered_under_its_name(self) -> None:
        assert getDecoderClass("point_cloud_decoder_v1") is PointCloudDecoder
        assert "point_cloud_decoder_v1" in listRegisteredDecoders()

    def test_they_implement_the_one_modality_interfaces(self) -> None:
        assert issubclass(PointCloudEncoder, AbstractEncoder)
        assert issubclass(PointCloudDecoder, AbstractDecoder)

    def test_the_modality_name_defaults_to_point_cloud_and_can_be_changed(self) -> None:
        assert _encoder().modality_name == "point_cloud"
        assert _decoder().modality_name == "point_cloud"
        assert _encoder(modality_name="shape").modality_name == "shape"
        assert _decoder(modality_name="shape").modality_name == "shape"

    def test_the_encoder_reports_its_latent_width_and_accepts_one_point(self) -> None:
        encoder = _encoder()
        assert encoder.latent_dim == LATENT
        assert encoder.minimal_input_length == 1


class TestSplitPoints:
    def test_it_zeroes_the_missing_points_and_marks_the_real_ones(self) -> None:
        x = _padded(_clouds(points=4), 2)
        filled, valid = splitPoints(x, 3)
        assert filled.shape == x.shape
        assert not torch.isnan(filled).any()
        assert valid.tolist() == [[True] * 4 + [False] * 2] * 3
        assert (filled[:, 4:] == 0).all()

    def test_a_nan_in_one_coordinate_makes_the_row_missing(self) -> None:
        x = _clouds(rows=1, points=3)
        x[0, 1, 2] = NAN
        assert splitPoints(x, 3)[1].tolist() == [[True, False, True]]

    @pytest.mark.parametrize(
        "x",
        [torch.zeros(4, 3), torch.zeros(2, 4, 2), torch.zeros(2, 4, 3, dtype=torch.long)],
    )
    def test_a_tensor_that_is_not_a_float_batch_of_clouds_is_refused(self, x: torch.Tensor) -> None:
        with pytest.raises(ValueError, match=r"\(batch, points, 3\)"):
            splitPoints(x, 3)


class TestPoolPoints:
    def test_max_and_mean_leave_the_missing_points_out(self) -> None:
        features = torch.tensor([[[1.0], [5.0], [100.0], [-50.0]]])
        valid = torch.tensor([[True, True, False, False]])
        assert poolPoints(features, valid, "max").item() == 5.0
        assert poolPoints(features, valid, "mean").item() == 3.0
        assert poolPoints(features, valid, "max_mean").tolist() == [[5.0, 3.0]]

    def test_a_cloud_with_no_point_pools_to_zeros(self) -> None:
        features = torch.randn(2, 5, 3)
        valid = torch.zeros(2, 5, dtype=torch.bool)
        for pooling in POOLING_CHOICES:
            assert (poolPoints(features, valid, pooling) == 0).all()

    def test_an_unknown_pooling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown pooling"):
            poolPoints(torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), "sum")


class TestEncoderOutput:
    def test_it_gives_one_mu_and_logvar_per_cloud(self) -> None:
        mu, logvar = _encoder()(_clouds())
        assert mu.shape == logvar.shape == (3, LATENT)

    @pytest.mark.parametrize("points", [1, 2, 7, 64])
    def test_any_number_of_points_is_accepted(self, points: int) -> None:
        mu, _ = _encoder()(_clouds(points=points))
        assert mu.shape == (3, LATENT)
        assert torch.isfinite(mu).all()

    def test_it_is_deterministic_in_eval_mode(self) -> None:
        encoder, x = _encoder(), _clouds()
        assert torch.equal(encoder(x)[0], encoder(x)[0])

    def test_extra_features_per_point_are_read_with_the_position(self) -> None:
        encoder = _encoder(in_dim=6)
        mu, _ = encoder(_clouds(dim=6))
        assert mu.shape == (3, LATENT)

    def test_a_layer_norm_and_an_empty_head_are_accepted(self) -> None:
        encoder = _encoder(normalization="layer", head_hidden_dims=())
        assert encoder(_clouds())[0].shape == (3, LATENT)


class TestInvariances:
    @pytest.mark.parametrize("pooling", POOLING_CHOICES)
    def test_the_order_of_the_points_does_not_change_the_latent_vector(self, pooling: str) -> None:
        encoder, x = _encoder(pooling=pooling), _clouds()
        for first, second in zip(encoder(x), encoder(_shuffled(x)), strict=True):
            assert torch.allclose(first, second, atol=1e-5)

    @pytest.mark.parametrize("pooling", POOLING_CHOICES)
    def test_rows_of_nan_after_the_points_do_not_change_the_latent_vector(
        self, pooling: str
    ) -> None:
        encoder, x = _encoder(pooling=pooling), _clouds(points=9)
        for first, second in zip(encoder(x), encoder(_padded(x, 7)), strict=True):
            assert torch.allclose(first, second, atol=1e-6)

    def test_padding_in_the_middle_is_the_same_as_padding_at_the_end(self) -> None:
        encoder, x = _encoder(), _clouds(points=9)
        gapped = torch.cat([x[:, :4], torch.full((3, 3, 3), NAN), x[:, 4:]], dim=1)
        assert torch.allclose(encoder(x)[0], encoder(gapped)[0], atol=1e-6)

    def test_clouds_of_different_sizes_share_a_batch_with_their_own_latent_vectors(self) -> None:
        encoder = _encoder()
        small, large = _clouds(rows=1, points=5), _clouds(rows=1, points=12, seed=1)
        batch = torch.cat([_padded(small, 7), large])
        mu = encoder(batch)[0]
        assert torch.allclose(mu[0], encoder(small)[0][0], atol=1e-6)
        assert torch.allclose(mu[1], encoder(large)[0][0], atol=1e-6)

    def test_a_cloud_with_no_point_stays_finite(self) -> None:
        mu, logvar = _encoder()(torch.full((2, 5, 3), NAN))
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_a_different_cloud_is_a_different_latent_vector(self) -> None:
        encoder = _encoder()
        first, second = encoder(_clouds(seed=1))[0], encoder(_clouds(seed=2))[0]
        assert not torch.allclose(first, second, atol=1e-3)


class TestEncoderTraining:
    def test_every_parameter_receives_a_finite_gradient(self) -> None:
        encoder = PointCloudEncoder(latent_dim=LATENT, **SMALL)  # type: ignore[arg-type]
        mu, logvar = encoder(_padded(_clouds(points=10), 4))
        backward(mu.sum() + logvar.sum())
        gradients = {name: p.grad for name, p in encoder.named_parameters()}
        assert [name for name, grad in gradients.items() if grad is None] == []
        assert all(torch.isfinite(grad).all() for grad in gradients.values() if grad is not None)

    def test_the_padding_receives_no_gradient(self) -> None:
        encoder = PointCloudEncoder(latent_dim=LATENT, **SMALL)  # type: ignore[arg-type]
        x = _padded(_clouds(points=6), 3).requires_grad_()
        backward(encoder(x)[0].sum())
        assert x.grad is not None
        assert torch.isfinite(torch.nan_to_num(x.grad, nan=0.0)).all()


class TestEncoderArguments:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"latent_dim": 0}, "latent_dim"),
            ({"in_dim": 0}, "in_dim"),
            ({"point_dims": ()}, "point_dims"),
            ({"pooling": "sum"}, "Unknown pooling"),
            ({"normalization": "batch"}, "mix the points"),
            ({"point_dims": (0,)}, "hidden_dims"),
            ({"activation": "swishy"}, "activation"),
        ],
    )
    def test_an_argument_out_of_range_is_refused_with_its_name(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        settings = {"latent_dim": LATENT, **kwargs}
        with pytest.raises(ValueError, match=match):
            PointCloudEncoder(**settings)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "x", [torch.zeros(4, 3), torch.zeros(2, 4, 2), torch.zeros(2, 4, 3, dtype=torch.long)]
    )
    def test_an_input_that_is_not_a_float_batch_of_clouds_is_refused(self, x: torch.Tensor) -> None:
        with pytest.raises(ValueError, match=r"\(batch, points, 3\)"):
            _encoder()(x)


class TestDecoderOutput:
    def test_it_writes_the_number_of_points_it_was_given(self) -> None:
        assert _decoder()(torch.randn(3, LATENT)).shape == (3, POINTS, 3)

    @pytest.mark.parametrize("points", [1, 7, 100])
    def test_any_number_of_points_is_possible(self, points: int) -> None:
        assert _decoder(num_points=points)(torch.randn(2, LATENT)).shape == (2, points, 3)

    def test_extra_values_per_point_are_written(self) -> None:
        assert _decoder(out_dim=6)(torch.randn(2, LATENT)).shape == (2, POINTS, 6)

    def test_it_is_deterministic_in_eval_mode(self) -> None:
        decoder, z = _decoder(), torch.randn(2, LATENT)
        assert torch.equal(decoder(z), decoder(z))

    def test_the_latent_vector_changes_the_cloud(self) -> None:
        decoder = _decoder()
        first, second = decoder(torch.randn(1, LATENT)), decoder(torch.randn(1, LATENT))
        assert not torch.allclose(first, second, atol=1e-3)

    def test_the_points_are_not_all_the_same_because_each_has_its_own_code(self) -> None:
        points = _decoder()(torch.randn(1, LATENT))[0].detach()
        assert float(points.std(dim=0).min()) > 1e-4

    def test_the_codes_are_trained_with_the_rest(self) -> None:
        decoder = PointCloudDecoder(latent_dim=LATENT, num_points=POINTS, hidden_dims=(16,))
        backward(decoder(torch.randn(2, LATENT)).pow(2).sum())
        assert decoder.codes.grad is not None
        assert float(decoder.codes.grad.abs().max()) > 0
        assert [n for n, p in decoder.named_parameters() if p.grad is None] == []

    def test_a_layer_norm_and_no_hidden_layer_are_accepted(self) -> None:
        assert _decoder(normalization="layer")(torch.randn(2, LATENT)).shape == (2, POINTS, 3)
        assert _decoder(hidden_dims=())(torch.randn(2, LATENT)).shape == (2, POINTS, 3)


class TestDecoderArguments:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"latent_dim": 0}, "latent_dim"),
            ({"num_points": 0}, "num_points"),
            ({"out_dim": 0}, "out_dim"),
            ({"code_dim": 0}, "code_dim"),
            ({"normalization": "batch"}, "mix the points"),
            ({"hidden_dims": (0,)}, "hidden_dims"),
        ],
    )
    def test_an_argument_out_of_range_is_refused_with_its_name(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        settings = {"latent_dim": LATENT, "num_points": POINTS, **kwargs}
        with pytest.raises(ValueError, match=match):
            PointCloudDecoder(**settings)  # type: ignore[arg-type]

    @pytest.mark.parametrize("z", [torch.zeros(LATENT), torch.zeros(2, LATENT + 1)])
    def test_a_latent_tensor_of_the_wrong_shape_is_refused(self, z: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="z must be"):
            _decoder()(z)
