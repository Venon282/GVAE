"""Unit tests for the time series encoder and its blocks (roadmap P3-7 and P3-8, ADR 0042)."""

import pytest
import torch
import torch.nn.functional as functional

import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders
from global_vae.encoders.time_series_encoder import TimeSeriesEncoder
from global_vae.utils.autograd import backward
from global_vae.utils.series_blocks import (
    ObservedPatchEmbed,
    computeEmptyPatchMask,
    computeMaskedSquaredError,
    splitObservations,
)

NAN = float("nan")
LATENT = 4
CHANNELS = 2
LENGTH = 24
PATCH = 4
SMALL = {"d_model": 16, "num_heads": 2, "num_layers": 2, "patch_size": PATCH}


def _encoder(**kwargs: object) -> TimeSeriesEncoder:
    settings = {**SMALL, "latent_dim": LATENT, "in_channels": CHANNELS, **kwargs}
    return TimeSeriesEncoder(**settings).eval()  # type: ignore[arg-type]


def _series(rows: int = 3, length: int = LENGTH, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, CHANNELS, length, generator=generator)


def _padded(x: torch.Tensor, extra: int) -> torch.Tensor:
    """`x` followed by `extra` unobserved steps, the padding of a short series."""
    return functional.pad(x, (0, extra), value=NAN)


class TestRegistration:
    def test_the_encoder_is_registered_under_its_name(self) -> None:
        assert getEncoderClass("time_series_encoder_v1") is TimeSeriesEncoder
        assert "time_series_encoder_v1" in listRegisteredEncoders()

    def test_it_implements_the_one_modality_interface(self) -> None:
        assert issubclass(TimeSeriesEncoder, AbstractEncoder)

    def test_the_modality_name_defaults_to_time_series_and_can_be_changed(self) -> None:
        assert _encoder().modality_name == "time_series"
        assert _encoder(modality_name="sensors").modality_name == "sensors"

    def test_it_reports_its_latent_width_and_accepts_one_time_step(self) -> None:
        encoder = _encoder()
        assert encoder.latent_dim == LATENT
        assert encoder.minimal_input_length == 1


class TestSplitObservations:
    def test_a_nan_is_unobserved_and_becomes_a_zero_value(self) -> None:
        x = torch.tensor([[[1.0, NAN, 3.0]]])
        values, observed = splitObservations(x, 1)
        assert values.tolist() == [[[1.0, 0.0, 3.0]]]
        assert observed.tolist() == [[[True, False, True]]]

    def test_a_real_zero_is_observed(self) -> None:
        _, observed = splitObservations(torch.zeros(1, 1, 3), 1)
        assert observed.all()

    def test_a_series_of_one_channel_may_leave_the_channel_axis_out(self) -> None:
        values, observed = splitObservations(torch.randn(2, 5), 1)
        assert values.shape == observed.shape == (2, 1, 5)

    def test_an_integer_tensor_is_refused_because_it_cannot_hold_a_nan(self) -> None:
        with pytest.raises(ValueError, match="float"):
            splitObservations(torch.zeros(2, 1, 5, dtype=torch.long), 1)

    def test_a_tensor_of_another_rank_is_refused_with_its_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\(2, 1, 5, 3\)"):
            splitObservations(torch.zeros(2, 1, 5, 3), 1)

    def test_another_number_of_channels_is_refused_and_names_both(self) -> None:
        with pytest.raises(ValueError, match="in_channels=2, got 3"):
            splitObservations(torch.zeros(2, 3, 5), 2)

    def test_a_series_with_no_time_step_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one time step"):
            splitObservations(torch.zeros(2, 1, 0), 1)


class TestEmptyPatchMask:
    def test_a_patch_with_no_observed_value_is_empty(self) -> None:
        observed = torch.ones(1, 1, 12, dtype=torch.bool)
        observed[..., 4:8] = False
        assert computeEmptyPatchMask(observed, 4).tolist() == [[False, True, False]]

    def test_one_observed_value_in_any_channel_keeps_the_patch(self) -> None:
        observed = torch.zeros(1, 2, 8, dtype=torch.bool)
        observed[0, 1, 5] = True
        assert computeEmptyPatchMask(observed, 4).tolist() == [[True, False]]

    def test_a_length_that_is_not_a_whole_number_of_patches_counts_its_last_partial_patch(
        self,
    ) -> None:
        observed = torch.ones(1, 1, 10, dtype=torch.bool)
        mask = computeEmptyPatchMask(observed, 4)
        assert mask.shape == (1, 3)
        assert not mask.any()

    def test_a_series_observed_nowhere_keeps_its_first_patch_so_attention_has_a_key(self) -> None:
        observed = torch.zeros(2, 1, 12, dtype=torch.bool)
        assert computeEmptyPatchMask(observed, 4).tolist() == [[False, True, True]] * 2


class TestObservedPatchEmbed:
    def test_it_gives_one_token_per_patch_and_a_mask_of_as_many(self) -> None:
        tokens, empty = ObservedPatchEmbed(CHANNELS, 16, PATCH)(_series())
        assert tokens.shape == (3, LENGTH // PATCH, 16)
        assert empty.shape == (3, LENGTH // PATCH)

    def test_the_last_patch_is_padded_when_the_length_is_not_a_multiple(self) -> None:
        tokens, empty = ObservedPatchEmbed(CHANNELS, 16, PATCH)(_series(length=10))
        assert tokens.shape[1] == empty.shape[1] == 3

    def test_padding_with_nan_to_a_whole_patch_does_not_change_the_tokens(self) -> None:
        embed = ObservedPatchEmbed(CHANNELS, 16, PATCH)
        x = _series(length=10)
        tokens, _ = embed(x)
        longer, _ = embed(_padded(x, 2))
        assert torch.allclose(tokens, longer)

    def test_a_gap_is_not_the_same_as_a_zero(self) -> None:
        """The mask channels tell them apart, so the model can."""
        embed = ObservedPatchEmbed(1, 16, PATCH)
        zeros = torch.zeros(1, 1, PATCH)
        gap = torch.full((1, 1, PATCH), NAN)
        assert not torch.allclose(embed(zeros)[0], embed(gap)[0])

    def test_a_series_that_is_nan_throughout_still_gives_finite_tokens(self) -> None:
        tokens, _ = ObservedPatchEmbed(1, 16, PATCH)(torch.full((1, 1, 8), NAN))
        assert torch.isfinite(tokens).all()

    def test_an_in_channels_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="in_channels"):
            ObservedPatchEmbed(0, 16, PATCH)


class TestMaskedSquaredError:
    def test_on_a_complete_series_it_is_the_sum_of_squares_averaged_over_the_batch(self) -> None:
        reconstruction, target = torch.randn(5, 2, 7), torch.randn(5, 2, 7)
        expected = (reconstruction - target).pow(2).sum() / 5
        assert torch.allclose(computeMaskedSquaredError(reconstruction, target), expected)

    def test_a_nan_target_adds_nothing(self) -> None:
        reconstruction = torch.randn(2, 1, 4)
        target = torch.randn(2, 1, 4)
        target[0, 0, 1] = NAN
        kept = torch.ones_like(target, dtype=torch.bool)
        kept[0, 0, 1] = False
        expected = (reconstruction - target).pow(2)[kept].sum() / 2
        assert torch.allclose(computeMaskedSquaredError(reconstruction, target), expected)

    def test_the_reconstruction_receives_no_gradient_where_the_target_is_unobserved(self) -> None:
        reconstruction = torch.randn(2, 1, 4, requires_grad=True)
        target = torch.randn(2, 1, 4)
        target[:, :, 2:] = NAN
        loss = computeMaskedSquaredError(reconstruction, target)
        backward(loss)
        assert reconstruction.grad is not None
        assert torch.isfinite(reconstruction.grad).all()
        assert (reconstruction.grad[:, :, 2:] == 0).all()
        assert (reconstruction.grad[:, :, :2] != 0).any()

    def test_a_target_observed_nowhere_gives_a_zero_loss_and_not_a_nan(self) -> None:
        loss = computeMaskedSquaredError(torch.randn(2, 1, 4), torch.full((2, 1, 4), NAN))
        assert loss.item() == 0.0

    def test_two_shapes_that_differ_are_refused_with_both(self) -> None:
        with pytest.raises(ValueError, match=r"\(2, 3\) and \(2, 4\)"):
            computeMaskedSquaredError(torch.randn(2, 3), torch.randn(2, 4))

    def test_a_tensor_with_no_batch_axis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="batch"):
            computeMaskedSquaredError(torch.randn(4), torch.randn(4))


class TestEncoder:
    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_it_returns_mu_and_logvar_of_the_latent_width(self, pooling: str) -> None:
        mu, logvar = _encoder(pooling=pooling)(_series())
        assert mu.shape == logvar.shape == (3, LATENT)

    def test_one_channel_may_leave_the_channel_axis_out(self) -> None:
        mu, _ = _encoder(in_channels=1)(torch.randn(3, LENGTH))
        assert mu.shape == (3, LATENT)

    @pytest.mark.parametrize("length", [1, 3, 4, 5, 50])
    def test_any_length_is_encoded(self, length: int) -> None:
        mu, _ = _encoder()(_series(length=length))
        assert mu.shape == (3, LATENT)
        assert torch.isfinite(mu).all()

    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_padding_with_nan_does_not_change_the_latent_vector(self, pooling: str) -> None:
        encoder = _encoder(pooling=pooling)
        x = _series(length=12)
        with torch.no_grad():
            mu, logvar = encoder(x)
            for extra in (3, 4, 12):
                padded_mu, padded_logvar = encoder(_padded(x, extra))
                assert torch.allclose(mu, padded_mu, atol=1e-5)
                assert torch.allclose(logvar, padded_logvar, atol=1e-5)

    def test_padding_with_nan_does_not_change_a_series_that_also_has_a_gap(self) -> None:
        encoder = _encoder()
        x = _series(length=12)
        x[:, :, 4:8] = NAN
        with torch.no_grad():
            assert torch.allclose(encoder(x)[0], encoder(_padded(x, 8))[0], atol=1e-5)

    def test_a_series_of_zeros_is_not_a_series_of_gaps(self) -> None:
        encoder = _encoder()
        with torch.no_grad():
            zeros = encoder(torch.zeros(1, CHANNELS, LENGTH))[0]
            gaps = encoder(torch.full((1, CHANNELS, LENGTH), NAN))[0]
        assert not torch.allclose(zeros, gaps, atol=1e-4)

    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_a_series_observed_nowhere_gives_a_finite_latent_vector(self, pooling: str) -> None:
        encoder = _encoder(pooling=pooling)
        mu, logvar = encoder(torch.full((2, CHANNELS, LENGTH), NAN))
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_a_gap_changes_the_latent_vector_it_is_not_ignored(self) -> None:
        encoder = _encoder()
        x = _series()
        gapped = x.clone()
        gapped[:, :, 8:16] = NAN
        with torch.no_grad():
            assert not torch.allclose(encoder(x)[0], encoder(gapped)[0], atol=1e-4)

    def test_it_is_deterministic_in_eval_mode(self) -> None:
        encoder = _encoder(dropout=0.3)
        x = _series()
        with torch.no_grad():
            assert torch.equal(encoder(x)[0], encoder(x)[0])

    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_every_parameter_receives_a_finite_gradient_on_a_series_with_gaps(
        self, pooling: str
    ) -> None:
        encoder = TimeSeriesEncoder(
            latent_dim=LATENT,
            in_channels=CHANNELS,
            pooling=pooling,
            head_hidden_dims=(8,),
            **SMALL,  # type: ignore[arg-type]
        )
        x = _series(length=22)
        x[0, :, 4:12] = NAN
        x[1, 0, :] = NAN
        mu, logvar = encoder(_padded(x, 6))
        backward(mu.sum() + logvar.sum())
        gradients = {name: parameter.grad for name, parameter in encoder.named_parameters()}
        assert [name for name, grad in gradients.items() if grad is None] == []
        assert all(torch.isfinite(grad).all() for grad in gradients.values() if grad is not None)

    def test_learned_positions_work_up_to_max_length_and_refuse_a_longer_series(self) -> None:
        encoder = _encoder(positional_encoding="learned", max_length=LENGTH)
        assert encoder(_series())[0].shape == (3, LATENT)
        with pytest.raises(ValueError, match="max_length"):
            encoder(_series(length=LENGTH + PATCH))

    def test_learned_positions_without_a_max_length_are_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            _encoder(positional_encoding="learned")

    def test_an_unknown_pooling_is_refused_and_lists_the_choices(self) -> None:
        with pytest.raises(ValueError, match="mean"):
            _encoder(pooling="max")

    def test_a_latent_width_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            _encoder(latent_dim=0)

    def test_a_batch_of_another_number_of_channels_is_refused_and_names_it(self) -> None:
        with pytest.raises(ValueError, match=f"in_channels={CHANNELS}"):
            _encoder()(torch.randn(3, CHANNELS + 1, LENGTH))

    def test_a_series_of_integers_is_refused(self) -> None:
        with pytest.raises(ValueError, match="float"):
            _encoder()(torch.zeros(3, CHANNELS, LENGTH, dtype=torch.long))

    def test_a_head_with_hidden_layers_keeps_the_output_shapes(self) -> None:
        mu, logvar = _encoder(head_hidden_dims=(8, 8))(_series())
        assert mu.shape == logvar.shape == (3, LATENT)
