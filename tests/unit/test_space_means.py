"""`collectSpaceMeans`: one space's posterior means over unpaired batches (ADR 0061)."""

import pytest
import torch

from global_vae.evaluation.latent_alignment import LatentView
from global_vae.evaluation.space_means import collectSpaceMeans
from tests.unit._tiny_pair_model import BATCH, LATENT, buildTinyModel, tinyBatches


def _signalOnly(count: int = 2, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    return [{"signal": batch["signal"]} for batch in tinyBatches(count, seed)]


def _imageOnly(count: int = 2, seed: int = 5) -> list[dict[str, torch.Tensor]]:
    return [{"image": batch["image"]} for batch in tinyBatches(count, seed)]


class TestCollect:
    def test_it_returns_the_posterior_means_of_every_sample(self) -> None:
        model = buildTinyModel()
        batches = _signalOnly(3)
        means = collectSpaceMeans(model, "z_signal", batches)
        assert means.shape == (3 * BATCH, LATENT) and means.dtype == torch.float32
        model.eval()
        with torch.no_grad():
            direct = torch.cat(
                [
                    model.encode({"signal": b["signal"]}, use_mean=True)["latent_params"][
                        "z_signal"
                    ][0]
                    for b in batches
                ]
            )
        assert torch.allclose(means, direct)

    def test_batches_without_the_input_are_skipped_so_a_mixed_stream_can_be_given(self) -> None:
        model = buildTinyModel()
        stream = [*_signalOnly(1), *_imageOnly(2), *tinyBatches(1, seed=9)]
        means = collectSpaceMeans(model, "z_signal", stream)
        assert means.shape[0] == 2 * BATCH

    def test_the_other_modality_reads_its_own_space(self) -> None:
        model = buildTinyModel()
        stream = [*_signalOnly(1), *_imageOnly(2)]
        assert collectSpaceMeans(model, "z_image", stream).shape[0] == 2 * BATCH

    def test_a_view_can_be_given_as_text_or_as_an_object(self) -> None:
        model = buildTinyModel()
        batches = _signalOnly(1)
        as_text = collectSpaceMeans(model, "z_signal@signal", batches)
        as_object = collectSpaceMeans(model, LatentView("z_signal", ("signal",)), batches)
        assert torch.equal(as_text, as_object)

    def test_max_samples_trims_the_last_batch_and_stops(self) -> None:
        means = collectSpaceMeans(
            buildTinyModel(), "z_signal", _signalOnly(4), max_samples=BATCH + 3
        )
        assert means.shape[0] == BATCH + 3

    def test_the_model_is_left_in_evaluation_mode_with_no_gradient_kept(self) -> None:
        model = buildTinyModel()
        model.train()
        means = collectSpaceMeans(model, "z_signal", _signalOnly(1))
        assert not model.training and not means.requires_grad


class TestRefusals:
    def test_a_loader_that_never_carries_the_modality_is_refused_by_name(self) -> None:
        with pytest.raises(
            ValueError, match=r"No batch .* \['signal'\] of the view 'z_signal@signal'"
        ):
            collectSpaceMeans(buildTinyModel(), "z_signal", _imageOnly())

    def test_an_empty_loader_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to fit a prior to"):
            collectSpaceMeans(buildTinyModel(), "z_signal", [])

    def test_a_space_the_model_does_not_have_is_refused(self) -> None:
        with pytest.raises(KeyError, match="z_audio"):
            collectSpaceMeans(buildTinyModel(), "z_audio", _signalOnly())

    @pytest.mark.parametrize("limit", [0, -3])
    def test_max_samples_below_one_is_refused(self, limit: int) -> None:
        with pytest.raises(ValueError, match="max_samples must be at least 1"):
            collectSpaceMeans(buildTinyModel(), "z_signal", _signalOnly(), max_samples=limit)
