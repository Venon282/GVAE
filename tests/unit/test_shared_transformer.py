"""Unit tests for the shared transformer encoder and decoder (spec §2.1, ADR 0027, roadmap P3-5).

They are the first concrete `AbstractSharedEncoder` and `AbstractSharedDecoder`: one token stream
for a series and an image, any non-empty subset of the modalities in, one reconstruction per
modality out.
"""

import itertools

import pytest
import torch
import torch.nn.functional as functional

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders
from global_vae.decoders.shared_base import AbstractSharedDecoder
from global_vae.decoders.shared_transformer_decoder import SharedTransformerDecoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders
from global_vae.encoders.shared_base import AbstractSharedEncoder
from global_vae.encoders.shared_transformer_encoder import SharedTransformerEncoder
from global_vae.utils.autograd import backward

LATENT = 8
LENGTH = 40
SHAPE = (12, 16)
SMALL = {"d_model": 16, "num_heads": 2, "num_layers": 2}
ENCODER_MODALITIES = {
    "signal": {"kind": "series", "patch_size": 8},
    "image": {"kind": "image", "patch_size": 4},
}
DECODER_MODALITIES = {
    "signal": {"kind": "series", "output_length": LENGTH, "patch_size": 8},
    "image": {"kind": "image", "output_shape": list(SHAPE), "patch_size": 4},
}


def _encoder(**kwargs: object) -> SharedTransformerEncoder:
    arguments = {"modalities": ENCODER_MODALITIES, **SMALL, **kwargs}
    return SharedTransformerEncoder(latent_dim=LATENT, **arguments)  # type: ignore[arg-type]


def _decoder(**kwargs: object) -> SharedTransformerDecoder:
    arguments = {"modalities": DECODER_MODALITIES, **SMALL, **kwargs}
    return SharedTransformerDecoder(latent_dim=LATENT, **arguments)  # type: ignore[arg-type]


def _batch(rows: int = 3) -> dict[str, torch.Tensor]:
    return {"signal": torch.randn(rows, LENGTH), "image": torch.randn(rows, *SHAPE)}


class TestRegistration:
    def test_the_encoder_is_registered_and_is_a_shared_encoder(self) -> None:
        assert getEncoderClass("shared_transformer_encoder_v1") is SharedTransformerEncoder
        assert "shared_transformer_encoder_v1" in listRegisteredEncoders()
        assert issubclass(SharedTransformerEncoder, AbstractSharedEncoder)

    def test_the_decoder_is_registered_and_is_a_shared_decoder(self) -> None:
        assert getDecoderClass("shared_transformer_decoder_v1") is SharedTransformerDecoder
        assert "shared_transformer_decoder_v1" in listRegisteredDecoders()
        assert issubclass(SharedTransformerDecoder, AbstractSharedDecoder)


class TestEncoder:
    def test_declares_its_modalities_in_the_order_of_the_config(self) -> None:
        assert _encoder().input_modalities == ("signal", "image")
        reversed_modalities = dict(reversed(list(ENCODER_MODALITIES.items())))
        assert _encoder(modalities=reversed_modalities).input_modalities == ("image", "signal")

    def test_latent_dim(self) -> None:
        assert _encoder().latent_dim == LATENT

    @pytest.mark.parametrize(
        "present",
        [subset for size in (1, 2) for subset in itertools.combinations(("signal", "image"), size)],
    )
    def test_encodes_every_non_empty_subset_of_its_modalities(
        self, present: tuple[str, ...]
    ) -> None:
        batch = _batch()
        mu, logvar = _encoder()({name: batch[name] for name in present})
        assert mu.shape == logvar.shape == (3, LATENT)

    def test_the_order_of_the_dict_does_not_matter(self) -> None:
        encoder = _encoder().eval()
        batch = _batch()
        forward = encoder({"signal": batch["signal"], "image": batch["image"]})
        backward_order = encoder({"image": batch["image"], "signal": batch["signal"]})
        assert torch.equal(forward[0], backward_order[0])

    def test_both_modalities_are_used_when_both_are_present(self) -> None:
        """Changing the image changes the posterior: the stream carries its tokens."""
        encoder = _encoder().eval()
        batch = _batch()
        other_image = {**batch, "image": torch.randn(3, *SHAPE)}
        assert not torch.allclose(encoder(batch)[0], encoder(other_image)[0], atol=1e-4)
        other_signal = {**batch, "signal": torch.randn(3, LENGTH)}
        assert not torch.allclose(encoder(batch)[0], encoder(other_signal)[0], atol=1e-4)

    def test_a_missing_modality_changes_the_posterior_but_not_the_shape(self) -> None:
        encoder = _encoder().eval()
        batch = _batch()
        assert not torch.allclose(encoder(batch)[0], encoder({"signal": batch["signal"]})[0])

    def test_an_absent_modality_leaves_no_trace_in_the_stream(self) -> None:
        """With the image missing, none of the image's own weights take part in the posterior."""
        encoder = _encoder().eval()
        signal = torch.randn(3, LENGTH)
        before = encoder({"signal": signal})[0]
        with torch.no_grad():
            for parameter in encoder.embeds["image"].parameters():
                parameter.add_(1.0)
            encoder.modality_embeddings["image"].add_(1.0)
        assert torch.equal(before, encoder({"signal": signal})[0])
        # ... while the same edit does change the posterior once the image is present.
        batch = {"signal": signal, "image": torch.randn(3, *SHAPE)}
        assert not torch.allclose(before, encoder(batch)[0], atol=1e-4)

    def test_gradient_reaches_the_tokenizer_of_a_present_modality_only(self) -> None:
        encoder = _encoder()
        mu, logvar = encoder({"signal": torch.randn(3, LENGTH)})
        backward(mu.sum() + logvar.sum())
        parameters = dict(encoder.named_parameters())
        assert parameters["embeds.signal.projection.weight"].grad is not None
        assert parameters["embeds.image.projection.weight"].grad is None
        assert parameters["modality_embeddings.image"].grad is None

    def test_gradient_reaches_every_parameter_when_all_modalities_are_present(self) -> None:
        encoder = _encoder()
        mu, logvar = encoder(_batch())
        backward(mu.sum() + logvar.sum())
        assert [name for name, p in encoder.named_parameters() if p.grad is None] == []

    def test_the_stream_is_one_transformer_not_one_per_modality(self) -> None:
        encoder = _encoder()
        layers = [name for name, _ in encoder.named_parameters() if "transformer.layers" in name]
        assert {name.split(".")[2] for name in layers} == {"0", "1"}  # num_layers, not 2 * layers

    @pytest.mark.parametrize("length", [1, 7, 40, 100])
    def test_any_series_length_and_any_image_size(self, length: int) -> None:
        encoder = _encoder()
        batch = {"signal": torch.randn(2, length), "image": torch.randn(2, 5 + length % 9, 21)}
        assert encoder(batch)[0].shape == (2, LATENT)

    def test_a_series_padded_by_hand_gives_the_same_posterior(self) -> None:
        encoder = _encoder().eval()
        image = torch.randn(2, *SHAPE)
        series = torch.randn(2, 1, 13)
        padded = functional.pad(series, (0, 3), mode="replicate")
        assert torch.allclose(
            encoder({"signal": series, "image": image})[0],
            encoder({"signal": padded, "image": image})[0],
            atol=1e-6,
        )

    def test_the_position_of_a_patch_matters_inside_each_modality(self) -> None:
        encoder = _encoder().eval()
        batch = _batch(2)
        signal = batch["signal"]
        swapped = torch.cat([signal[:, 8:16], signal[:, :8], signal[:, 16:]], dim=1)
        assert not torch.allclose(
            encoder(batch)[0], encoder({**batch, "signal": swapped})[0], atol=1e-4
        )

    def test_channels_are_checked_per_modality(self) -> None:
        encoder = _encoder(
            modalities={
                "signal": {"kind": "series", "in_channels": 2},
                "image": {"kind": "image", "in_channels": 3},
            }
        )
        mu, _ = encoder({"signal": torch.randn(2, 2, 30), "image": torch.randn(2, 3, 8, 8)})
        assert mu.shape == (2, LATENT)
        with pytest.raises(ValueError, match="modality 'image' was built for in_channels=3"):
            encoder({"image": torch.randn(2, 8, 8)})

    def test_default_patch_sizes_are_sixteen_for_series_and_eight_for_images(self) -> None:
        encoder = _encoder(modalities={"s": {"kind": "series"}, "i": {"kind": "image"}})
        assert encoder.embeds["s"].patch_size == 16
        assert encoder.embeds["i"].patch_size == (8, 8)

    def test_an_image_patch_can_be_rectangular(self) -> None:
        encoder = _encoder(modalities={"i": {"kind": "image", "patch_size": [2, 8]}})
        assert encoder.embeds["i"].patch_size == (2, 8)
        assert encoder({"i": torch.randn(2, 8, 16)})[0].shape == (2, LATENT)

    def test_one_modality_alone_is_a_valid_shared_encoder(self) -> None:
        encoder = _encoder(modalities={"only": {"kind": "series", "patch_size": 4}})
        assert encoder.input_modalities == ("only",)
        assert encoder({"only": torch.randn(2, 20)})[0].shape == (2, LATENT)

    def test_mean_pooling_and_the_head_mlp(self) -> None:
        encoder = _encoder(pooling="mean", head_hidden_dims=(24,))
        assert encoder.cls_token is None
        assert encoder(_batch())[0].shape == (3, LATENT)

    def test_eval_mode_is_deterministic_even_with_dropout(self) -> None:
        encoder = _encoder(dropout=0.5).eval()
        batch = _batch()
        assert torch.equal(encoder(batch)[0], encoder(batch)[0])


class TestEncoderValidation:
    def test_rejects_no_modality(self) -> None:
        with pytest.raises(ValueError, match="at least one modality"):
            _encoder(modalities={})

    def test_rejects_an_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="needs kind"):
            _encoder(modalities={"x": {"kind": "audio"}})

    def test_rejects_a_key_that_is_not_an_encoder_setting(self) -> None:
        with pytest.raises(ValueError, match="unknown key"):
            _encoder(modalities={"x": {"kind": "series", "output_length": 8}})

    def test_rejects_an_unknown_pooling(self) -> None:
        with pytest.raises(ValueError, match="Unknown pooling"):
            _encoder(pooling="max")

    def test_rejects_heads_that_do_not_divide_the_width(self) -> None:
        with pytest.raises(ValueError, match="not evenly divisible"):
            _encoder(d_model=20, num_heads=8)

    def test_an_image_modality_needs_a_width_that_is_a_multiple_of_four(self) -> None:
        with pytest.raises(ValueError, match="multiple of 4"):
            _encoder(d_model=18, num_heads=2)

    def test_series_only_accepts_a_width_that_is_merely_even(self) -> None:
        encoder = _encoder(d_model=18, num_heads=2, modalities={"s": {"kind": "series"}})
        assert encoder({"s": torch.randn(2, 20)})[0].shape == (2, LATENT)

    def test_rejects_no_input(self) -> None:
        with pytest.raises(ValueError, match="no modality"):
            _encoder()({})

    def test_rejects_a_modality_it_does_not_ingest(self) -> None:
        with pytest.raises(ValueError, match=r"\['mask'\] it does not ingest"):
            _encoder()({"signal": torch.randn(2, LENGTH), "mask": torch.randn(2, 4)})

    def test_rejects_a_batch_of_the_wrong_rank(self) -> None:
        encoder = _encoder()
        with pytest.raises(ValueError, match="modality 'signal' expects 2 or 3 dimensions"):
            encoder({"signal": torch.randn(LENGTH)})
        with pytest.raises(ValueError, match="modality 'image' expects 3 or 4 dimensions"):
            encoder({"image": torch.randn(2, 1, 1, 8, 8)})


class TestDecoder:
    def test_declares_its_modalities_in_the_order_of_the_config(self) -> None:
        assert _decoder().output_modalities == ("signal", "image")

    def test_returns_exactly_one_tensor_per_declared_modality(self) -> None:
        out = _decoder()(torch.randn(3, LATENT))
        assert list(out) == ["signal", "image"]
        assert out["signal"].shape == (3, LENGTH)
        assert out["image"].shape == (3, *SHAPE)

    def test_lengths_and_shapes_need_not_be_multiples_of_the_patch(self) -> None:
        out = _decoder(
            modalities={
                "s": {"kind": "series", "output_length": 13, "patch_size": 8},
                "i": {"kind": "image", "output_shape": [5, 7], "patch_size": [4, 4]},
            }
        )(torch.randn(2, LATENT))
        assert out["s"].shape == (2, 13)
        assert out["i"].shape == (2, 5, 7)

    def test_several_channels_keep_their_axis_and_one_is_squeezed(self) -> None:
        out = _decoder(
            modalities={
                "s": {"kind": "series", "output_length": 16, "out_channels": 2, "patch_size": 8},
                "i": {"kind": "image", "output_shape": [8, 8], "out_channels": 3, "patch_size": 4},
                "m": {"kind": "image", "output_shape": [8, 8], "patch_size": 4},
            }
        )(torch.randn(2, LATENT))
        assert out["s"].shape == (2, 2, 16)
        assert out["i"].shape == (2, 3, 8, 8)
        assert out["m"].shape == (2, 8, 8)

    def test_each_modality_depends_on_the_latent_vector(self) -> None:
        decoder = _decoder().eval()
        first, second = decoder(torch.randn(2, LATENT)), decoder(torch.randn(2, LATENT))
        for name in ("signal", "image"):
            assert not torch.allclose(first[name], second[name], atol=1e-4)

    def test_modalities_are_decoded_jointly_not_by_two_separate_decoders(self) -> None:
        """One transformer serves both: its layers appear once, whatever the modality count."""
        decoder = _decoder()
        layers = [name for name, _ in decoder.named_parameters() if "transformer.layers" in name]
        assert {name.split(".")[2] for name in layers} == {"0", "1"}

    def test_rows_of_a_batch_do_not_influence_each_other(self) -> None:
        decoder = _decoder().eval()
        z = torch.randn(3, LATENT)
        batch = decoder(z)
        for row in range(3):
            single = decoder(z[row : row + 1])
            for name in batch:
                assert torch.allclose(batch[name][row], single[name][0], atol=1e-5)

    def test_gradient_reaches_every_parameter(self) -> None:
        decoder = _decoder()
        out = decoder(torch.randn(3, LATENT))
        backward(out["signal"].sum() + out["image"].sum())
        assert [name for name, p in decoder.named_parameters() if p.grad is None] == []

    def test_a_loss_on_one_modality_alone_still_reaches_the_shared_layers(self) -> None:
        decoder = _decoder()
        backward(decoder(torch.randn(3, LATENT))["signal"].sum())
        parameters = dict(decoder.named_parameters())
        assert parameters["transformer.layers.0.linear1.weight"].grad is not None
        assert parameters["unembeds.image.projection.weight"].grad is None

    def test_eval_mode_is_deterministic_even_with_dropout(self) -> None:
        decoder = _decoder(dropout=0.5).eval()
        z = torch.randn(2, LATENT)
        assert torch.equal(decoder(z)["image"], decoder(z)["image"])


class TestDecoderValidation:
    def test_rejects_no_modality(self) -> None:
        with pytest.raises(ValueError, match="at least one modality"):
            _decoder(modalities={})

    def test_a_series_needs_output_length(self) -> None:
        with pytest.raises(ValueError, match=r"missing required key\(s\) \['output_length'\]"):
            _decoder(modalities={"s": {"kind": "series"}})

    def test_an_image_needs_output_shape(self) -> None:
        with pytest.raises(ValueError, match=r"missing required key\(s\) \['output_shape'\]"):
            _decoder(modalities={"i": {"kind": "image"}})

    def test_output_shape_needs_two_sides(self) -> None:
        with pytest.raises(ValueError, match=r"output_shape=\[height, width\]"):
            _decoder(modalities={"i": {"kind": "image", "output_shape": [8, 8, 8]}})

    def test_rejects_a_key_that_is_not_a_decoder_setting(self) -> None:
        with pytest.raises(ValueError, match="unknown key"):
            _decoder(modalities={"s": {"kind": "series", "output_length": 8, "in_channels": 1}})

    def test_rejects_an_output_size_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _decoder(modalities={"s": {"kind": "series", "output_length": 0}})
        with pytest.raises(ValueError, match="at least 1"):
            _decoder(modalities={"i": {"kind": "image", "output_shape": [0, 8]}})

    def test_rejects_heads_that_do_not_divide_the_width(self) -> None:
        with pytest.raises(ValueError, match="not evenly divisible"):
            _decoder(d_model=20, num_heads=8)

    def test_an_image_modality_needs_a_width_that_is_a_multiple_of_four(self) -> None:
        with pytest.raises(ValueError, match="multiple of 4"):
            _decoder(d_model=18, num_heads=2)

    def test_rejects_a_latent_width_below_one(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            SharedTransformerDecoder(latent_dim=0, modalities=DECODER_MODALITIES)


class TestEncoderDecoderTogether:
    def test_the_decoder_returns_what_the_encoder_read(self) -> None:
        batch = _batch()
        mu, _ = _encoder()(batch)
        out = _decoder()(mu)
        assert {name: tensor.shape for name, tensor in out.items()} == {
            name: tensor.shape for name, tensor in batch.items()
        }

    def test_an_overfit_run_lowers_the_reconstruction_error_of_both_modalities(self) -> None:
        torch.manual_seed(0)
        encoder, decoder = _encoder(), _decoder()
        rows = torch.linspace(0, 1, SHAPE[0]).view(-1, 1) * torch.linspace(1, 0, SHAPE[1])
        batch = {
            "signal": torch.sin(torch.linspace(0, 5.0, LENGTH)).repeat(4, 1),
            "image": rows.repeat(4, 1, 1),
        }
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)

        def errors() -> dict[str, torch.Tensor]:
            out = decoder(encoder(batch)[0])
            return {name: functional.mse_loss(out[name], batch[name]) for name in batch}

        first = {name: value.item() for name, value in errors().items()}
        for _ in range(80):
            optimizer.zero_grad()
            backward(sum(errors().values()))  # type: ignore[arg-type]
            optimizer.step()
        last = {name: value.item() for name, value in errors().items()}
        assert all(last[name] < 0.5 * first[name] for name in batch)
