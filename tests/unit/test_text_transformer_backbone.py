"""Unit tests for the text transformer encoder and decoder and their blocks (roadmap P3-8)."""

import pytest
import torch
import torch.nn.functional as functional

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders
from global_vae.decoders.text_transformer_decoder import TextTransformerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders
from global_vae.encoders.text_transformer_encoder import TextTransformerEncoder
from global_vae.utils.autograd import backward
from global_vae.utils.text_blocks import (
    TokenEmbedding,
    computeMaskedMean,
    computePaddingMask,
    computeTokenCrossEntropy,
)

LATENT = 4
VOCAB = 12
LENGTH = 7
PAD = 0
SMALL = {"d_model": 16, "num_heads": 2, "num_layers": 2}


def _encoder(**kwargs: object) -> TextTransformerEncoder:
    settings = {**SMALL, "latent_dim": LATENT, "vocab_size": VOCAB, **kwargs}
    return TextTransformerEncoder(**settings)  # type: ignore[arg-type]


def _decoder(**kwargs: object) -> TextTransformerDecoder:
    settings = {
        **SMALL,
        "latent_dim": LATENT,
        "vocab_size": VOCAB,
        "output_length": LENGTH,
        **kwargs,
    }
    return TextTransformerDecoder(**settings)  # type: ignore[arg-type]


def _ids(rows: int = 3, length: int = LENGTH, seed: int = 0) -> torch.Tensor:
    """Random ids of the real tokens, `1` to `VOCAB - 1`, so `0` stays the padding token."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (rows, length), generator=generator)


def _oneHot(ids: torch.Tensor) -> torch.Tensor:
    return functional.one_hot(ids, VOCAB).float()


class TestRegistration:
    def test_the_encoder_is_registered_under_its_name(self) -> None:
        assert getEncoderClass("text_transformer_encoder_v1") is TextTransformerEncoder
        assert "text_transformer_encoder_v1" in listRegisteredEncoders()

    def test_the_decoder_is_registered_under_its_name(self) -> None:
        assert getDecoderClass("text_transformer_decoder_v1") is TextTransformerDecoder
        assert "text_transformer_decoder_v1" in listRegisteredDecoders()

    def test_they_implement_the_one_modality_interfaces(self) -> None:
        assert issubclass(TextTransformerEncoder, AbstractEncoder)
        assert issubclass(TextTransformerDecoder, AbstractDecoder)

    def test_the_modality_name_defaults_to_text_and_can_be_changed(self) -> None:
        assert _encoder().modality_name == _decoder().modality_name == "text"
        assert _encoder(modality_name="review").modality_name == "review"
        assert _decoder(modality_name="review").modality_name == "review"


class TestTokenEmbedding:
    def test_ids_and_one_hot_rows_give_the_same_vectors(self) -> None:
        embedding = TokenEmbedding(VOCAB, 8)
        ids = _ids()
        assert torch.allclose(embedding(ids), embedding(_oneHot(ids)), atol=1e-6)

    def test_the_output_is_one_vector_per_token(self) -> None:
        assert TokenEmbedding(VOCAB, 8)(_ids(2, 5)).shape == (2, 5, 8)

    def test_a_soft_row_is_the_weighted_sum_of_the_vectors(self) -> None:
        embedding = TokenEmbedding(VOCAB, 8)
        soft = torch.zeros(1, 1, VOCAB)
        soft[0, 0, 3], soft[0, 0, 5] = 0.25, 0.75
        expected = 0.25 * embedding.table.weight[3] + 0.75 * embedding.table.weight[5]
        assert torch.allclose(embedding(soft)[0, 0], expected, atol=1e-6)

    def test_an_int32_tensor_is_read_as_ids(self) -> None:
        embedding = TokenEmbedding(VOCAB, 8)
        ids = _ids()
        assert torch.equal(embedding(ids.to(torch.int32)), embedding(ids))

    def test_a_float_length_by_batch_tensor_is_refused_not_read_as_ids(self) -> None:
        with pytest.raises(ValueError, match="one-hot"):
            TokenEmbedding(VOCAB, 8)(torch.rand(2, 5))

    def test_a_one_hot_tensor_of_another_vocabulary_is_refused(self) -> None:
        with pytest.raises(ValueError, match=f"vocab_size={VOCAB}"):
            TokenEmbedding(VOCAB, 8)(torch.zeros(2, 5, VOCAB + 1))

    def test_ids_of_the_wrong_rank_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\(batch, length\)"):
            TokenEmbedding(VOCAB, 8)(torch.zeros(2, 5, 3, dtype=torch.long))

    def test_a_dtype_that_is_neither_ids_nor_float_is_refused(self) -> None:
        with pytest.raises(ValueError, match="int32 or int64"):
            TokenEmbedding(VOCAB, 8)(torch.zeros(2, 5, dtype=torch.bool))

    @pytest.mark.parametrize(("vocab_size", "d_model"), [(1, 8), (0, 8), (VOCAB, 0)])
    def test_a_bad_constructor_argument_is_refused_and_named(
        self, vocab_size: int, d_model: int
    ) -> None:
        with pytest.raises(ValueError, match=r"vocab_size|d_model"):
            TokenEmbedding(vocab_size, d_model)


class TestPaddingMask:
    def test_no_pad_id_means_no_mask(self) -> None:
        assert computePaddingMask(_ids(), None, VOCAB) is None

    def test_ids_equal_to_the_pad_id_are_padding(self) -> None:
        ids = torch.tensor([[4, 5, PAD, PAD], [PAD, 6, 7, PAD]])
        mask = computePaddingMask(ids, PAD, VOCAB)
        assert mask is not None
        assert mask.tolist() == [[False, False, True, True], [True, False, False, True]]

    def test_a_one_hot_row_is_read_like_its_id(self) -> None:
        ids = torch.tensor([[4, 5, PAD, PAD], [PAD, 6, 7, PAD]])
        assert torch.equal(
            computePaddingMask(_oneHot(ids), PAD, VOCAB),  # type: ignore[arg-type]
            computePaddingMask(ids, PAD, VOCAB),  # type: ignore[arg-type]
        )

    def test_a_row_of_zeros_is_padding_too(self) -> None:
        one_hot = _oneHot(torch.tensor([[4, 5, 6]]))
        one_hot[0, 2] = 0.0
        mask = computePaddingMask(one_hot, PAD, VOCAB)
        assert mask is not None
        assert mask.tolist() == [[False, False, True]]

    def test_a_soft_row_is_padding_when_most_of_its_mass_is_the_pad_class(self) -> None:
        soft = torch.zeros(1, 2, VOCAB)
        soft[0, 0, PAD], soft[0, 0, 4] = 0.7, 0.3
        soft[0, 1, PAD], soft[0, 1, 4] = 0.3, 0.7
        mask = computePaddingMask(soft, PAD, VOCAB)
        assert mask is not None
        assert mask.tolist() == [[True, False]]

    def test_an_empty_text_keeps_its_first_position_so_attention_has_a_key(self) -> None:
        mask = computePaddingMask(torch.full((2, 4), PAD), PAD, VOCAB)
        assert mask is not None
        assert mask.tolist() == [[False, True, True, True]] * 2

    def test_a_pad_id_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(ValueError, match="pad_token_id"):
            computePaddingMask(_ids(), VOCAB, VOCAB)


class TestMaskedMean:
    def test_without_a_mask_it_is_the_plain_mean(self) -> None:
        encoded = torch.randn(2, 5, 3)
        assert torch.allclose(computeMaskedMean(encoded, None), encoded.mean(dim=1))

    def test_padded_positions_are_left_out(self) -> None:
        encoded = torch.randn(1, 4, 3)
        mask = torch.tensor([[False, False, True, True]])
        assert torch.allclose(computeMaskedMean(encoded, mask)[0], encoded[0, :2].mean(dim=0))

    def test_a_mask_of_nothing_is_the_plain_mean(self) -> None:
        encoded = torch.randn(2, 5, 3)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        assert torch.allclose(computeMaskedMean(encoded, mask), encoded.mean(dim=1), atol=1e-6)


class TestTokenCrossEntropy:
    def test_it_is_the_summed_cross_entropy_of_the_tokens_averaged_over_the_texts(self) -> None:
        scores = torch.randn(5, LENGTH, VOCAB)
        ids = torch.randint(0, VOCAB, (5, LENGTH))
        expected = functional.cross_entropy(scores.transpose(1, 2), ids, reduction="sum") / 5
        assert torch.allclose(computeTokenCrossEntropy(scores, _oneHot(ids)), expected, atol=1e-4)

    def test_equal_scores_cost_log_vocab_size_per_token(self) -> None:
        loss = computeTokenCrossEntropy(
            torch.zeros(3, LENGTH, VOCAB), _oneHot(torch.randint(0, VOCAB, (3, LENGTH)))
        )
        assert loss.item() == pytest.approx(LENGTH * torch.tensor(float(VOCAB)).log().item())

    def test_a_row_of_zeros_adds_nothing(self) -> None:
        scores = torch.randn(2, 3, VOCAB)
        target = _oneHot(torch.randint(0, VOCAB, (2, 3)))
        padded = torch.cat([target, torch.zeros(2, 2, VOCAB)], dim=1)
        longer = torch.cat([scores, torch.randn(2, 2, VOCAB)], dim=1)
        assert torch.allclose(
            computeTokenCrossEntropy(scores, target), computeTokenCrossEntropy(longer, padded)
        )

    def test_a_soft_target_is_the_weighted_cross_entropy_of_its_classes(self) -> None:
        scores = torch.randn(1, 1, VOCAB)
        soft = torch.zeros(1, 1, VOCAB)
        soft[0, 0, 2], soft[0, 0, 7] = 0.25, 0.75
        log_probabilities = torch.log_softmax(scores, dim=-1)[0, 0]
        expected = -(0.25 * log_probabilities[2] + 0.75 * log_probabilities[7])
        assert torch.allclose(computeTokenCrossEntropy(scores, soft), expected, atol=1e-6)

    def test_it_is_differentiable_and_prefers_the_right_token(self) -> None:
        ids = torch.randint(0, VOCAB, (4, LENGTH))
        right = (_oneHot(ids) * 10.0).requires_grad_(True)
        wrong = (_oneHot((ids + 1) % VOCAB) * 10.0).requires_grad_(True)
        loss_right = computeTokenCrossEntropy(right, _oneHot(ids))
        loss_wrong = computeTokenCrossEntropy(wrong, _oneHot(ids))
        backward(loss_right)
        assert right.grad is not None
        assert loss_right < loss_wrong

    @pytest.mark.parametrize(
        ("scores_shape", "target_shape"),
        [((2, LENGTH, VOCAB), (2, LENGTH, VOCAB + 1)), ((2, VOCAB), (2, VOCAB))],
    )
    def test_a_shape_that_is_not_a_matching_text_batch_is_refused(
        self, scores_shape: tuple[int, ...], target_shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(ValueError, match=r"\(batch, length, vocab_size\)"):
            computeTokenCrossEntropy(torch.zeros(scores_shape), torch.zeros(target_shape))


class TestEncoder:
    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_output_shapes(self, pooling: str) -> None:
        mu, logvar = _encoder(pooling=pooling)(_oneHot(_ids(5)))
        assert mu.shape == logvar.shape == (5, LATENT)

    def test_a_batch_of_one_text_of_one_token_is_encoded(self) -> None:
        mu, _ = _encoder()(_ids(1, 1))
        assert mu.shape == (1, LATENT)

    def test_ids_and_one_hot_give_the_same_output(self) -> None:
        encoder = _encoder(pad_token_id=PAD).eval()
        ids = _ids()
        mu_ids, logvar_ids = encoder(ids)
        mu_one_hot, logvar_one_hot = encoder(_oneHot(ids))
        assert torch.allclose(mu_ids, mu_one_hot, atol=1e-5)
        assert torch.allclose(logvar_ids, logvar_one_hot, atol=1e-5)

    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_padding_a_text_does_not_change_its_encoding(self, pooling: str) -> None:
        """The point of `pad_token_id`: the length a loader pads to is not information."""
        encoder = _encoder(pooling=pooling, pad_token_id=PAD).eval()
        ids = _ids(4, 5)
        padded = functional.pad(ids, (0, 6), value=PAD)
        mu, logvar = encoder(ids)
        mu_padded, logvar_padded = encoder(padded)
        assert torch.allclose(mu, mu_padded, atol=1e-5)
        assert torch.allclose(logvar, logvar_padded, atol=1e-5)
        mu_one_hot, _ = encoder(_oneHot(padded))
        assert torch.allclose(mu, mu_one_hot, atol=1e-5)

    def test_without_pad_token_id_padding_is_read_like_any_token(self) -> None:
        encoder = _encoder().eval()
        ids = _ids(2, 5)
        padded = functional.pad(ids, (0, 6), value=PAD)
        assert not torch.allclose(encoder(ids)[0], encoder(padded)[0], atol=1e-4)

    def test_a_padded_text_that_differs_in_a_real_token_is_encoded_differently(self) -> None:
        encoder = _encoder(pad_token_id=PAD).eval()
        ids = _ids(2, 5)
        changed = ids.clone()
        changed[:, 0] = torch.where(ids[:, 0] == 1, 2, 1)
        assert not torch.allclose(encoder(ids)[0], encoder(changed)[0], atol=1e-4)

    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_an_empty_text_gives_a_finite_encoding(self, pooling: str) -> None:
        encoder = _encoder(pooling=pooling, pad_token_id=PAD)
        mu, logvar = encoder(torch.full((2, 6), PAD))
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_a_text_is_encoded_the_same_in_training_and_eval_without_dropout(self) -> None:
        """Guards the masked fast path of `nn.TransformerEncoder` in eval mode."""
        encoder = _encoder(pad_token_id=PAD)
        padded = functional.pad(_ids(3, 4), (0, 3), value=PAD)
        encoder.train()
        trained, _ = encoder(padded)
        encoder.eval()
        with torch.no_grad():
            evaluated, _ = encoder(padded)
        assert torch.allclose(trained, evaluated, atol=1e-5)

    def test_sinusoidal_positions_have_no_length_limit(self) -> None:
        encoder = _encoder()
        for length in (1, 3, 200):
            assert encoder(_ids(2, length))[0].shape == (2, LATENT)

    def test_learned_positions_cover_up_to_max_length(self) -> None:
        encoder = _encoder(positional_encoding="learned", max_length=LENGTH)
        assert encoder(_ids(2, LENGTH))[0].shape == (2, LATENT)
        assert encoder(_ids(2, 3))[0].shape == (2, LATENT)
        with pytest.raises(ValueError, match="learned positions"):
            encoder(_ids(2, LENGTH + 1))

    def test_learned_positions_need_max_length(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            _encoder(positional_encoding="learned")

    def test_the_head_can_have_hidden_layers(self) -> None:
        mu, _ = _encoder(head_hidden_dims=[8, 8])(_ids())
        assert mu.shape == (3, LATENT)

    def test_it_reports_its_latent_width_and_a_minimal_length_of_one(self) -> None:
        encoder = _encoder()
        assert encoder.latent_dim == LATENT
        assert encoder.minimal_input_length == 1

    @pytest.mark.parametrize(
        "x",
        [
            torch.rand(3, 5),
            torch.zeros(3, 5, VOCAB + 1),
            torch.zeros(3, 5, 2, 2, VOCAB),
            torch.zeros(3, 5, 4, dtype=torch.long),
        ],
        ids=["float-2d", "wrong-vocabulary", "float-5d", "ids-3d"],
    )
    def test_an_input_that_is_not_a_token_batch_is_refused(self, x: torch.Tensor) -> None:
        with pytest.raises(ValueError, match=r"one-hot|vocab_size|ids"):
            _encoder()(x)

    def test_a_text_with_no_token_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            _encoder()(torch.zeros(3, 0, dtype=torch.long))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"pooling": "max"}, "pooling"),
            ({"latent_dim": 0}, "latent_dim"),
            ({"vocab_size": 1}, "vocab_size"),
            ({"pad_token_id": VOCAB}, "pad_token_id"),
            ({"pad_token_id": -1}, "pad_token_id"),
            ({"num_heads": 3}, "divisible"),
            ({"num_layers": 0}, "num_layers"),
            ({"dropout": 1.0}, "dropout"),
        ],
    )
    def test_a_bad_constructor_argument_is_refused_and_named(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _encoder(**kwargs)

    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_every_parameter_receives_a_gradient(self, pooling: str) -> None:
        encoder = _encoder(pooling=pooling, pad_token_id=PAD, head_hidden_dims=[8])
        mu, logvar = encoder(_ids())
        backward((mu**2).sum() + (logvar**2).sum())
        assert [n for n, p in encoder.named_parameters() if p.grad is None] == []

    def test_dropout_is_off_in_eval_mode(self) -> None:
        encoder = _encoder(dropout=0.5).eval()
        ids = _ids()
        assert torch.equal(encoder(ids)[0], encoder(ids)[0])


class TestDecoder:
    def test_output_shape_is_length_by_vocabulary(self) -> None:
        assert _decoder()(torch.randn(5, LATENT)).shape == (5, LENGTH, VOCAB)

    def test_a_batch_of_one_is_decoded(self) -> None:
        assert _decoder()(torch.randn(1, LATENT)).shape == (1, LENGTH, VOCAB)

    def test_the_default_output_is_a_distribution_over_the_vocabulary_at_each_position(
        self,
    ) -> None:
        probabilities = _decoder()(torch.randn(4, LATENT))
        assert (probabilities >= 0).all()
        assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4, LENGTH), atol=1e-5)

    def test_no_output_activation_gives_raw_scores_whose_softmax_is_the_default(self) -> None:
        z = torch.randn(4, LATENT)
        torch.manual_seed(0)
        probabilities = _decoder()(z)
        torch.manual_seed(0)  # the same weights for both
        scores = _decoder(output_activation=None)(z)
        assert not torch.allclose(scores.sum(dim=-1), torch.ones(4, LENGTH), atol=1e-3)
        assert torch.allclose(torch.softmax(scores, dim=-1), probabilities, atol=1e-6)

    @pytest.mark.parametrize("length", [1, 2, 7, 100])
    def test_any_output_length_is_reached(self, length: int) -> None:
        assert _decoder(output_length=length)(torch.randn(2, LATENT)).shape == (2, length, VOCAB)

    def test_two_latent_vectors_give_different_texts(self) -> None:
        """The latent vector is the only input, so it cannot be ignored (no posterior collapse)."""
        decoder = _decoder().eval()
        first, second = decoder(torch.randn(1, LATENT)), decoder(torch.randn(1, LATENT))
        assert not torch.allclose(first, second, atol=1e-4)

    def test_learned_positions_are_one_vector_per_output_token(self) -> None:
        decoder = _decoder(positional_encoding="learned")
        assert decoder.positions.table is not None
        assert decoder.positions.table.shape == (LENGTH, SMALL["d_model"])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"output_activation": "tanh"}, "output_activation"),
            ({"latent_dim": 0}, "latent_dim"),
            ({"vocab_size": 1}, "vocab_size"),
            ({"output_length": 0}, "output_length"),
            ({"num_heads": 3}, "divisible"),
            ({"positional_encoding": "rotary"}, "positional"),
        ],
    )
    def test_a_bad_constructor_argument_is_refused_and_named(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _decoder(**kwargs)

    def test_every_parameter_receives_a_gradient(self) -> None:
        decoder = _decoder()
        backward((decoder(torch.randn(3, LATENT)) ** 2).sum())
        assert [n for n, p in decoder.named_parameters() if p.grad is None] == []


class TestEncoderAndDecoderTogether:
    def test_a_text_goes_through_both_and_comes_back_at_its_shape(self) -> None:
        encoder, decoder = _encoder(pad_token_id=PAD), _decoder()
        one_hot = _oneHot(_ids())
        mu, _ = encoder(one_hot)
        assert decoder(mu).shape == one_hot.shape

    def test_the_decoder_output_can_be_encoded_again(self) -> None:
        """Probabilities are a legal encoder input, so a reconstruction can be re-encoded."""
        encoder, decoder = _encoder(pad_token_id=PAD), _decoder()
        mu, _ = encoder(_oneHot(_ids()))
        again, _ = encoder(decoder(mu))
        assert again.shape == mu.shape
        assert torch.isfinite(again).all()

    def test_a_tiny_model_learns_to_copy_a_few_texts(self) -> None:
        """End to end through the encoder, the decoder and a squared error on probabilities."""
        torch.manual_seed(0)
        encoder, decoder = _encoder(pad_token_id=PAD), _decoder()
        ids = functional.pad(_ids(6, 5, seed=3), (0, LENGTH - 5), value=PAD)
        target = _oneHot(ids)
        parameters = [*encoder.parameters(), *decoder.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=3e-3)
        for _ in range(300):
            optimizer.zero_grad()
            mu, _ = encoder(target)
            loss = functional.binary_cross_entropy(decoder(mu), target)
            backward(loss)
            optimizer.step()
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            predicted = decoder(encoder(target)[0]).argmax(dim=-1)
        assert (predicted == ids).float().mean().item() > 0.95
