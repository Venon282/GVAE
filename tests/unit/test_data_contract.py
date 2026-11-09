"""The facts the loader owns against what the model declares (roadmap P5-7, ADR 0049).

`checkDataContract` is a pure function of two configs, so everything here builds the configs in
Python and builds no model. The runs that go through `loadExperimentConfig` and `scripts/train.py`
are in `tests/integration/test_data_contract.py`.
"""

import dataclasses
from typing import Any

import pytest
from omegaconf import OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import DataConfig, LoaderFactsConfig
from global_vae.config.data_contract import FACT_NAMES, checkDataContract, validateLoaderFacts
from global_vae.config.model import DecoderConfig, EncoderConfig, ModalityConfig, ModelConfig


def _model(**modalities: tuple[str, dict[str, Any], str, dict[str, Any]]) -> ModelConfig:
    """A model config from `name=(encoder, encoder kwargs, decoder, decoder kwargs)`."""
    return ModelConfig(
        modalities={
            name: ModalityConfig(
                encoder=EncoderConfig(name=encoder, kwargs=dict(encoder_kwargs)),
                decoder=DecoderConfig(name=decoder, kwargs=dict(decoder_kwargs)),
            )
            for name, (encoder, encoder_kwargs, decoder, decoder_kwargs) in modalities.items()
        }
    )


def _audio(rate: int | None = 16000, encoder_rate: int | None = None) -> ModelConfig:
    """An audio model that declares `rate` on its decoder (and `encoder_rate` on its encoder)."""
    encoder_kwargs: dict[str, Any] = {} if encoder_rate is None else {"sample_rate": encoder_rate}
    decoder_kwargs: dict[str, Any] = {"output_length": 2048}
    if rate is not None:
        decoder_kwargs["sample_rate"] = rate
    return _model(
        audio=(
            "audio_waveform_encoder_v1",
            encoder_kwargs,
            "audio_waveform_decoder_v1",
            decoder_kwargs,
        )
    )


def _text(encoder_vocab: int = 32, decoder_vocab: int = 32) -> ModelConfig:
    return _model(
        text=(
            "text_transformer_encoder_v1",
            {"vocab_size": encoder_vocab},
            "text_transformer_decoder_v1",
            {"vocab_size": decoder_vocab, "output_length": 8},
        )
    )


def _table(sizes: list[int] | tuple[int, ...] | None = (3, 5)) -> ModelConfig:
    decoder_kwargs: dict[str, Any] = {"num_features": 16}
    if sizes is not None:
        decoder_kwargs["categorical_sizes"] = sizes
    return _model(
        tabular=(
            "tabular_mlp_encoder_v1",
            {"num_features": 16},
            "tabular_mlp_decoder_v1",
            decoder_kwargs,
        )
    )


def _data(**facts_by_modality: dict[str, Any]) -> DataConfig:
    return DataConfig(
        loader_factory="unused.module:unused",
        train_path="unused",
        loader_facts={
            name: LoaderFactsConfig(**facts) for name, facts in facts_by_modality.items()
        },
    )


class TestTheBlock:
    def test_it_is_optional_and_empty_by_default(self) -> None:
        assert DataConfig(loader_factory="a:b", train_path="x").loader_facts is None

    def test_it_has_one_field_per_fact_and_each_is_optional(self) -> None:
        names = [field.name for field in dataclasses.fields(LoaderFactsConfig)]
        assert names == list(FACT_NAMES)
        assert LoaderFactsConfig() == LoaderFactsConfig(None, None, None)

    def test_it_is_materialized_as_typed_instances_from_a_yaml_mapping(self) -> None:
        text = """
        loader_factory: unused.module:unused
        train_path: unused
        loader_facts:
          audio: {sample_rate: 22050}
          text: {vocab_size: 5000}
          tabular: {categorical_sizes: [3, 5]}
        """
        merged = OmegaConf.merge(OmegaConf.structured(DataConfig), OmegaConf.create(text))
        data = OmegaConf.to_object(merged)
        assert isinstance(data, DataConfig)
        assert data.loader_facts == {
            "audio": LoaderFactsConfig(sample_rate=22050),
            "text": LoaderFactsConfig(vocab_size=5000),
            "tabular": LoaderFactsConfig(categorical_sizes=[3, 5]),
        }

    def test_a_field_that_is_not_a_fact_is_refused_by_the_schema(self) -> None:
        text = """
        loader_facts:
          audio: {sampling_rate: 22050}
        """
        with pytest.raises(Exception, match="sampling_rate"):
            OmegaConf.merge(OmegaConf.structured(DataConfig), OmegaConf.create(text))


class TestTheValuesOfTheBlock:
    @pytest.mark.parametrize("fact", ["sample_rate", "vocab_size"])
    @pytest.mark.parametrize("bad", [0, -1, 1.5, "16000", True])
    def test_a_number_that_is_not_a_positive_integer_is_refused_with_its_path(
        self, fact: str, bad: object
    ) -> None:
        data = _data(audio={})
        facts = data.loader_facts["audio"]  # type: ignore[index]
        setattr(facts, fact, bad)
        with pytest.raises(
            ValueError, match=rf"^data\.loader_facts\.audio\.{fact}: expected a positive integer"
        ):
            validateLoaderFacts(data)

    @pytest.mark.parametrize("bad", [[], [1, 3], [3, 0], [3, "5"], [3, True], "35", 3])
    def test_categorical_sizes_must_be_a_list_of_integers_of_at_least_two(
        self, bad: object
    ) -> None:
        data = _data(tabular={})
        data.loader_facts["tabular"].categorical_sizes = bad  # type: ignore[index]
        with pytest.raises(
            ValueError,
            match=r"^data\.loader_facts\.tabular\.categorical_sizes: expected a non-empty list",
        ):
            validateLoaderFacts(data)

    def test_an_entry_that_is_not_a_block_is_refused(self) -> None:
        data = _data()
        data.loader_facts = {"audio": {"sample_rate": 16000}}  # type: ignore[dict-item]
        with pytest.raises(ValueError, match=r"^data\.loader_facts\.audio: expected a block"):
            validateLoaderFacts(data)

    def test_the_values_are_checked_before_any_model_is_read(self) -> None:
        data = _data(audio={"sample_rate": -1})
        with pytest.raises(ValueError, match=r"^data\.loader_facts\.audio\.sample_rate"):
            checkDataContract(ModelConfig(), data)

    def test_valid_values_pass(self) -> None:
        validateLoaderFacts(
            _data(
                audio={"sample_rate": 22050},
                text={"vocab_size": 5000},
                tabular={"categorical_sizes": [2, 7]},
            )
        )

    def test_the_prefix_starts_the_path(self) -> None:
        data = _data(audio={"sample_rate": -1})
        with pytest.raises(ValueError, match=r"^cfg\.data\.loader_facts\.audio\.sample_rate"):
            validateLoaderFacts(data, prefix="cfg.data")


class TestAnExperimentThatSetsNoFact:
    def test_a_missing_block_is_not_checked_and_the_model_is_not_read(self) -> None:
        """No block, no comparison: even a model that could not be resolved passes."""
        data = DataConfig(loader_factory="a:b", train_path="x")
        checkDataContract(ModelConfig(), data)
        checkDataContract(_audio(rate=16000), data)

    def test_an_empty_block_is_not_checked_either(self) -> None:
        data = _data()
        assert data.loader_facts == {}
        checkDataContract(ModelConfig(), data)

    def test_a_block_with_no_value_in_it_compares_nothing(self) -> None:
        checkDataContract(_audio(rate=16000), _data(audio={}))


class TestSampleRate:
    def test_a_model_and_a_loader_at_the_same_rate_agree(self) -> None:
        checkDataContract(_audio(rate=16000), _data(audio={"sample_rate": 16000}))

    def test_a_16_khz_model_and_a_22_05_khz_loader_disagree_with_the_model_field_first(
        self,
    ) -> None:
        with pytest.raises(ValueError) as caught:
            checkDataContract(_audio(rate=16000), _data(audio={"sample_rate": 22050}))
        message = str(caught.value)
        assert message.startswith("model.modalities.audio.decoder.kwargs.sample_rate: ")
        assert "declares 16000" in message
        assert "data.loader_facts.audio.sample_rate" in message
        assert "22050" in message

    def test_the_encoder_is_compared_too(self) -> None:
        model = _audio(rate=22050, encoder_rate=16000)
        with pytest.raises(
            ValueError, match=r"^model\.modalities\.audio\.encoder\.kwargs\.sample_rate: "
        ):
            checkDataContract(model, _data(audio={"sample_rate": 22050}))

    def test_a_model_that_declares_no_rate_is_not_compared(self) -> None:
        checkDataContract(_audio(rate=None), _data(audio={"sample_rate": 22050}))

    def test_a_null_rate_in_the_model_means_not_declared(self) -> None:
        model = _audio(rate=None)
        model.modalities["audio"].decoder.kwargs["sample_rate"] = None
        checkDataContract(model, _data(audio={"sample_rate": 22050}))

    def test_a_float_that_is_a_whole_number_is_the_same_rate(self) -> None:
        model = _audio(rate=None)
        model.modalities["audio"].decoder.kwargs["sample_rate"] = 16000.0
        checkDataContract(model, _data(audio={"sample_rate": 16000}))


class TestVocabularySize:
    def test_a_model_and_a_loader_with_the_same_vocabulary_agree(self) -> None:
        checkDataContract(_text(32, 32), _data(text={"vocab_size": 32}))

    def test_the_encoder_is_compared(self) -> None:
        with pytest.raises(ValueError) as caught:
            checkDataContract(
                _text(encoder_vocab=64, decoder_vocab=32), _data(text={"vocab_size": 32})
            )
        message = str(caught.value)
        assert message.startswith("model.modalities.text.encoder.kwargs.vocab_size: ")
        assert "declares 64" in message
        assert "data.loader_facts.text.vocab_size" in message
        assert "32" in message

    def test_the_decoder_is_compared(self) -> None:
        with pytest.raises(
            ValueError, match=r"^model\.modalities\.text\.decoder\.kwargs\.vocab_size: "
        ):
            checkDataContract(_text(32, 31), _data(text={"vocab_size": 32}))


class TestCategoricalSizes:
    def test_the_same_blocks_agree(self) -> None:
        checkDataContract(_table([3, 5]), _data(tabular={"categorical_sizes": [3, 5]}))

    def test_a_tuple_and_a_list_are_the_same_blocks(self) -> None:
        checkDataContract(_table((3, 5)), _data(tabular={"categorical_sizes": [3, 5]}))

    def test_other_blocks_disagree(self) -> None:
        with pytest.raises(ValueError) as caught:
            checkDataContract(_table([3, 4]), _data(tabular={"categorical_sizes": [3, 5]}))
        message = str(caught.value)
        assert message.startswith("model.modalities.tabular.decoder.kwargs.categorical_sizes: ")
        assert "[3, 4]" in message
        assert "[3, 5]" in message

    def test_the_same_blocks_in_another_order_disagree(self) -> None:
        with pytest.raises(ValueError, match="categorical_sizes"):
            checkDataContract(_table([5, 3]), _data(tabular={"categorical_sizes": [3, 5]}))

    def test_a_decoder_that_leaves_them_out_treats_every_column_as_numeric(self) -> None:
        """The class default is `()`, a value: the table has blocks and the decoder has none."""
        with pytest.raises(ValueError) as caught:
            checkDataContract(_table(None), _data(tabular={"categorical_sizes": [3, 5]}))
        message = str(caught.value)
        assert message.startswith("model.modalities.tabular.decoder.kwargs.categorical_sizes: ")
        assert "leaves unset, so the class uses ()" in message

    def test_only_the_decoder_has_the_argument_so_the_encoder_is_skipped(self) -> None:
        checkDataContract(_table([3, 5]), _data(tabular={"categorical_sizes": [3, 5]}))


class TestWhatIsNotCompared:
    def test_a_required_argument_that_is_left_out_is_left_to_the_class(self) -> None:
        """`vocab_size` has no default: the build raises its own error, nothing to compare."""
        model = _model(
            text=(
                "text_transformer_encoder_v1",
                {},
                "text_transformer_decoder_v1",
                {"output_length": 8},
            )
        )
        checkDataContract(model, _data(text={"vocab_size": 32}))

    def test_a_fact_that_no_class_of_the_modality_takes_is_not_compared(self) -> None:
        model = _model(image=("2d_cnn_encoder_v1", {}, "2d_cnn_decoder_v1", {}))
        checkDataContract(model, _data(image={"sample_rate": 16000}))

    def test_an_unknown_registry_name_is_left_to_the_model_builder(self) -> None:
        model = _model(audio=("no_such_encoder", {}, "no_such_decoder", {"sample_rate": 16000}))
        with pytest.raises(
            ValueError, match=r"^model\.modalities\.audio\.decoder\.kwargs\.sample_rate"
        ):
            checkDataContract(model, _data(audio={"sample_rate": 22050}))
        checkDataContract(model, _data(audio={"sample_rate": 16000}))


class TestTheModalityOfAnEntry:
    def test_a_name_that_is_not_in_the_model_is_refused_with_the_names_that_are(self) -> None:
        with pytest.raises(ValueError) as caught:
            checkDataContract(_audio(), _data(autio={"sample_rate": 16000}))
        message = str(caught.value)
        assert message.startswith("data.loader_facts.autio: ")
        assert "['audio']" in message

    def test_the_encoders_and_decoders_sections_name_modalities_too(self) -> None:
        model = ModelConfig(
            encoders={
                "text": EncoderConfig(name="text_transformer_encoder_v1", kwargs={"vocab_size": 64})
            },
            decoders={
                "text": DecoderConfig(
                    name="text_transformer_decoder_v1",
                    kwargs={"vocab_size": 32, "output_length": 8},
                )
            },
        )
        with pytest.raises(ValueError, match=r"^model\.encoders\.text\.kwargs\.vocab_size: "):
            checkDataContract(model, _data(text={"vocab_size": 32}))

    def test_a_decoder_with_no_encoder_of_its_name_still_counts(self) -> None:
        model = ModelConfig(
            decoders={
                "audio_out": DecoderConfig(
                    name="audio_waveform_decoder_v1",
                    kwargs={"output_length": 2048, "sample_rate": 16000},
                )
            },
        )
        with pytest.raises(ValueError, match=r"^model\.decoders\.audio_out\.kwargs\.sample_rate: "):
            checkDataContract(model, _data(audio_out={"sample_rate": 22050}))

    def test_two_modalities_are_each_compared_with_their_own_facts(self) -> None:
        model = ModelConfig(
            modalities={
                **_audio(rate=16000).modalities,
                **_text(32, 32).modalities,
            }
        )
        data = _data(audio={"sample_rate": 16000}, text={"vocab_size": 32})
        checkDataContract(model, data)
        data.loader_facts["text"].vocab_size = 33  # type: ignore[index]
        with pytest.raises(
            ValueError, match=r"^model\.modalities\.text\.encoder\.kwargs\.vocab_size"
        ):
            checkDataContract(model, data)

    def test_the_prefixes_start_the_paths(self) -> None:
        with pytest.raises(
            ValueError, match=r"^cfg\.model\.modalities\.audio\.decoder\.kwargs\.sample_rate"
        ):
            checkDataContract(
                _audio(16000),
                _data(audio={"sample_rate": 22050}),
                model_prefix="cfg.model",
                data_prefix="cfg.data",
            )
        with pytest.raises(ValueError, match=r"^cfg\.data\.loader_facts\.autio"):
            checkDataContract(
                _audio(16000),
                _data(autio={"sample_rate": 22050}),
                model_prefix="cfg.model",
                data_prefix="cfg.data",
            )
