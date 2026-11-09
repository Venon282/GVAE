"""The data contract, end to end (roadmap P5-7, ADR 0049).

A model that declares a sample rate, a vocabulary size or categorical block sizes that the loader's
`data.loader_facts` contradicts fails before the model is built, with the path of the field first:
through `loadExperimentConfig` and through `scripts/train.py`, on the shipped experiments. An
experiment that sets none of them is the experiment it was: every shipped file composes with no
block, and a declared value changes no parameter.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import LoaderFactsConfig
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]
_AUDIO_LOADER = "tests.integration._train_script_fixtures:buildDummyAudioDataloaders"
_AUDIO_ENCODER = "+model.modalities.audio.encoder.kwargs.sample_rate"
_AUDIO_DECODER = "+model.modalities.audio.decoder.kwargs.sample_rate"
_EXPERIMENTS = sorted(path.stem for path in (_CONFIG_DIR / "experiment").glob("*.yaml"))


def _load(experiment: str, *overrides: str) -> ExperimentConfig:
    return loadExperimentConfig(
        f"experiment/{experiment}", overrides=[*_DATA_OVERRIDES, *overrides]
    )


class TestAnExperimentThatSetsNeitherValueIsUnchanged:
    @pytest.mark.parametrize("experiment", _EXPERIMENTS)
    def test_every_shipped_experiment_composes_with_no_block(self, experiment: str) -> None:
        assert _load(experiment).data.loader_facts is None

    def test_a_declared_rate_changes_no_parameter_of_the_built_model(self) -> None:
        plain = _load("audio_vae")
        declared = _load("audio_vae", _AUDIO_ENCODER + "=16000", _AUDIO_DECODER + "=16000")
        torch.manual_seed(0)
        plain_model = buildModelFromConfig(plain.model)
        torch.manual_seed(0)
        declared_model = buildModelFromConfig(declared.model)
        plain_state = plain_model.state_dict()
        declared_state = declared_model.state_dict()
        assert list(plain_state) == list(declared_state)
        assert all(torch.equal(plain_state[key], declared_state[key]) for key in plain_state)

    def test_a_block_with_no_model_value_to_compare_changes_nothing(self) -> None:
        config = _load("audio_vae", "++data.loader_facts.audio.sample_rate=44100")
        assert config.data.loader_facts == {"audio": LoaderFactsConfig(sample_rate=44100)}


class TestAnAudioModelAndALoaderAtAnotherRate:
    def test_a_16_khz_model_and_a_22_05_khz_block_fail_before_the_model_is_built(self) -> None:
        with pytest.raises(ValueError) as caught:
            _load(
                "audio_vae",
                _AUDIO_ENCODER + "=16000",
                _AUDIO_DECODER + "=16000",
                "++data.loader_facts.audio.sample_rate=22050",
            )
        message = str(caught.value)
        assert message.startswith("model.modalities.audio.encoder.kwargs.sample_rate: ")
        assert "16000" in message
        assert "data.loader_facts.audio.sample_rate" in message
        assert "22050" in message

    def test_the_decoder_alone_is_enough(self) -> None:
        with pytest.raises(
            ValueError, match=r"^model\.modalities\.audio\.decoder\.kwargs\.sample_rate: "
        ):
            _load(
                "audio_vae",
                _AUDIO_DECODER + "=16000",
                "++data.loader_facts.audio.sample_rate=22050",
            )

    def test_the_same_rate_on_both_sides_loads_and_builds(self) -> None:
        config = _load(
            "audio_vae",
            _AUDIO_ENCODER + "=22050",
            _AUDIO_DECODER + "=22050",
            "++data.loader_facts.audio.sample_rate=22050",
        )
        model = buildModelFromConfig(config.model)
        assert model.encoders["audio"].sample_rate == 22050
        assert model.decoders["audio"].sample_rate == 22050

    def test_a_block_for_a_modality_the_model_does_not_have_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"^data\.loader_facts\.speech: "):
            _load("audio_vae", "++data.loader_facts.speech.sample_rate=16000")

    def test_a_value_that_is_not_a_rate_is_refused_with_its_path(self) -> None:
        with pytest.raises(ValueError, match=r"^data\.loader_facts\.audio\.sample_rate: "):
            _load("audio_vae", "++data.loader_facts.audio.sample_rate=-1")


class TestATextModelAndALoaderWithAnotherVocabulary:
    def test_the_shipped_vocabulary_agrees_with_itself(self) -> None:
        config = _load("text_vae", "++data.loader_facts.text.vocab_size=32")
        assert config.model.modalities["text"].encoder.kwargs["vocab_size"] == 32

    def test_another_vocabulary_in_the_loader_fails_on_the_encoder_first(self) -> None:
        with pytest.raises(ValueError) as caught:
            _load("text_vae", "++data.loader_facts.text.vocab_size=64")
        message = str(caught.value)
        assert message.startswith("model.modalities.text.encoder.kwargs.vocab_size: ")
        assert "declares 32" in message
        assert "64" in message

    def test_a_decoder_that_drifted_from_the_encoder_is_caught(self) -> None:
        with pytest.raises(
            ValueError, match=r"^model\.modalities\.text\.decoder\.kwargs\.vocab_size: "
        ):
            _load(
                "text_vae",
                "model.modalities.text.decoder.kwargs.vocab_size=31",
                "++data.loader_facts.text.vocab_size=32",
            )


class TestATableAndADecoderThatDisagreeAboutItsCategories:
    _DECODER = "+model.modalities.tabular.decoder.kwargs.categorical_sizes"

    def test_the_blocks_of_the_loader_and_of_the_decoder_agree(self) -> None:
        config = _load(
            "tabular_vae",
            self._DECODER + "=[3,5]",
            "++data.loader_facts.tabular.categorical_sizes=[3,5]",
        )
        assert config.data.loader_facts == {"tabular": LoaderFactsConfig(categorical_sizes=[3, 5])}

    def test_other_blocks_in_the_decoder_fail(self) -> None:
        with pytest.raises(ValueError) as caught:
            _load(
                "tabular_vae",
                self._DECODER + "=[3,4]",
                "++data.loader_facts.tabular.categorical_sizes=[3,5]",
            )
        assert str(caught.value).startswith(
            "model.modalities.tabular.decoder.kwargs.categorical_sizes: "
        )

    def test_a_decoder_with_no_blocks_for_a_table_that_has_some_fails(self) -> None:
        with pytest.raises(ValueError, match="leaves unset, so the class uses"):
            _load("tabular_vae", "++data.loader_facts.tabular.categorical_sizes=[3,5]")


def _runTrain(tmp_path: Path, *overrides: str) -> subprocess.CompletedProcess[str]:
    """Run `scripts/train.py --config-name experiment/audio_vae` on dummy batches."""
    arguments = [
        "--config-name",
        "experiment/audio_vae",
        f"data.loader_factory={_AUDIO_LOADER}",
        "data.train_path=unused",
        f"output_dir={tmp_path / 'run'}",
        "training.num_epochs=1",
        f"hydra.run.dir={tmp_path / 'hydra_run'}",
        *overrides,
    ]
    python_path = [str(_REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    return subprocess.run(
        [sys.executable, str(_TRAIN_SCRIPT), *arguments],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(python_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.slow
class TestTheTrainingScript:
    def test_a_16_khz_model_and_a_22_05_khz_block_stop_the_run_before_the_first_step(
        self, tmp_path: Path
    ) -> None:
        completed = _runTrain(
            tmp_path,
            _AUDIO_ENCODER + "=16000",
            _AUDIO_DECODER + "=16000",
            "++data.loader_facts.audio.sample_rate=22050",
        )
        assert completed.returncode != 0
        output = completed.stdout + completed.stderr
        assert "model.modalities.audio.encoder.kwargs.sample_rate: " in output
        assert "22050" in output
        assert "Starting training" not in output
        assert not (tmp_path / "run" / "metrics.csv").exists()

    def test_the_same_run_with_the_same_rate_trains(self, tmp_path: Path) -> None:
        completed = _runTrain(
            tmp_path,
            _AUDIO_ENCODER + "=22050",
            _AUDIO_DECODER + "=22050",
            "++data.loader_facts.audio.sample_rate=22050",
        )
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in completed.stdout + completed.stderr

    def test_a_run_that_sets_neither_value_trains_as_before(self, tmp_path: Path) -> None:
        completed = _runTrain(tmp_path)
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in completed.stdout + completed.stderr
