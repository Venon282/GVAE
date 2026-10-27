"""The audio model config and its experiment (roadmap P3-8, spec §6 and §10).

`configs/model/audio_waveform_single_latent.yaml` is the third modality added through the
registries after the 1D signal and the image, and the second proof, with the tabular one, that the
core does not change. `test_model_config_schema.py` already checks that every shipped file validates
and builds. This file checks what is specific to this one: that the checklist held (no core module
knows the new classes), the wiring, the shapes a YAML list produces, a clip of any length, an
output of any number of samples, a stereo model, the bound of the waveform, that a spectrogram goes
through the 2D classes with no audio code, the gradient, the loss scale the experiment sets (and
the arithmetic behind it), the data contract, and a run of `scripts/train.py` on dummy batches.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import buildTransformPipeline
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import ModelConfig, buildModelFromConfig, validateModelConfig
from global_vae.decoders.registry import listRegisteredDecoders
from global_vae.encoders.registry import listRegisteredEncoders
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_SRC_DIR = _REPO_ROOT / "src" / "global_vae"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_DUMMY_AUDIO_LOADER = "tests.integration._train_script_fixtures:buildDummyAudioDataloaders"
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

MODEL_FILE = "audio_waveform_single_latent"
EXPERIMENT = "audio_vae"
BATCH_SIZE = 4
SAMPLES = 2048

# The packages that make a model, a loss, a training loop or a config. Adding a modality must not
# touch them (spec §10: zero changes to core framework code), so none may name this one: neither
# its registered classes nor its batch key.
_MODALITY_REFERENCE = re.compile(r"audio_waveform|AudioWaveform|[\"']audio[\"']")
_CORE_PACKAGES = (
    "assemblers",
    "config",
    "data",
    "evaluation",
    "fusion",
    "heads",
    "latent",
    "losses",
    "models",
    "training",
    "visualization",
)


def _modelConfig() -> ModelConfig:
    """Read the model file the way a YAML/Hydra config reaches the builders: through OmegaConf."""
    text = (_CONFIG_DIR / "model" / f"{MODEL_FILE}.yaml").read_text()
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(text))
    )
    assert isinstance(config, ModelConfig)
    return config


def _model(**overrides: object) -> GlobalVae:
    """The shipped model, with `overrides` set on the encoder or decoder kwargs.

    A key starting with `encoder_` or `decoder_` goes to that side, with the prefix removed.
    """
    config = _modelConfig()
    pair = config.modalities["audio"]
    for key, value in overrides.items():
        side, name = key.split("_", 1)
        (pair.encoder if side == "encoder" else pair.decoder).kwargs[name] = value
    return buildModelFromConfig(config)


def _experiment(*overrides: str) -> ExperimentConfig:
    return loadExperimentConfig(
        f"experiment/{EXPERIMENT}", overrides=[*_DATA_OVERRIDES, *overrides]
    )


def _batch(samples: int = SAMPLES, rows: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    """A batch of random clips within `[-1, 1]`."""
    return {"audio": torch.tanh(torch.randn(rows, samples))}


class TestTheNewModalityChecklist:
    """The practical test of the registry design: no core file is needed for a new modality."""

    def test_the_file_validates_and_builds(self) -> None:
        config = _modelConfig()
        validateModelConfig(config)
        model = buildModelFromConfig(config)
        assert set(model.latent_spaces) == {"z_fused"}
        assert model.input_modalities == ("audio",)
        assert model.output_modalities == ("audio",)

    def test_it_names_the_two_registered_classes(self) -> None:
        pair = _modelConfig().modalities["audio"]
        assert pair.encoder.name == "audio_waveform_encoder_v1"
        assert pair.decoder.name == "audio_waveform_decoder_v1"
        assert pair.encoder.name in listRegisteredEncoders()
        assert pair.decoder.name in listRegisteredDecoders()

    @pytest.mark.parametrize("package", _CORE_PACKAGES)
    def test_no_core_package_mentions_the_modality(self, package: str) -> None:
        mentions = [
            str(path.relative_to(_SRC_DIR))
            for path in (_SRC_DIR / package).rglob("*.py")
            if _MODALITY_REFERENCE.search(path.read_text())
        ]
        assert mentions == []

    def test_the_top_level_modules_do_not_mention_it_either(self) -> None:
        mentions = [
            path.name
            for path in _SRC_DIR.glob("*.py")
            if _MODALITY_REFERENCE.search(path.read_text())
        ]
        assert mentions == []


class TestTheModelFile:
    def test_it_is_one_modality_in_one_unfused_latent_space(self) -> None:
        config = _modelConfig()
        assert config.latent_mode == "single"
        assert config.single_latent is not None
        assert config.single_latent.name == "z_fused"
        assert config.single_latent.fusion is None  # one encoder needs no fusion (spec §4)
        assert len(_model().fusions) == 0

    def test_a_yaml_value_gives_the_reconstruction_it_declares(self) -> None:
        reconstruction = _model()(_batch())["reconstructions"]["audio"]
        assert reconstruction.shape == (BATCH_SIZE, SAMPLES)

    def test_the_declared_length_is_the_one_of_the_data_file(self) -> None:
        """One loader must serve the data file and the model: same number of samples."""
        declared = _modelConfig().modalities["audio"].decoder.kwargs["output_length"]
        assert _experiment().data.sequence_length == {"audio": declared}

    def test_the_latent_width_is_set_in_one_place(self) -> None:
        config = _modelConfig()
        assert config.single_latent is not None
        mu, logvar = _model()(_batch())["latent_params"]["z_fused"]
        assert mu.shape == logvar.shape == (BATCH_SIZE, config.single_latent.dim)

    @pytest.mark.parametrize("samples", [1, 100, 3000])
    def test_a_clip_of_another_length_is_encoded_but_reconstructed_at_the_declared_one(
        self, samples: int
    ) -> None:
        """The reduction over time makes the encoder length-agnostic; the decoder never resizes."""
        outputs = _model()(_batch(samples))
        assert outputs["reconstructions"]["audio"].shape == (BATCH_SIZE, SAMPLES)

    def test_the_waveform_stays_within_the_bounds_of_audio(self) -> None:
        reconstruction = _model()(_batch())["reconstructions"]["audio"]
        assert reconstruction.abs().max() <= 1.0

    def test_every_parameter_receives_a_gradient(self) -> None:
        model = _model()
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(_batch(), step=0)
        backward(losses.total)
        assert torch.isfinite(losses.total)
        assert [n for n, p in model.named_parameters() if p.grad is None] == []

    def test_the_loss_falls_over_a_few_steps(self) -> None:
        torch.manual_seed(0)
        model = _model()
        trainer = Trainer(model, optimizer_kwargs={"lr": 3e-3}, modality_dropout_p=0.0)
        t = torch.arange(SAMPLES).float()
        batch = {"audio": (0.8 * torch.sin(2 * torch.pi * t / 64)).repeat(BATCH_SIZE, 1)}
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(20):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        assert trainer.computeLosses(batch, step=20).reconstruction.item() < first

    def test_a_key_with_no_encoder_or_decoder_is_ignored(self) -> None:
        """A loader that yields more keys (a paired one, say) serves this model as well."""
        model = _model()
        batch = {**_batch(), "image": torch.randn(BATCH_SIZE, 64, 64)}
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(batch, step=0)
        assert torch.isfinite(losses.total)
        assert set(model(model.selectEncoderInputs(batch))["reconstructions"]) == {"audio"}


class TestWhatTheYamlLetsYouChange:
    @pytest.mark.parametrize("samples", [1000, 1999, 4097])
    def test_any_number_of_samples_is_reached_by_changing_one_value(self, samples: int) -> None:
        """No stack of doublings reaches 1000 or 1999: the seed is rounded up and cropped."""
        outputs = _model(decoder_output_length=samples)(_batch(samples))
        assert outputs["reconstructions"]["audio"].shape == (BATCH_SIZE, samples)

    def test_a_stereo_model_reads_and_writes_two_channels(self) -> None:
        model = _model(encoder_in_channels=2, decoder_out_channels=2)
        clip = torch.tanh(torch.randn(BATCH_SIZE, 2, SAMPLES))
        outputs = model({"audio": clip})
        assert outputs["reconstructions"]["audio"].shape == (BATCH_SIZE, 2, SAMPLES)
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses({"audio": clip}, step=0)
        assert torch.isfinite(losses.total)

    def test_an_unbounded_output_is_one_value_away(self) -> None:
        torch.manual_seed(0)
        model = _model(decoder_output_activation=None)
        reconstruction = model(_batch())["reconstructions"]["audio"]
        assert reconstruction.shape == (BATCH_SIZE, SAMPLES)
        bounded = _model()(_batch())["reconstructions"]["audio"]
        assert bounded.abs().max() <= 1.0

    def test_other_strides_and_widths_build(self) -> None:
        model = _model(
            encoder_hidden_channels=[4, 8],
            encoder_strides=[8, 4],
            decoder_hidden_channels=[8, 4],
            decoder_strides=[8, 4],
        )
        assert model(_batch())["reconstructions"]["audio"].shape == (BATCH_SIZE, SAMPLES)


class TestASpectrogramIsAnImage:
    """Audio as a time-frequency picture needs no audio class: the 2D classes already take it."""

    def test_a_log_magnitude_spectrogram_goes_through_the_image_model(self) -> None:
        text = (_CONFIG_DIR / "model" / "image_single_latent.yaml").read_text()
        config = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(text))
        )
        assert isinstance(config, ModelConfig)
        config.modalities["spectrogram"] = config.modalities.pop("image")
        model = buildModelFromConfig(config)

        clips = _batch()["audio"]
        window = torch.hann_window(126)
        spectrum = torch.stft(clips, n_fft=126, hop_length=32, window=window, return_complex=True)
        spectrogram = torch.log1p(spectrum.abs())[..., :64]
        assert spectrogram.shape == (BATCH_SIZE, 64, 64)  # (frequency bins, frames) is an image

        outputs = model({"spectrogram": spectrogram})
        assert outputs["reconstructions"]["spectrogram"].shape == spectrogram.shape
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(
            {"spectrogram": spectrogram}, step=0
        )
        backward(losses.total)
        assert [n for n, p in model.named_parameters() if p.grad is None] == []


class TestTheExperiment:
    def test_it_composes_its_model_data_and_training(self) -> None:
        config = _experiment()
        assert config.model == _modelConfig()
        assert config.output_dir == f"outputs/{EXPERIMENT}"
        assert set(config.training.beta_schedules) == {"z_fused"}

    def test_the_latent_width_follows_one_override(self) -> None:
        config = _experiment("model.single_latent.dim=3")
        mu, _ = buildModelFromConfig(config.model)(_batch())["latent_params"]["z_fused"]
        assert mu.shape == (BATCH_SIZE, 3)

    def test_the_clip_length_follows_the_overrides_of_the_data_and_the_decoder(self) -> None:
        config = _experiment(
            "data.sequence_length.audio=1000",
            "model.modalities.audio.decoder.kwargs.output_length=1000",
        )
        assert config.data.sequence_length == {"audio": 1000}
        out = buildModelFromConfig(config.model)(_batch(1000))
        assert out["reconstructions"]["audio"].shape == (BATCH_SIZE, 1000)

    def test_it_trains_on_the_summed_loss_with_the_elbo_weight(self) -> None:
        training = _experiment().training
        assert training.reconstruction_reduction == "sum"
        assert training.reconstruction_weight == 0.5

    def test_the_summed_loss_weighs_the_reconstruction_by_half_the_values_it_reconstructs(
        self,
    ) -> None:
        """The arithmetic the experiment file quotes: `sum` at 0.5 is `N / 2` times `mean`."""
        training = _experiment().training
        model = _model()
        batch = _batch()

        def reconstruction(reduction: str, weight: float) -> float:
            trainer = Trainer(
                model,
                modality_dropout_p=0.0,
                reconstruction_reduction=reduction,
                reconstruction_weights=weight,
            )
            torch.manual_seed(0)  # the same reparameterisation noise for both
            return trainer.computeLosses(batch, step=0).reconstruction.item()

        summed = reconstruction(training.reconstruction_reduction, training.reconstruction_weight)
        averaged = reconstruction("mean", 1.0)
        assert summed / averaged == pytest.approx(SAMPLES / 2, rel=1e-4)


class TestTheDataContract:
    def test_the_data_config_has_an_empty_pipeline_so_the_waveform_stays_in_range(self) -> None:
        data = _experiment().data
        assert data.transforms == {"audio": []}

    def test_the_configured_pipeline_is_the_identity(self) -> None:
        pipeline = buildTransformPipeline(_experiment().data)["audio"]
        x = _batch()["audio"]
        assert torch.equal(pipeline(x), x)
        assert torch.equal(pipeline.inverse(x), x)


def _runTrain(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/audio_vae` for two short epochs."""
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{EXPERIMENT}",
        f"data.loader_factory={_DUMMY_AUDIO_LOADER}",
        "data.train_path=unused",
        f"output_dir={output_dir}",
        "training.num_epochs=2",
        "training.beta_schedules.z_fused.kwargs.warmup_steps=5",
        f"hydra.run.dir={tmp_path / 'hydra_run'}",
    ]
    python_path = [str(_REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    completed = subprocess.run(
        [sys.executable, str(_TRAIN_SCRIPT), *overrides],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(python_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed, output_dir


@pytest.mark.slow
class TestTheExperimentTrainsThroughTheScript:
    def test_trains_on_dummy_batches_and_writes_a_checkpoint_the_model_loads(
        self, tmp_path: Path
    ) -> None:
        completed, output_dir = _runTrain(tmp_path)
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in (completed.stdout + completed.stderr)
        assert (output_dir / "metrics.csv").exists()
        metadata = loadCheckpoint(output_dir / "checkpoints" / "best.pt", model=_model())
        assert metadata.global_step > 0
