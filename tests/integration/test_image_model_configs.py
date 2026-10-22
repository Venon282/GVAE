"""The image-only model configs and their experiments (roadmap P3-3, spec §6.1 milestone 2).

`configs/model/image_single_latent.yaml` and `image_resnet_single_latent.yaml` are the image
counterparts of the two signal files (the plain 2D pair of ADR 0017 and the residual pair of ADR
0018), `configs/data/image.yaml` is their data contract, and `configs/experiment/image_vae.yaml`
and `image_resnet_vae.yaml` assemble them. `test_model_config_schema.py` already checks that every
shipped file validates and builds, and that a single-latent file equals what
`GlobalVae.createSingleLatent` builds. This file checks what is specific to these files, in the
style of `test_yaml_shapes.py`: the shapes a YAML list produces, the wiring, the gradient, the
data contract, the loss scale the experiments set for images (and the arithmetic behind it), and
a run of `scripts/train.py` per experiment, once on dummy images and once on the paired loader of
`examples/04_paired_loader.py`, which also yields `signal` and so shows that an extra key is
ignored.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import buildTransformPipeline
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import ModelConfig, buildModelFromConfig, validateModelConfig
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_DUMMY_IMAGE_LOADER = "tests.integration._train_script_fixtures:buildDummyImageDataloaders"
_PAIRED_LOADER = "examples.04_paired_loader:buildPairedDataloaders"
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

BATCH_SIZE = 4
IMAGE_SIZE = 64

# Model file -> (encoder name, decoder name) of its one `image` modality.
_MODELS = {
    "image_single_latent": ("2d_cnn_encoder_v1", "2d_cnn_decoder_v1"),
    "image_resnet_single_latent": ("2d_cnn_resnet_encoder_v1", "2d_cnn_resnet_decoder_v1"),
}
# Experiment file -> the model file it composes.
_EXPERIMENTS = {
    "image_vae": "image_single_latent",
    "image_resnet_vae": "image_resnet_single_latent",
}


def _modelConfig(name: str) -> ModelConfig:
    """Read a model file the way a YAML/Hydra config reaches the builders: through OmegaConf."""
    text = (_CONFIG_DIR / "model" / f"{name}.yaml").read_text()
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(text))
    )
    assert isinstance(config, ModelConfig)
    return config


def _model(name: str) -> GlobalVae:
    return buildModelFromConfig(_modelConfig(name))


def _experiment(name: str, *overrides: str) -> ExperimentConfig:
    return loadExperimentConfig(f"experiment/{name}", overrides=[*_DATA_OVERRIDES, *overrides])


def _loadPairedLoaderExample() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "global_vae_example_04_for_config", _REPO_ROOT / "examples" / "04_paired_loader.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", list(_MODELS))
class TestImageModelFiles:
    def test_the_file_is_one_image_modality_in_one_unfused_latent_space(self, name: str) -> None:
        config = _modelConfig(name)
        validateModelConfig(config)
        assert config.latent_mode == "single"
        assert set(config.modalities) == {"image"}
        assert config.single_latent is not None
        assert config.single_latent.name == "z_fused"
        assert config.single_latent.fusion is None  # one encoder needs no fusion (spec §4)
        model = buildModelFromConfig(config)
        assert set(model.encoders) == {"image"} and set(model.decoders) == {"image"}
        assert set(model.latent_spaces) == {"z_fused"}
        assert len(model.fusions) == 0

    def test_it_names_the_2d_pair_it_mirrors_the_signal_file_with(self, name: str) -> None:
        encoder_name, decoder_name = _MODELS[name]
        config = _modelConfig(name)
        assert config.modalities["image"].encoder.name == encoder_name
        assert config.modalities["image"].decoder.name == decoder_name

    def test_a_yaml_list_shape_gives_the_reconstruction_it_declares(self, name: str) -> None:
        """`output_shape: [64, 64]` reaches the decoder as a list and must still be reached."""
        config = _modelConfig(name)
        declared = config.modalities["image"].decoder.kwargs["output_shape"]
        assert isinstance(declared, list)
        model = buildModelFromConfig(config)
        reconstruction = model({"image": torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)})[
            "reconstructions"
        ]["image"]
        assert tuple(reconstruction.shape) == (BATCH_SIZE, *declared)

    def test_an_image_with_an_explicit_channel_axis_is_accepted(self, name: str) -> None:
        model = _model(name)
        outputs = model({"image": torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)})
        assert outputs["reconstructions"]["image"].shape == (2, IMAGE_SIZE, IMAGE_SIZE)

    def test_the_image_size_is_the_one_of_the_signal_plus_image_model(self, name: str) -> None:
        """One paired loader must serve `default.yaml` and these files: same image size."""
        image = _modelConfig(name).modalities["image"].decoder.kwargs["output_shape"]
        paired = _modelConfig("default").modalities["image"].decoder.kwargs["output_shape"]
        assert image == paired == [IMAGE_SIZE, IMAGE_SIZE]

    def test_the_latent_width_is_set_in_one_place(self, name: str) -> None:
        config = _modelConfig(name)
        assert config.single_latent is not None
        model = buildModelFromConfig(config)
        mu, logvar = model({"image": torch.randn(2, IMAGE_SIZE, IMAGE_SIZE)})["latent_params"][
            "z_fused"
        ]
        assert mu.shape == logvar.shape == (2, config.single_latent.dim)

    def test_every_parameter_receives_a_gradient(self, name: str) -> None:
        model = _model(name)
        trainer = Trainer(model, modality_dropout_p=0.0)
        batch = {"image": torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)}
        losses = trainer.computeLosses(batch, step=0)
        backward(losses.total)
        assert torch.isfinite(losses.total)
        assert [n for n, p in model.named_parameters() if p.grad is None] == []

    def test_the_loss_falls_over_a_few_steps(self, name: str) -> None:
        torch.manual_seed(0)
        model = _model(name)
        trainer = Trainer(model, optimizer_kwargs={"lr": 1e-2}, modality_dropout_p=0.0)
        batch = {"image": torch.rand(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)}
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(15):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        assert trainer.computeLosses(batch, step=15).reconstruction.item() < first

    def test_a_key_with_no_encoder_or_decoder_is_ignored(self, name: str) -> None:
        """The paired loader yields `signal` too: an image model neither fails on it nor uses it."""
        model = _model(name)
        batch = {
            "signal": torch.randn(BATCH_SIZE, 256),
            "image": torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE),
        }
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(batch, step=0)
        assert torch.isfinite(losses.total)
        assert set(model(model.selectEncoderInputs(batch))["reconstructions"]) == {"image"}


class TestResidualVariantDiffersOnlyInItsBackbone:
    def test_the_two_files_agree_on_everything_but_the_modules(self) -> None:
        plain, residual = (
            _modelConfig("image_single_latent"),
            _modelConfig("image_resnet_single_latent"),
        )
        assert plain.single_latent == residual.single_latent
        assert (
            plain.modalities["image"].decoder.kwargs["output_shape"]
            == residual.modalities["image"].decoder.kwargs["output_shape"]
        )

    def test_the_residual_model_is_a_different_network(self) -> None:
        plain_state, residual_state = (
            _model("image_single_latent"),
            _model("image_resnet_single_latent"),
        )
        assert set(plain_state.state_dict()) != set(residual_state.state_dict())


class TestImageDataContract:
    def test_a_loader_and_a_path_are_required(self) -> None:
        with pytest.raises(MissingMandatoryValue):
            loadExperimentConfig("experiment/image_vae")

    def test_only_the_image_modality_is_configured(self) -> None:
        data = _experiment("image_vae").data
        assert set(data.transforms) == {"image"}
        assert data.sequence_length is None  # a 1D signal length: an image's size is the decoder's

    def test_the_transform_pipeline_builds_and_round_trips(self) -> None:
        pipeline = buildTransformPipeline(_experiment("image_vae").data)["image"]
        image = torch.rand(IMAGE_SIZE, IMAGE_SIZE)
        assert torch.allclose(pipeline.inverse(pipeline.apply(image)), image, atol=1e-6)

    def test_the_split_defaults_match_the_signal_contracts(self) -> None:
        image, signal = (
            _experiment("image_vae").data,
            loadExperimentConfig("experiment/signal_vae", overrides=_DATA_OVERRIDES).data,
        )
        assert (image.batch_size, image.val_split, image.test_split, image.shuffle_train) == (
            signal.batch_size,
            signal.val_split,
            signal.test_split,
            signal.shuffle_train,
        )


@pytest.mark.parametrize(("experiment", "model_file"), list(_EXPERIMENTS.items()))
class TestImageExperiments:
    def test_the_experiment_composes_its_model_data_and_training(
        self, experiment: str, model_file: str
    ) -> None:
        config = _experiment(experiment)
        assert config.model == _modelConfig(model_file)
        assert config.output_dir == f"outputs/{experiment}"
        assert set(config.training.beta_schedules) == {"z_fused"}

    def test_the_model_group_can_be_swapped_from_the_command_line(
        self, experiment: str, model_file: str
    ) -> None:
        other = next(name for name in _MODELS if name != model_file)
        config = _experiment(experiment, f"model={other}")
        assert config.model == _modelConfig(other)

    def test_the_latent_width_follows_one_override(self, experiment: str, model_file: str) -> None:
        config = _experiment(experiment, "model.single_latent.dim=8")
        model = buildModelFromConfig(config.model)
        mu, _ = model({"image": torch.randn(2, IMAGE_SIZE, IMAGE_SIZE)})["latent_params"]["z_fused"]
        assert mu.shape == (2, 8)

    def test_images_train_on_the_summed_loss_with_the_elbo_weight(
        self, experiment: str, model_file: str
    ) -> None:
        training = _experiment(experiment).training
        assert training.reconstruction_reduction == "sum"
        assert training.reconstruction_weight == 0.5

    def test_the_summed_loss_weighs_the_reconstruction_by_half_the_pixel_count(
        self, experiment: str, model_file: str
    ) -> None:
        """The arithmetic the experiment files quote: `sum` at 0.5 is `N / 2` times `mean`."""
        training = _experiment(experiment).training
        model = _model(model_file)
        batch = {"image": torch.rand(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)}

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
        assert summed / averaged == pytest.approx(IMAGE_SIZE * IMAGE_SIZE / 2, rel=1e-4)


def _runTrain(
    tmp_path: Path, experiment: str, loader_factory: str, train_path: str, *extra: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/<experiment>` for two short epochs."""
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{experiment}",
        f"data.loader_factory={loader_factory}",
        f"data.train_path={train_path}",
        f"output_dir={output_dir}",
        "training.num_epochs=2",
        "training.beta_schedules.z_fused.kwargs.warmup_steps=5",
        f"hydra.run.dir={tmp_path / 'hydra_run'}",
        *extra,
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


# Every test below runs `scripts/train.py` in a subprocess.
@pytest.mark.slow
@pytest.mark.parametrize("experiment", list(_EXPERIMENTS))
class TestImageExperimentsTrainThroughTheScript:
    def test_trains_on_dummy_images_and_writes_a_checkpoint_the_model_loads(
        self, tmp_path: Path, experiment: str
    ) -> None:
        completed, output_dir = _runTrain(tmp_path, experiment, _DUMMY_IMAGE_LOADER, "unused")
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in (completed.stdout + completed.stderr)
        assert (output_dir / "metrics.csv").exists()
        model = buildModelFromConfig(_experiment(experiment).model)
        metadata = loadCheckpoint(output_dir / "checkpoints" / "best.pt", model=model)
        assert metadata.global_step > 0

    def test_trains_on_the_paired_loader_which_also_yields_a_signal(
        self, tmp_path: Path, experiment: str
    ) -> None:
        _loadPairedLoaderExample().writeSyntheticPairedDataset(
            tmp_path / "data", num_pairs=24, num_signal_only=2, seed=3
        )
        completed, output_dir = _runTrain(
            tmp_path, experiment, _PAIRED_LOADER, str(tmp_path / "data"), "data.batch_size=8"
        )
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in (completed.stdout + completed.stderr)
        assert (output_dir / "checkpoints" / "best.pt").is_file()
