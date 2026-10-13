"""The model config schema against the files the repository ships (roadmap P1-3, ADR 0025).

`tests/unit/test_model_config_validation.py` covers the rules on hand-made dataclasses. This file
covers what only real files can show:

* every `configs/model/*.yaml` and every `configs/experiment/*.yaml` validates and builds, so a
  new file that breaks the schema fails here and not on someone's first run;
* the second and third patterns of spec §9 (several independent latent spaces, shared plus
  private) are YAML files that build, run, and send a gradient to every parameter, heads
  included;
* the model of `examples/03_signal_image_to_image.py`, which `modalities` could not express, is a
  YAML file whose model has the same parameters as the Python one;
* single-latent files build the same model as `GlobalVae.createSingleLatent`;
* a mistake in a file is reported with the path of the field to edit.

Training through `scripts/train.py` is in `test_train_script.py`.
"""

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

import global_vae.assemblers.registry as assembler_registry
import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.assemblers.base import AbstractAssembler
from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import ModelConfig, buildModelFromConfig, validateModelConfig
from global_vae.heads.linear import LinearHead
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_MODEL_FILES = sorted((_CONFIG_DIR / "model").glob("*.yaml"))
_EXPERIMENT_FILES = sorted((_CONFIG_DIR / "experiment").glob("*.yaml"))
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

BATCH_SIZE = 4


def _modelConfigFromText(text: str) -> ModelConfig:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(text))
    )
    assert isinstance(config, ModelConfig)
    return config


def _modelConfigFromFile(path: Path) -> ModelConfig:
    return _modelConfigFromText(path.read_text())


def _modelFromFile(name: str) -> GlobalVae:
    return buildModelFromConfig(_modelConfigFromFile(_CONFIG_DIR / "model" / f"{name}.yaml"))


def _batch() -> dict[str, torch.Tensor]:
    image = torch.randn(BATCH_SIZE, 64, 64)
    return {
        "signal": torch.randn(BATCH_SIZE, 256),
        "image": image,
        "image_out": image.clone(),
    }


def _loadExample03() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "global_vae_example_03_for_config", _REPO_ROOT / "examples" / "03_signal_image_to_image.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assertDecoderTakes(model: GlobalVae, name: str, width: int) -> None:
    """The decoder accepts latent vectors of exactly `width` (decoders expose no `latent_dim`)."""
    decoder = model.decoders[name]
    with torch.no_grad():
        assert decoder(torch.randn(2, width)).shape[0] == 2
        with pytest.raises(RuntimeError):
            decoder(torch.randn(2, width + 1))


def _shapes(model: GlobalVae) -> dict[str, tuple[int, ...]]:
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


class TestEveryShippedFile:
    def test_the_directories_are_not_empty(self) -> None:
        assert len(_MODEL_FILES) >= 6
        assert len(_EXPERIMENT_FILES) >= 5

    @pytest.mark.parametrize("path", _MODEL_FILES, ids=lambda path: path.name)
    def test_a_model_file_validates_and_builds(self, path: Path) -> None:
        config = _modelConfigFromFile(path)
        validateModelConfig(config)
        model = buildModelFromConfig(config)
        assert len(model.encoders) >= 1
        assert len(model.decoders) >= 1

    @pytest.mark.parametrize("path", _EXPERIMENT_FILES, ids=lambda path: path.name)
    def test_an_experiment_file_composes_validates_and_builds(self, path: Path) -> None:
        config = loadExperimentConfig(f"experiment/{path.stem}", overrides=_DATA_OVERRIDES)
        validateModelConfig(config.model)
        model = buildModelFromConfig(config.model)
        # A beta schedule for a latent space the model lacks would be dropped silently (the
        # trainer only warns), so a shipped experiment must not carry one.
        assert set(config.training.beta_schedules) <= set(model.latent_spaces)

    @pytest.mark.parametrize("path", _EXPERIMENT_FILES, ids=lambda path: path.name)
    def test_a_shipped_experiment_gives_every_latent_space_a_beta_schedule(
        self, path: Path
    ) -> None:
        config = loadExperimentConfig(f"experiment/{path.stem}", overrides=_DATA_OVERRIDES)
        model = buildModelFromConfig(config.model)
        assert set(config.training.beta_schedules) == set(model.latent_spaces)


class TestSeveralIndependentLatentSpaces:
    """Spec §9, second pattern: no encoder fan-out, one joint decoder."""

    def test_the_wiring_is_the_one_the_file_describes(self) -> None:
        model = _modelFromFile("signal_image_independent_latents")
        assert set(model.encoders) == {"signal", "image"}
        assert set(model.decoders) == {"image_out"}
        assert set(model.latent_spaces) == {"z_signal", "z_image"}
        assert model.routing_graph.encoder_to_latents == {
            "signal": ["z_signal"],
            "image": ["z_image"],
        }
        assert model.routing_graph.latent_to_decoders == {
            "z_signal": ["image_out"],
            "z_image": ["image_out"],
        }
        assert model.routing_graph.decoder_assemblers == {"image_out": "concat"}
        assert len(model.heads) == 0
        assert len(model.fusions) == 0

    def test_widths_come_from_the_latent_spaces(self) -> None:
        model = _modelFromFile("signal_image_independent_latents")
        assert model.encoders["signal"].latent_dim == 16
        assert model.encoders["image"].latent_dim == 16
        _assertDecoderTakes(model, "image_out", 32)

    def test_the_joint_decoder_reads_both_latent_spaces(self) -> None:
        model = _modelFromFile("signal_image_independent_latents")
        batch = _batch()
        outputs = model({"signal": batch["signal"], "image": batch["image"]})
        assert outputs["reconstructions"]["image_out"].shape == (BATCH_SIZE, 64, 64)
        assert set(outputs["latent_params"]) == {"z_signal", "z_image"}

    def test_the_joint_decoder_is_skipped_when_one_modality_is_missing(self) -> None:
        model = _modelFromFile("signal_image_independent_latents")
        outputs = model({"signal": _batch()["signal"]})
        assert set(outputs["latent_params"]) == {"z_signal"}
        assert outputs["reconstructions"] == {}

    def test_every_parameter_receives_a_gradient(self) -> None:
        model = _modelFromFile("signal_image_independent_latents")
        trainer = Trainer(model, modality_dropout_p=0.0)
        backward(trainer.computeLosses(_batch(), step=0).total)
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []


class TestSharedPrivateLatentSpaces:
    """Spec §9, third pattern: encoder fan-out through latent heads."""

    def test_the_wiring_is_the_one_the_file_describes(self) -> None:
        model = _modelFromFile("signal_image_shared_private")
        assert set(model.encoders) == {"signal", "image"}
        assert set(model.decoders) == {"signal", "image"}
        assert set(model.latent_spaces) == {"z_shared", "z_signal_private", "z_image_private"}
        assert model.routing_graph.encoder_to_latents == {
            "signal": ["z_shared", "z_signal_private"],
            "image": ["z_shared", "z_image_private"],
        }
        assert model.routing_graph.decoder_assemblers == {"signal": "concat", "image": "concat"}
        assert set(model.fusions) == {"z_shared"}

    def test_each_private_edge_has_a_linear_head_and_each_shared_edge_none(self) -> None:
        model = _modelFromFile("signal_image_shared_private")
        assert isinstance(model.getHead("signal", "z_signal_private"), LinearHead)
        assert isinstance(model.getHead("image", "z_image_private"), LinearHead)
        assert model.getHead("signal", "z_shared") is None
        assert model.getHead("image", "z_shared") is None

    def test_widths_come_from_the_shared_space_and_the_assembler(self) -> None:
        model = _modelFromFile("signal_image_shared_private")
        assert model.encoders["signal"].latent_dim == 32
        assert model.encoders["image"].latent_dim == 32
        _assertDecoderTakes(model, "signal", 40)
        _assertDecoderTakes(model, "image", 40)

    def test_both_decoders_reconstruct_and_the_three_latent_spaces_differ(self) -> None:
        model = _modelFromFile("signal_image_shared_private")
        batch = _batch()
        outputs = model({"signal": batch["signal"], "image": batch["image"]}, use_mean=True)
        assert outputs["reconstructions"]["signal"].shape == (BATCH_SIZE, 256)
        assert outputs["reconstructions"]["image"].shape == (BATCH_SIZE, 64, 64)
        mu_signal, _ = outputs["latent_params"]["z_signal_private"]
        mu_image, _ = outputs["latent_params"]["z_image_private"]
        assert mu_signal.shape == mu_image.shape == (BATCH_SIZE, 8)
        assert not torch.allclose(mu_signal, mu_image)

    def test_a_hidden_modality_removes_its_private_space_and_its_decoder(self) -> None:
        model = _modelFromFile("signal_image_shared_private")
        outputs = model({"signal": _batch()["signal"]})
        assert set(outputs["latent_params"]) == {"z_shared", "z_signal_private"}
        assert set(outputs["reconstructions"]) == {"signal"}

    def test_every_parameter_receives_a_gradient_heads_included(self) -> None:
        model = _modelFromFile("signal_image_shared_private")
        trainer = Trainer(model, modality_dropout_p=0.0)
        backward(trainer.computeLosses(_batch(), step=0).total)
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []
        assert any(name.startswith("heads.signal.z_signal_private") for name in model.state_dict())

    def test_the_loss_falls_over_a_few_steps(self) -> None:
        torch.manual_seed(0)
        model = _modelFromFile("signal_image_shared_private")
        trainer = Trainer(model, optimizer_kwargs={"lr": 1e-2}, modality_dropout_p=0.0)
        batch = _batch()
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(20):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        last = trainer.computeLosses(batch, step=20).reconstruction.item()
        assert last < first


class TestTranslationModelInYaml:
    """The model of `examples/03_signal_image_to_image.py`, written as a file."""

    def test_the_model_has_the_same_parameters_as_the_python_one(self) -> None:
        example = _loadExample03()
        from_yaml = _modelFromFile("signal_image_to_image")
        from_python = example.buildModel()
        assert _shapes(from_yaml) == _shapes(from_python)

    def test_the_wiring_matches_the_python_one(self) -> None:
        example = _loadExample03()
        from_yaml = _modelFromFile("signal_image_to_image")
        from_python = example.buildModel()
        assert from_yaml.routing_graph == from_python.routing_graph
        assert list(from_yaml.fusions) == list(from_python.fusions)
        assert type(from_yaml.regularizers["z_fused"]) is type(from_python.regularizers["z_fused"])

    def test_the_decoder_has_no_encoder_of_the_same_name(self) -> None:
        model = _modelFromFile("signal_image_to_image")
        assert set(model.encoders) == {"signal", "image_in"}
        assert set(model.decoders) == {"image_out"}

    def test_it_reconstructs_the_target_from_either_input_alone(self) -> None:
        model = _modelFromFile("signal_image_to_image")
        signal = torch.randn(2, 64)
        image_in = torch.randn(2, 32, 32)
        for inputs in (
            {"signal": signal},
            {"image_in": image_in},
            {"signal": signal, "image_in": image_in},
        ):
            outputs = model(inputs)
            assert outputs["reconstructions"]["image_out"].shape == (2, 32, 32)

    def test_a_trainer_step_accepts_a_target_that_is_not_an_input(self) -> None:
        model = _modelFromFile("signal_image_to_image")
        batch = {
            "signal": torch.randn(2, 64),
            "image_in": torch.randn(2, 32, 32),
            "image_out": torch.randn(2, 32, 32),
        }
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(batch, step=0)
        backward(losses.total)
        assert torch.isfinite(losses.total)


class TestSingleLatentFilesAreUnchanged:
    """`single` files go through `GlobalVae.__init__` now, and must build what they always did."""

    @pytest.mark.parametrize(
        "name", ["default", "signal_single_latent", "signal_resnet_single_latent"]
    )
    def test_the_model_equals_what_create_single_latent_builds(self, name: str) -> None:
        config = _modelConfigFromFile(_CONFIG_DIR / "model" / f"{name}.yaml")
        assert config.single_latent is not None
        latent_dim = config.single_latent.dim
        fusion = config.single_latent.fusion
        expected = GlobalVae.createSingleLatent(
            modality_configs={
                modality: {"encoder": cfg.encoder.name, "decoder": cfg.decoder.name}
                for modality, cfg in config.modalities.items()
            },
            latent_dim=latent_dim,
            fusion_strategy=fusion.strategy if fusion is not None else None,
            latent_name=config.single_latent.name,
            regularizer_strategy=config.single_latent.regularizer.strategy,
            encoder_kwargs={
                modality: {"latent_dim": latent_dim, **cfg.encoder.kwargs}
                for modality, cfg in config.modalities.items()
            },
            decoder_kwargs={
                modality: {"latent_dim": latent_dim, **cfg.decoder.kwargs}
                for modality, cfg in config.modalities.items()
            },
            regularizer_kwargs={config.single_latent.name: config.single_latent.regularizer.kwargs},
        )
        built = buildModelFromConfig(config)
        assert _shapes(built) == _shapes(expected)
        assert list(built.state_dict()) == list(expected.state_dict())
        assert built.routing_graph == expected.routing_graph


class TestMixingSections:
    _TEXT = """
    latent_mode: several
    modalities:
      signal:
        encoder: {name: 1d_cnn_encoder_v1}
        decoder:
          name: 1d_cnn_decoder_v1
          kwargs:
            output_length: 64
            hidden_channels: [32, 16, 8]
            upsample_modes: conv_transpose
    encoders:
      image: {name: 2d_cnn_encoder_v1}
    decoders:
      image_out:
        name: 2d_cnn_decoder_v1
        kwargs:
          output_shape: [32, 32]
          hidden_channels: [32, 16, 8]
          seed_shape: [4, 4]
          upsample_modes: conv_transpose
    latent:
      spaces:
        z_signal: {dim: 8, fed_by: [signal]}
        z_image: {dim: 8, fed_by: [image]}
      decoders_consume:
        signal: {spaces: [z_signal]}
        image_out: {spaces: [z_signal, z_image], assembler: sum}
    """

    def test_modalities_and_the_two_sections_can_be_used_together(self) -> None:
        model = buildModelFromConfig(_modelConfigFromText(self._TEXT))
        assert list(model.encoders) == ["signal", "image"]
        assert list(model.decoders) == ["signal", "image_out"]
        outputs = model({"signal": torch.randn(2, 64), "image": torch.randn(2, 32, 32)})
        assert outputs["reconstructions"]["signal"].shape == (2, 64)
        assert outputs["reconstructions"]["image_out"].shape == (2, 32, 32)
        _assertDecoderTakes(model, "image_out", 8)  # sum keeps the common width

    def test_a_decoder_of_one_space_still_runs_when_the_other_modality_is_missing(self) -> None:
        model = buildModelFromConfig(_modelConfigFromText(self._TEXT))
        outputs = model({"signal": torch.randn(2, 64)})
        assert set(outputs["reconstructions"]) == {"signal", "image_out"}  # sum tolerates a gap


class TestErrorsNameTheField:
    def test_an_unknown_key_carries_the_omegaconf_full_key(self) -> None:
        text = """
        latent_mode: several
        latent:
          spaces:
            z: {dim: 4, fed_by: [a], fuse: poe}
        """
        with pytest.raises(ConfigKeyError, match=r"full_key: latent.spaces.z.fuse"):
            _modelConfigFromText(text)

    def test_a_malformed_fed_by_in_a_file_is_reported_with_its_path(self) -> None:
        config = _modelConfigFromFile(_CONFIG_DIR / "model" / "signal_image_shared_private.yaml")
        assert config.latent is not None
        config.latent.spaces["z_signal_private"].fed_by = {"signal": {"heads": "linear"}}
        with pytest.raises(
            ValueError, match=r"model.latent.spaces.z_signal_private.fed_by.signal.heads: unknown"
        ):
            buildModelFromConfig(config)

    def test_a_cli_override_that_breaks_the_wiring_is_reported_with_its_path(self) -> None:
        config = loadExperimentConfig(
            "experiment/signal_image_shared_private",
            overrides=[*_DATA_OVERRIDES, "model.latent.spaces.z_shared.fed_by=[signal,audio]"],
        )
        with pytest.raises(
            ValueError, match=r"model.latent.spaces.z_shared.fed_by: unknown encoder 'audio'"
        ):
            buildModelFromConfig(config.model)

    def test_a_second_identity_edge_from_one_encoder_is_reported_with_its_path(self) -> None:
        config = _modelConfigFromFile(_CONFIG_DIR / "model" / "signal_image_shared_private.yaml")
        assert config.latent is not None
        config.latent.spaces["z_signal_private"].dim = 32
        config.latent.spaces["z_signal_private"].fed_by = {"signal": {"head": "identity"}}
        with pytest.raises(
            ValueError,
            match=r"model.latent.spaces.z_signal_private.fed_by: encoder 'signal' feeds 2",
        ):
            buildModelFromConfig(config)

    def test_an_experiment_override_can_change_a_dimension_everything_follows(self) -> None:
        config = loadExperimentConfig(
            "experiment/signal_image_shared_private",
            overrides=[*_DATA_OVERRIDES, "model.latent.spaces.z_shared.dim=16"],
        )
        model = buildModelFromConfig(config.model)
        assert model.encoders["signal"].latent_dim == 16
        _assertDecoderTakes(model, "signal", 24)


class TestAssemblerKwargs:
    """`decoders_consume.<decoder>.kwargs` reach the assembler the model builds (ADR 0025)."""

    class _Weighted(AbstractAssembler):
        """Test assembler: concatenates the latents after scaling the first one."""

        def __init__(self, input_dims: Sequence[int], first_weight: float) -> None:
            super().__init__(input_dims)
            self.first_weight = first_weight

        @property
        def output_dim(self) -> int:
            return sum(self.input_dims)

        def forward(
            self, latents: list[torch.Tensor], present: Sequence[int] | None = None
        ) -> torch.Tensor:
            return torch.cat([latents[0] * self.first_weight, *latents[1:]], dim=-1)

    def test_kwargs_from_the_config_reach_the_built_assembler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            assembler_registry._ASSEMBLER_REGISTRY, "weighted_for_test", self._Weighted
        )
        config = _modelConfigFromFile(
            _CONFIG_DIR / "model" / "signal_image_independent_latents.yaml"
        )
        assert config.latent is not None
        consumption = config.latent.decoders_consume["image_out"]
        consumption.assembler = "weighted_for_test"
        consumption.kwargs = {"first_weight": 0.5}
        model = buildModelFromConfig(config)
        assembler = model.assemblers["image_out"]
        assert isinstance(assembler, self._Weighted)
        assert assembler.first_weight == 0.5
        batch = _batch()
        outputs = model({"signal": batch["signal"], "image": batch["image"]})
        assert outputs["reconstructions"]["image_out"].shape == (BATCH_SIZE, 64, 64)
