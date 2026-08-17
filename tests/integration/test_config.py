"""Integration tests for `global_vae.config` (spec §9, §10 "Config management").

Uses the repository's real `configs/` directory and real built-in encoder/decoder/
regularizer/beta-schedule/logger implementations throughout (unlike most other
integration tests in this suite, which register trivial dummies): the whole point of
this config layer is turning `configs/model/signal_single_latent.yaml` +
`configs/data/signal.yaml` + `configs/training/default.yaml` into a real, working
`GlobalVae`/`Trainer` pair, so exercising it against anything else would not actually
test the thing being built.

`loadExperimentConfig` and `initialize_config_dir` (which it uses internally) are safe
to call repeatedly within one process/pytest session: each call's `with
initialize_config_dir(...)` block clears Hydra's global state on exit, verified by the
`TestRepeatedLoading` class below.
"""

from pathlib import Path

import pytest
import torch
from hydra.errors import MissingConfigException
from omegaconf.errors import MissingMandatoryValue

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import DataConfig, DataloaderBundle, buildDataloadersFromConfig
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import ModelConfig, buildModelFromConfig
from global_vae.config.training import (
    BetaScheduleConfig,
    CheckpointConfig,
    LoggerEntryConfig,
    TrainingConfig,
    buildBetaSchedules,
    buildCallbacksFromConfig,
    buildTrainerFromConfig,
    listSupportedOptimizerNames,
    listSupportedReconstructionLossNames,
    resolveOptimizerClass,
    resolveReconstructionLossFn,
)
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedules.linear_warmup import LinearWarmupBetaSchedule
from global_vae.training.checkpoint import BestCheckpointCallback, CheckpointCallback
from global_vae.training.loggers.csv_logger import CsvLogger
from global_vae.training.loggers.tensorboard_logger import TensorBoardLogger
from global_vae.training.trainer import Trainer

_LOADER_FACTORY = "tests.integration._train_script_fixtures:buildDummyDataloaders"
_BASE_OVERRIDES = [f"data.loader_factory={_LOADER_FACTORY}", "data.train_path=/unused"]


def _loadSignalVaeConfig(overrides: list[str] | None = None) -> ExperimentConfig:
    return loadExperimentConfig(overrides=[*_BASE_OVERRIDES, *(overrides or [])])


class TestLoadExperimentConfig:
    def test_composes_model_data_and_training_groups(self) -> None:
        cfg = _loadSignalVaeConfig()
        assert isinstance(cfg, ExperimentConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.training, TrainingConfig)

    def test_signal_single_latent_model_fields(self) -> None:
        cfg = _loadSignalVaeConfig()
        assert cfg.model.latent_mode == "single"
        assert set(cfg.model.modalities) == {"signal"}
        assert cfg.model.modalities["signal"].encoder.name == "1d_cnn_encoder_v1"
        assert cfg.model.single_latent is not None
        assert cfg.model.single_latent.dim == 16
        assert cfg.model.single_latent.fusion is None  # single modality: no fusion needed

    def test_output_dir_interpolation_reaches_nested_training_fields(self) -> None:
        """`${output_dir}` inside configs/training/default.yaml must resolve against the
        experiment-level output_dir, not fail or stay a literal string."""
        cfg = _loadSignalVaeConfig(overrides=["output_dir=/tmp/some_run"])
        assert cfg.training.checkpoint.directory == "/tmp/some_run/checkpoints"
        assert cfg.training.checkpoint.best_path == "/tmp/some_run/checkpoints/best.pt"
        logger_paths = {entry.name: entry.kwargs for entry in cfg.training.loggers}
        assert logger_paths["csv"]["path"] == "/tmp/some_run/metrics.csv"
        assert logger_paths["tensorboard"]["log_dir"] == "/tmp/some_run/tensorboard"

    def test_cli_style_overrides_reach_every_config_group(self) -> None:
        cfg = _loadSignalVaeConfig(
            overrides=[
                "training.num_epochs=7",
                "training.optimizer.kwargs.lr=0.005",
                "data.batch_size=64",
                "model.single_latent.dim=32",
            ]
        )
        assert cfg.training.num_epochs == 7
        assert cfg.training.optimizer.kwargs["lr"] == pytest.approx(0.005)
        assert cfg.data.batch_size == 64
        assert cfg.model.single_latent.dim == 32

    def test_missing_required_data_fields_raises(self) -> None:
        with pytest.raises(MissingMandatoryValue):
            loadExperimentConfig()  # no data.loader_factory / data.train_path override

    def test_unknown_config_name_raises(self) -> None:
        with pytest.raises(MissingConfigException):
            loadExperimentConfig(config_name="experiment/does_not_exist")

    def test_explicit_config_dir_is_respected(self, tmp_path: Path) -> None:
        (tmp_path / "experiment").mkdir()
        (tmp_path / "experiment" / "tiny.yaml").write_text(
            "# @package _global_\n"
            "seed: 123\n"
            "model: {latent_mode: single}\n"
            f"data: {{loader_factory: '{_LOADER_FACTORY}', train_path: x}}\n"
        )
        cfg = loadExperimentConfig(config_name="experiment/tiny", config_dir=tmp_path)
        assert cfg.seed == 123


class TestRepeatedLoading:
    def test_can_be_called_many_times_in_one_process(self) -> None:
        for step in range(5):
            cfg = _loadSignalVaeConfig(overrides=[f"training.num_epochs={step + 1}"])
            assert cfg.training.num_epochs == step + 1


class TestBuildModelFromConfig:
    def test_builds_a_real_global_vae(self) -> None:
        cfg = _loadSignalVaeConfig()
        model = buildModelFromConfig(cfg.model)
        assert isinstance(model, GlobalVae)
        assert set(model.encoders) == {"signal"}
        assert set(model.decoders) == {"signal"}
        assert "z_fused" not in model.fusions  # single encoder: no fusion module built

    def test_forward_pass_shapes_match_configured_dims(self) -> None:
        cfg = _loadSignalVaeConfig()
        model = buildModelFromConfig(cfg.model)
        output = model({"signal": torch.randn(3, 256)})
        assert output["reconstructions"]["signal"].shape == (3, 256)
        mu, logvar = output["latent_params"]["z_fused"]
        assert mu.shape == (3, 16)
        assert logvar.shape == (3, 16)

    def test_latent_dim_is_auto_injected_into_encoder_and_decoder_kwargs(self) -> None:
        """The whole point of the auto-fill: signal_single_latent.yaml never repeats
        latent_dim in the encoder/decoder kwargs, yet it must still end up correct."""
        cfg = _loadSignalVaeConfig(overrides=["model.single_latent.dim=24"])
        model = buildModelFromConfig(cfg.model)
        assert model.encoders["signal"].latent_dim == 24
        mu, _ = model.encoders["signal"](torch.randn(2, 256))
        assert mu.shape == (2, 24)

    def test_explicit_latent_dim_override_in_encoder_kwargs_is_respected(self) -> None:
        """An explicit kwargs.latent_dim must win over the auto-fill, not be clobbered.

        `+` is Hydra's "add a new key" override syntax: `kwargs.latent_dim` does not
        already exist in `signal_single_latent.yaml`, unlike a plain value override.
        """
        cfg = _loadSignalVaeConfig(
            overrides=["+model.modalities.signal.encoder.kwargs.latent_dim=8"]
        )
        model = buildModelFromConfig(cfg.model)
        assert model.encoders["signal"].latent_dim == 8

    def test_two_modality_config_needs_a_fusion_strategy(self) -> None:
        """default.yaml's two-modality example is schema-valid; only its unregistered
        image encoder/decoder makes it unbuildable today (see next test)."""
        cfg = loadExperimentConfig(
            config_name="experiment/signal_vae",
            overrides=[*_BASE_OVERRIDES, "model=default"],
        )
        assert cfg.model.single_latent.fusion is not None
        assert cfg.model.single_latent.fusion.strategy == "poe"

    def test_unregistered_encoder_name_raises_key_error(self) -> None:
        cfg = loadExperimentConfig(
            config_name="experiment/signal_vae",
            overrides=[*_BASE_OVERRIDES, "model=default"],
        )
        with pytest.raises(KeyError, match="resnet_encoder_v1"):
            buildModelFromConfig(cfg.model)

    def test_several_latent_mode_raises_not_implemented(self) -> None:
        cfg = _loadSignalVaeConfig(overrides=["model.latent_mode=several"])
        with pytest.raises(NotImplementedError, match="several"):
            buildModelFromConfig(cfg.model)

    def test_single_mode_without_single_latent_raises_value_error(self) -> None:
        config = ModelConfig(
            modalities={
                "signal": _loadSignalVaeConfig().model.modalities["signal"],
            },
            latent_mode="single",
            single_latent=None,
        )
        with pytest.raises(ValueError, match="single_latent"):
            buildModelFromConfig(config)

    def test_empty_modalities_raises_value_error(self) -> None:
        config = ModelConfig(
            modalities={},
            latent_mode="single",
            single_latent=_loadSignalVaeConfig().model.single_latent,
        )
        with pytest.raises(ValueError, match="modalities"):
            buildModelFromConfig(config)


class TestDataConfig:
    def test_build_dataloaders_from_config_resolves_and_calls_the_factory(self) -> None:
        cfg = _loadSignalVaeConfig()
        bundle = buildDataloadersFromConfig(cfg.data)
        assert isinstance(bundle, DataloaderBundle)
        assert len(list(bundle.train)) == 3
        assert bundle.val is not None and len(list(bundle.val)) == 1

    def test_sequence_length_reaches_the_dummy_factory(self) -> None:
        cfg = _loadSignalVaeConfig(overrides=["data.sequence_length=64"])
        bundle = buildDataloadersFromConfig(cfg.data)
        first_batch = next(iter(bundle.train))
        assert first_batch["signal"].shape[1] == 64

    def test_invalid_loader_factory_spec_raises_value_error(self) -> None:
        config = DataConfig(loader_factory="not_a_valid_spec", train_path="x")
        with pytest.raises(ValueError, match="module.path:function_name"):
            buildDataloadersFromConfig(config)

    def test_unknown_loader_factory_module_raises(self) -> None:
        config = DataConfig(loader_factory="does.not.exist:build", train_path="x")
        with pytest.raises(ModuleNotFoundError):
            buildDataloadersFromConfig(config)


class TestTrainingConfigLookups:
    def test_supported_optimizer_names_include_adam(self) -> None:
        assert "adam" in listSupportedOptimizerNames()
        assert resolveOptimizerClass("adam") is torch.optim.Adam

    def test_supported_reconstruction_loss_names_include_mse(self) -> None:
        assert "mse" in listSupportedReconstructionLossNames()
        assert resolveReconstructionLossFn("mse") is torch.nn.functional.mse_loss

    def test_unknown_optimizer_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="does_not_exist"):
            resolveOptimizerClass("does_not_exist")

    def test_unknown_reconstruction_loss_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="does_not_exist"):
            resolveReconstructionLossFn("does_not_exist")


class TestBuildBetaSchedules:
    def test_builds_one_schedule_per_configured_latent_space(self) -> None:
        config = TrainingConfig(
            beta_schedules={
                "z_fused": BetaScheduleConfig(strategy="linear_warmup", kwargs={"warmup_steps": 10})
            }
        )
        schedules = buildBetaSchedules(config)
        assert set(schedules) == {"z_fused"}
        assert isinstance(schedules["z_fused"], LinearWarmupBetaSchedule)
        assert schedules["z_fused"](10) == pytest.approx(1.0)

    def test_empty_beta_schedules_gives_empty_dict(self) -> None:
        assert buildBetaSchedules(TrainingConfig()) == {}


class TestBuildCallbacksFromConfig:
    def test_empty_config_gives_no_callbacks(self) -> None:
        assert buildCallbacksFromConfig(TrainingConfig()) == []

    def test_loggers_are_instantiated_in_order(self, tmp_path: Path) -> None:
        config = TrainingConfig(
            loggers=[
                LoggerEntryConfig(name="csv", kwargs={"path": str(tmp_path / "m.csv")}),
                LoggerEntryConfig(name="tensorboard", kwargs={"log_dir": str(tmp_path / "tb")}),
            ]
        )
        callbacks = buildCallbacksFromConfig(config)
        assert [type(callback) for callback in callbacks] == [CsvLogger, TensorBoardLogger]

    def test_checkpoint_directory_adds_checkpoint_callback(self, tmp_path: Path) -> None:
        config = TrainingConfig(checkpoint=CheckpointConfig(directory=str(tmp_path)))
        callbacks = buildCallbacksFromConfig(config)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], CheckpointCallback)

    def test_best_path_adds_best_checkpoint_callback(self, tmp_path: Path) -> None:
        config = TrainingConfig(checkpoint=CheckpointConfig(best_path=str(tmp_path / "best.pt")))
        callbacks = buildCallbacksFromConfig(config)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], BestCheckpointCallback)

    def test_unknown_logger_name_raises_key_error(self) -> None:
        config = TrainingConfig(loggers=[LoggerEntryConfig(name="does_not_exist")])
        with pytest.raises(KeyError, match="does_not_exist"):
            buildCallbacksFromConfig(config)


class TestBuildTrainerFromConfig:
    def test_builds_a_real_trainer_with_resolved_optimizer(self) -> None:
        cfg = _loadSignalVaeConfig()
        model = buildModelFromConfig(cfg.model)
        trainer = buildTrainerFromConfig(model, cfg.training)
        assert isinstance(trainer, Trainer)
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)

    def test_beta_schedule_is_wired_and_used(self) -> None:
        cfg = _loadSignalVaeConfig(
            overrides=["training.beta_schedules.z_fused.kwargs.warmup_steps=100"]
        )
        model = buildModelFromConfig(cfg.model)
        trainer = buildTrainerFromConfig(model, cfg.training)
        assert trainer._computeBeta(0)["z_fused"] == pytest.approx(0.0)
        assert trainer._computeBeta(100)["z_fused"] == pytest.approx(1.0)

    def test_callbacks_include_configured_loggers_and_checkpoints(self, tmp_path: Path) -> None:
        cfg = _loadSignalVaeConfig(overrides=[f"output_dir={tmp_path}"])
        model = buildModelFromConfig(cfg.model)
        trainer = buildTrainerFromConfig(model, cfg.training, config_snapshot=cfg)
        callback_types = {type(callback) for callback in trainer.callbacks}
        assert callback_types == {
            CsvLogger,
            TensorBoardLogger,
            CheckpointCallback,
            BestCheckpointCallback,
        }

    def test_full_fit_run_end_to_end(self, tmp_path: Path) -> None:
        """The actual point of this whole module: config in, a trained model out."""
        cfg = _loadSignalVaeConfig(
            overrides=[
                f"output_dir={tmp_path}",
                "training.num_epochs=2",
                "training.beta_schedules.z_fused.kwargs.warmup_steps=5",
            ]
        )
        model = buildModelFromConfig(cfg.model)
        dataloaders = buildDataloadersFromConfig(cfg.data)
        trainer = buildTrainerFromConfig(model, cfg.training, config_snapshot=cfg)

        history = trainer.fit(
            dataloaders.train, num_epochs=cfg.training.num_epochs, val_dataloader=dataloaders.val
        )

        assert len(history) == 2
        for entry in history:
            assert torch.isfinite(torch.tensor(entry["train/loss/total"]))
        assert (tmp_path / "metrics.csv").exists()
        assert (tmp_path / "checkpoints" / "best.pt").exists()
