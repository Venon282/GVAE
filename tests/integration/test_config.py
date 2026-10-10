"""Integration tests for `global_vae.config` (spec §9, §10 "Config management").

Uses the repository's real `configs/` directory and real built-in encoder/decoder/
regularizer/beta-schedule/logger/callback implementations throughout (unlike most other
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
from hydra.errors import ConfigCompositionException, MissingConfigException
from omegaconf.errors import MissingMandatoryValue

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import DataConfig, DataloaderBundle, buildDataloadersFromConfig
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import ModelConfig, buildModelFromConfig
from global_vae.config.training import (
    BetaScheduleConfig,
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
from global_vae.training.callbacks.best_checkpoint import BestCheckpointCallback
from global_vae.training.callbacks.checkpoint import CheckpointCallback
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
        """`${output_dir}` in configs/training/default.yaml must resolve against `output_dir`.

        It resolves against the experiment-level output_dir: it must not fail or stay a
        literal string.
        """
        cfg = _loadSignalVaeConfig(overrides=["output_dir=/tmp/some_run"])
        assert cfg.training.callbacks["checkpoint"]["directory"] == "/tmp/some_run/checkpoints"
        assert (
            cfg.training.callbacks["best_checkpoint"]["path"] == "/tmp/some_run/checkpoints/best.pt"
        )
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

    def test_overriding_an_already_present_callback_needs_no_plus_prefix(self) -> None:
        cfg = _loadSignalVaeConfig(
            overrides=["training.callbacks.best_checkpoint.monitor=val/loss/reconstruction"]
        )
        assert cfg.training.callbacks["best_checkpoint"]["monitor"] == "val/loss/reconstruction"

    def test_adding_a_brand_new_callback_from_the_cli_needs_a_plus_prefix(self) -> None:
        """Hydra's own struct-mode rule for any dict field, not specific to `callbacks`.

        A dotlist override can change an existing key, but adding a key absent from every
        composed YAML file needs the `+` prefix. Writing the same key directly into a YAML
        config file (rather than a CLI override) needs no such prefix.
        """
        with pytest.raises(ConfigCompositionException, match="early_stopping"):
            _loadSignalVaeConfig(
                overrides=["training.callbacks.early_stopping.monitor=val/loss/total"]
            )

        cfg = _loadSignalVaeConfig(
            overrides=["+training.callbacks.early_stopping.monitor=val/loss/total"]
        )
        assert cfg.training.callbacks["early_stopping"] == {"monitor": "val/loss/total"}

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
        """signal_single_latent.yaml never repeats latent_dim in the encoder/decoder kwargs.

        That is the whole point of the auto-fill: it must still end up correct.
        """
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
        """default.yaml's two-modality example declares the fusion its two encoders need."""
        cfg = loadExperimentConfig(
            config_name="experiment/signal_vae",
            overrides=[*_BASE_OVERRIDES, "model=default"],
        )
        assert cfg.model.single_latent.fusion is not None
        assert cfg.model.single_latent.fusion.strategy == "poe"

    def test_two_modality_config_builds_a_signal_and_image_model(self) -> None:
        """default.yaml is buildable: a signal and an image encoder fused by PoE (ADR 0017)."""
        cfg = loadExperimentConfig(
            config_name="experiment/signal_vae",
            overrides=[*_BASE_OVERRIDES, "model=default"],
        )
        model = buildModelFromConfig(cfg.model)
        assert set(model.encoders) == {"signal", "image"}
        assert set(model.decoders) == {"signal", "image"}
        assert "z_fused" in model.fusions  # two encoders feed one latent space

        output = model({"signal": torch.randn(3, 256), "image": torch.randn(3, 1, 64, 64)})
        assert output["reconstructions"]["signal"].shape == (3, 256)
        assert output["reconstructions"]["image"].shape == (3, 64, 64)
        mu, logvar = output["latent_params"]["z_fused"]
        assert mu.shape == (3, 32)
        assert logvar.shape == (3, 32)

    def test_two_modality_config_still_encodes_one_modality_alone(self) -> None:
        """Hiding a modality at the encoder input still gives a latent (spec §5)."""
        cfg = loadExperimentConfig(
            config_name="experiment/signal_vae",
            overrides=[*_BASE_OVERRIDES, "model=default"],
        )
        model = buildModelFromConfig(cfg.model)
        output = model({"image": torch.randn(2, 1, 64, 64)})
        mu, _ = output["latent_params"]["z_fused"]
        assert mu.shape == (2, 32)

    def test_unregistered_encoder_name_raises_key_error(self) -> None:
        cfg = loadExperimentConfig(
            config_name="experiment/signal_vae",
            overrides=[
                *_BASE_OVERRIDES,
                "model=default",
                "model.modalities.image.encoder.name=not_a_registered_encoder_v1",
            ],
        )
        with pytest.raises(KeyError, match="not_a_registered_encoder_v1"):
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


class TestSignalResnetVaeExperiment:
    """The residual-variant configs: the same spec §6.1 milestone 1 shape, residual modules.

    `configs/model/signal_resnet_single_latent.yaml` and
    `configs/experiment/signal_resnet_vae.yaml` (spec §7,
    `docs/adr/0014-residual-1d-encoder-decoder.md`) have the same spec §6.1 milestone 1 shape as
    `signal_vae.yaml`/`signal_single_latent.yaml`, but compose the residual encoder/decoder
    instead of the plain conv stack. They are selectable either as their own named experiment
    file, or as a `model=...` override on top of the existing `signal_vae.yaml` (both ways are
    exercised below, since both are documented as supported entry points).
    """

    def test_dedicated_experiment_file_selects_the_resnet_model(self) -> None:
        cfg = loadExperimentConfig(
            config_name="experiment/signal_resnet_vae", overrides=_BASE_OVERRIDES
        )
        assert cfg.model.name == "global_vae_signal_resnet_single_latent"
        assert cfg.model.modalities["signal"].encoder.name == "1d_cnn_resnet_encoder_v1"
        assert cfg.model.modalities["signal"].decoder.name == "1d_cnn_resnet_decoder_v1"

    def test_model_group_override_selects_the_same_resnet_model(self) -> None:
        """The second documented way to reach the same model: override the `model` config group.

        This is done directly on top of the default experiment file, exactly like
        `test_two_modality_config_needs_a_fusion_strategy` already does for `model=default`.
        """
        cfg = _loadSignalVaeConfig(overrides=["model=signal_resnet_single_latent"])
        assert cfg.model.name == "global_vae_signal_resnet_single_latent"

    def test_builds_the_real_residual_encoder_and_decoder(self) -> None:
        cfg = loadExperimentConfig(
            config_name="experiment/signal_resnet_vae", overrides=_BASE_OVERRIDES
        )
        model = buildModelFromConfig(cfg.model)
        assert type(model.encoders["signal"]).__name__ == "OneDCnnResidualEncoder"
        assert type(model.decoders["signal"]).__name__ == "OneDCnnResidualDecoder"
        assert "z_fused" not in model.fusions  # single encoder: no fusion module built

    def test_forward_pass_shapes_match_configured_dims(self) -> None:
        cfg = loadExperimentConfig(
            config_name="experiment/signal_resnet_vae", overrides=_BASE_OVERRIDES
        )
        model = buildModelFromConfig(cfg.model)
        output = model({"signal": torch.randn(3, 256)})
        assert output["reconstructions"]["signal"].shape == (3, 256)
        mu, logvar = output["latent_params"]["z_fused"]
        assert mu.shape == (3, 16)
        assert logvar.shape == (3, 16)

    def test_per_stage_block_depths_reach_the_model(self) -> None:
        """`block_depths` must actually reach the constructed modules, not just be schema-valid.

        This is the whole point of the residual variant.
        """
        cfg = loadExperimentConfig(
            config_name="experiment/signal_resnet_vae", overrides=_BASE_OVERRIDES
        )
        assert cfg.model.modalities["signal"].encoder.kwargs["block_depths"] == [2, 2, 3, 3, 2]
        model = buildModelFromConfig(cfg.model)
        # 5 stages configured -> 5 residual blocks in the encoder's conv stack
        # (each stage contributes exactly one Residual1DBlock plus, optionally, one
        # pooling layer; block_depths controls each block's own internal layer count,
        # not how many blocks exist).
        residual_blocks = [
            module
            for module in model.encoders["signal"].conv
            if type(module).__name__ == "Residual1DBlock"
        ]
        assert len(residual_blocks) == 5

    def test_gradients_flow_through_the_residual_stack(self) -> None:
        cfg = loadExperimentConfig(
            config_name="experiment/signal_resnet_vae", overrides=_BASE_OVERRIDES
        )
        model = buildModelFromConfig(cfg.model)
        output = model({"signal": torch.randn(2, 256)})
        reconstruction_loss = output["reconstructions"]["signal"].pow(2).mean()
        regularization_loss = model.computeRegularizationLoss(output["latent_params"])
        (reconstruction_loss + regularization_loss).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"parameter '{name}' got no gradient"

    def test_full_fit_run_end_to_end(self, tmp_path: Path) -> None:
        cfg = loadExperimentConfig(
            config_name="experiment/signal_resnet_vae",
            overrides=[
                *_BASE_OVERRIDES,
                f"output_dir={tmp_path}",
                "training.num_epochs=2",
                "training.beta_schedules.z_fused.kwargs.warmup_steps=5",
            ],
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
        assert (tmp_path / "checkpoints" / "best.pt").exists()


class TestDataConfig:
    def test_build_dataloaders_from_config_resolves_and_calls_the_factory(self) -> None:
        cfg = _loadSignalVaeConfig()
        bundle = buildDataloadersFromConfig(cfg.data)
        assert isinstance(bundle, DataloaderBundle)
        assert len(list(bundle.train)) == 3
        assert bundle.val is not None and len(list(bundle.val)) == 1

    def test_sequence_length_reaches_the_dummy_factory(self) -> None:
        cfg = _loadSignalVaeConfig(overrides=["data.sequence_length.signal=64"])
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

    def test_unknown_logger_name_raises_key_error(self) -> None:
        config = TrainingConfig(loggers=[LoggerEntryConfig(name="does_not_exist")])
        with pytest.raises(KeyError, match="does_not_exist"):
            buildCallbacksFromConfig(config)

    def test_callbacks_are_instantiated_in_order(self, tmp_path: Path) -> None:
        config = TrainingConfig(
            callbacks={
                "best_checkpoint": {"path": str(tmp_path / "best.pt")},
                "checkpoint": {"directory": str(tmp_path)},
            }
        )
        callbacks = buildCallbacksFromConfig(config)
        assert [type(callback) for callback in callbacks] == [
            BestCheckpointCallback,
            CheckpointCallback,
        ]

    def test_reordering_the_mapping_reorders_the_callbacks(self, tmp_path: Path) -> None:
        config = TrainingConfig(
            callbacks={
                "checkpoint": {"directory": str(tmp_path)},
                "best_checkpoint": {"path": str(tmp_path / "best.pt")},
            }
        )
        callbacks = buildCallbacksFromConfig(config)
        assert [type(callback) for callback in callbacks] == [
            CheckpointCallback,
            BestCheckpointCallback,
        ]

    def test_loggers_always_come_before_registry_callbacks(self, tmp_path: Path) -> None:
        config = TrainingConfig(
            loggers=[LoggerEntryConfig(name="csv", kwargs={"path": str(tmp_path / "m.csv")})],
            callbacks={"checkpoint": {"directory": str(tmp_path)}},
        )
        callbacks = buildCallbacksFromConfig(config)
        assert [type(callback) for callback in callbacks] == [CsvLogger, CheckpointCallback]

    def test_checkpoint_entry_builds_a_checkpoint_callback(self, tmp_path: Path) -> None:
        config = TrainingConfig(callbacks={"checkpoint": {"directory": str(tmp_path)}})
        callbacks = buildCallbacksFromConfig(config)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], CheckpointCallback)

    def test_best_checkpoint_entry_builds_a_best_checkpoint_callback(self, tmp_path: Path) -> None:
        config = TrainingConfig(callbacks={"best_checkpoint": {"path": str(tmp_path / "best.pt")}})
        callbacks = buildCallbacksFromConfig(config)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], BestCheckpointCallback)

    def test_none_value_means_every_default(self, tmp_path: Path) -> None:
        """A bare `some_name:` key in YAML (no nested kwargs) parses to `None`."""
        config = TrainingConfig(callbacks={"early_stopping": None})
        callbacks = buildCallbacksFromConfig(config)
        assert len(callbacks) == 1
        assert callbacks[0].monitor == "val/loss/total"  # EarlyStopping's own default

    def test_unknown_callback_name_raises_key_error(self) -> None:
        config = TrainingConfig(callbacks={"does_not_exist": {}})
        with pytest.raises(KeyError, match="does_not_exist"):
            buildCallbacksFromConfig(config)

    def test_config_snapshot_is_forwarded_only_to_callbacks_that_accept_it(
        self, tmp_path: Path
    ) -> None:
        """`CheckpointCallback` declares a `config` parameter and receives the snapshot.

        `EarlyStopping` does not, and is built with no such kwarg (it would raise
        `TypeError` if one were forwarded).
        """
        config = TrainingConfig(
            callbacks={
                "early_stopping": {},
                "checkpoint": {"directory": str(tmp_path)},
            }
        )
        callbacks = buildCallbacksFromConfig(config, config_snapshot={"some": "snapshot"})
        checkpoint_callback = next(c for c in callbacks if isinstance(c, CheckpointCallback))
        assert checkpoint_callback.config == {"some": "snapshot"}

    def test_an_explicit_config_kwarg_is_not_overridden(self, tmp_path: Path) -> None:
        config = TrainingConfig(
            callbacks={"checkpoint": {"directory": str(tmp_path), "config": "explicit"}}
        )
        callbacks = buildCallbacksFromConfig(config, config_snapshot={"some": "snapshot"})
        assert callbacks[0].config == "explicit"


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
