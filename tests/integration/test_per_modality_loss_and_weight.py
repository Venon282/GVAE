"""Integration tests for a reconstruction loss and weight per modality (roadmap P2-2, ADR 0030).

`training.reconstruction_loss` and `training.reconstruction_weight` take a single value for every
modality or a mapping decoder name -> value. Tested here, through several components:

- Hydra overrides, through `buildTrainerFromConfig`, into `Trainer`;
- the resolvers, against the decoders of a real model;
- the numbers: a per-modality setting reaches the loss the trainer computes, checked by
  linearity against the same model with one modality weighted at a time;
- the settings applied together with a reduction (P2-1), in training;
- `scripts/evaluate.py` with `--reconstruction-loss` and `--reconstruction-weight`.

The two-modality model of `_script_fixtures.py` (decoders `x` and `y`, 16 values each) gives the
numbers; the signal and image experiment of the shipped configs gives the names a real config
resolves against.
"""

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from omegaconf import DictConfig, OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    OptimizerConfig,
    TrainingConfig,
    buildTrainerFromConfig,
    resolveReconstructionLoss,
    resolveReconstructionReduction,
    resolveReconstructionWeight,
)
from global_vae.evaluation.evaluate import evaluate
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import saveCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from tests.integration._script_fixtures import (
    INPUT_DIM,
    buildModelForScript,
    buildTwoModalityDataloaderForScript,
    buildTwoModalityModelForScript,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "evaluate.py"
_LOADER_FACTORY = "tests.integration._train_script_fixtures:buildDummyDataloaders"
_SIGNAL_IMAGE_LOADER_FACTORY = (
    "tests.integration._train_script_fixtures:buildDummySignalImageDataloaders"
)


def _loadSingle(tmp_path: Path, *overrides: str) -> ExperimentConfig:
    """The default experiment: one `signal` modality."""
    return loadExperimentConfig(
        overrides=[
            f"data.loader_factory={_LOADER_FACTORY}",
            "data.train_path=/unused",
            f"output_dir={tmp_path}",
            *overrides,
        ]
    )


def _loadSignalImage(tmp_path: Path, *overrides: str) -> ExperimentConfig:
    """The shipped signal and image experiment: decoders `image` and `signal`."""
    return loadExperimentConfig(
        config_name="experiment/signal_image_shared_private",
        overrides=[
            f"data.loader_factory={_SIGNAL_IMAGE_LOADER_FACTORY}",
            "data.train_path=/unused",
            f"output_dir={tmp_path}",
            *overrides,
        ],
    )


def _buildTrainer(model: GlobalVae, **fields: object) -> Trainer:
    """A trainer from a `TrainingConfig` with these fields, on the CPU."""
    return buildTrainerFromConfig(model, TrainingConfig(device="cpu", **fields))  # type: ignore[arg-type]


def _reconstructionLoss(model: GlobalVae, **fields: object) -> float:
    """The reconstruction term of the first batch, with the posterior sample fixed by a seed."""
    trainer = _buildTrainer(model, **fields)
    torch.manual_seed(0)
    batch = buildTwoModalityDataloaderForScript()[0]
    return trainer.computeLosses(batch, 0, apply_dropout=False).reconstruction.item()


class TestShippedConfigs:
    @pytest.mark.parametrize("name", ["default", "several_latents"])
    def test_the_shipped_training_files_write_the_defaults_explicitly(self, name: str) -> None:
        raw = cast(DictConfig, OmegaConf.load(_REPO_ROOT / "configs" / "training" / f"{name}.yaml"))
        assert raw.reconstruction_loss == "mse"
        assert raw.reconstruction_weight == 1.0

    def test_a_config_that_does_not_mention_them_loads_the_defaults(self, tmp_path: Path) -> None:
        cfg = _loadSingle(tmp_path)
        assert cfg.training.reconstruction_loss == "mse"
        assert cfg.training.reconstruction_weight == 1.0


class TestConfigToTrainer:
    def test_the_defaults_give_the_trainer_the_loss_and_weight_it_always_had(
        self, tmp_path: Path
    ) -> None:
        cfg = _loadSingle(tmp_path)
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_loss_fn is F.mse_loss
        assert trainer.reconstruction_weights == 1.0
        assert type(trainer.reconstruction_weights) is float

    def test_the_defaults_compute_the_same_loss_as_a_trainer_built_without_a_config(
        self, tmp_path: Path
    ) -> None:
        """Bit for bit: the new resolvers change nothing for a run that sets neither field."""
        model = buildTwoModalityModelForScript()
        batch = buildTwoModalityDataloaderForScript()[0]
        plain = Trainer(model, device="cpu")
        configured = _buildTrainer(model)
        values = []
        for trainer in (plain, configured):
            torch.manual_seed(0)
            values.append(trainer.computeLosses(batch, 0, apply_dropout=False).reconstruction)
        assert torch.equal(values[0], values[1])

    def test_a_loss_name_override_reaches_the_trainer(self, tmp_path: Path) -> None:
        cfg = _loadSingle(tmp_path, "training.reconstruction_loss=l1")
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_loss_fn is F.l1_loss

    def test_a_weight_override_reaches_the_trainer(self, tmp_path: Path) -> None:
        cfg = _loadSingle(tmp_path, "training.reconstruction_weight=0.5")
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_weights == 0.5

    def test_an_integer_weight_becomes_a_float(self, tmp_path: Path) -> None:
        cfg = _loadSingle(tmp_path, "training.reconstruction_weight=2")
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_weights == 2.0
        assert type(trainer.reconstruction_weights) is float

    def test_a_loss_mapping_override_is_completed_with_the_default_for_the_other_decoders(
        self, tmp_path: Path
    ) -> None:
        cfg = _loadSignalImage(tmp_path, "training.reconstruction_loss={signal:l1}")
        assert cfg.training.reconstruction_loss == {"signal": "l1"}
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_loss_fn == {"signal": F.l1_loss, "image": F.mse_loss}

    def test_a_weight_mapping_override_reaches_the_trainer_as_a_plain_dict_of_floats(
        self, tmp_path: Path
    ) -> None:
        cfg = _loadSignalImage(tmp_path, "training.reconstruction_weight={image:2,signal:0.5}")
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        weights = trainer.reconstruction_weights
        assert type(weights) is dict
        assert weights == {"image": 2.0, "signal": 0.5}
        assert all(type(value) is float for value in weights.values())

    def test_a_decoder_left_out_of_the_weight_mapping_is_not_given_an_entry(
        self, tmp_path: Path
    ) -> None:
        """`computeTotalReconstructionLoss` gives a decoder with no entry the weight 1.0."""
        cfg = _loadSignalImage(tmp_path, "training.reconstruction_weight={image:2}")
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_weights == {"image": 2.0}

    def test_the_three_mappings_can_be_set_together(self, tmp_path: Path) -> None:
        cfg = _loadSignalImage(
            tmp_path,
            "training.reconstruction_loss={image:smooth_l1}",
            "training.reconstruction_weight={image:2}",
            "training.reconstruction_reduction={image:sum}",
        )
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_loss_fn == {"signal": F.mse_loss, "image": F.smooth_l1_loss}
        assert trainer.reconstruction_weights == {"image": 2.0}
        assert trainer.reconstruction_reduction == {"image": "sum"}

    def test_an_unknown_loss_name_is_a_key_error_naming_the_field(self, tmp_path: Path) -> None:
        cfg = _loadSingle(tmp_path, "training.reconstruction_loss=huber")
        with pytest.raises(KeyError) as excinfo:
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert str(excinfo.value.args[0]).startswith(
            "training.reconstruction_loss: unknown loss 'huber'"
        )

    def test_an_unknown_loss_name_in_a_mapping_names_the_decoder(self, tmp_path: Path) -> None:
        cfg = _loadSingle(tmp_path, "training.reconstruction_loss={signal:huber}")
        with pytest.raises(KeyError) as excinfo:
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert str(excinfo.value.args[0]).startswith(
            "training.reconstruction_loss['signal']: unknown loss 'huber'"
        )

    def test_a_mistyped_decoder_in_the_loss_is_an_error_and_not_a_silent_mse(
        self, tmp_path: Path
    ) -> None:
        cfg = _loadSignalImage(tmp_path, "training.reconstruction_loss={sinal:l1}")
        with pytest.raises(ValueError, match=r"\['sinal'\].*decoders: \['image', 'signal'\]"):
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)

    def test_a_mistyped_decoder_in_the_weight_is_an_error_and_not_a_silent_one(
        self, tmp_path: Path
    ) -> None:
        cfg = _loadSignalImage(tmp_path, "training.reconstruction_weight={imge:2}")
        with pytest.raises(ValueError, match=r"^training\.reconstruction_weight names.*\['imge'\]"):
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)

    @pytest.mark.parametrize("value", ["-1", "-0.5", "true", "abc", "null", "[1,2]"])
    def test_a_weight_that_is_not_a_number_at_least_zero_is_refused(
        self, tmp_path: Path, value: str
    ) -> None:
        cfg = _loadSingle(tmp_path, f"training.reconstruction_weight={value}")
        with pytest.raises(ValueError, match=r"^training\.reconstruction_weight"):
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)

    def test_a_negative_weight_inside_a_mapping_names_the_decoder(self, tmp_path: Path) -> None:
        cfg = _loadSingle(tmp_path, "training.reconstruction_weight={signal:-1}")
        with pytest.raises(ValueError, match=r"^training\.reconstruction_weight\['signal'\]"):
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)

    @pytest.mark.parametrize("value", ["3", "null", "[l1]"])
    def test_a_loss_that_is_not_a_name_or_a_mapping_is_refused(
        self, tmp_path: Path, value: str
    ) -> None:
        cfg = _loadSingle(tmp_path, f"training.reconstruction_loss={value}")
        with pytest.raises(ValueError, match=r"^training\.reconstruction_loss: expected a loss"):
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)


class TestResolvers:
    def test_a_loss_name_resolves_to_one_function(self) -> None:
        model = buildTwoModalityModelForScript()
        assert resolveReconstructionLoss("l1", model) is F.l1_loss

    def test_a_loss_mapping_resolves_to_an_entry_for_every_decoder(self) -> None:
        model = buildTwoModalityModelForScript()
        resolved = resolveReconstructionLoss({"y": "l1"}, model)
        assert resolved == {"x": F.mse_loss, "y": F.l1_loss}

    def test_an_empty_loss_mapping_resolves_to_the_default_for_every_decoder(self) -> None:
        model = buildTwoModalityModelForScript()
        assert resolveReconstructionLoss({}, model) == {"x": F.mse_loss, "y": F.mse_loss}

    def test_a_weight_resolves_to_a_float(self) -> None:
        model = buildTwoModalityModelForScript()
        assert resolveReconstructionWeight(2, model) == 2.0
        assert type(resolveReconstructionWeight(2, model)) is float

    def test_a_weight_mapping_resolves_to_a_new_dict_of_floats(self) -> None:
        model = buildTwoModalityModelForScript()
        original = {"x": 2}
        resolved = resolveReconstructionWeight(original, model)
        assert resolved == {"x": 2.0}
        assert resolved is not original
        assert original == {"x": 2}

    @pytest.mark.parametrize("value", [OmegaConf.create({"x": 2}), {"x": 2}])
    def test_a_weight_mapping_can_be_an_omegaconf_container(self, value: object) -> None:
        resolved = resolveReconstructionWeight(value, buildTwoModalityModelForScript())
        assert type(resolved) is dict
        assert resolved == {"x": 2.0}

    def test_a_loss_mapping_can_be_an_omegaconf_container(self) -> None:
        resolved = resolveReconstructionLoss(
            OmegaConf.create({"y": "l1"}), buildTwoModalityModelForScript()
        )
        assert type(resolved) is dict
        assert resolved == {"x": F.mse_loss, "y": F.l1_loss}

    @pytest.mark.parametrize(
        ("resolver", "value", "default"),
        [
            (resolveReconstructionLoss, {"z": "l1"}, "'mse'"),
            (resolveReconstructionWeight, {"z": 2.0}, "1.0"),
            (resolveReconstructionReduction, {"z": "sum"}, "'mean'"),
        ],
    )
    def test_an_unknown_decoder_is_refused_and_the_message_says_what_it_would_have_kept(
        self, resolver: object, value: object, default: str
    ) -> None:
        model = buildTwoModalityModelForScript()
        with pytest.raises(ValueError) as excinfo:
            resolver(value, model)  # type: ignore[operator]
        message = str(excinfo.value)
        assert "['z']" in message
        assert "decoders: ['x', 'y']" in message
        assert f"keeps {default}" in message

    @pytest.mark.parametrize(
        ("resolver", "value"),
        [
            (resolveReconstructionLoss, {"z": "l1"}),
            (resolveReconstructionWeight, {"z": 2.0}),
            (resolveReconstructionReduction, {"z": "sum"}),
        ],
    )
    def test_the_field_name_in_the_message_can_be_changed(
        self, resolver: object, value: object
    ) -> None:
        with pytest.raises(ValueError, match=r"^--some-flag names decoder"):
            resolver(value, buildTwoModalityModelForScript(), "--some-flag")  # type: ignore[operator]

    def test_a_decoder_that_is_right_but_a_value_that_is_wrong_is_refused_for_the_value(
        self,
    ) -> None:
        model = buildTwoModalityModelForScript()
        with pytest.raises(KeyError, match="unknown loss 'huber'"):
            resolveReconstructionLoss({"x": "huber"}, model)
        with pytest.raises(ValueError, match="must be a finite number >= 0"):
            resolveReconstructionWeight({"x": -1}, model)

    def test_the_trainer_does_not_complete_a_loss_dict_itself(self) -> None:
        """Why the resolver adds an entry for every decoder: the `Trainer` would raise."""
        model = buildTwoModalityModelForScript()
        trainer = Trainer(model, device="cpu", reconstruction_loss_fn={"x": F.mse_loss})
        batch = buildTwoModalityDataloaderForScript()[0]
        with pytest.raises(KeyError, match="No loss function provided for reconstruction 'y'"):
            trainer.computeLosses(batch, 0, apply_dropout=False)


class TestTheNumbers:
    """A mapping reaches the loss.

    Each expected value is built from the same model with one modality weighted at a time
    (weights `{x: 1, y: 0}` and `{x: 0, y: 1}`), never from the code under test.
    """

    def test_the_two_isolated_terms_add_up_to_the_default_total(self) -> None:
        model = buildTwoModalityModelForScript()
        loss_x = _reconstructionLoss(model, reconstruction_weight={"x": 1.0, "y": 0.0})
        loss_y = _reconstructionLoss(model, reconstruction_weight={"x": 0.0, "y": 1.0})
        assert loss_x > 0
        assert loss_y > 0
        assert _reconstructionLoss(model) == pytest.approx(loss_x + loss_y, rel=1e-5)

    def test_a_single_weight_scales_every_modality(self) -> None:
        model = buildTwoModalityModelForScript()
        assert _reconstructionLoss(model, reconstruction_weight=0.5) == pytest.approx(
            0.5 * _reconstructionLoss(model), rel=1e-5
        )

    def test_a_weight_for_one_decoder_leaves_the_other_at_one(self) -> None:
        model = buildTwoModalityModelForScript()
        loss_x = _reconstructionLoss(model, reconstruction_weight={"x": 1.0, "y": 0.0})
        loss_y = _reconstructionLoss(model, reconstruction_weight={"x": 0.0, "y": 1.0})
        weighted = _reconstructionLoss(model, reconstruction_weight={"x": 2.0})
        assert weighted == pytest.approx(2.0 * loss_x + loss_y, rel=1e-5)

    def test_a_weight_for_each_decoder(self) -> None:
        model = buildTwoModalityModelForScript()
        loss_x = _reconstructionLoss(model, reconstruction_weight={"x": 1.0, "y": 0.0})
        loss_y = _reconstructionLoss(model, reconstruction_weight={"x": 0.0, "y": 1.0})
        weighted = _reconstructionLoss(model, reconstruction_weight={"x": 0.25, "y": 3.0})
        assert weighted == pytest.approx(0.25 * loss_x + 3.0 * loss_y, rel=1e-5)

    def test_a_zero_weight_removes_the_modality_from_the_loss(self) -> None:
        model = buildTwoModalityModelForScript()
        loss_y = _reconstructionLoss(model, reconstruction_weight={"x": 0.0, "y": 1.0})
        assert _reconstructionLoss(model, reconstruction_weight={"x": 0.0}) == pytest.approx(
            loss_y, rel=1e-5
        )
        assert _reconstructionLoss(model, reconstruction_weight=0.0) == 0.0

    def test_a_loss_for_one_decoder_changes_that_term_only(self) -> None:
        model = buildTwoModalityModelForScript()
        only_x = {"x": 1.0, "y": 0.0}
        only_y = {"x": 0.0, "y": 1.0}
        loss_x_mse = _reconstructionLoss(model, reconstruction_weight=only_x)
        loss_y_mse = _reconstructionLoss(model, reconstruction_weight=only_y)
        loss_y_l1 = _reconstructionLoss(
            model, reconstruction_loss="l1", reconstruction_weight=only_y
        )
        assert loss_y_l1 != pytest.approx(loss_y_mse, rel=1e-3)  # the two losses do differ here
        mixed = _reconstructionLoss(model, reconstruction_loss={"y": "l1"})
        assert mixed == pytest.approx(loss_x_mse + loss_y_l1, rel=1e-5)

    def test_every_supported_loss_can_be_chosen_for_one_decoder(self) -> None:
        """`bce` needs targets and reconstructions in [0, 1], which these dummy decoders lack."""
        model = buildTwoModalityModelForScript()
        for name in ("mse", "l1", "smooth_l1"):
            value = _reconstructionLoss(model, reconstruction_loss={"x": name})
            assert math.isfinite(value)
            assert value > 0

    def test_loss_weight_and_reduction_compose_in_that_order(self) -> None:
        """`x` is the l1 loss times `N` (sum) times 2.0 (weight); `y` is the mse loss times 3.0."""
        model = buildTwoModalityModelForScript()
        loss_x_l1 = _reconstructionLoss(
            model, reconstruction_loss="l1", reconstruction_weight={"x": 1.0, "y": 0.0}
        )
        loss_y_mse = _reconstructionLoss(model, reconstruction_weight={"x": 0.0, "y": 1.0})
        trainer = _buildTrainer(
            model,
            reconstruction_loss={"x": "l1"},
            reconstruction_weight={"x": 2.0, "y": 3.0},
            reconstruction_reduction={"x": "sum"},
        )
        torch.manual_seed(0)
        batch = buildTwoModalityDataloaderForScript()[0]
        combined = trainer.computeLosses(batch, 0, apply_dropout=False).reconstruction.item()
        assert combined == pytest.approx(2.0 * INPUT_DIM * loss_x_l1 + 3.0 * loss_y_mse, rel=1e-5)

    def test_the_regularization_term_does_not_depend_on_the_loss_or_the_weight(self) -> None:
        model = buildTwoModalityModelForScript()
        batch = buildTwoModalityDataloaderForScript()[0]
        values = []
        for fields in (
            {},
            {"reconstruction_loss": {"x": "l1"}, "reconstruction_weight": {"y": 5.0}},
        ):
            torch.manual_seed(0)
            values.append(
                _buildTrainer(model, **fields)
                .computeLosses(batch, 0, apply_dropout=False)
                .regularization.item()
            )
        assert values[0] == pytest.approx(values[1], rel=1e-6)

    def test_the_validation_loss_uses_the_same_weight(self) -> None:
        model = buildTwoModalityModelForScript()
        batches = buildTwoModalityDataloaderForScript()

        def validationLoss(weight: float | dict[str, float]) -> float:
            torch.manual_seed(0)
            trainer = _buildTrainer(model, reconstruction_weight=weight)
            return trainer.evaluate(batches)["val/loss/reconstruction"]

        only_x = validationLoss({"x": 1.0, "y": 0.0})
        assert validationLoss({"x": 2.0, "y": 0.0}) == pytest.approx(2.0 * only_x, rel=1e-5)
        assert validationLoss(1.0) > only_x


class TestTraining:
    def test_a_zero_weight_decoder_gets_no_gradient_and_the_other_does(self) -> None:
        model = buildTwoModalityModelForScript()
        trainer = _buildTrainer(model, reconstruction_weight={"x": 0.0})
        batch = buildTwoModalityDataloaderForScript()[0]
        backward(trainer.computeLosses(batch, 0, apply_dropout=False).total)
        for parameter in model.decoders["x"].parameters():
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) == 0
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in model.decoders["y"].parameters()
        )

    def test_training_with_a_loss_a_weight_and_a_reduction_per_decoder_runs_and_the_loss_falls(
        self,
    ) -> None:
        trainer = _buildTrainer(
            buildTwoModalityModelForScript(),
            reconstruction_loss={"y": "smooth_l1"},
            reconstruction_weight={"x": 0.5, "y": 2.0},
            reconstruction_reduction={"x": "sum"},
            optimizer=OptimizerConfig(name="adam", kwargs={"lr": 1e-2}),
        )
        history = trainer.fit(buildTwoModalityDataloaderForScript(), num_epochs=15)
        assert all(math.isfinite(entry["train/loss/total"]) for entry in history)
        assert history[-1]["train/loss/total"] < history[0]["train/loss/total"]


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_evaluate_script_p2_2", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def two_modality_checkpoint(tmp_path: Path) -> tuple[GlobalVae, Path]:
    """A saved two-modality model, and that same model, so that a report can be recomputed."""
    model = buildTwoModalityModelForScript()
    path = tmp_path / "two.pt"
    saveCheckpoint(path, model=model)
    return model, path


def _runScript(script: ModuleType, checkpoint: Path, *extra: str, two_modalities: bool) -> None:
    factory = "buildTwoModalityModelForScript" if two_modalities else "buildModelForScript"
    loader = "buildTwoModalityDataloaderForScript" if two_modalities else "buildDataloaderForScript"
    script.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--model-factory",
            f"tests.integration._script_fixtures:{factory}",
            "--dataloader-factory",
            f"tests.integration._script_fixtures:{loader}",
            "--device",
            "cpu",
            *extra,
        ]
    )


def _report(
    script: ModuleType, checkpoint: Path, output_dir: Path, *flags: str, two_modalities: bool = True
) -> dict[str, Any]:
    _runScript(
        script,
        checkpoint,
        *flags,
        "--output-dir",
        str(output_dir),
        "--no-figures",
        two_modalities=two_modalities,
    )
    report: dict[str, Any] = json.loads((output_dir / "results.json").read_text())
    return report


class TestEvaluateScript:
    def test_the_defaults_are_the_flags_written_out(
        self,
        script: ModuleType,
        two_modality_checkpoint: tuple[GlobalVae, Path],
        tmp_path: Path,
    ) -> None:
        _, checkpoint = two_modality_checkpoint
        implicit = _report(script, checkpoint, tmp_path / "a")
        explicit = _report(
            script,
            checkpoint,
            tmp_path / "b",
            "--reconstruction-loss",
            "mse",
            "--reconstruction-weight",
            "1",
        )
        assert implicit == explicit

    def test_a_weight_scales_the_total_and_nothing_else(
        self,
        script: ModuleType,
        two_modality_checkpoint: tuple[GlobalVae, Path],
        tmp_path: Path,
    ) -> None:
        _, checkpoint = two_modality_checkpoint
        base = _report(script, checkpoint, tmp_path / "a")
        half = _report(script, checkpoint, tmp_path / "b", "--reconstruction-weight", "0.5")
        assert half["total_reconstruction_loss"] == pytest.approx(
            0.5 * base["total_reconstruction_loss"], rel=1e-5
        )
        assert half["reconstruction_metrics"] == base["reconstruction_metrics"]
        assert half["latent_diagnostics"] == base["latent_diagnostics"]

    def test_a_weight_per_modality_matches_the_per_modality_metrics(
        self,
        script: ModuleType,
        two_modality_checkpoint: tuple[GlobalVae, Path],
        tmp_path: Path,
    ) -> None:
        """The metrics are unweighted mean squared errors, so the total is `2 * mse_x + mse_y`.

        Both batches have the same size, so the mean of the batch means is the pooled mean.
        """
        _, checkpoint = two_modality_checkpoint
        report = _report(script, checkpoint, tmp_path, "--reconstruction-weight", "x=2")
        metrics = report["reconstruction_metrics"]
        expected = 2.0 * metrics["x"]["mse"] + metrics["y"]["mse"]
        assert report["total_reconstruction_loss"] == pytest.approx(expected, rel=1e-4)

    def test_a_loss_per_modality_matches_the_library_call(
        self,
        script: ModuleType,
        two_modality_checkpoint: tuple[GlobalVae, Path],
        tmp_path: Path,
    ) -> None:
        model, checkpoint = two_modality_checkpoint
        report = _report(script, checkpoint, tmp_path, "--reconstruction-loss", "y=l1")
        expected = evaluate(
            model,
            buildTwoModalityDataloaderForScript(),
            reconstruction_loss_fn={"x": F.mse_loss, "y": F.l1_loss},
            device="cpu",
        )
        assert report["total_reconstruction_loss"] == pytest.approx(
            expected.total_reconstruction_loss, rel=1e-5
        )
        mse_everywhere = _report(script, checkpoint, tmp_path / "mse")
        assert report["total_reconstruction_loss"] != pytest.approx(
            mse_everywhere["total_reconstruction_loss"], rel=1e-3
        )

    def test_the_three_flags_compose_loss_then_reduction_then_weight(
        self,
        script: ModuleType,
        two_modality_checkpoint: tuple[GlobalVae, Path],
        tmp_path: Path,
    ) -> None:
        """`x` is the mse loss times `N` (sum) times 0.5; `y` is the l1 loss times 2.0.

        Each isolated term comes from `evaluate()` with the other modality weighted by zero.
        """
        model, checkpoint = two_modality_checkpoint
        report = _report(
            script,
            checkpoint,
            tmp_path,
            "--reconstruction-loss",
            "y=l1",
            "--reconstruction-weight",
            "x=0.5,y=2",
            "--reconstruction-reduction",
            "x=sum",
        )
        loader = buildTwoModalityDataloaderForScript()
        loss_x = evaluate(
            model, loader, reconstruction_weights={"x": 1.0, "y": 0.0}, device="cpu"
        ).total_reconstruction_loss
        loss_y = evaluate(
            model,
            loader,
            reconstruction_weights={"x": 0.0, "y": 1.0},
            reconstruction_loss_fn=F.l1_loss,
            device="cpu",
        ).total_reconstruction_loss
        assert report["total_reconstruction_loss"] == pytest.approx(
            0.5 * INPUT_DIM * loss_x + 2.0 * loss_y, rel=1e-5
        )

    def test_the_single_modality_flags_take_a_bare_value(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        path = tmp_path / "single.pt"
        saveCheckpoint(path, model=buildModelForScript())
        base = _report(script, path, tmp_path / "a", two_modalities=False)
        weighted = _report(
            script,
            path,
            tmp_path / "b",
            "--reconstruction-weight",
            "3",
            "--reconstruction-loss",
            "mse",
            two_modalities=False,
        )
        assert weighted["total_reconstruction_loss"] == pytest.approx(
            3.0 * base["total_reconstruction_loss"], rel=1e-5
        )

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--reconstruction-loss", "z=l1"),
            ("--reconstruction-weight", "z=2"),
            ("--reconstruction-reduction", "z=sum"),
        ],
    )
    def test_a_modality_the_model_does_not_have_is_a_usage_error(
        self,
        script: ModuleType,
        two_modality_checkpoint: tuple[GlobalVae, Path],
        capsys: pytest.CaptureFixture[str],
        flag: str,
        value: str,
    ) -> None:
        _, checkpoint = two_modality_checkpoint
        with pytest.raises(SystemExit) as excinfo:
            _runScript(script, checkpoint, flag, value, two_modalities=True)
        assert excinfo.value.code == 2
        error = capsys.readouterr().err
        assert f"{flag} names decoder(s) ['z']" in error
        assert "decoders: ['x', 'y']" in error

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--reconstruction-loss", "huber"),
            ("--reconstruction-loss", "x=huber"),
            ("--reconstruction-loss", "x=l1,y"),
            ("--reconstruction-weight", "abc"),
            ("--reconstruction-weight", "-1"),
            ("--reconstruction-weight", "nan"),
            ("--reconstruction-weight", "x=-1"),
            ("--reconstruction-weight", "x=abc"),
            ("--reconstruction-weight", "x=1,y"),
            ("--reconstruction-weight", "=1"),
        ],
    )
    def test_a_malformed_value_is_a_usage_error(
        self,
        script: ModuleType,
        two_modality_checkpoint: tuple[GlobalVae, Path],
        flag: str,
        value: str,
    ) -> None:
        _, checkpoint = two_modality_checkpoint
        with pytest.raises(SystemExit) as excinfo:
            _runScript(script, checkpoint, flag, value, two_modalities=True)
        assert excinfo.value.code == 2

    def test_the_help_names_the_new_flags_and_the_adr(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "--reconstruction-loss" in out
        assert "--reconstruction-weight" in out
        unwrapped = "".join(out.split())  # the help wraps a long path at its hyphens
        assert "0030-per-modality-reconstruction-loss-and-weight.md" in unwrapped


class TestParsers:
    def test_a_loss_name(self, script: ModuleType) -> None:
        assert script._parseLoss("l1") == "l1"
        assert script._parseLoss(" bce ") == "bce"

    def test_loss_pairs(self, script: ModuleType) -> None:
        assert script._parseLoss("signal = mse , mask=bce") == {"signal": "mse", "mask": "bce"}

    def test_a_weight_number(self, script: ModuleType) -> None:
        assert script._parseWeight("0.5") == 0.5
        assert script._parseWeight(" 2 ") == 2.0
        assert type(script._parseWeight("2")) is float
        assert script._parseWeight("0") == 0.0

    def test_weight_pairs(self, script: ModuleType) -> None:
        assert script._parseWeight("signal=0.5, image = 2") == {"signal": 0.5, "image": 2.0}

    def test_the_errors_are_argparse_errors(self, script: ModuleType) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="--reconstruction-loss: unknown"):
            script._parseLoss("huber")
        with pytest.raises(argparse.ArgumentTypeError, match="modality=name"):
            script._parseLoss("x=l1,y")
        with pytest.raises(argparse.ArgumentTypeError, match="modality=number"):
            script._parseWeight("x=1,y")

    @pytest.mark.parametrize("text", ["abc", "", "-1", "inf", "nan", "x=abc", "x=-0.1"])
    def test_a_weight_that_is_not_a_number_at_least_zero_names_the_option(
        self, script: ModuleType, text: str
    ) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match=re.escape("--reconstruction-weight")):
            script._parseWeight(text)
