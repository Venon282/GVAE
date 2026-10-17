"""Integration tests for the loss scale convention (roadmap P2-1, ADR 0029).

Three things are wired through several components here, so they are tested together:

- `training.reconstruction_reduction` from the shipped YAML files and Hydra overrides, through
  `buildTrainerFromConfig`, into `Trainer` (training and validation loss);
- `evaluate()` with a reduction, and the latent diagnostics it adds to `EvaluationResults`, to
  `summary()` and to the JSON report;
- `scripts/evaluate.py` with `--reconstruction-reduction` and `--active-unit-threshold`.

The dummy model and dataloaders come from `_script_fixtures.py` (a single `signal` modality of
16 values, so `N = 16`, and a two-modality `x`/`y` model of the same width).
"""

import argparse
import importlib.util
import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.patches import Rectangle
from omegaconf import DictConfig, OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import buildDataloadersFromConfig
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    TrainingConfig,
    buildTrainerFromConfig,
    resolveReconstructionReduction,
)
from global_vae.evaluation.evaluate import EvaluationResults, evaluate
from global_vae.evaluation.latent_diagnostics import LatentDiagnostics, computeLatentDiagnostics
from global_vae.losses.regularizers.kl_standard_normal import computeKlPerDimension
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import saveCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from global_vae.visualization.latent_plot import plotPerDimensionKl
from tests.integration._script_fixtures import (
    INPUT_DIM,
    LATENT_DIM,
    buildDataloaderForScript,
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
_BASE_OVERRIDES = [f"data.loader_factory={_LOADER_FACTORY}", "data.train_path=/unused"]


def _loadConfig(tmp_path: Path, *overrides: str) -> ExperimentConfig:
    return loadExperimentConfig(overrides=[*_BASE_OVERRIDES, f"output_dir={tmp_path}", *overrides])


def _trainer(model: GlobalVae, reduction: str | dict[str, str] = "mean") -> Trainer:
    """A trainer around `model`. Pass the same model to compare reductions on equal weights."""
    return Trainer(
        model, optimizer_kwargs={"lr": 1e-2}, device="cpu", reconstruction_reduction=reduction
    )


def _explodingDataloader() -> Iterator[dict[str, torch.Tensor]]:
    """A dataloader that fails when read: for checks that must happen before any batch."""
    raise AssertionError("the dataloader must not be read")
    yield {}


class TestShippedConfigs:
    @pytest.mark.parametrize("name", ["default", "several_latents"])
    def test_the_shipped_training_files_write_the_default_explicitly(self, name: str) -> None:
        """Read without resolving: the files hold `${output_dir}` interpolations."""
        raw = cast(DictConfig, OmegaConf.load(_REPO_ROOT / "configs" / "training" / f"{name}.yaml"))
        assert raw.reconstruction_reduction == "mean"

    def test_the_dataclass_default_is_mean(self) -> None:
        assert TrainingConfig().reconstruction_reduction == "mean"

    def test_a_config_that_does_not_mention_it_loads_as_mean(self, tmp_path: Path) -> None:
        cfg = _loadConfig(tmp_path)
        assert cfg.training.reconstruction_reduction == "mean"

    def test_a_several_latent_experiment_loads_as_mean_and_takes_a_mapping_per_decoder(
        self, tmp_path: Path
    ) -> None:
        cfg = loadExperimentConfig(
            config_name="experiment/signal_image_shared_private",
            overrides=[
                f"data.loader_factory={_SIGNAL_IMAGE_LOADER_FACTORY}",
                "data.train_path=/unused",
                f"output_dir={tmp_path}",
            ],
        )
        assert cfg.training.reconstruction_reduction == "mean"

        model = buildModelFromConfig(cfg.model)
        decoder_names = sorted(model.decoders)
        assert len(decoder_names) > 1
        cfg.training.reconstruction_reduction = {decoder_names[0]: "sum"}
        trainer = buildTrainerFromConfig(model, cfg.training)
        assert trainer.reconstruction_reduction == {decoder_names[0]: "sum"}

        cfg.training.reconstruction_reduction = {"not_a_decoder": "sum"}
        with pytest.raises(ValueError, match=re.escape(f"decoders: {decoder_names}")):
            buildTrainerFromConfig(model, cfg.training)


class TestConfigToTrainer:
    def test_a_name_override_reaches_the_trainer(self, tmp_path: Path) -> None:
        cfg = _loadConfig(tmp_path, "training.reconstruction_reduction=sum")
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_reduction == "sum"

    def test_the_default_config_gives_a_mean_trainer(self, tmp_path: Path) -> None:
        cfg = _loadConfig(tmp_path)
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert trainer.reconstruction_reduction == "mean"

    def test_a_mapping_override_reaches_the_trainer_as_a_plain_dict(self, tmp_path: Path) -> None:
        cfg = _loadConfig(tmp_path, "training.reconstruction_reduction={signal:sum}")
        assert cfg.training.reconstruction_reduction == {"signal": "sum"}
        trainer = buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)
        assert type(trainer.reconstruction_reduction) is dict
        assert trainer.reconstruction_reduction == {"signal": "sum"}

    def test_the_config_value_reaches_the_loss(self, tmp_path: Path) -> None:
        """A `sum` config makes the reconstruction term `N` times the `mean` config's."""
        mean_cfg = _loadConfig(tmp_path / "mean")
        sum_cfg = _loadConfig(tmp_path / "sum", "training.reconstruction_reduction=sum")
        batch = next(iter(buildDataloadersFromConfig(mean_cfg.data).train))
        losses = {}
        for name, cfg in (("mean", mean_cfg), ("sum", sum_cfg)):
            torch.manual_seed(0)  # the same initial weights for both models
            model = buildModelFromConfig(cfg.model)
            trainer = buildTrainerFromConfig(model, cfg.training)
            torch.manual_seed(1)  # the same posterior sample for both passes
            losses[name] = trainer.computeLosses(batch, 0, apply_dropout=False).reconstruction
        num_values = math.prod(batch["signal"].shape[1:])
        assert losses["sum"].item() == pytest.approx(num_values * losses["mean"].item(), rel=1e-4)

    def test_an_unsupported_name_is_an_error_naming_the_field(self, tmp_path: Path) -> None:
        cfg = _loadConfig(tmp_path, "training.reconstruction_reduction=average")
        with pytest.raises(ValueError, match=r"^training\.reconstruction_reduction: 'average'"):
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)

    def test_an_unsupported_name_inside_a_mapping_is_an_error_naming_the_key(
        self, tmp_path: Path
    ) -> None:
        cfg = _loadConfig(tmp_path, "training.reconstruction_reduction={signal:average}")
        with pytest.raises(
            ValueError, match=r"^training\.reconstruction_reduction\['signal'\]: 'average'"
        ):
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)

    def test_a_mistyped_decoder_is_an_error_and_not_a_silent_mean(self, tmp_path: Path) -> None:
        cfg = _loadConfig(tmp_path, "training.reconstruction_reduction={signall:sum}")
        model = buildModelFromConfig(cfg.model)
        with pytest.raises(ValueError, match=r"\['signall'\].*decoders: \['signal'\]"):
            buildTrainerFromConfig(model, cfg.training)

    def test_a_value_of_the_wrong_type_is_an_error(self, tmp_path: Path) -> None:
        cfg = _loadConfig(tmp_path, "training.reconstruction_reduction=3")
        with pytest.raises(
            ValueError, match=r"^training\.reconstruction_reduction: expected one of"
        ):
            buildTrainerFromConfig(buildModelFromConfig(cfg.model), cfg.training)

    def test_the_resolver_returns_a_copy_of_the_mapping(self) -> None:
        model = buildModelForScript()
        original = {"signal": "sum"}
        resolved = resolveReconstructionReduction(original, model)
        assert resolved == original
        assert resolved is not original

    def test_the_resolver_accepts_an_omegaconf_mapping(self) -> None:
        model = buildModelForScript()
        resolved = resolveReconstructionReduction(OmegaConf.create({"signal": "sum"}), model)
        assert type(resolved) is dict
        assert resolved == {"signal": "sum"}


class TestTrainer:
    def test_the_default_is_mean(self) -> None:
        trainer = Trainer(buildModelForScript(), device="cpu")
        assert trainer.reconstruction_reduction == "mean"

    def test_an_invalid_reduction_is_rejected_when_the_trainer_is_built(self) -> None:
        with pytest.raises(ValueError, match="^reconstruction_reduction: 'average'"):
            _trainer(buildModelForScript(), "average")

    def test_a_mapping_naming_no_decoder_of_the_model_is_not_checked_by_the_trainer(self) -> None:
        """The decoder names are checked where a config is turned into a trainer."""
        trainer = _trainer(buildModelForScript(), {"other": "sum"})
        assert trainer.reconstruction_reduction == {"other": "sum"}

    def test_the_reduction_is_the_last_argument_so_positional_callers_are_unaffected(self) -> None:
        trainer = Trainer(
            buildModelForScript(),
            torch.optim.Adam,
            None,
            "cpu",
            0.5,
            torch.nn.functional.l1_loss,
            2.0,
            None,
            0.0,
            None,
            None,
            7,
        )
        assert trainer.log_every_n_steps == 7
        assert trainer.reconstruction_reduction == "mean"

    def test_sum_scales_the_training_reconstruction_loss_by_n(self) -> None:
        model = buildModelForScript()
        batch = buildDataloaderForScript()[0]
        values = {}
        for reduction in ("mean", "sum"):
            torch.manual_seed(0)
            values[reduction] = _trainer(model, reduction).computeLosses(
                batch, 0, apply_dropout=False
            )
        assert values["sum"].reconstruction.item() == pytest.approx(
            INPUT_DIM * values["mean"].reconstruction.item(), rel=1e-5
        )

    def test_the_regularization_term_does_not_depend_on_the_reduction(self) -> None:
        model = buildModelForScript()
        batch = buildDataloaderForScript()[0]
        values = {}
        for reduction in ("mean", "sum"):
            torch.manual_seed(0)
            values[reduction] = _trainer(model, reduction).computeLosses(
                batch, 0, apply_dropout=False
            )
        assert values["sum"].regularization.item() == pytest.approx(
            values["mean"].regularization.item(), rel=1e-6
        )

    def test_the_validation_loss_uses_the_same_reduction(self) -> None:
        model = buildModelForScript()
        batches = buildDataloaderForScript()
        metrics = {}
        for reduction in ("mean", "sum"):
            torch.manual_seed(0)
            metrics[reduction] = _trainer(model, reduction).evaluate(batches)
        assert metrics["sum"]["val/loss/reconstruction"] == pytest.approx(
            INPUT_DIM * metrics["mean"]["val/loss/reconstruction"], rel=1e-5
        )

    def test_training_under_sum_runs_and_the_loss_falls(self) -> None:
        trainer = _trainer(buildModelForScript(), "sum")
        history = trainer.fit(buildDataloaderForScript(), num_epochs=15)
        assert all(math.isfinite(entry["train/loss/total"]) for entry in history)
        assert history[-1]["train/loss/total"] < history[0]["train/loss/total"]

    def test_every_parameter_receives_a_gradient_under_sum(self) -> None:
        trainer = _trainer(buildModelForScript(), "sum")
        batch = buildDataloaderForScript()[0]
        backward(trainer.computeLosses(batch, 0).total)
        assert all(parameter.grad is not None for parameter in trainer.model.parameters())

    def test_a_mapping_mixes_the_reductions_of_two_modalities(self) -> None:
        """On equal weights and an equal posterior sample: `x` is scaled by `N`, `y` is not."""
        model = buildTwoModalityModelForScript()
        batch = buildTwoModalityDataloaderForScript()[0]
        cases: list[tuple[str, str | dict[str, str]]] = [
            ("mean", "mean"),
            ("mixed", {"x": "sum"}),
            ("sum", "sum"),
        ]
        values = {}
        for name, reduction in cases:
            torch.manual_seed(0)
            values[name] = (
                _trainer(model, reduction)
                .computeLosses(batch, 0, apply_dropout=False)
                .reconstruction.item()
            )
        assert values["mean"] < values["mixed"] < values["sum"]
        assert values["sum"] == pytest.approx(INPUT_DIM * values["mean"], rel=1e-5)


class TestEvaluate:
    def test_the_default_records_mean(self) -> None:
        results = evaluate(buildModelForScript(), buildDataloaderForScript(), device="cpu")
        assert results.reconstruction_reduction == "mean"
        assert "reconstruction reduction" not in results.summary()

    def test_sum_scales_the_total_reconstruction_loss_by_n(self) -> None:
        model = buildModelForScript()
        data = buildDataloaderForScript()
        mean = evaluate(model, data, device="cpu")
        total = evaluate(model, data, device="cpu", reconstruction_reduction="sum")
        assert total.total_reconstruction_loss == pytest.approx(
            INPUT_DIM * mean.total_reconstruction_loss, rel=1e-5
        )

    def test_the_metrics_and_the_regularization_do_not_depend_on_the_reduction(self) -> None:
        model = buildModelForScript()
        data = buildDataloaderForScript()
        mean = evaluate(model, data, device="cpu")
        total = evaluate(model, data, device="cpu", reconstruction_reduction="sum")
        assert total.reconstruction_metrics == mean.reconstruction_metrics
        assert total.regularization_metrics == mean.regularization_metrics
        assert total.total_regularization_loss == mean.total_regularization_loss
        assert total.latent_diagnostics == mean.latent_diagnostics

    def test_the_reduction_is_recorded_and_written_in_the_summary_and_the_json(
        self, tmp_path: Path
    ) -> None:
        results = evaluate(
            buildModelForScript(),
            buildDataloaderForScript(),
            device="cpu",
            reconstruction_reduction="sum",
        )
        assert results.reconstruction_reduction == "sum"
        assert "  reconstruction reduction: sum" in results.summary()
        results.save(tmp_path / "report.json")
        assert (
            json.loads((tmp_path / "report.json").read_text())["reconstruction_reduction"] == "sum"
        )

    def test_a_mapping_is_recorded_as_a_plain_dict_and_mixes_the_reductions(
        self, tmp_path: Path
    ) -> None:
        model = buildTwoModalityModelForScript()
        data = buildTwoModalityDataloaderForScript()
        mean = evaluate(model, data, device="cpu")
        mixed = evaluate(model, data, device="cpu", reconstruction_reduction={"x": "sum"})
        everything = evaluate(model, data, device="cpu", reconstruction_reduction="sum")

        mse_x = mean.reconstruction_metrics["x"]["mse"]
        mse_y = mean.reconstruction_metrics["y"]["mse"]
        assert mean.total_reconstruction_loss == pytest.approx(mse_x + mse_y, rel=1e-4)
        assert mixed.total_reconstruction_loss == pytest.approx(INPUT_DIM * mse_x + mse_y, rel=1e-4)
        assert everything.total_reconstruction_loss == pytest.approx(
            INPUT_DIM * (mse_x + mse_y), rel=1e-4
        )
        assert mixed.reconstruction_reduction == {"x": "sum"}
        assert "reconstruction reduction: x=sum" in mixed.summary()
        mixed.save(tmp_path / "report.json")
        assert json.loads((tmp_path / "report.json").read_text())["reconstruction_reduction"] == {
            "x": "sum"
        }

    def test_a_read_only_mapping_is_copied_into_a_plain_dict(self) -> None:
        results = evaluate(
            buildTwoModalityModelForScript(),
            buildTwoModalityDataloaderForScript(),
            device="cpu",
            reconstruction_reduction=MappingProxyType({"y": "sum"}),
        )
        assert type(results.reconstruction_reduction) is dict

    def test_an_invalid_reduction_fails_before_the_dataloader_is_read(self) -> None:
        with pytest.raises(ValueError, match="^reconstruction_reduction: 'average'"):
            evaluate(
                buildModelForScript(),
                _explodingDataloader(),
                device="cpu",
                reconstruction_reduction="average",
            )

    @pytest.mark.parametrize("threshold", [-1.0, math.inf, math.nan])
    def test_an_invalid_threshold_fails_before_the_dataloader_is_read(
        self, threshold: float
    ) -> None:
        with pytest.raises(ValueError, match="active_unit_threshold"):
            evaluate(
                buildModelForScript(),
                _explodingDataloader(),
                device="cpu",
                active_unit_threshold=threshold,
            )


class TestLatentDiagnosticsInTheReport:
    def test_there_is_one_entry_per_latent_space(self) -> None:
        model = buildModelForScript()
        results = evaluate(model, buildDataloaderForScript(), device="cpu")
        assert set(results.latent_diagnostics) == set(model.latent_spaces)
        diagnostics = results.latent_diagnostics["z_fused"]
        assert isinstance(diagnostics, LatentDiagnostics)
        assert diagnostics.num_dimensions == LATENT_DIM
        assert len(diagnostics.kl_per_dimension) == LATENT_DIM
        assert len(diagnostics.mu_variance_per_dimension) == LATENT_DIM
        assert 0 <= diagnostics.active_units <= LATENT_DIM

    def test_the_kl_per_dimension_adds_up_to_the_reported_kl(self) -> None:
        results = evaluate(buildModelForScript(), buildDataloaderForScript(), device="cpu")
        reported = results.regularization_metrics["z_fused"]["kl_standard_normal"]
        assert sum(results.latent_diagnostics["z_fused"].kl_per_dimension) == pytest.approx(
            reported, rel=1e-5
        )

    def test_the_values_are_those_of_the_pooled_posterior(self) -> None:
        model = buildModelForScript()
        data = buildDataloaderForScript()
        results = evaluate(model, data, device="cpu")

        model.eval()
        with torch.no_grad():
            params = [
                model(model.selectEncoderInputs(batch), use_mean=True)["latent_params"]["z_fused"]
                for batch in data
            ]
        mu = torch.cat([mu for mu, _ in params])
        logvar = torch.cat([logvar for _, logvar in params])
        expected = computeLatentDiagnostics(mu, logvar)
        assert results.latent_diagnostics["z_fused"] == expected

    def test_max_samples_limits_the_data_the_diagnostics_are_computed_from(self) -> None:
        model = buildModelForScript()
        data = buildDataloaderForScript()
        results = evaluate(model, data, device="cpu", max_samples=len(data[0]["signal"]))

        model.eval()
        with torch.no_grad():
            mu, logvar = model(model.selectEncoderInputs(data[0]), use_mean=True)["latent_params"][
                "z_fused"
            ]
        assert results.latent_diagnostics["z_fused"] == computeLatentDiagnostics(mu, logvar)

    def test_the_threshold_is_applied_and_recorded(self) -> None:
        model = buildModelForScript()
        data = buildDataloaderForScript()
        nothing_active = evaluate(model, data, device="cpu", active_unit_threshold=1e9)
        everything_active = evaluate(model, data, device="cpu", active_unit_threshold=0.0)
        assert nothing_active.latent_diagnostics["z_fused"].active_units == 0
        assert nothing_active.latent_diagnostics["z_fused"].active_threshold == 1e9
        assert everything_active.latent_diagnostics["z_fused"].active_units == LATENT_DIM
        assert everything_active.latent_diagnostics["z_fused"].active_threshold == 0.0

    def test_the_report_numbers_are_the_bars_of_the_per_dimension_kl_figure(self) -> None:
        """One formula: the figure of `scripts/visualize_latent.py` and the report cannot differ."""
        torch.manual_seed(0)
        mu, logvar = torch.randn(40, LATENT_DIM), torch.randn(40, LATENT_DIM)
        diagnostics = computeLatentDiagnostics(mu, logvar)
        fig = plotPerDimensionKl(mu, logvar)
        bars = [cast(Rectangle, patch).get_height() for patch in fig.axes[0].patches]
        plt.close(fig)
        assert bars == pytest.approx(diagnostics.kl_per_dimension, abs=1e-6)
        assert diagnostics.kl_per_dimension == pytest.approx(
            computeKlPerDimension(mu, logvar).mean(dim=0).tolist(), abs=1e-6
        )

    def test_a_two_modality_model_with_a_shared_latent_space(self) -> None:
        model = buildTwoModalityModelForScript()
        results = evaluate(model, buildTwoModalityDataloaderForScript(), device="cpu")
        assert set(results.latent_diagnostics) == set(model.latent_spaces)

    def test_the_diagnostics_are_in_the_json_report(self, tmp_path: Path) -> None:
        results = evaluate(buildModelForScript(), buildDataloaderForScript(), device="cpu")
        results.save(tmp_path / "report.json")
        report = json.loads((tmp_path / "report.json").read_text())
        entry = report["latent_diagnostics"]["z_fused"]
        assert set(entry) == {
            "num_dimensions",
            "active_threshold",
            "active_units",
            "kl_per_dimension",
            "mu_variance_per_dimension",
        }
        assert entry["num_dimensions"] == LATENT_DIM
        assert entry["kl_per_dimension"] == pytest.approx(
            results.latent_diagnostics["z_fused"].kl_per_dimension
        )

    def test_to_dict_stays_json_serializable(self) -> None:
        results = evaluate(buildModelForScript(), buildDataloaderForScript(), device="cpu")
        json.dumps(results.toDict())


class TestSummary:
    def test_the_summary_lists_the_active_units_and_every_kl_of_a_narrow_space(self) -> None:
        results = EvaluationResults(
            num_samples=10,
            total_reconstruction_loss=0.5,
            total_regularization_loss=0.1,
            latent_diagnostics={
                "z_fused": LatentDiagnostics(
                    num_dimensions=3,
                    active_threshold=0.01,
                    active_units=2,
                    kl_per_dimension=[0.5, 0.25, 0.0],
                    mu_variance_per_dimension=[1.0, 0.5, 0.0],
                )
            },
        )
        summary = results.summary()
        assert "  latent diagnostics:" in summary
        assert "[z_fused] active units: 2/3 (variance of mu > 0.01)" in summary
        assert "kl per dimension: 0.5000, 0.2500, 0.0000" in summary

    def test_a_wide_space_prints_the_minimum_median_and_maximum_only(self) -> None:
        kl_values = [float(index) for index in range(40)]
        results = EvaluationResults(
            num_samples=1,
            total_reconstruction_loss=0.0,
            total_regularization_loss=0.0,
            latent_diagnostics={
                "z": LatentDiagnostics(
                    num_dimensions=40,
                    active_threshold=0.05,
                    active_units=40,
                    kl_per_dimension=kl_values,
                    mu_variance_per_dimension=[1.0] * 40,
                )
            },
        )
        summary = results.summary()
        assert "[z] active units: 40/40 (variance of mu > 0.05)" in summary
        assert "kl per dimension: min=0.0000, median=19.5000, max=39.0000" in summary
        assert "0.0000, 1.0000" not in summary

    def test_sixteen_dimensions_are_still_listed_in_full(self) -> None:
        results = EvaluationResults(
            num_samples=1,
            total_reconstruction_loss=0.0,
            total_regularization_loss=0.0,
            latent_diagnostics={
                "z": LatentDiagnostics(
                    num_dimensions=16,
                    active_threshold=0.01,
                    active_units=16,
                    kl_per_dimension=[1.0] * 16,
                    mu_variance_per_dimension=[1.0] * 16,
                )
            },
        )
        assert "min=" not in results.summary()

    def test_a_result_built_by_hand_without_the_new_fields_is_still_valid(self) -> None:
        results = EvaluationResults(0, 0.0, 0.0)
        assert results.latent_diagnostics == {}
        assert results.reconstruction_reduction == "mean"
        assert "latent diagnostics" not in results.summary()
        assert results.toDict()["latent_diagnostics"] == {}

    def test_the_summary_of_a_real_evaluation_has_the_block(self) -> None:
        results = evaluate(buildModelForScript(), buildDataloaderForScript(), device="cpu")
        summary = results.summary()
        assert f"[z_fused] active units: {results.latent_diagnostics['z_fused'].active_units}/" in (
            summary
        )


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_evaluate_script_p2_1", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    saveCheckpoint(path, model=buildModelForScript())
    return path


def _runScript(script: ModuleType, checkpoint: Path, *extra: str) -> None:
    script.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--model-factory",
            "tests.integration._script_fixtures:buildModelForScript",
            "--dataloader-factory",
            "tests.integration._script_fixtures:buildDataloaderForScript",
            "--device",
            "cpu",
            *extra,
        ]
    )


class TestEvaluateScript:
    def test_the_default_run_prints_the_diagnostics_and_no_reduction_line(
        self, script: ModuleType, checkpoint_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _runScript(script, checkpoint_path)
        out = capsys.readouterr().out
        assert "latent diagnostics:" in out
        assert "reconstruction reduction" not in out

    def test_the_reduction_flag_reaches_the_report(
        self,
        script: ModuleType,
        checkpoint_path: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        output_dir = tmp_path / "results"
        _runScript(
            script,
            checkpoint_path,
            "--reconstruction-reduction",
            "sum",
            "--output-dir",
            str(output_dir),
            "--no-figures",
        )
        assert "reconstruction reduction: sum" in capsys.readouterr().out
        report = json.loads((output_dir / "results.json").read_text())
        assert report["reconstruction_reduction"] == "sum"

    def test_the_flag_changes_the_scale_of_the_total_loss_and_nothing_else(
        self, script: ModuleType, checkpoint_path: Path, tmp_path: Path
    ) -> None:
        reports = {}
        for reduction in ("mean", "sum"):
            output_dir = tmp_path / reduction
            _runScript(
                script,
                checkpoint_path,
                "--reconstruction-reduction",
                reduction,
                "--output-dir",
                str(output_dir),
                "--no-figures",
            )
            reports[reduction] = json.loads((output_dir / "results.json").read_text())
        assert reports["sum"]["total_reconstruction_loss"] == pytest.approx(
            INPUT_DIM * reports["mean"]["total_reconstruction_loss"], rel=1e-5
        )
        assert reports["sum"]["reconstruction_metrics"] == reports["mean"]["reconstruction_metrics"]
        assert reports["sum"]["latent_diagnostics"] == reports["mean"]["latent_diagnostics"]

    def test_modality_equals_name_pairs_are_accepted(
        self, script: ModuleType, checkpoint_path: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "results"
        _runScript(
            script,
            checkpoint_path,
            "--reconstruction-reduction",
            "signal=sum",
            "--output-dir",
            str(output_dir),
            "--no-figures",
        )
        report = json.loads((output_dir / "results.json").read_text())
        assert report["reconstruction_reduction"] == {"signal": "sum"}

    def test_the_threshold_flag_reaches_the_report(
        self, script: ModuleType, checkpoint_path: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "results"
        _runScript(
            script,
            checkpoint_path,
            "--active-unit-threshold",
            "0.5",
            "--output-dir",
            str(output_dir),
            "--no-figures",
        )
        report = json.loads((output_dir / "results.json").read_text())
        assert report["latent_diagnostics"]["z_fused"]["active_threshold"] == 0.5

    def test_an_unsupported_reduction_is_a_usage_error(
        self, script: ModuleType, checkpoint_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _runScript(script, checkpoint_path, "--reconstruction-reduction", "average")
        assert excinfo.value.code == 2
        assert "Available: mean, sum" in capsys.readouterr().err

    @pytest.mark.parametrize("text", ["signal=sum,image", "=sum", "signal=avg", "signal=sum,,"])
    def test_a_malformed_pair_list_is_a_usage_error(
        self, script: ModuleType, checkpoint_path: Path, text: str
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _runScript(script, checkpoint_path, "--reconstruction-reduction", text)
        assert excinfo.value.code == 2


class TestParseReduction:
    def test_a_name(self, script: ModuleType) -> None:
        assert script._parseReduction("sum") == "sum"
        assert script._parseReduction("mean") == "mean"

    def test_surrounding_spaces_are_ignored(self, script: ModuleType) -> None:
        assert script._parseReduction(" sum ") == "sum"
        assert script._parseReduction("signal = sum , image = mean") == {
            "signal": "sum",
            "image": "mean",
        }

    def test_pairs(self, script: ModuleType) -> None:
        assert script._parseReduction("signal=sum,image=mean") == {
            "signal": "sum",
            "image": "mean",
        }

    def test_the_error_is_an_argparse_error_that_names_the_option(self, script: ModuleType) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="--reconstruction-reduction: 'avg'"):
            script._parseReduction("avg")
        with pytest.raises(argparse.ArgumentTypeError, match="modality=name"):
            script._parseReduction("signal=sum,image")
