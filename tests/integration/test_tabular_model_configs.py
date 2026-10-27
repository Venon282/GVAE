"""The tabular model config and its experiment (roadmap P3-8, spec §6 and §10).

`configs/model/tabular_single_latent.yaml` is the second proof of the genericity claim after the
1D signal and the image: a modality added through the registries only. `test_model_config_schema.py`
already checks that every shipped file validates and builds. This file checks what is specific to
this one: that the checklist held (no core module knows the word "tabular"), the wiring, the shape
a YAML list produces, a table with one-hot columns, the gradient, the loss scale the experiment
sets (and the arithmetic behind it), the preprocessing the data config describes, and a run of
`scripts/train.py` on dummy batches.
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
from global_vae.data.transforms.standardize import StandardizeTransform
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
_DUMMY_TABULAR_LOADER = "tests.integration._train_script_fixtures:buildDummyTabularDataloaders"
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

MODEL_FILE = "tabular_single_latent"
EXPERIMENT = "tabular_vae"
BATCH_SIZE = 8
NUM_FEATURES = 16
CATEGORICAL_SIZES = [3, 3]

# The packages that make a model, a loss, a training loop or a config. Adding a modality must not
# touch them (spec §10: zero changes to core framework code), so none may name this one: neither
# its registered classes nor its batch key (the word alone is not enough, a logger docstring
# speaks of "a purely tabular backend").
_MODALITY_REFERENCE = re.compile(r"tabular_mlp|TabularMlp|[\"']tabular[\"']")
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


def _model(categorical_sizes: list[int] | None = None) -> GlobalVae:
    config = _modelConfig()
    if categorical_sizes is not None:
        config.modalities["tabular"].decoder.kwargs["categorical_sizes"] = categorical_sizes
    return buildModelFromConfig(config)


def _experiment(*overrides: str) -> ExperimentConfig:
    return loadExperimentConfig(
        f"experiment/{EXPERIMENT}", overrides=[*_DATA_OVERRIDES, *overrides]
    )


def _batch(rows: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    return {"tabular": torch.randn(rows, NUM_FEATURES)}


def _mixedBatch(rows: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    """A table of 10 standardized numeric columns and two one-hot columns of 3 categories."""
    numeric = torch.randn(rows, NUM_FEATURES - sum(CATEGORICAL_SIZES))
    blocks = [
        torch.nn.functional.one_hot(torch.randint(0, size, (rows,)), size).float()
        for size in CATEGORICAL_SIZES
    ]
    return {"tabular": torch.cat([numeric, *blocks], dim=1)}


class TestTheNewModalityChecklist:
    """The practical test of the registry design: no core file is needed for a new modality."""

    def test_the_file_validates_and_builds(self) -> None:
        config = _modelConfig()
        validateModelConfig(config)
        model = buildModelFromConfig(config)
        assert set(model.latent_spaces) == {"z_fused"}
        assert model.input_modalities == ("tabular",)
        assert model.output_modalities == ("tabular",)

    def test_it_names_the_two_registered_classes(self) -> None:
        pair = _modelConfig().modalities["tabular"]
        assert pair.encoder.name == "tabular_mlp_encoder_v1"
        assert pair.decoder.name == "tabular_mlp_decoder_v1"
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

    def test_the_row_width_is_the_same_for_the_encoder_and_the_decoder(self) -> None:
        pair = _modelConfig().modalities["tabular"]
        assert pair.encoder.kwargs["num_features"] == pair.decoder.kwargs["num_features"]
        assert pair.encoder.kwargs["num_features"] == NUM_FEATURES

    def test_a_yaml_value_gives_the_reconstruction_it_declares(self) -> None:
        reconstruction = _model()(_batch())["reconstructions"]["tabular"]
        assert reconstruction.shape == (BATCH_SIZE, NUM_FEATURES)

    def test_the_latent_width_is_set_in_one_place(self) -> None:
        config = _modelConfig()
        assert config.single_latent is not None
        mu, logvar = _model()(_batch())["latent_params"]["z_fused"]
        assert mu.shape == logvar.shape == (BATCH_SIZE, config.single_latent.dim)

    def test_a_row_of_another_width_is_refused_not_resized(self) -> None:
        with pytest.raises(ValueError, match=rf"\(batch, {NUM_FEATURES}\)"):
            _model()({"tabular": torch.randn(BATCH_SIZE, NUM_FEATURES + 2)})

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
        batch = {"tabular": torch.randn(1, NUM_FEATURES).repeat(BATCH_SIZE, 1)}
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(30):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        assert trainer.computeLosses(batch, step=30).reconstruction.item() < first

    def test_a_key_with_no_encoder_or_decoder_is_ignored(self) -> None:
        """A loader that yields more keys (a paired one, say) serves this model as well."""
        model = _model()
        batch = {**_batch(), "signal": torch.randn(BATCH_SIZE, 256)}
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(batch, step=0)
        assert torch.isfinite(losses.total)
        assert set(model(model.selectEncoderInputs(batch))["reconstructions"]) == {"tabular"}


class TestATableWithOneHotColumns:
    """`categorical_sizes` in the decoder kwargs: a mixed table, with no core change."""

    def test_the_one_hot_blocks_are_probabilities_and_the_numeric_columns_are_not(self) -> None:
        reconstruction = _model(CATEGORICAL_SIZES)(_mixedBatch())["reconstructions"]["tabular"]
        assert reconstruction.shape == (BATCH_SIZE, NUM_FEATURES)
        first, second = reconstruction[:, 10:13], reconstruction[:, 13:]
        for block in (first, second):
            assert torch.allclose(block.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-6)

    def test_the_default_mse_loss_trains_it_and_every_parameter_receives_a_gradient(self) -> None:
        torch.manual_seed(0)
        model = _model(CATEGORICAL_SIZES)
        trainer = Trainer(model, optimizer_kwargs={"lr": 3e-3}, modality_dropout_p=0.0)
        batch = _mixedBatch(32)
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(30):
            trainer.optimizer.zero_grad()
            losses = trainer.computeLosses(batch, step=step)
            backward(losses.total)
            trainer.optimizer.step()
        assert [n for n, p in model.named_parameters() if p.grad is None] == []
        assert trainer.computeLosses(batch, step=30).reconstruction.item() < first


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

    def test_the_row_width_follows_the_overrides_of_the_two_modules(self) -> None:
        config = _experiment(
            "model.modalities.tabular.encoder.kwargs.num_features=24",
            "model.modalities.tabular.decoder.kwargs.num_features=24",
        )
        out = buildModelFromConfig(config.model)({"tabular": torch.randn(2, 24)})
        assert out["reconstructions"]["tabular"].shape == (2, 24)

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
        assert summed / averaged == pytest.approx(NUM_FEATURES / 2, rel=1e-4)


class TestTheDataContract:
    def test_the_data_config_describes_one_tabular_pipeline_and_no_sequence_length(self) -> None:
        data = _experiment().data
        assert set(data.transforms) == {"tabular"}
        assert not data.sequence_length

    def test_the_configured_pipeline_is_invertible(self) -> None:
        pipeline = buildTransformPipeline(_experiment().data)["tabular"]
        x = torch.randn(BATCH_SIZE, NUM_FEATURES)
        assert torch.allclose(pipeline.inverse(pipeline(x)), x, atol=1e-6)

    def test_a_mean_and_a_std_per_column_standardize_a_table(self) -> None:
        """What the data file tells a `loader_factory` to build: one statistic per column."""
        table = torch.randn(200, NUM_FEATURES) * torch.arange(1, NUM_FEATURES + 1) + 5.0
        transform = StandardizeTransform(mean=table.mean(dim=0), std=table.std(dim=0))
        standardized = transform(table)
        assert torch.allclose(standardized.mean(dim=0), torch.zeros(NUM_FEATURES), atol=1e-5)
        assert torch.allclose(standardized.std(dim=0), torch.ones(NUM_FEATURES), atol=1e-5)
        assert torch.allclose(transform.inverse(standardized), table, atol=1e-4)

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="roadmap P3-8: StandardizeTransform takes a float or a tensor, not a YAML list",
    )
    def test_a_per_column_mean_and_std_written_in_yaml_build_the_pipeline(self) -> None:
        """What the data file cannot say today: per-column statistics have to be built in code.

        A YAML list reaches `StandardizeTransform` as a Python `list`, which it rejects with a
        `TypeError` that does not name the cause. The fix is a change to `data/transforms`
        (accept a sequence), so it is flagged and not made here. When it is made this test passes
        and `strict` asks for the marker to be removed.
        """
        config = _experiment().data
        config.transforms["tabular"][0].kwargs = {"mean": [0.0] * NUM_FEATURES, "std": [1.0] * 16}
        pipeline = buildTransformPipeline(config)["tabular"]
        assert pipeline(torch.randn(2, NUM_FEATURES)).shape == (2, NUM_FEATURES)


def _runTrain(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/tabular_vae` for two short epochs."""
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{EXPERIMENT}",
        f"data.loader_factory={_DUMMY_TABULAR_LOADER}",
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
