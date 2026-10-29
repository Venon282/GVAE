"""The time series model config and its experiment (roadmap P3-7 and P3-8, spec §6 and §10).

`configs/model/time_series_single_latent.yaml` is the sixth modality added through the registries
after the 1D signal, the image, the table, the waveform and the text, and the fifth proof that the
core does not change. `test_model_config_schema.py` already checks that every shipped file
validates and builds. This file checks what is specific to this one: that the checklist held (no
core module knows the new class), the wiring, the shapes a YAML value produces, a series padded
with `NaN` to another length, what the loss of the experiment is (and is not) on a gap, the masked
loss written in Python, a model that learns series with gaps, what the evaluation does with a
`NaN`, the data contract, and a run of `scripts/train.py` on dummy batches.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional
from omegaconf import OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.data import buildTransformPipeline
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import ModelConfig, buildModelFromConfig, validateModelConfig
from global_vae.config.training import buildTrainerFromConfig
from global_vae.data.transforms.standardize import StandardizeTransform
from global_vae.decoders.registry import listRegisteredDecoders
from global_vae.encoders.registry import listRegisteredEncoders
from global_vae.evaluation.evaluate import evaluate
from global_vae.evaluation.visual_export import exportEvaluationFigures
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from global_vae.utils.series_blocks import computeMaskedSquaredError
from tests.integration._time_series_corpus import CHANNELS as CORPUS_CHANNELS
from tests.integration._time_series_corpus import LENGTH as CORPUS_LENGTH
from tests.integration._time_series_corpus import makeSeries

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_SRC_DIR = _REPO_ROOT / "src" / "global_vae"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_DUMMY_SERIES_LOADER = "tests.integration._train_script_fixtures:buildDummyTimeSeriesDataloaders"
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

MODEL_FILE = "time_series_single_latent"
EXPERIMENT = "time_series_vae"
BATCH_SIZE = 4
CHANNELS = 3
LENGTH = 96
NAN = float("nan")

# The packages that make a model, a loss, a training loop or a config. Adding a modality must not
# touch them (spec §10: zero changes to core framework code), so none may name this one: neither
# its registered class nor its batch key.
_MODALITY_REFERENCE = re.compile(r"time_series_encoder|TimeSeries|[\"']time_series[\"']")
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
    pair = config.modalities["time_series"]
    for key, value in overrides.items():
        side, name = key.split("_", 1)
        (pair.encoder if side == "encoder" else pair.decoder).kwargs[name] = value
    return buildModelFromConfig(config)


def _experiment(*overrides: str, output_dir: Path | None = None) -> ExperimentConfig:
    """The shipped experiment. Pass `output_dir` when a trainer built from it writes anywhere."""
    where = [f"output_dir={output_dir}"] if output_dir is not None else []
    return loadExperimentConfig(
        f"experiment/{EXPERIMENT}", overrides=[*_DATA_OVERRIDES, *where, *overrides]
    )


def _batch(rows: int = BATCH_SIZE, length: int = LENGTH) -> dict[str, torch.Tensor]:
    """A batch of random complete series."""
    return {"time_series": torch.randn(rows, CHANNELS, length)}


def _gappy(rows: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    """A batch with gaps, short series padded with `NaN`, and dropped values."""
    observed, _, _ = makeSeries(rows, seed=7)
    assert torch.isnan(observed).any()
    return {"time_series": observed}


class TestTheNewModalityChecklist:
    """The practical test of the registry design: no core file is needed for a new modality."""

    def test_the_file_validates_and_builds(self) -> None:
        config = _modelConfig()
        validateModelConfig(config)
        model = buildModelFromConfig(config)
        assert set(model.latent_spaces) == {"z_fused"}
        assert model.input_modalities == ("time_series",)
        assert model.output_modalities == ("time_series",)

    def test_it_names_a_new_encoder_and_a_decoder_that_already_existed(self) -> None:
        """The decoder is reused on purpose: it measured better than the alternative (ADR 0042)."""
        pair = _modelConfig().modalities["time_series"]
        assert pair.encoder.name == "time_series_encoder_v1"
        assert pair.decoder.name == "1d_cnn_decoder_v1"
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
        reconstruction = _model()(_batch())["reconstructions"]["time_series"]
        assert reconstruction.shape == (BATCH_SIZE, CHANNELS, LENGTH)

    def test_the_declared_length_is_the_one_of_the_data_file(self) -> None:
        """One loader must serve the data file and the model: same number of time steps."""
        declared = _modelConfig().modalities["time_series"].decoder.kwargs["output_length"]
        assert _experiment().data.sequence_length == {"time_series": declared}

    def test_the_channels_are_declared_by_the_encoder_and_the_decoder_alike(self) -> None:
        pair = _modelConfig().modalities["time_series"]
        assert pair.encoder.kwargs["in_channels"] == pair.decoder.kwargs["out_channels"] == CHANNELS

    def test_the_latent_width_is_set_in_one_place(self) -> None:
        config = _modelConfig()
        assert config.single_latent is not None
        mu, logvar = _model()(_batch())["latent_params"]["z_fused"]
        assert mu.shape == logvar.shape == (BATCH_SIZE, config.single_latent.dim)

    @pytest.mark.parametrize("length", [1, 20, 200])
    def test_a_series_of_another_length_is_encoded_but_reconstructed_at_the_declared_one(
        self, length: int
    ) -> None:
        """The encoder has no length limit; the decoder never resizes."""
        outputs = _model()(_batch(length=length))
        assert outputs["reconstructions"]["time_series"].shape == (BATCH_SIZE, CHANNELS, LENGTH)

    def test_padding_a_series_with_nan_does_not_change_its_latent_vector_or_its_reconstruction(
        self,
    ) -> None:
        model = _model().eval()
        series = _batch(length=60)["time_series"]
        padded = functional.pad(series, (0, LENGTH - 60), value=NAN)
        with torch.no_grad():
            short = model({"time_series": series}, use_mean=True)
            long = model({"time_series": padded}, use_mean=True)
        assert torch.allclose(
            short["latent_params"]["z_fused"][0], long["latent_params"]["z_fused"][0], atol=1e-5
        )
        assert torch.allclose(
            short["reconstructions"]["time_series"],
            long["reconstructions"]["time_series"],
            atol=1e-4,
        )

    def test_every_parameter_receives_a_gradient(self, tmp_path: Path) -> None:
        model = _model()
        trainer = buildTrainerFromConfig(model, _experiment(output_dir=tmp_path).training)
        losses = trainer.computeLosses(_batch(), step=0)
        backward(losses.total)
        assert torch.isfinite(losses.total)
        assert [n for n, p in model.named_parameters() if p.grad is None] == []

    def test_the_loss_falls_over_a_few_steps(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = _model()
        trainer = buildTrainerFromConfig(model, _experiment(output_dir=tmp_path).training)
        batch = _batch()
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(30):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        assert trainer.computeLosses(batch, step=30).reconstruction.item() < first

    def test_a_key_with_no_encoder_or_decoder_is_ignored(self) -> None:
        """A loader that yields more keys (a paired one, say) serves this model as well."""
        model = _model()
        batch = {**_batch(), "image": torch.randn(BATCH_SIZE, 64, 64)}
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(batch, step=0)
        assert torch.isfinite(losses.total)
        assert set(model(model.selectEncoderInputs(batch))["reconstructions"]) == {"time_series"}


class TestWhatTheYamlLetsYouChange:
    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_both_poolings_build_and_run(self, pooling: str) -> None:
        reconstruction = _model(encoder_pooling=pooling)(_batch())["reconstructions"]["time_series"]
        assert reconstruction.shape == (BATCH_SIZE, CHANNELS, LENGTH)

    def test_learned_positions_are_a_yaml_choice(self) -> None:
        model = _model(encoder_positional_encoding="learned", encoder_max_length=LENGTH)
        assert model(_batch())["reconstructions"]["time_series"].shape == (
            BATCH_SIZE,
            CHANNELS,
            LENGTH,
        )

    def test_another_length_is_set_where_it_is_written(self) -> None:
        """Seed 6 with the three doubling stages is 6 -> 12 -> 24 -> 48."""
        model = _model(decoder_output_length=48, decoder_seed_length=6)
        reconstruction = model(_batch(length=48))["reconstructions"]["time_series"]
        assert reconstruction.shape == (BATCH_SIZE, CHANNELS, 48)

    def test_a_decoder_that_does_not_land_on_the_length_is_refused_when_the_model_is_built(
        self,
    ) -> None:
        """The CNN decoder checks the exact length and never resizes one away."""
        with pytest.raises(ValueError, match="output_length=100"):
            _model(decoder_output_length=100)

    def test_one_channel_is_set_in_the_encoder_and_the_decoder_and_has_no_channel_axis(
        self,
    ) -> None:
        """With `out_channels: 1` the CNN decoder returns `(batch, length)`, like the batch."""
        model = _model(encoder_in_channels=1, decoder_out_channels=1)
        batch = {"time_series": torch.randn(BATCH_SIZE, LENGTH)}
        assert model(batch)["reconstructions"]["time_series"].shape == (BATCH_SIZE, LENGTH)
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(batch, step=0)
        assert torch.isfinite(losses.total)

    def test_a_batch_of_another_number_of_channels_is_refused_by_the_encoder_and_names_it(
        self,
    ) -> None:
        with pytest.raises(ValueError, match=f"in_channels={CHANNELS}"):
            _model()({"time_series": torch.randn(BATCH_SIZE, CHANNELS + 1, LENGTH)})

    def test_a_decoder_of_another_number_of_channels_is_refused_by_the_loss_with_both_shapes(
        self,
    ) -> None:
        """Encoder and decoder disagreeing is caught where the shapes are compared."""
        model = _model(decoder_out_channels=CHANNELS + 1)
        trainer = Trainer(model, modality_dropout_p=0.0)
        with pytest.raises(ValueError, match="shape"):
            trainer.computeLosses(_batch(), step=0)


class TestTheLossOfTheExperiment:
    """What `mse` with a sum is on a series, what it is on a gap, and the masked loss."""

    def test_the_experiment_sets_a_squared_error_summed_per_series_with_weight_fifty(self) -> None:
        training = _experiment().training
        assert training.reconstruction_loss == "mse"
        assert training.reconstruction_reduction == "sum"
        assert training.reconstruction_weight == 50.0

    def test_the_yaml_loss_is_nan_as_soon_as_one_value_is_a_gap(self, tmp_path: Path) -> None:
        """Pins the finding of ADR 0042: the loss names of a YAML file know no mask."""
        trainer = buildTrainerFromConfig(_model(), _experiment(output_dir=tmp_path).training)
        assert torch.isfinite(trainer.computeLosses(_batch(), step=0).reconstruction)
        assert torch.isnan(trainer.computeLosses(_gappy(), step=0).reconstruction)

    def test_the_masked_loss_trains_the_same_model_on_a_batch_with_gaps(self) -> None:
        model = _model()
        trainer = Trainer(
            model,
            reconstruction_loss_fn=computeMaskedSquaredError,
            reconstruction_weights=50.0,
            modality_dropout_p=0.0,
        )
        losses = trainer.computeLosses(_gappy(), step=0)
        backward(losses.total)
        assert torch.isfinite(losses.total)
        gradients = [parameter.grad for parameter in model.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)

    def test_on_complete_series_the_masked_loss_is_the_sum_reduction_of_the_mse(self) -> None:
        """The masked loss is a drop-in where there is no gap: same value as the YAML route."""
        batch = _batch()

        def reconstruction(**settings: object) -> float:
            torch.manual_seed(0)  # the same weights and reparameterisation noise for both
            trainer = Trainer(_model(), modality_dropout_p=0.0, **settings)  # type: ignore[arg-type]
            torch.manual_seed(1)
            return trainer.computeLosses(batch, step=0).reconstruction.item()

        yaml_route = reconstruction(
            reconstruction_loss_fn=functional.mse_loss, reconstruction_reduction="sum"
        )
        python_route = reconstruction(reconstruction_loss_fn=computeMaskedSquaredError)
        assert yaml_route == pytest.approx(python_route, rel=1e-4)


def _recovered(
    model: GlobalVae, observed: torch.Tensor, complete: torch.Tensor, valid: torch.Tensor
) -> dict[str, float]:
    """Share of the variance of the complete series recovered, from the mean latent vector.

    Returns the share on the observed values and on the gaps inside the recording (the values
    the model never saw, which only the latent vector of the rest can give).
    """
    model.eval()
    with torch.no_grad():
        reconstruction = model({"time_series": observed}, use_mean=True)["reconstructions"][
            "time_series"
        ]
    steps = torch.arange(observed.shape[-1])[None, None, :]
    inside = (steps < valid[:, None, None]).expand_as(observed)
    seen = ~torch.isnan(observed)

    def share(mask: torch.Tensor) -> float:
        truth = complete[mask]
        error = (reconstruction[mask] - truth).pow(2).sum()
        return float(1 - error / (truth - truth.mean()).pow(2).sum())

    return {"observed": share(seen), "gap": share(inside & ~seen)}


def _batchesOf(series: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Batches of 64 series."""
    return [{"time_series": series[start : start + 64]} for start in range(0, len(series), 64)]


def _constantGuess(
    training: torch.Tensor, held_out: torch.Tensor, complete: torch.Tensor, valid: torch.Tensor
) -> dict[str, float]:
    """The same shares for the best answer that ignores the series: the mean series of training."""
    mean_series = torch.nanmean(training, dim=0, keepdim=True).expand_as(held_out)
    steps = torch.arange(held_out.shape[-1])[None, None, :]
    inside = (steps < valid[:, None, None]).expand_as(held_out)
    seen = ~torch.isnan(held_out)

    def share(mask: torch.Tensor) -> float:
        truth = complete[mask]
        error = (mean_series[mask] - truth).pow(2).sum()
        return float(1 - error / (truth - truth.mean()).pow(2).sum())

    return {"observed": share(seen), "gap": share(inside & ~seen)}


class TestAModelThatLearnsSeriesWithGaps:
    """The settings of the experiment on synthetic series, not on a named recording."""

    def test_the_masked_loss_recovers_the_gaps_it_never_saw(self) -> None:
        assert (CORPUS_CHANNELS, CORPUS_LENGTH) == (CHANNELS, LENGTH)
        torch.manual_seed(0)
        observed, _, _ = makeSeries(1024, seed=1)
        held_out, complete, valid = makeSeries(256, seed=2)
        guess = _constantGuess(observed, held_out, complete, valid)
        model = _model()
        trainer = Trainer(
            model,
            optimizer_kwargs={"lr": 3e-3},
            reconstruction_loss_fn=computeMaskedSquaredError,
            reconstruction_weights=50.0,
            modality_dropout_p=0.0,
        )
        trainer.fit(_batchesOf(observed), num_epochs=30)
        recovered = _recovered(model, held_out, complete, valid)
        assert recovered["observed"] > guess["observed"] + 0.2
        assert recovered["gap"] > guess["gap"] + 0.2

    def test_the_experiment_learns_complete_series_through_the_yaml_route(
        self, tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        _, complete, _ = makeSeries(1024, seed=1, gaps=False)
        held_out, held_complete, valid = makeSeries(256, seed=2, gaps=False)
        guess = _constantGuess(complete, held_out, held_complete, valid)
        training = _experiment(
            "training.beta_schedules.z_fused.kwargs.warmup_steps=20",
            "training.optimizer.kwargs.lr=0.003",
            output_dir=tmp_path,
        ).training
        model = _model()
        trainer = buildTrainerFromConfig(model, training)
        trainer.fit(_batchesOf(complete), num_epochs=30, val_dataloader=_batchesOf(held_out))
        recovered = _recovered(model, held_out, held_complete, valid)
        assert recovered["observed"] > guess["observed"] + 0.2


class TestEvaluation:
    """`scripts/evaluate.py` and its figures read a time series model with no code of their own."""

    def test_the_total_loss_and_the_figures_work_on_series_with_gaps(self, tmp_path: Path) -> None:
        model = _model().eval()
        loader = [_gappy(6)]
        results = evaluate(model, loader, reconstruction_loss_fn=computeMaskedSquaredError)
        assert results.num_samples == 6
        assert set(results.reconstruction_metrics) == {"time_series"}
        assert torch.isfinite(torch.tensor(results.total_reconstruction_loss))
        paths = exportEvaluationFigures(model, loader, tmp_path)
        assert [path.name for path in paths] == [
            "reconstructions_time_series.png",
            "latent_z_fused.png",
            "kl_z_fused.png",
        ]
        assert all(path.exists() for path in paths)

    def test_the_per_modality_metrics_are_nan_on_series_with_gaps(self) -> None:
        """Pins a finding of ADR 0042: the metrics of `evaluate` know no mask.

        They are fine on complete series. On a gap they are `NaN`, and the way round is to compute
        them over the observed values in Python (docs/how-to/train-on-time-series.md).
        """
        model = _model().eval()
        complete = evaluate(model, [_batch(6)]).reconstruction_metrics["time_series"]
        gapped = evaluate(model, [_gappy(6)]).reconstruction_metrics["time_series"]
        assert all(torch.isfinite(torch.tensor(value)) for value in complete.values())
        assert all(torch.isnan(torch.tensor(value)) for value in gapped.values())


class TestTheDataContract:
    def test_the_data_config_has_an_empty_pipeline_because_the_standardization_is_per_channel(
        self,
    ) -> None:
        assert _experiment().data.transforms == {"time_series": []}

    def test_the_configured_pipeline_is_the_identity(self) -> None:
        pipeline = buildTransformPipeline(_experiment().data)["time_series"]
        x = _batch()["time_series"]
        assert torch.equal(pipeline(x), x)
        assert torch.equal(pipeline.inverse(x), x)

    def test_a_per_channel_standardization_built_in_code_leaves_a_gap_a_gap_and_inverts(
        self,
    ) -> None:
        """What the how-to tells a loader to do, since a YAML file can only write a scalar."""
        observed = _gappy(8)["time_series"]
        mean = torch.tensor([1.0, -2.0, 0.5]).view(CHANNELS, 1)
        std = torch.tensor([2.0, 0.5, 1.5]).view(CHANNELS, 1)
        transform = StandardizeTransform(mean=mean, std=std)
        scaled = transform(observed)
        assert torch.equal(torch.isnan(scaled), torch.isnan(observed))
        restored = transform.inverse(scaled)
        seen = ~torch.isnan(observed)
        assert torch.allclose(restored[seen], observed[seen], atol=1e-5)


def _runTrain(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/time_series_vae` for two short epochs."""
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{EXPERIMENT}",
        f"data.loader_factory={_DUMMY_SERIES_LOADER}",
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
