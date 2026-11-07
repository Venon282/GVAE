"""The point cloud model config and its experiment (roadmap P3-8, spec §6 and §10).

`configs/model/point_cloud_single_latent.yaml` is the modality added after the table, the waveform,
the text, the time series and the graph, and one more proof that the core does not change.
`test_model_config_schema.py` already checks that every shipped file validates and builds. This
file checks what is specific to this one: that the checklist held (no core module knows the new
classes), the wiring, the shapes a YAML value produces, what a set is to the encoder (order and
padding), what the loss of the experiment is (and why it is a Python function that the file only
points to), a model that learns clouds of several shapes, evaluation, the data contract, and a run
of `scripts/train.py` on dummy batches.
"""

import math
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
from global_vae.config.training import buildTrainerFromConfig
from global_vae.decoders.registry import listRegisteredDecoders
from global_vae.encoders.registry import listRegisteredEncoders
from global_vae.evaluation.evaluate import evaluate
from global_vae.evaluation.visual_export import exportEvaluationFigures
from global_vae.losses.reconstruction.chamfer import ChamferLoss
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from global_vae.utils.point_cloud_metrics import computeChamferDistance, computeCloudScores
from tests.integration._point_cloud_corpus import N_MAX as CORPUS_POINTS
from tests.integration._point_cloud_corpus import makeClouds, sortPoints

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_SRC_DIR = _REPO_ROOT / "src" / "global_vae"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_DUMMY_POINT_CLOUD_LOADER = (
    "tests.integration._train_script_fixtures:buildDummyPointCloudDataloaders"
)
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

MODEL_FILE = "point_cloud_single_latent"
EXPERIMENT = "point_cloud_vae"
BATCH_SIZE = 4
POINTS = 64

# The packages that make a model, a loss, a training loop or a config. Adding a modality must not
# touch them (spec §10: zero changes to core framework code), so none may name this one: neither
# its registered classes nor its batch key.
# `utils.point_cloud_metrics` is the shared helper that the `chamfer` loss and the cloud metrics
# wrap (roadmap P5-3, ADR 0048), a utility and not a modality class or a batch key.
_MODALITY_REFERENCE = re.compile(r"point_cloud(?!_metrics)|PointCloud")
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
    pair = config.modalities["point_cloud"]
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


def _chamferTrainer(model: GlobalVae, *overrides: str, output_dir: Path) -> Trainer:
    """The trainer of the experiment, which names the Chamfer distance in its YAML (ADR 0048).

    The loss, the optimizer, the beta schedule and the weight all come from the file. The Chamfer
    distance is already a sum over points and counts as one term, so `sum` does not rescale it.
    """
    trainer = buildTrainerFromConfig(model, _experiment(*overrides, output_dir=output_dir).training)
    assert isinstance(trainer.reconstruction_loss_fn, ChamferLoss)
    return trainer


def _clouds(rows: int = BATCH_SIZE, points: int = POINTS, seed: int = 0) -> torch.Tensor:
    """Random complete clouds of `points` points in the unit ball, in no particular order."""
    generator = torch.Generator().manual_seed(seed)
    clouds = torch.randn(rows, points, 3, generator=generator)
    inside: torch.Tensor = clouds / clouds.norm(dim=-1, keepdim=True).clamp_min(1.0)
    return inside


def _batch(rows: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    return {"point_cloud": _clouds(rows)}


def _meanLatent(model: GlobalVae, clouds: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        mu: torch.Tensor = model({"point_cloud": clouds}, use_mean=True)["latent_params"][
            "z_fused"
        ][0]
    return mu


class TestTheNewModalityChecklist:
    """The practical test of the registry design: no core file is needed for a new modality."""

    def test_the_file_validates_and_builds(self) -> None:
        config = _modelConfig()
        validateModelConfig(config)
        model = buildModelFromConfig(config)
        assert set(model.latent_spaces) == {"z_fused"}
        assert model.input_modalities == ("point_cloud",)
        assert model.output_modalities == ("point_cloud",)

    def test_it_names_two_classes_registered_by_the_modality_and_nothing_else_new(self) -> None:
        pair = _modelConfig().modalities["point_cloud"]
        assert pair.encoder.name == "point_cloud_encoder_v1"
        assert pair.decoder.name == "point_cloud_decoder_v1"
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
        reconstruction = _model()(_batch())["reconstructions"]["point_cloud"]
        assert reconstruction.shape == (BATCH_SIZE, POINTS, 3)

    def test_the_declared_number_of_points_is_the_one_of_the_data_file(self) -> None:
        """One loader must serve the data file and the model: the same number of rows."""
        declared = _modelConfig().modalities["point_cloud"].decoder.kwargs["num_points"]
        assert declared == POINTS
        assert _experiment().data.sequence_length == {"point_cloud": POINTS}

    def test_the_corpus_of_the_tests_has_the_size_the_files_declare(self) -> None:
        assert CORPUS_POINTS == POINTS

    def test_the_latent_width_is_set_in_one_place(self) -> None:
        config = _modelConfig()
        assert config.single_latent is not None
        mu, logvar = _model()(_batch())["latent_params"]["z_fused"]
        assert mu.shape == logvar.shape == (BATCH_SIZE, config.single_latent.dim)

    def test_every_parameter_receives_a_gradient_on_clouds_with_padding(
        self, tmp_path: Path
    ) -> None:
        clouds, _, _ = makeClouds(BATCH_SIZE, seed=3)
        assert torch.isnan(clouds).any()  # some rows are padding
        model = _model()
        trainer = _chamferTrainer(model, output_dir=tmp_path)
        losses = trainer.computeLosses({"point_cloud": clouds}, step=0)
        backward(losses.total)
        assert torch.isfinite(losses.total)
        assert [n for n, p in model.named_parameters() if p.grad is None] == []
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    def test_the_loss_falls_over_a_few_steps(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = _model()
        trainer = _chamferTrainer(model, output_dir=tmp_path)
        batch = {"point_cloud": makeClouds(16, seed=4)[0]}
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(30):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        assert trainer.computeLosses(batch, step=30).reconstruction.item() < first

    def test_a_key_with_no_encoder_or_decoder_is_ignored(self, tmp_path: Path) -> None:
        """A loader can carry the number of points of each cloud beside the clouds (ADR 0019)."""
        model = _model()
        batch = {**_batch(), "num_points": torch.full((BATCH_SIZE,), 50)}
        trainer = _chamferTrainer(model, output_dir=tmp_path)
        assert torch.isfinite(trainer.computeLosses(batch, step=0).total)
        selected = model.selectEncoderInputs(batch)
        assert set(model(selected)["reconstructions"]) == {"point_cloud"}


class TestASetIsNotASequence:
    """What the encoder does with the order of the points and with rows of `NaN` (ADR 0044)."""

    def test_the_same_points_in_another_order_are_the_same_latent_vector(self) -> None:
        model = _model()
        clouds = _clouds()
        order = torch.randperm(POINTS, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(
            _meanLatent(model, clouds), _meanLatent(model, clouds[:, order]), atol=1e-5
        )

    def test_rows_of_nan_are_not_points_wherever_they_are(self) -> None:
        """A cloud padded to 64 rows equals the same cloud with only its points, at any place."""
        model = _model()
        clouds = _clouds(points=40)
        padded = torch.cat([clouds, torch.full((BATCH_SIZE, 24, 3), float("nan"))], dim=1)
        scattered = padded[:, torch.randperm(POINTS, generator=torch.Generator().manual_seed(1))]
        expected = _meanLatent(model, clouds)
        assert torch.isfinite(_meanLatent(model, padded)).all()
        assert torch.allclose(_meanLatent(model, padded), expected, atol=1e-5)
        assert torch.allclose(_meanLatent(model, scattered), expected, atol=1e-5)

    def test_clouds_of_different_sizes_in_one_batch_are_encoded_each_by_its_own_points(
        self,
    ) -> None:
        model = _model()
        clouds, num_points, _ = makeClouds(BATCH_SIZE, seed=5)
        assert int(num_points.min()) < int(num_points.max())
        together = _meanLatent(model, clouds)
        for index in range(BATCH_SIZE):
            cloud = clouds[index]
            points = cloud[~torch.isnan(cloud).any(dim=-1)]
            assert len(points) == int(num_points[index])
            alone = _meanLatent(model, points[None])[0]
            assert torch.allclose(together[index], alone, atol=1e-5), index

    def test_a_cloud_with_no_point_is_a_finite_latent_vector(self) -> None:
        clouds = _clouds()
        clouds[1] = float("nan")
        assert torch.isfinite(_meanLatent(_model(), clouds)).all()

    def test_the_reconstruction_has_no_order_to_read_either(self) -> None:
        """The decoder writes points from learned codes, not from a copy of the input's order."""
        model = _model().eval()
        clouds = _clouds()
        order = torch.randperm(POINTS, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            first = model({"point_cloud": clouds}, use_mean=True)["reconstructions"]["point_cloud"]
            second = model({"point_cloud": clouds[:, order]}, use_mean=True)["reconstructions"][
                "point_cloud"
            ]
        assert torch.allclose(first, second, atol=1e-5)


class TestWhatTheYamlLetsYouChange:
    def test_points_that_carry_more_than_coordinates_are_a_yaml_choice(self) -> None:
        """Normals or colors: `(batch, N, 3 + k)` in, the same out, set in the two files' kwargs."""
        model = _model(encoder_in_dim=6, decoder_out_dim=6)
        batch = {"point_cloud": torch.randn(BATCH_SIZE, POINTS, 6)}
        assert model(batch)["reconstructions"]["point_cloud"].shape == (BATCH_SIZE, POINTS, 6)

    def test_another_number_of_points_is_set_where_it_is_written(self) -> None:
        model = _model(decoder_num_points=128)
        reconstruction = model({"point_cloud": _clouds(points=128)})["reconstructions"]
        assert reconstruction["point_cloud"].shape == (BATCH_SIZE, 128, 3)

    @pytest.mark.parametrize("pooling", ["max_mean", "max", "mean"])
    def test_every_pooling_gives_a_latent_vector_that_does_not_read_the_order(
        self, pooling: str
    ) -> None:
        model = _model(encoder_pooling=pooling)
        clouds = _clouds()
        order = torch.randperm(POINTS, generator=torch.Generator().manual_seed(3))
        assert torch.allclose(
            _meanLatent(model, clouds), _meanLatent(model, clouds[:, order]), atol=1e-5
        )

    def test_layer_normalization_is_a_yaml_choice_and_batch_normalization_is_refused(
        self,
    ) -> None:
        model = _model(encoder_normalization="layer", decoder_normalization="layer")
        assert model(_batch())["reconstructions"]["point_cloud"].shape == (BATCH_SIZE, POINTS, 3)
        with pytest.raises(ValueError, match="mix the points"):
            _model(encoder_normalization="batch")

    def test_a_pooling_that_does_not_exist_is_refused_when_the_model_is_built(self) -> None:
        with pytest.raises(ValueError, match="pooling"):
            _model(encoder_pooling="attention")

    def test_a_decoder_of_another_size_trains_with_chamfer_and_is_refused_with_mse(
        self, tmp_path: Path
    ) -> None:
        """The Chamfer distance takes any two sizes of cloud, and so does the loss that wraps it.

        A decoder that writes 32 points is scored against clouds of 64 by the experiment's
        `chamfer` (it opts out of the shape check, ADR 0048). A position-by-position loss such as
        `mse` still needs equal shapes, and the trainer refuses with both shapes (ADR 0044).
        """
        model = _model(decoder_num_points=32)
        guess = model(_batch())["reconstructions"]["point_cloud"]
        assert guess.shape == (BATCH_SIZE, 32, 3)
        trainer = _chamferTrainer(model, output_dir=tmp_path)
        losses = trainer.computeLosses(_batch(), step=0)
        backward(losses.total)
        assert torch.isfinite(losses.total)
        mse_trainer = buildTrainerFromConfig(
            model, _experiment("training.reconstruction_loss=mse", output_dir=tmp_path).training
        )
        with pytest.raises(ValueError, match="shape"):
            mse_trainer.computeLosses(_batch(), step=0)


class TestTheLossOfTheExperiment:
    """Why the loss of a point cloud is the Chamfer distance, and that the YAML file names it."""

    def test_the_file_names_the_chamfer_distance_with_weight_one_hundred(self) -> None:
        training = _experiment().training
        assert training.reconstruction_loss == "chamfer"
        assert training.reconstruction_reduction == "sum"
        assert training.reconstruction_weight == 100.0

    def test_the_file_has_no_placeholder_loss_and_no_warning_header(self) -> None:
        text = (_CONFIG_DIR / "experiment" / f"{EXPERIMENT}.yaml").read_text()
        assert "READ THIS BEFORE TRAINING" not in text
        assert "Python route" not in text
        assert "placeholder" not in text

    def test_a_position_by_position_loss_depends_on_the_order_of_the_points_the_chamfer_does_not(
        self,
    ) -> None:
        clouds = _clouds()
        order = torch.randperm(POINTS, generator=torch.Generator().manual_seed(4))
        shuffled = clouds[:, order]
        assert float(torch.nn.functional.mse_loss(shuffled, clouds)) > 0.05
        assert float(computeChamferDistance(shuffled, clouds)) < 1e-4

    def test_the_mse_is_nan_on_a_padded_cloud_and_the_chamfer_of_the_file_is_not(
        self, tmp_path: Path
    ) -> None:
        clouds, _, _ = makeClouds(BATCH_SIZE, seed=6)
        batch = {"point_cloud": clouds}
        model = _model()
        mse_trainer = buildTrainerFromConfig(
            model, _experiment("training.reconstruction_loss=mse", output_dir=tmp_path).training
        )
        assert torch.isnan(mse_trainer.computeLosses(batch, step=0).reconstruction)
        chamfer_trainer = _chamferTrainer(model, output_dir=tmp_path)
        assert torch.isfinite(chamfer_trainer.computeLosses(batch, step=0).reconstruction)

    def test_the_registered_chamfer_counts_as_one_term_so_sum_leaves_it_alone(self) -> None:
        batch = {"point_cloud": makeClouds(BATCH_SIZE, seed=7)[0]}

        def reconstruction(reduction: str) -> float:
            torch.manual_seed(0)  # the same weights and reparameterisation noise for both
            trainer = Trainer(
                _model(), reconstruction_loss_fn=ChamferLoss(), reconstruction_reduction=reduction
            )
            torch.manual_seed(1)
            return trainer.computeLosses(batch, step=0).reconstruction.item()

        assert reconstruction("sum") == reconstruction("mean")

    def test_sum_multiplies_the_plain_chamfer_function_by_the_values_of_a_cloud(self) -> None:
        """The plain function, passed by hand, is rescaled by `sum` like any callable (ADR 0029)."""
        batch = {"point_cloud": makeClouds(BATCH_SIZE, seed=7)[0]}

        def reconstruction(reduction: str) -> float:
            torch.manual_seed(0)  # the same weights and reparameterisation noise for both
            trainer = Trainer(
                _model(),
                reconstruction_loss_fn=computeChamferDistance,
                reconstruction_reduction=reduction,
            )
            torch.manual_seed(1)
            return trainer.computeLosses(batch, step=0).reconstruction.item()

        assert reconstruction("sum") / reconstruction("mean") == pytest.approx(POINTS * 3, rel=1e-4)


def _batchesOf(clouds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Batches of 64 clouds."""
    return [{"point_cloud": clouds[start : start + 64]} for start in range(0, len(clouds), 64)]


def _trainedScores(
    train: torch.Tensor, held_out: torch.Tensor, output_dir: Path, *, chamfer: bool = True
) -> dict[str, float]:
    """Train the shipped experiment on `train` and score the held-out clouds' reconstructions.

    `chamfer=False` trains it with `mse` instead (summed), with the same optimizer.
    """
    overrides = (
        "training.beta_schedules.z_fused.kwargs.warmup_steps=20",
        "training.optimizer.kwargs.lr=0.002",
    )
    model = _model()
    if chamfer:
        trainer = _chamferTrainer(model, *overrides, output_dir=output_dir)
    else:
        trainer = buildTrainerFromConfig(
            model,
            _experiment(
                *overrides, "training.reconstruction_loss=mse", output_dir=output_dir
            ).training,
        )
    trainer.fit(_batchesOf(train), num_epochs=40, val_dataloader=_batchesOf(held_out))
    model.eval()
    with torch.no_grad():
        reconstruction = model({"point_cloud": held_out}, use_mean=True)["reconstructions"][
            "point_cloud"
        ]
    return computeCloudScores(reconstruction, held_out)


def _constantCloud(train: torch.Tensor, held_out: torch.Tensor) -> dict[str, float]:
    """The best answer that ignores the cloud: one fixed cloud of `POINTS` points.

    Fitted to `train` by gradient descent on the Chamfer distance to random batches of it.
    """
    generator = torch.Generator().manual_seed(0)
    cloud = torch.nn.Parameter(0.5 * torch.randn(POINTS, 3, generator=generator))
    optimizer = torch.optim.Adam([cloud], lr=1e-2)
    for _ in range(300):
        rows = torch.randint(len(train), (128,), generator=generator)
        loss = computeChamferDistance(cloud[None].expand(128, -1, -1), train[rows])
        optimizer.zero_grad()
        backward(loss)
        optimizer.step()
    return computeCloudScores(cloud.detach()[None].expand(len(held_out), -1, -1), held_out)


class TestAModelThatLearnsPointClouds:
    """The settings of the experiment on synthetic clouds, not on a named collection."""

    def test_the_chamfer_route_beats_the_best_cloud_that_ignores_the_input(
        self, tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        train, _, _ = makeClouds(512, seed=1)
        held_out, _, _ = makeClouds(128, seed=2)
        guess = _constantCloud(train, held_out)
        scores = _trainedScores(train, held_out, tmp_path)
        assert scores["chamfer"] < 0.8 * guess["chamfer"]
        assert scores["f_score"] > guess["f_score"] + 0.05

    def test_a_position_by_position_loss_trains_a_model_worse_than_that_cloud(
        self, tmp_path: Path
    ) -> None:
        """Pins the main finding of ADR 0044: `mse` compares points that have nothing in common.

        The clouds are complete, since on a padded one `mse` is `NaN` (tested above). The same
        model, optimizer and weight, with the loss of the file instead of the Chamfer distance,
        ends up several times farther from the clouds than a cloud that ignores them, which is
        why the file names `chamfer`.
        """
        torch.manual_seed(0)
        train, _, _ = makeClouds(512, seed=1, variable=False)
        held_out, _, _ = makeClouds(128, seed=2, variable=False)
        guess = _constantCloud(train, held_out)
        scores = _trainedScores(train, held_out, tmp_path, chamfer=False)
        assert scores["chamfer"] > 2 * guess["chamfer"]

    def test_sorting_the_points_does_not_rescue_a_position_by_position_loss(
        self, tmp_path: Path
    ) -> None:
        """A canonical order helps, and is still worse than that cloud (ADR 0044)."""
        torch.manual_seed(0)
        train, _, _ = makeClouds(512, seed=1, variable=False)
        held_out, _, _ = makeClouds(128, seed=2, variable=False)
        guess = _constantCloud(train, held_out)
        scores = _trainedScores(sortPoints(train), sortPoints(held_out), tmp_path, chamfer=False)
        assert scores["chamfer"] > 2 * guess["chamfer"]


class TestEvaluation:
    """`scripts/evaluate.py` and its figures read a point cloud model with no code of their own."""

    def test_the_report_and_the_figures_run(self, tmp_path: Path) -> None:
        model = _model().eval()
        clouds, _, _ = makeClouds(6, seed=8)
        loader = [{"point_cloud": clouds}]
        results = evaluate(model, loader, reconstruction_loss_fn=ChamferLoss())
        assert results.num_samples == 6
        assert set(results.reconstruction_metrics) == {"point_cloud"}
        assert torch.isfinite(torch.tensor(results.total_reconstruction_loss))
        scores = results.reconstruction_metrics["point_cloud"]
        assert set(scores) == {"chamfer", "precision", "recall", "f_score"}
        assert all(math.isfinite(value) for value in scores.values())
        paths = exportEvaluationFigures(model, loader, tmp_path)
        assert [path.name for path in paths] == [
            "reconstructions_point_cloud.png",
            "latent_z_fused.png",
            "kl_z_fused.png",
        ]
        assert all(path.exists() for path in paths)

    def test_a_plain_function_says_nothing_so_the_five_metrics_are_nan_on_a_padded_cloud(
        self,
    ) -> None:
        """The metrics follow a registered loss; a bare function gets the five, as in 2.0.0."""
        model = _model().eval()
        clouds, _, _ = makeClouds(6, seed=8)
        results = evaluate(
            model, [{"point_cloud": clouds}], reconstruction_loss_fn=computeChamferDistance
        )
        assert torch.isnan(torch.tensor(results.reconstruction_metrics["point_cloud"]["mse"]))

    def test_the_scores_read_the_reconstruction_of_a_padded_batch(self) -> None:
        model = _model().eval()
        clouds, num_points, _ = makeClouds(6, seed=9)
        with torch.no_grad():
            reconstruction = model({"point_cloud": clouds}, use_mean=True)["reconstructions"][
                "point_cloud"
            ]
        scores = computeCloudScores(reconstruction, clouds)
        assert set(scores) == {"chamfer", "precision", "recall", "f_score", "num_points"}
        assert scores["num_points"] == int(num_points.sum())
        assert all(torch.isfinite(torch.tensor(float(value))) for value in scores.values())


class TestSamplingFromThePrior:
    def test_a_point_cloud_is_decoded_from_a_draw_of_the_prior(self) -> None:
        model = _model().eval()
        width = model.latent_spaces["z_fused"].dim
        with torch.no_grad():
            clouds = model.decoders["point_cloud"](torch.randn(8, width))
        assert clouds.shape == (8, POINTS, 3)
        assert torch.isfinite(clouds).all()


class TestTheDataContract:
    def test_the_data_config_has_an_empty_pipeline_because_a_cloud_is_normalized_one_at_a_time(
        self,
    ) -> None:
        assert _experiment().data.transforms == {"point_cloud": []}

    def test_the_configured_pipeline_is_the_identity(self) -> None:
        pipeline = buildTransformPipeline(_experiment().data)["point_cloud"]
        x = _clouds()
        assert torch.equal(pipeline(x), x)
        assert torch.equal(pipeline.inverse(x), x)


def _runTrain(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/point_cloud_vae` for two short epochs."""
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{EXPERIMENT}",
        f"data.loader_factory={_DUMMY_POINT_CLOUD_LOADER}",
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
