"""The graph model config and its experiment (roadmap P3-8, spec §6 and §10).

`configs/model/graph_cnn_single_latent.yaml` is the modality added after the table, the waveform,
the text and the time series, and one more proof that the core does not change.
`test_model_config_schema.py` already checks that every shipped file validates and builds. This
file checks what is specific to this one: that the checklist held (no core module knows the new
class), the wiring, the shapes a YAML value produces, padding, the node order, why a `NaN` is not
the padding value, what the loss of the experiment is (and why the class that gives it
probabilities exists), a model that learns graphs of several families, what node order costs,
evaluation, the data contract, and a run of `scripts/train.py` on dummy batches.
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
from global_vae.decoders.registry import listRegisteredDecoders
from global_vae.encoders.registry import listRegisteredEncoders
from global_vae.evaluation.evaluate import evaluate
from global_vae.evaluation.visual_export import exportEvaluationFigures
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from global_vae.utils.graph_metrics import computeEdgeScores
from tests.integration._graph_corpus import N_MAX as CORPUS_NODES
from tests.integration._graph_corpus import makeGraphs, shuffleNodes

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_SRC_DIR = _REPO_ROOT / "src" / "global_vae"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_DUMMY_GRAPH_LOADER = "tests.integration._train_script_fixtures:buildDummyGraphDataloaders"
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

MODEL_FILE = "graph_cnn_single_latent"
EXPERIMENT = "graph_vae"
BATCH_SIZE = 4
NODES = 16

# The packages that make a model, a loss, a training loop or a config. Adding a modality must not
# touch them (spec §10: zero changes to core framework code), so none may name this one: neither
# its registered class nor its batch key.
_MODALITY_REFERENCE = re.compile(r"graph_cnn|GraphCnn|[\"']graph[\"']")
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
    pair = config.modalities["graph"]
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


def _graphs(rows: int = BATCH_SIZE, size: int = NODES, seed: int = 0) -> torch.Tensor:
    """Random undirected graphs with a zero diagonal."""
    generator = torch.Generator().manual_seed(seed)
    upper = torch.triu((torch.rand(rows, size, size, generator=generator) < 0.3).float(), 1)
    return upper + upper.transpose(1, 2)


def _batch(rows: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    return {"graph": _graphs(rows)}


class TestTheNewModalityChecklist:
    """The practical test of the registry design: no core file is needed for a new modality."""

    def test_the_file_validates_and_builds(self) -> None:
        config = _modelConfig()
        validateModelConfig(config)
        model = buildModelFromConfig(config)
        assert set(model.latent_spaces) == {"z_fused"}
        assert model.input_modalities == ("graph",)
        assert model.output_modalities == ("graph",)

    def test_it_names_a_registered_encoder_that_already_existed_and_a_new_decoder(self) -> None:
        """The encoder is the 2D CNN one as it is: an adjacency matrix is an image (ADR 0043)."""
        pair = _modelConfig().modalities["graph"]
        assert pair.encoder.name == "2d_cnn_encoder_v1"
        assert pair.decoder.name == "graph_cnn_decoder_v1"
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
        reconstruction = _model()(_batch())["reconstructions"]["graph"]
        assert reconstruction.shape == (BATCH_SIZE, NODES, NODES)

    def test_the_reconstruction_is_a_symmetric_matrix_of_probabilities_with_a_zero_diagonal(
        self,
    ) -> None:
        reconstruction = _model()(_batch())["reconstructions"]["graph"]
        assert (reconstruction >= 0).all()
        assert (reconstruction <= 1).all()
        assert torch.equal(reconstruction, reconstruction.transpose(1, 2))
        assert (reconstruction.diagonal(dim1=1, dim2=2) == 0).all()

    def test_the_declared_size_is_the_one_of_the_data_file(self) -> None:
        """One loader must serve the data file and the model: the same padded number of nodes."""
        declared = _modelConfig().modalities["graph"].decoder.kwargs["output_shape"]
        assert list(declared) == [NODES, NODES]
        assert _experiment().data.sequence_length == {"graph": NODES}

    def test_the_corpus_of_the_tests_has_the_size_the_files_declare(self) -> None:
        assert CORPUS_NODES == NODES

    def test_the_latent_width_is_set_in_one_place(self) -> None:
        config = _modelConfig()
        assert config.single_latent is not None
        mu, logvar = _model()(_batch())["latent_params"]["z_fused"]
        assert mu.shape == logvar.shape == (BATCH_SIZE, config.single_latent.dim)

    def test_a_graph_padded_with_isolated_nodes_is_encoded_and_reconstructed(self) -> None:
        graphs, _, _ = makeGraphs(BATCH_SIZE, seed=3)
        assert (graphs.sum(dim=-1) == 0).any()  # some rows are padding
        outputs = _model()({"graph": graphs})
        assert torch.isfinite(outputs["latent_params"]["z_fused"][0]).all()
        assert outputs["reconstructions"]["graph"].shape == (BATCH_SIZE, NODES, NODES)

    def test_a_nan_is_not_a_padding_value_it_turns_the_whole_latent_vector_into_nan(self) -> None:
        """Pins why a padded node is a row of zeros and not of `NaN` (ADR 0043)."""
        graphs = _graphs()
        graphs[:, 12:, :] = float("nan")
        graphs[:, :, 12:] = float("nan")
        mu, _ = _model()({"graph": graphs})["latent_params"]["z_fused"]
        assert torch.isnan(mu).all()

    def test_the_same_graph_in_another_node_order_is_another_latent_vector(self) -> None:
        """Pins the order dependence of ADR 0043: nodes must come in a meaningful order."""
        model = _model().eval()
        graphs = _graphs()
        order = torch.randperm(NODES, generator=torch.Generator().manual_seed(0))
        permuted = graphs[:, order][:, :, order]
        with torch.no_grad():
            first = model({"graph": graphs}, use_mean=True)["latent_params"]["z_fused"][0]
            second = model({"graph": permuted}, use_mean=True)["latent_params"]["z_fused"][0]
        assert not torch.allclose(first, second, atol=1e-3)

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
        """A loader can carry the number of nodes of each graph beside the matrices (ADR 0019)."""
        model = _model()
        batch = {**_batch(), "num_nodes": torch.full((BATCH_SIZE,), 10)}
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(batch, step=0)
        assert torch.isfinite(losses.total)
        assert set(model(model.selectEncoderInputs(batch))["reconstructions"]) == {"graph"}


class TestWhatTheYamlLetsYouChange:
    def test_a_directed_graph_is_a_yaml_choice(self) -> None:
        reconstruction = _model(decoder_symmetric=False)(_batch())["reconstructions"]["graph"]
        assert not torch.allclose(reconstruction, reconstruction.transpose(1, 2))

    def test_self_loops_are_a_yaml_choice(self) -> None:
        reconstruction = _model(decoder_zero_diagonal=False)(_batch())["reconstructions"]["graph"]
        assert (reconstruction.diagonal(dim1=1, dim2=2) > 0).all()

    def test_several_kinds_of_edge_are_set_in_the_encoder_and_the_decoder(self) -> None:
        model = _model(encoder_in_channels=2, decoder_out_channels=2)
        batch = {"graph": torch.stack([_graphs(), _graphs(seed=1)], dim=1)}
        assert model(batch)["reconstructions"]["graph"].shape == (BATCH_SIZE, 2, NODES, NODES)

    def test_another_padded_size_is_set_where_it_is_written(self) -> None:
        """A seed of 8 by 8 with the two doubling stages is 8 -> 16 -> 32."""
        model = _model(decoder_output_shape=[32, 32], decoder_seed_shape=[8, 8])
        reconstruction = model({"graph": _graphs(size=32)})["reconstructions"]["graph"]
        assert reconstruction.shape == (BATCH_SIZE, 32, 32)

    def test_a_decoder_that_does_not_land_on_the_size_is_refused_when_the_model_is_built(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="output_shape"):
            _model(decoder_output_shape=[20, 20])

    def test_a_matrix_that_is_not_square_is_refused_when_the_model_is_built(self) -> None:
        with pytest.raises(ValueError, match="square"):
            _model(decoder_output_shape=[16, 8])

    def test_a_decoder_of_another_size_is_refused_by_the_loss_with_both_shapes(self) -> None:
        """The encoder takes a graph of any size, so the loss is where a mismatch shows."""
        model = _model(decoder_output_shape=[32, 32], decoder_seed_shape=[8, 8])
        trainer = Trainer(model, modality_dropout_p=0.0)
        with pytest.raises(ValueError, match="shape"):
            trainer.computeLosses(_batch(), step=0)


class TestTheLossOfTheExperiment:
    """What `bce` with a sum on a 0/1 adjacency matrix is, and why the decoder has a sigmoid."""

    def test_the_experiment_sets_a_bernoulli_loss_summed_per_graph_with_weight_four(self) -> None:
        training = _experiment().training
        assert training.reconstruction_loss == "bce"
        assert training.reconstruction_reduction == "sum"
        assert training.reconstruction_weight == 4.0

    def test_the_registered_2d_decoder_cannot_be_used_with_bce_which_is_why_the_class_exists(
        self, tmp_path: Path
    ) -> None:
        """Pins the finding of ADR 0043: the parent's raw output is outside `[0, 1]`."""
        config = _modelConfig()
        config.modalities["graph"].decoder.name = "2d_cnn_decoder_v1"
        model = buildModelFromConfig(config)
        trainer = buildTrainerFromConfig(model, _experiment(output_dir=tmp_path).training)
        with pytest.raises((RuntimeError, ValueError)):
            trainer.computeLosses(_batch(), step=0)

    def test_sum_is_the_number_of_pairs_of_a_matrix_times_the_mean(self) -> None:
        """`sum` multiplies the element-wise mean by `N * N` (ADR 0029)."""
        batch = _batch()

        def reconstruction(reduction: str) -> float:
            torch.manual_seed(0)  # the same weights and reparameterisation noise for both
            trainer = Trainer(
                _model(),
                reconstruction_loss_fn=functional.binary_cross_entropy,
                reconstruction_reduction=reduction,
            )
            torch.manual_seed(1)
            return trainer.computeLosses(batch, step=0).reconstruction.item()

        assert reconstruction("sum") / reconstruction("mean") == pytest.approx(
            NODES * NODES, rel=1e-4
        )


def _batchesOf(graphs: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Batches of 64 graphs."""
    return [{"graph": graphs[start : start + 64]} for start in range(0, len(graphs), 64)]


def _trainedScores(
    train: torch.Tensor, held_out: torch.Tensor, num_nodes: torch.Tensor, output_dir: Path
) -> dict[str, float]:
    """Train the shipped experiment on `train` and score the held-out graphs' edges."""
    training = _experiment(
        "training.beta_schedules.z_fused.kwargs.warmup_steps=20",
        "training.optimizer.kwargs.lr=0.003",
        output_dir=output_dir,
    ).training
    model = _model()
    trainer = buildTrainerFromConfig(model, training)
    trainer.fit(_batchesOf(train), num_epochs=30, val_dataloader=_batchesOf(held_out))
    model.eval()
    with torch.no_grad():
        probabilities = model({"graph": held_out}, use_mean=True)["reconstructions"]["graph"]
    return computeEdgeScores(probabilities, held_out, num_nodes)


def _constantGuess(
    train: torch.Tensor, train_nodes: torch.Tensor, held_out: torch.Tensor, num_nodes: torch.Tensor
) -> dict[str, float]:
    """The best answer that ignores the graph: the mean matrix over the graphs that have a pair."""
    exists = torch.arange(NODES)[None, :] < train_nodes[:, None]
    pair = exists[:, :, None] & exists[:, None, :]
    mean_matrix = (train * pair).sum(dim=0) / pair.sum(dim=0).clamp_min(1)
    return computeEdgeScores(mean_matrix.expand_as(held_out), held_out, num_nodes)


class TestAModelThatLearnsGraphs:
    """The settings of the experiment on synthetic graphs, not on a named collection."""

    def test_the_experiment_beats_the_best_guess_that_ignores_the_graph(
        self, tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        train, train_nodes, _ = makeGraphs(1024, seed=1)
        held_out, held_nodes, _ = makeGraphs(256, seed=2)
        guess = _constantGuess(train, train_nodes, held_out, held_nodes)
        scores = _trainedScores(train, held_out, held_nodes, tmp_path)
        assert scores["auc"] > guess["auc"] + 0.15
        assert scores["bce"] < guess["bce"] - 0.1

    def test_the_order_of_the_nodes_is_worth_more_than_anything_the_model_can_do_about_it(
        self, tmp_path: Path
    ) -> None:
        """Pins the main finding of ADR 0043: the same graphs in a random order are far harder."""
        torch.manual_seed(0)
        train, train_nodes, _ = makeGraphs(1024, seed=1)
        held_out, held_nodes, _ = makeGraphs(256, seed=2)
        ordered = _trainedScores(train, held_out, held_nodes, tmp_path / "ordered")
        torch.manual_seed(0)
        shuffled = _trainedScores(
            shuffleNodes(train, train_nodes, seed=5),
            shuffleNodes(held_out, held_nodes, seed=6),
            held_nodes,
            tmp_path / "shuffled",
        )
        assert ordered["auc"] > shuffled["auc"] + 0.1


class TestEvaluation:
    """`scripts/evaluate.py` and its figures read a graph model with no graph code of their own."""

    def test_the_report_and_the_figures_run(self, tmp_path: Path) -> None:
        """With the number of nodes in the batch, as a loader that scores padded graphs has it."""
        model = _model().eval()
        loader = [{**_batch(6), "num_nodes": torch.full((6,), 12)}]
        results = evaluate(
            model,
            loader,
            reconstruction_loss_fn=functional.binary_cross_entropy,
            reconstruction_reduction="sum",
        )
        assert results.num_samples == 6
        assert set(results.reconstruction_metrics) == {"graph"}
        assert torch.isfinite(torch.tensor(results.total_reconstruction_loss))
        paths = exportEvaluationFigures(model, loader, tmp_path)
        assert [path.name for path in paths] == [
            "reconstructions_graph.png",
            "latent_z_fused.png",
            "kl_z_fused.png",
        ]
        assert all(path.exists() for path in paths)

    def test_the_edge_scores_read_the_reconstruction_of_a_batch(self) -> None:
        model = _model().eval()
        graphs, num_nodes, _ = makeGraphs(6, seed=4)
        with torch.no_grad():
            probabilities = model({"graph": graphs}, use_mean=True)["reconstructions"]["graph"]
        scores = computeEdgeScores(probabilities, graphs, num_nodes)
        assert set(scores) == {"bce", "auc", "f1", "density", "num_pairs"}
        assert all(torch.isfinite(torch.tensor(value)) for value in scores.values())


class TestTheDataContract:
    def test_the_data_config_has_an_empty_pipeline_because_an_adjacency_matrix_is_not_a_quantity(
        self,
    ) -> None:
        assert _experiment().data.transforms == {"graph": []}

    def test_the_configured_pipeline_is_the_identity(self) -> None:
        pipeline = buildTransformPipeline(_experiment().data)["graph"]
        x = _graphs()
        assert torch.equal(pipeline(x), x)
        assert torch.equal(pipeline.inverse(x), x)


def _runTrain(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/graph_vae` for two short epochs."""
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{EXPERIMENT}",
        f"data.loader_factory={_DUMMY_GRAPH_LOADER}",
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
