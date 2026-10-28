"""The text model config and its experiment (roadmap P3-8, spec §6 and §10).

`configs/model/text_transformer_single_latent.yaml` is the fifth modality added through the
registries after the 1D signal, the image, the table and the waveform, and the fourth proof that
the core does not change. `test_model_config_schema.py` already checks that every shipped file
validates and builds. This file checks what is specific to this one: that the checklist held (no
core module knows the new classes), the wiring, the shapes a YAML value produces, ids and one-hot
batches, padding, a text of any length, a vocabulary mismatch, what the loss of the experiment is
(and is not), a cross-entropy written in Python, a model that learns a small grammar, the data
contract, and a run of `scripts/train.py` on dummy batches.
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
from global_vae.utils.text_blocks import computeTokenCrossEntropy
from tests.integration._text_corpus import LENGTH as CORPUS_LENGTH
from tests.integration._text_corpus import PAD as CORPUS_PAD
from tests.integration._text_corpus import VOCAB as CORPUS_VOCAB
from tests.integration._text_corpus import makeIds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_SRC_DIR = _REPO_ROOT / "src" / "global_vae"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_DUMMY_TEXT_LOADER = "tests.integration._train_script_fixtures:buildDummyTextDataloaders"
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

MODEL_FILE = "text_transformer_single_latent"
EXPERIMENT = "text_vae"
BATCH_SIZE = 4
LENGTH = 8
VOCAB = 32
PAD = 0

# The packages that make a model, a loss, a training loop or a config. Adding a modality must not
# touch them (spec §10: zero changes to core framework code), so none may name this one: neither
# its registered classes nor its batch key.
_MODALITY_REFERENCE = re.compile(r"text_transformer|TextTransformer|[\"']text[\"']")
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
    pair = config.modalities["text"]
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


def _ids(rows: int = BATCH_SIZE, length: int = LENGTH) -> torch.Tensor:
    """Random ids of real tokens, `1` to `VOCAB - 1`, so that `0` stays the padding token."""
    return torch.randint(1, VOCAB, (rows, length))


def _batch(rows: int = BATCH_SIZE, length: int = LENGTH) -> dict[str, torch.Tensor]:
    """A batch of random one-hot texts."""
    return {"text": functional.one_hot(_ids(rows, length), VOCAB).float()}


class TestTheNewModalityChecklist:
    """The practical test of the registry design: no core file is needed for a new modality."""

    def test_the_file_validates_and_builds(self) -> None:
        config = _modelConfig()
        validateModelConfig(config)
        model = buildModelFromConfig(config)
        assert set(model.latent_spaces) == {"z_fused"}
        assert model.input_modalities == ("text",)
        assert model.output_modalities == ("text",)

    def test_it_names_the_two_registered_classes(self) -> None:
        pair = _modelConfig().modalities["text"]
        assert pair.encoder.name == "text_transformer_encoder_v1"
        assert pair.decoder.name == "text_transformer_decoder_v1"
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
        reconstruction = _model()(_batch())["reconstructions"]["text"]
        assert reconstruction.shape == (BATCH_SIZE, LENGTH, VOCAB)

    def test_the_reconstruction_is_a_distribution_over_the_vocabulary_at_each_position(
        self,
    ) -> None:
        reconstruction = _model()(_batch())["reconstructions"]["text"]
        assert (reconstruction >= 0).all()
        assert torch.allclose(reconstruction.sum(dim=-1), torch.ones(BATCH_SIZE, LENGTH), atol=1e-5)

    def test_the_declared_length_is_the_one_of_the_data_file(self) -> None:
        """One loader must serve the data file and the model: same number of tokens."""
        declared = _modelConfig().modalities["text"].decoder.kwargs["output_length"]
        assert _experiment().data.sequence_length == {"text": declared}

    def test_the_vocabulary_is_declared_by_the_encoder_and_the_decoder_alike(self) -> None:
        pair = _modelConfig().modalities["text"]
        assert pair.encoder.kwargs["vocab_size"] == pair.decoder.kwargs["vocab_size"] == VOCAB

    def test_the_latent_width_is_set_in_one_place(self) -> None:
        config = _modelConfig()
        assert config.single_latent is not None
        mu, logvar = _model()(_batch())["latent_params"]["z_fused"]
        assert mu.shape == logvar.shape == (BATCH_SIZE, config.single_latent.dim)

    @pytest.mark.parametrize("length", [1, 3, 20])
    def test_a_text_of_another_length_is_encoded_but_reconstructed_at_the_declared_one(
        self, length: int
    ) -> None:
        """The encoder has no length limit; the decoder never resizes."""
        outputs = _model()(_batch(length=length))
        assert outputs["reconstructions"]["text"].shape == (BATCH_SIZE, LENGTH, VOCAB)

    def test_the_whole_model_takes_ids_as_well_as_one_hot_rows(self) -> None:
        """Ids spare the memory of a one-hot tensor at inference, and encode identically."""
        model = _model().eval()
        ids = _ids()
        with torch.no_grad():
            from_ids = model({"text": ids}, use_mean=True)["latent_params"]["z_fused"][0]
            from_rows = model({"text": functional.one_hot(ids, VOCAB).float()}, use_mean=True)[
                "latent_params"
            ]["z_fused"][0]
        assert torch.allclose(from_ids, from_rows, atol=1e-5)

    def test_padding_a_text_does_not_change_its_latent_vector(self) -> None:
        model = _model().eval()
        ids = _ids(length=5)
        padded = functional.pad(ids, (0, LENGTH - 5), value=PAD)
        with torch.no_grad():
            short = model({"text": ids}, use_mean=True)["latent_params"]["z_fused"][0]
            long = model({"text": padded}, use_mean=True)["latent_params"]["z_fused"][0]
        assert torch.allclose(short, long, atol=1e-5)

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
        assert set(model(model.selectEncoderInputs(batch))["reconstructions"]) == {"text"}


class TestWhatTheYamlLetsYouChange:
    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_both_poolings_build_and_run(self, pooling: str) -> None:
        reconstruction = _model(encoder_pooling=pooling)(_batch())["reconstructions"]["text"]
        assert reconstruction.shape == (BATCH_SIZE, LENGTH, VOCAB)

    def test_learned_positions_are_a_yaml_choice(self) -> None:
        model = _model(
            encoder_positional_encoding="learned",
            encoder_max_length=LENGTH,
            decoder_positional_encoding="learned",
        )
        assert model(_batch())["reconstructions"]["text"].shape == (BATCH_SIZE, LENGTH, VOCAB)

    def test_a_null_output_activation_gives_raw_scores(self) -> None:
        reconstruction = _model(decoder_output_activation=None)(_batch())["reconstructions"]["text"]
        assert not torch.allclose(
            reconstruction.sum(dim=-1), torch.ones(BATCH_SIZE, LENGTH), atol=1e-3
        )

    def test_a_longer_text_and_a_bigger_vocabulary_are_set_where_they_are_written(self) -> None:
        model = _model(encoder_vocab_size=50, decoder_vocab_size=50, decoder_output_length=12)
        batch = {"text": functional.one_hot(torch.randint(1, 50, (BATCH_SIZE, 12)), 50).float()}
        assert model(batch)["reconstructions"]["text"].shape == (BATCH_SIZE, 12, 50)

    def test_a_batch_of_another_vocabulary_is_refused_by_the_encoder_and_names_it(self) -> None:
        batch = {"text": functional.one_hot(torch.randint(1, 40, (BATCH_SIZE, LENGTH)), 40).float()}
        with pytest.raises(ValueError, match=f"vocab_size={VOCAB}"):
            _model()(batch)

    def test_a_decoder_of_another_vocabulary_is_refused_by_the_loss_with_both_shapes(
        self,
    ) -> None:
        """Encoder and decoder disagreeing is caught where the shapes are compared."""
        model = _model(decoder_vocab_size=VOCAB + 1)
        trainer = Trainer(model, modality_dropout_p=0.0)
        with pytest.raises(ValueError, match="shape"):
            trainer.computeLosses(_batch(), step=0)


class TestTheLossOfTheExperiment:
    """What `bce` with a sum on one-hot targets is, and what it is not."""

    def test_the_experiment_sets_a_bernoulli_loss_summed_per_text_with_weight_two(self) -> None:
        training = _experiment().training
        assert training.reconstruction_loss == "bce"
        assert training.reconstruction_reduction == "sum"
        assert training.reconstruction_weight == 2.0

    def test_sum_is_the_number_of_values_of_a_text_times_the_mean(self) -> None:
        """`sum` multiplies the element-wise mean by `length * vocab_size` (ADR 0029)."""
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
            LENGTH * VOCAB, rel=1e-4
        )

    def test_the_bernoulli_loss_is_never_below_the_categorical_one(self) -> None:
        """Pins the finding of ADR 0041: `bce` adds a term on the classes that are not the token.

        The cross-entropy of the token is `-log p[token]` and the Bernoulli loss of the one-hot
        row is that plus `-log(1 - p[v])` summed over the other classes, which is not negative.
        """
        probabilities = torch.softmax(torch.randn(6, LENGTH, VOCAB), dim=-1)
        target = functional.one_hot(torch.randint(0, VOCAB, (6, LENGTH)), VOCAB).float()
        bernoulli = functional.binary_cross_entropy(probabilities, target, reduction="sum")
        categorical = -(target * probabilities.log()).sum()
        assert bernoulli > categorical

    def test_the_exact_cross_entropy_trains_the_same_model_on_raw_scores(self) -> None:
        """The way to the exact likelihood needs no core change: a Python loss and no softmax."""
        torch.manual_seed(0)
        model = _model(decoder_output_activation=None)
        trainer = Trainer(
            model,
            optimizer_kwargs={"lr": 3e-3},
            reconstruction_loss_fn=computeTokenCrossEntropy,
            modality_dropout_p=0.0,
        )
        batch = _batch()
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(30):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        last = trainer.computeLosses(batch, step=30).reconstruction.item()
        assert last < first
        assert first == pytest.approx(LENGTH * torch.tensor(float(VOCAB)).log().item(), rel=0.5)


def _tokenAccuracy(model: GlobalVae, ids: torch.Tensor) -> float:
    """Share of the positions whose most likely token is the right one, from the mean latent."""
    model.eval()
    with torch.no_grad():
        batch = {"text": functional.one_hot(ids, CORPUS_VOCAB).float()}
        reconstruction = model(batch, use_mean=True)["reconstructions"]["text"]
    return float((reconstruction.argmax(dim=-1) == ids).float().mean().item())


def _batchesOf(ids: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """One-hot batches of 64 texts."""
    chunks = [ids[start : start + 64] for start in range(0, len(ids), 64)]
    return [{"text": functional.one_hot(chunk, VOCAB).float()} for chunk in chunks]


class TestAModelThatLearnsAGrammar:
    """The experiment's settings on a small synthetic language, not on a named corpus."""

    def test_the_experiment_beats_the_best_guess_that_ignores_the_text(
        self, tmp_path: Path
    ) -> None:
        assert (CORPUS_LENGTH, CORPUS_VOCAB, CORPUS_PAD) == (LENGTH, VOCAB, PAD)
        torch.manual_seed(0)
        train_ids, valid_ids = makeIds(1024, seed=1), makeIds(256, seed=2)
        # The best constant answer: the most frequent token at each position of the training set.
        constant = (valid_ids == torch.mode(train_ids, dim=0).values).float().mean().item()

        training = _experiment(
            "training.beta_schedules.z_fused.kwargs.warmup_steps=20",
            "training.optimizer.kwargs.lr=0.003",
            output_dir=tmp_path,
        ).training
        model = _model()
        trainer = buildTrainerFromConfig(model, training)
        trainer.fit(_batchesOf(train_ids), num_epochs=40, val_dataloader=_batchesOf(valid_ids))

        assert _tokenAccuracy(model, valid_ids) > constant + 0.2

    def test_the_exact_cross_entropy_learns_most_of_it_in_the_same_time(self) -> None:
        """The Python route of ADR 0041: raw scores, `computeTokenCrossEntropy`, a weight of 4."""
        torch.manual_seed(0)
        train_ids, valid_ids = makeIds(1024, seed=1), makeIds(256, seed=2)
        model = _model(decoder_output_activation=None)
        trainer = Trainer(
            model,
            optimizer_kwargs={"lr": 3e-3},
            reconstruction_loss_fn=computeTokenCrossEntropy,
            reconstruction_weights=4.0,
            modality_dropout_p=0.0,
        )
        trainer.fit(_batchesOf(train_ids), num_epochs=40)

        assert _tokenAccuracy(model, valid_ids) > 0.8


class TestEvaluation:
    """`scripts/evaluate.py` and its figures read a text model with no text code of their own."""

    def test_the_report_and_the_figures_run(self, tmp_path: Path) -> None:
        model = _model().eval()
        loader = [_batch(6)]
        results = evaluate(
            model,
            loader,
            reconstruction_loss_fn=functional.binary_cross_entropy,
            reconstruction_reduction="sum",
        )
        assert results.num_samples == 6
        assert set(results.reconstruction_metrics) == {"text"}
        assert torch.isfinite(torch.tensor(results.total_reconstruction_loss))
        paths = exportEvaluationFigures(model, loader, tmp_path)
        assert [path.name for path in paths] == [
            "reconstructions_text.png",
            "latent_z_fused.png",
            "kl_z_fused.png",
        ]
        assert all(path.exists() for path in paths)


class TestTheDataContract:
    def test_the_data_config_has_an_empty_pipeline_so_the_one_hot_rows_stay_intact(self) -> None:
        data = _experiment().data
        assert data.transforms == {"text": []}

    def test_the_configured_pipeline_is_the_identity(self) -> None:
        pipeline = buildTransformPipeline(_experiment().data)["text"]
        x = _batch()["text"]
        assert torch.equal(pipeline(x), x)
        assert torch.equal(pipeline.inverse(x), x)


def _runTrain(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/text_vae` for two short epochs."""
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{EXPERIMENT}",
        f"data.loader_factory={_DUMMY_TEXT_LOADER}",
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
