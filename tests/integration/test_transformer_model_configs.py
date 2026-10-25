"""The transformer and ViT model configs and their experiments (roadmap P3-5, spec §6).

`configs/model/signal_transformer_single_latent.yaml` and `image_vit_single_latent.yaml` are the
transformer counterparts of the CNN and residual pairs, `signal_image_transformer_cross_attention
.yaml` fuses a transformer and a ViT by `cross_attention`, and `signal_image_shared_transformer
.yaml` is the first concrete shared backbone (ADR 0027): one encoder and one decoder for both
modalities. `test_model_config_schema.py` already checks that every shipped file validates and
builds, and that a single-latent file equals what `GlobalVae.createSingleLatent` builds. This file
checks what is specific to these files: the wiring, the shapes a YAML list produces, any subset of
the inputs, the gradient, the loss scale the experiments set (and the arithmetic behind it), how
the loss settings behave for a shared decoder (a known limit, pinned here), and a run of
`scripts/train.py` per experiment, once on dummy batches and once on the paired loader of
`examples/04_paired_loader.py`.
"""

import importlib.util
import itertools
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from omegaconf import OmegaConf

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import ModelConfig, buildModelFromConfig, validateModelConfig
from global_vae.config.training import resolveReconstructionLoss, resolveReconstructionWeight
from global_vae.fusion.cross_attention import CrossAttentionFusion
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from global_vae.visualization.reconstruction_plot import collectCrossModalReconstructions

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_DUMMY_SIGNAL_LOADER = "tests.integration._train_script_fixtures:buildDummyDataloaders"
_DUMMY_IMAGE_LOADER = "tests.integration._train_script_fixtures:buildDummyImageDataloaders"
_DUMMY_PAIRED_LOADER = "tests.integration._train_script_fixtures:buildDummySignalImageDataloaders"
_PAIRED_LOADER = "examples.04_paired_loader:buildPairedDataloaders"
_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]

BATCH_SIZE = 4
SIGNAL_LENGTH = 256
IMAGE_SIZE = 64
SINGLE_MODALITY_FILES = ("signal_transformer_single_latent", "image_vit_single_latent")
TWO_MODALITY_FILES = ("signal_image_transformer_cross_attention", "signal_image_shared_transformer")

# Model file -> {modality: (encoder name, decoder name)} for the files that pair them per modality.
_PAIRS = {
    "signal_transformer_single_latent": {
        "signal": ("1d_transformer_encoder_v1", "1d_transformer_decoder_v1"),
    },
    "image_vit_single_latent": {
        "image": ("2d_vit_encoder_v1", "2d_vit_decoder_v1"),
    },
    "signal_image_transformer_cross_attention": {
        "signal": ("1d_transformer_encoder_v1", "1d_transformer_decoder_v1"),
        "image": ("2d_vit_encoder_v1", "2d_vit_decoder_v1"),
    },
}
# Experiment file -> (the model file it composes, the dummy loader that serves it).
_EXPERIMENTS = {
    "signal_transformer_vae": ("signal_transformer_single_latent", _DUMMY_SIGNAL_LOADER),
    "image_vit_vae": ("image_vit_single_latent", _DUMMY_IMAGE_LOADER),
    "signal_image_transformer_vae": (
        "signal_image_transformer_cross_attention",
        _DUMMY_PAIRED_LOADER,
    ),
    "signal_image_shared_transformer_vae": (
        "signal_image_shared_transformer",
        _DUMMY_PAIRED_LOADER,
    ),
}
# Experiment file -> another model file that fits its data contract (CNN or the other fusion).
_SWAPS = {
    "signal_transformer_vae": "signal_single_latent",
    "image_vit_vae": "image_single_latent",
    "signal_image_transformer_vae": "signal_image_shared_transformer",
    "signal_image_shared_transformer_vae": "signal_image_transformer_cross_attention",
}


def _modelConfig(name: str) -> ModelConfig:
    """Read a model file the way a YAML/Hydra config reaches the builders: through OmegaConf."""
    text = (_CONFIG_DIR / "model" / f"{name}.yaml").read_text()
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(text))
    )
    assert isinstance(config, ModelConfig)
    return config


def _model(name: str) -> GlobalVae:
    return buildModelFromConfig(_modelConfig(name))


def _experiment(name: str, *overrides: str) -> ExperimentConfig:
    return loadExperimentConfig(f"experiment/{name}", overrides=[*_DATA_OVERRIDES, *overrides])


def _batch(*modalities: str) -> dict[str, torch.Tensor]:
    """A random batch of the shipped data contract, restricted to `modalities`."""
    tensors = {
        "signal": torch.randn(BATCH_SIZE, SIGNAL_LENGTH),
        "image": torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE),
    }
    return {name: tensors[name] for name in modalities}


def _loadPairedLoaderExample() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "global_vae_example_04_for_transformer_config",
        _REPO_ROOT / "examples" / "04_paired_loader.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modalitiesOf(name: str) -> tuple[str, ...]:
    return ("signal", "image") if name in TWO_MODALITY_FILES else (name.split("_")[0],)


class TestEveryFileFollowsTheNewModalityChecklist:
    """The practical test of the registry design: no core file is needed for these backbones."""

    @pytest.mark.parametrize("name", [*SINGLE_MODALITY_FILES, *TWO_MODALITY_FILES])
    def test_the_file_validates_and_builds(self, name: str) -> None:
        config = _modelConfig(name)
        validateModelConfig(config)
        model = buildModelFromConfig(config)
        assert set(model.latent_spaces) == {"z_fused"}
        assert set(model.input_modalities) == set(_modalitiesOf(name))
        assert set(model.output_modalities) == set(_modalitiesOf(name))

    @pytest.mark.parametrize("name", list(_PAIRS))
    def test_it_names_the_transformer_pair_of_each_modality(self, name: str) -> None:
        config = _modelConfig(name)
        assert set(config.modalities) == set(_PAIRS[name])
        for modality, (encoder_name, decoder_name) in _PAIRS[name].items():
            assert config.modalities[modality].encoder.name == encoder_name
            assert config.modalities[modality].decoder.name == decoder_name


@pytest.mark.parametrize("name", SINGLE_MODALITY_FILES)
class TestSingleModalityFiles:
    def test_the_file_is_one_modality_in_one_unfused_latent_space(self, name: str) -> None:
        config = _modelConfig(name)
        assert config.latent_mode == "single"
        assert config.single_latent is not None
        assert config.single_latent.name == "z_fused"
        assert config.single_latent.fusion is None  # one encoder needs no fusion (spec §4)
        assert len(_model(name).fusions) == 0

    def test_a_yaml_value_gives_the_reconstruction_it_declares(self, name: str) -> None:
        (modality,) = _modalitiesOf(name)
        kwargs = _modelConfig(name).modalities[modality].decoder.kwargs
        declared = (
            (kwargs["output_length"],) if modality == "signal" else tuple(kwargs["output_shape"])
        )
        reconstruction = _model(name)(_batch(modality))["reconstructions"][modality]
        assert tuple(reconstruction.shape) == (BATCH_SIZE, *declared)

    def test_the_latent_width_is_set_in_one_place(self, name: str) -> None:
        (modality,) = _modalitiesOf(name)
        config = _modelConfig(name)
        assert config.single_latent is not None
        mu, logvar = _model(name)(_batch(modality))["latent_params"]["z_fused"]
        assert mu.shape == logvar.shape == (BATCH_SIZE, config.single_latent.dim)

    def test_every_parameter_receives_a_gradient(self, name: str) -> None:
        (modality,) = _modalitiesOf(name)
        model = _model(name)
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(_batch(modality), step=0)
        backward(losses.total)
        assert torch.isfinite(losses.total)
        assert [n for n, p in model.named_parameters() if p.grad is None] == []

    def test_the_loss_falls_over_a_few_steps(self, name: str) -> None:
        torch.manual_seed(0)
        (modality,) = _modalitiesOf(name)
        model = _model(name)
        trainer = Trainer(model, optimizer_kwargs={"lr": 3e-3}, modality_dropout_p=0.0)
        batch = {modality: torch.rand_like(_batch(modality)[modality])}
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(20):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        assert trainer.computeLosses(batch, step=20).reconstruction.item() < first

    def test_a_key_with_no_encoder_or_decoder_is_ignored(self, name: str) -> None:
        """The paired loader yields both keys: a one-modality model neither fails on the other."""
        (modality,) = _modalitiesOf(name)
        model = _model(name)
        batch = _batch("signal", "image")
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(batch, step=0)
        assert torch.isfinite(losses.total)
        assert set(model(model.selectEncoderInputs(batch))["reconstructions"]) == {modality}


class TestSignalTransformerFile:
    def test_a_series_of_another_length_is_encoded_but_reconstructed_at_the_declared_one(
        self,
    ) -> None:
        """Sinusoidal positions are length-agnostic; the decoder's output length is fixed."""
        model = _model("signal_transformer_single_latent")
        outputs = model({"signal": torch.randn(2, 100)})
        assert outputs["reconstructions"]["signal"].shape == (2, SIGNAL_LENGTH)

    def test_the_series_is_sixteen_patch_tokens_of_sixteen_values(self) -> None:
        kwargs = (
            _modelConfig("signal_transformer_single_latent").modalities["signal"].decoder.kwargs
        )
        assert SIGNAL_LENGTH // kwargs["patch_size"] == 16


class TestVitFile:
    def test_an_image_with_an_explicit_channel_axis_is_accepted(self) -> None:
        outputs = _model("image_vit_single_latent")(
            {"image": torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)}
        )
        assert outputs["reconstructions"]["image"].shape == (2, IMAGE_SIZE, IMAGE_SIZE)

    def test_an_image_of_another_size_is_encoded_but_reconstructed_at_the_declared_one(
        self,
    ) -> None:
        outputs = _model("image_vit_single_latent")({"image": torch.randn(2, 40, 56)})
        assert outputs["reconstructions"]["image"].shape == (2, IMAGE_SIZE, IMAGE_SIZE)

    def test_the_image_size_is_the_one_of_the_signal_plus_image_model(self) -> None:
        """One paired loader must serve `default.yaml` and these files: same image size."""
        sizes = {
            name: _modelConfig(name).modalities["image"].decoder.kwargs["output_shape"]
            for name in ("image_vit_single_latent", "signal_image_transformer_cross_attention")
        }
        assert list(sizes.values()) == [[IMAGE_SIZE, IMAGE_SIZE]] * 2
        assert _modelConfig("default").modalities["image"].decoder.kwargs["output_shape"] == [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ]


class TestCrossAttentionFile:
    name = "signal_image_transformer_cross_attention"

    def test_the_fusion_is_cross_attention_over_the_two_modalities(self) -> None:
        config = _modelConfig(self.name)
        assert config.single_latent is not None and config.single_latent.fusion is not None
        fusion = config.single_latent.fusion
        assert fusion.strategy == "cross_attention"
        assert fusion.kwargs["latent_dim"] == config.single_latent.dim  # not filled in for you
        assert fusion.kwargs["known_modalities"] == ["signal", "image"]
        built = _model(self.name).fusions["z_fused"]
        assert isinstance(built, CrossAttentionFusion)

    def test_each_modality_has_its_own_transformer_pair(self) -> None:
        model = _model(self.name)
        assert type(model.encoders["signal"]).__name__ == "OneDTransformerEncoder"
        assert type(model.encoders["image"]).__name__ == "TwoDVitEncoder"
        assert set(model.decoders) == {"signal", "image"}

    @pytest.mark.parametrize(
        "present",
        [subset for size in (1, 2) for subset in itertools.combinations(("signal", "image"), size)],
    )
    def test_any_subset_of_the_inputs_gives_both_reconstructions(
        self, present: tuple[str, ...]
    ) -> None:
        outputs = _model(self.name)(_batch(*present))
        assert outputs["reconstructions"]["signal"].shape == (BATCH_SIZE, SIGNAL_LENGTH)
        assert outputs["reconstructions"]["image"].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)

    def test_the_cross_modal_reconstructions_are_collected(self) -> None:
        model = _model(self.name).eval()
        batches = [{**_batch("signal", "image")}]
        pairs = collectCrossModalReconstructions(model, batches)
        assert set(pairs) == {
            frozenset({"signal"}),
            frozenset({"image"}),
            frozenset({"signal", "image"}),
        }
        original, reconstruction = pairs[frozenset({"signal"})]["image"]
        assert original.shape == reconstruction.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)

    def test_every_parameter_receives_a_gradient(self) -> None:
        model = _model(self.name)
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(
            _batch("signal", "image"), step=0
        )
        backward(losses.total)
        assert [n for n, p in model.named_parameters() if p.grad is None] == []

    def test_training_with_modality_dropout_stays_finite(self) -> None:
        torch.manual_seed(0)
        model = _model(self.name)
        trainer = Trainer(model, optimizer_kwargs={"lr": 1e-3}, modality_dropout_p=0.5)
        batch = _batch("signal", "image")
        for step in range(6):
            trainer.optimizer.zero_grad()
            losses = trainer.computeLosses(batch, step=step)
            assert torch.isfinite(losses.total)
            backward(losses.total)
            trainer.optimizer.step()

    def test_the_fusion_attends_over_one_token_per_modality_not_over_patches(self) -> None:
        """The scale of `cross_attention`: its input is one `(mu, logvar)` per encoder.

        A bigger backbone changes the encoders and leaves the fusion untouched: its parameters
        are the same for a 3-layer and a 6-layer transformer. Attention over the patch tokens of
        two modalities is what the shared transformer does.
        """
        shallow = _modelConfig(self.name)
        deep = _modelConfig(self.name)
        deep.modalities["signal"].encoder.kwargs["num_layers"] = 6
        deep.modalities["image"].encoder.kwargs["num_layers"] = 6

        def fusionParameters(config: ModelConfig) -> int:
            fusion = buildModelFromConfig(config).fusions["z_fused"]
            return sum(p.numel() for p in fusion.parameters())

        assert fusionParameters(shallow) == fusionParameters(deep)


class TestSharedTransformerFile:
    name = "signal_image_shared_transformer"

    def test_one_encoder_and_one_decoder_serve_both_modalities(self) -> None:
        model = _model(self.name)
        assert set(model.encoders) == {"backbone"}
        assert set(model.decoders) == {"backbone_out"}
        assert model.input_modalities == ("signal", "image")
        assert model.output_modalities == ("signal", "image")

    def test_there_is_no_fusion_strategy(self) -> None:
        config = _modelConfig(self.name)
        assert config.single_latent is not None and config.single_latent.fusion is None
        assert len(_model(self.name).fusions) == 0

    def test_the_file_declares_its_instances_under_encoders_and_decoders(self) -> None:
        config = _modelConfig(self.name)
        assert not config.modalities
        assert config.encoders["backbone"].name == "shared_transformer_encoder_v1"
        assert config.decoders["backbone_out"].name == "shared_transformer_decoder_v1"

    @pytest.mark.parametrize(
        "present",
        [subset for size in (1, 2) for subset in itertools.combinations(("signal", "image"), size)],
    )
    def test_any_subset_of_the_inputs_gives_both_reconstructions(
        self, present: tuple[str, ...]
    ) -> None:
        """A missing modality is a shorter token stream (spec §5): no mask, no placeholder."""
        outputs = _model(self.name)(_batch(*present))
        assert outputs["reconstructions"]["signal"].shape == (BATCH_SIZE, SIGNAL_LENGTH)
        assert outputs["reconstructions"]["image"].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        mu, logvar = outputs["latent_params"]["z_fused"]
        assert mu.shape == logvar.shape == (BATCH_SIZE, 32)

    def test_a_decoder_only_target_key_is_not_encoded(self) -> None:
        model = _model(self.name)
        batch = {**_batch("signal", "image"), "image_out": torch.randn(BATCH_SIZE, 8, 8)}
        assert set(model.selectEncoderInputs(batch)) == {"signal", "image"}

    def test_the_two_modalities_interact_inside_the_encoder(self) -> None:
        model = _model(self.name).eval()
        batch = _batch("signal", "image")
        other_image = {**batch, "image": torch.randn_like(batch["image"])}
        mu = model(batch, use_mean=True)["latent_params"]["z_fused"][0]
        other = model(other_image, use_mean=True)["latent_params"]["z_fused"][0]
        assert not torch.allclose(mu, other, atol=1e-4)

    def test_every_parameter_receives_a_gradient(self) -> None:
        model = _model(self.name)
        losses = Trainer(model, modality_dropout_p=0.0).computeLosses(
            _batch("signal", "image"), step=0
        )
        backward(losses.total)
        assert torch.isfinite(losses.total)
        assert [n for n, p in model.named_parameters() if p.grad is None] == []

    def test_the_loss_falls_over_a_few_steps(self) -> None:
        torch.manual_seed(0)
        model = _model(self.name)
        trainer = Trainer(model, optimizer_kwargs={"lr": 3e-3}, modality_dropout_p=0.0)
        batch = {
            "signal": torch.sin(torch.linspace(0, 12.0, SIGNAL_LENGTH)).repeat(BATCH_SIZE, 1),
            "image": torch.rand(1, IMAGE_SIZE, IMAGE_SIZE).repeat(BATCH_SIZE, 1, 1),
        }
        first = trainer.computeLosses(batch, step=0).reconstruction.item()
        for step in range(20):
            trainer.optimizer.zero_grad()
            backward(trainer.computeLosses(batch, step=step).total)
            trainer.optimizer.step()
        assert trainer.computeLosses(batch, step=20).reconstruction.item() < first

    def test_training_with_modality_dropout_stays_finite(self) -> None:
        torch.manual_seed(1)
        model = _model(self.name)
        trainer = Trainer(model, optimizer_kwargs={"lr": 1e-3}, modality_dropout_p=0.5)
        batch = _batch("signal", "image")
        for step in range(6):
            trainer.optimizer.zero_grad()
            losses = trainer.computeLosses(batch, step=step)
            assert torch.isfinite(losses.total)
            backward(losses.total)
            trainer.optimizer.step()

    def test_the_cross_modal_reconstructions_are_collected(self) -> None:
        model = _model(self.name).eval()
        pairs = collectCrossModalReconstructions(model, [_batch("signal", "image")])
        assert set(pairs) == {
            frozenset({"signal"}),
            frozenset({"image"}),
            frozenset({"signal", "image"}),
        }
        original, reconstruction = pairs[frozenset({"image"})]["signal"]
        assert original.shape == reconstruction.shape == (BATCH_SIZE, SIGNAL_LENGTH)

    def test_it_is_a_smaller_model_than_the_two_separate_backbones(self) -> None:
        """One transformer for both modalities, against one per modality plus a fusion."""
        shared = sum(p.numel() for p in _model(self.name).parameters())
        separate = sum(
            p.numel() for p in _model("signal_image_transformer_cross_attention").parameters()
        )
        assert shared < separate


class TestLossSettingsOfASharedDecoder:
    """What a model with a shared decoder does with `training.reconstruction_*` (roadmap P3-5).

    The config layer keys a per-decoder mapping by decoder instance name (`backbone_out`), and the
    loss reads the reconstruction keys, which are the modalities (`signal`, `image`). A single
    value works and covers every modality, which is what the shipped experiment uses; a mapping
    does not. The `xfail` tests state the behaviour a mapping should have: when the config layer
    or the loss is changed to agree, they pass and `strict` asks for the marker to be removed.
    """

    name = "signal_image_shared_transformer"

    def _reconstruction(self, model: GlobalVae, **trainer_kwargs: object) -> float:
        """The reconstruction term for one batch, with the same sampling noise on every call."""
        trainer = Trainer(model, modality_dropout_p=0.0, **trainer_kwargs)  # type: ignore[arg-type]
        torch.manual_seed(0)
        return trainer.computeLosses(_batch("signal", "image"), step=0).reconstruction.item()

    def test_a_single_weight_covers_both_modalities(self) -> None:
        model = _model(self.name)
        full = self._reconstruction(model)
        half = self._reconstruction(model, reconstruction_weights=0.5)
        assert half == pytest.approx(full / 2, rel=1e-4)

    def test_the_summed_loss_is_the_sum_of_each_modality_scaled_by_its_size(self) -> None:
        model = _model(self.name)
        batch = _batch("signal", "image")
        torch.manual_seed(0)
        outputs = model(batch)["reconstructions"]
        expected = sum(
            torch.nn.functional.mse_loss(outputs[name], batch[name]).item() * batch[name][0].numel()
            for name in batch
        )
        trainer = Trainer(model, modality_dropout_p=0.0, reconstruction_reduction="sum")
        torch.manual_seed(0)
        assert trainer.computeLosses(batch, step=0).reconstruction.item() == pytest.approx(
            expected, rel=1e-4
        )

    def test_a_mapping_by_modality_is_refused_with_the_decoder_names(self) -> None:
        model = _model(self.name)
        with pytest.raises(ValueError, match=r"\['signal'\].*decoders: \['backbone_out'\]"):
            resolveReconstructionWeight({"signal": 0.5}, model)

    def test_a_mapping_by_decoder_name_is_accepted(self) -> None:
        assert resolveReconstructionWeight({"backbone_out": 0.5}, _model(self.name)) == {
            "backbone_out": 0.5
        }

    @pytest.mark.xfail(
        strict=True,
        reason="roadmap P3-5: a weight keyed by shared decoder name is ignored by the loss",
    )
    def test_a_weight_mapping_by_decoder_name_is_applied(self) -> None:
        model = _model(self.name)
        weights = resolveReconstructionWeight({"backbone_out": 0.5}, model)
        assert self._reconstruction(model, reconstruction_weights=weights) == pytest.approx(
            self._reconstruction(model) / 2, rel=1e-4
        )

    @pytest.mark.xfail(
        strict=True,
        raises=KeyError,
        reason="roadmap P3-5: a loss function keyed by shared decoder name has no entry for signal",
    )
    def test_a_loss_function_mapping_by_decoder_name_trains(self) -> None:
        model = _model(self.name)
        loss_fn = resolveReconstructionLoss({"backbone_out": "l1"}, model)
        assert self._reconstruction(model, reconstruction_loss_fn=loss_fn) > 0


@pytest.mark.parametrize(("experiment", "expected"), list(_EXPERIMENTS.items()))
class TestExperiments:
    def test_the_experiment_composes_its_model_data_and_training(
        self, experiment: str, expected: tuple[str, str]
    ) -> None:
        model_file, _ = expected
        config = _experiment(experiment)
        assert config.model == _modelConfig(model_file)
        assert config.output_dir == f"outputs/{experiment}"
        assert set(config.training.beta_schedules) == {"z_fused"}

    def test_the_model_group_can_be_swapped_from_the_command_line(
        self, experiment: str, expected: tuple[str, str]
    ) -> None:
        other = _SWAPS[experiment]
        assert _experiment(experiment, f"model={other}").model == _modelConfig(other)

    def test_the_latent_width_follows_one_override(
        self, experiment: str, expected: tuple[str, str]
    ) -> None:
        model_file, _ = expected
        overrides = ["model.single_latent.dim=8"]
        if model_file == "signal_image_transformer_cross_attention":
            overrides.append("model.single_latent.fusion.kwargs.latent_dim=8")
        config = _experiment(experiment, *overrides)
        model = buildModelFromConfig(config.model)
        mu, _ = model(_batch(*_modalitiesOf(model_file)))["latent_params"]["z_fused"]
        assert mu.shape == (BATCH_SIZE, 8)

    def test_the_experiment_trains_on_the_summed_loss_with_the_elbo_weight(
        self, experiment: str, expected: tuple[str, str]
    ) -> None:
        training = _experiment(experiment).training
        assert training.reconstruction_reduction == "sum"
        assert training.reconstruction_weight == 0.5


@pytest.mark.parametrize(
    ("experiment", "expected"),
    [(name, _EXPERIMENTS[name]) for name in ("signal_transformer_vae", "image_vit_vae")],
)
class TestLossScaleOfTheSingleModalityExperiments:
    def test_the_summed_loss_weighs_the_reconstruction_by_half_the_values_it_reconstructs(
        self, experiment: str, expected: tuple[str, str]
    ) -> None:
        """The arithmetic the experiment files quote: `sum` at 0.5 is `N / 2` times `mean`."""
        model_file, _ = expected
        (modality,) = _modalitiesOf(model_file)
        training = _experiment(experiment).training
        model = _model(model_file)
        batch = _batch(modality)

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
        assert summed / averaged == pytest.approx(batch[modality][0].numel() / 2, rel=1e-4)


def _runTrain(
    tmp_path: Path, experiment: str, loader_factory: str, train_path: str, *extra: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `scripts/train.py --config-name experiment/<experiment>` for two short epochs."""
    output_dir = tmp_path / "run"
    overrides = [
        "--config-name",
        f"experiment/{experiment}",
        f"data.loader_factory={loader_factory}",
        f"data.train_path={train_path}",
        f"output_dir={output_dir}",
        "training.num_epochs=2",
        "training.beta_schedules.z_fused.kwargs.warmup_steps=5",
        f"hydra.run.dir={tmp_path / 'hydra_run'}",
        *extra,
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


# Every test below runs `scripts/train.py` in a subprocess.
@pytest.mark.slow
@pytest.mark.parametrize(("experiment", "expected"), list(_EXPERIMENTS.items()))
class TestExperimentsTrainThroughTheScript:
    def test_trains_on_dummy_batches_and_writes_a_checkpoint_the_model_loads(
        self, tmp_path: Path, experiment: str, expected: tuple[str, str]
    ) -> None:
        _, loader = expected
        completed, output_dir = _runTrain(tmp_path, experiment, loader, "unused")
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in (completed.stdout + completed.stderr)
        assert (output_dir / "metrics.csv").exists()
        model = buildModelFromConfig(_experiment(experiment).model)
        metadata = loadCheckpoint(output_dir / "checkpoints" / "best.pt", model=model)
        assert metadata.global_step > 0

    def test_trains_on_the_paired_loader(
        self, tmp_path: Path, experiment: str, expected: tuple[str, str]
    ) -> None:
        _loadPairedLoaderExample().writeSyntheticPairedDataset(
            tmp_path / "data", num_pairs=24, num_signal_only=2, seed=3
        )
        completed, output_dir = _runTrain(
            tmp_path, experiment, _PAIRED_LOADER, str(tmp_path / "data"), "data.batch_size=8"
        )
        assert completed.returncode == 0, completed.stderr
        assert "Training complete" in (completed.stdout + completed.stderr)
        assert (output_dir / "checkpoints" / "best.pt").is_file()
