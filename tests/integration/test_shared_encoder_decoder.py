"""Shared encoders and shared decoders end to end (spec §2.1, roadmap P1-5, ADR 0027).

`test_configuration_matrix.py` (stage 3) checks that the four `E1-*` rows build and train. This
file checks what the shared modules change around them, following the "done when" of roadmap P1-5:

* the rows build from YAML text, with the widths the config layer derives;
* a batch key names an input modality, and `GlobalVae` hands a shared encoder exactly the
  modalities that are present, in the order it declares them;
* modality dropout hides modalities inside a shared encoder's stream, and a hidden modality gets
  no gradient while every modality is still reconstructed;
* the cross-modal reports, the evaluation and the figure export work with a shared encoder;
* the guards: a modality owned twice, a bad declaration and a shared decoder that returns the
  wrong modalities are all refused;
* checkpoints name the shared modules by instance name and round-trip.

The modules are the dummies of `_shared_modules.py`. The overlay plots draw 1D series only
(roadmap section 2.3), so the figure tests use two signal modalities.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import pytest
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
import global_vae.fusion  # noqa: F401  (registers the built-in fusion strategies)
import global_vae.heads  # noqa: F401  (registers the built-in latent heads)
from global_vae.config.model import (
    DecoderConfig,
    DecoderConsumptionConfig,
    EncoderConfig,
    FusionConfig,
    LatentConfig,
    LatentSpaceConfig,
    ModelConfig,
    SingleLatentConfig,
    buildModelFromConfig,
)
from global_vae.evaluation.cross_modal import (
    computeCrossModalReconstructionMetrics,
    exportCrossModalFigures,
)
from global_vae.evaluation.evaluate import evaluate
from global_vae.evaluation.visual_export import exportEvaluationFigures
from global_vae.losses.reconstruction import computeTotalReconstructionLoss
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint, saveCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from global_vae.visualization.latent_plot import collectLatentParams
from global_vae.visualization.reconstruction_plot import (
    collectCrossModalReconstructions,
    resolveDefaultInputSubsets,
)
from tests.integration._shared_modules import (
    MISDECLARED_DECODER,
    MISDECLARED_ENCODER,
    SHARED_DECODER,
    SHARED_ENCODER,
    DummySharedDecoder,
    sharedDecoderConfig,
    sharedEncoderConfig,
)

BATCH_SIZE = 8
SIGNAL_LENGTH = 32
IMAGE_SHAPE = (32, 32)
LATENT_DIM = 8

SHAPES: dict[str, tuple[int, ...]] = {"signal": (SIGNAL_LENGTH,), "image": IMAGE_SHAPE}
PAIR_SHAPES: dict[str, tuple[int, ...]] = {
    "signal_a": (SIGNAL_LENGTH,),
    "signal_b": (SIGNAL_LENGTH,),
}

Batch = dict[str, torch.Tensor]


def _oneModalityDecoders() -> dict[str, DecoderConfig]:
    """A signal decoder and an image decoder, without `latent_dim` (the config layer fills it)."""
    common: dict[str, Any] = {"upsample_modes": "conv_transpose", "hidden_channels": (16, 8)}
    return {
        "signal": DecoderConfig(
            name="1d_cnn_decoder_v1", kwargs={**common, "output_length": SIGNAL_LENGTH}
        ),
        "image": DecoderConfig(
            name="2d_cnn_decoder_v1",
            kwargs={**common, "output_shape": IMAGE_SHAPE, "seed_shape": (8, 8)},
        ),
    }


def _buildModel(
    decoders: str = "D1", shapes: Mapping[str, tuple[int, ...]] | None = None
) -> GlobalVae:
    """An `E1-L1-D*` model over the dummy shared encoder, one latent space, no fusion.

    Args:
        decoders: `"D1"` for the dummy shared decoder over every modality, `"DN"` for a real 1D
            and a real 2D decoder (only with the default signal and image `shapes`).
        shapes: Modality name -> shape. Defaults to `SHAPES`.

    Returns:
        The model.
    """
    shapes = shapes or SHAPES
    decoder_configs = (
        {"shared_out": sharedDecoderConfig(shapes)} if decoders == "D1" else _oneModalityDecoders()
    )
    return buildModelFromConfig(
        ModelConfig(
            latent_mode="single",
            encoders={"backbone": sharedEncoderConfig(shapes)},
            decoders=decoder_configs,
            single_latent=SingleLatentConfig(dim=LATENT_DIM),
        )
    )


def _batches(
    num_batches: int = 4, shapes: Mapping[str, tuple[int, ...]] | None = None, seed: int = 0
) -> list[Batch]:
    """A small fixed dataset whose modalities share a few latent factors.

    Low-rank structure keeps the reconstruction loss learnable in a handful of steps.
    """
    shapes = shapes or SHAPES
    generator = torch.Generator().manual_seed(seed)
    bases = {name: torch.randn(3, *shape, generator=generator) for name, shape in shapes.items()}
    batches: list[Batch] = []
    for _ in range(num_batches):
        factors = torch.randn(BATCH_SIZE, 3, generator=generator)
        batches.append(
            {name: torch.einsum("bk,k...->b...", factors, basis) for name, basis in bases.items()}
        )
    return batches


def _projectionWidth(model: GlobalVae, modality: str) -> int:
    """Input width of the dummy shared decoder's projection for one modality."""
    decoder = model.decoders["shared_out"]
    assert isinstance(decoder, DummySharedDecoder)
    projection = decoder.projections[modality]
    assert isinstance(projection, torch.nn.Linear)
    return projection.in_features


def _recordEncoderCalls(model: GlobalVae) -> list[tuple[str, ...]]:
    """Make the shared encoder log the modality names of every call it receives."""
    calls: list[tuple[str, ...]] = []
    model.encoders["backbone"].register_forward_hook(
        lambda _module, args, _output: calls.append(tuple(args[0]))
    )
    return calls


def _lossOf(model: GlobalVae, batch: Batch, inputs: Batch | None = None) -> torch.Tensor:
    """Total loss of `batch` (reconstruction plus regularization), encoding only `inputs`."""
    outputs = model(inputs if inputs is not None else batch)
    reconstruction = computeTotalReconstructionLoss(outputs["reconstructions"], batch)
    return reconstruction + model.computeRegularizationLoss(outputs["latent_params"])


@pytest.fixture(params=["D1", "DN"])
def model(request: pytest.FixtureRequest) -> GlobalVae:
    """A freshly built, seeded `E1-L1-D1` or `E1-L1-DN` model over a signal and an image."""
    torch.manual_seed(0)
    return _buildModel(decoders=request.param)


def _modelConfigFromText(text: str) -> ModelConfig:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(text))
    )
    assert isinstance(config, ModelConfig)
    return config


class TestBuiltFromYaml:
    """The `E1` rows are written with the existing `encoders`, `decoders` and `latent` sections."""

    _TEXT = f"""
latent_mode: several
encoders:
  backbone:
    name: {SHARED_ENCODER}
    kwargs:
      input_dims: {{signal: {SIGNAL_LENGTH}, image: 1024}}
decoders:
  shared_out:
    name: {SHARED_DECODER}
    kwargs:
      output_shapes: {{signal: [{SIGNAL_LENGTH}], image: [32, 32]}}
latent:
  spaces:
    z_a: {{dim: 6, fed_by: [backbone]}}
    z_b:
      dim: 4
      fed_by:
        backbone: {{head: linear}}
  decoders_consume:
    shared_out: {{spaces: [z_a, z_b], assembler: concat}}
"""

    def test_an_e1_ln_d1_model_builds_with_the_widths_the_config_layer_derives(self) -> None:
        """No `latent_dim` is written: the encoder takes `z_a`'s width, the decoder the sum."""
        model = buildModelFromConfig(_modelConfigFromText(self._TEXT))
        assert set(model.encoders) == {"backbone"}
        assert set(model.decoders) == {"shared_out"}
        assert model.encoders["backbone"].latent_dim == 6
        assert model.assemblers["shared_out"].output_dim == 10
        assert _projectionWidth(model, "image") == 10
        assert model.input_modalities == ("signal", "image")
        assert model.output_modalities == ("signal", "image")

    def test_the_yaml_model_runs_and_reconstructs_both_modalities(self) -> None:
        """A forward pass over the batch gives one reconstruction per modality."""
        model = buildModelFromConfig(_modelConfigFromText(self._TEXT))
        outputs = model(_batches(1)[0])
        assert {k: v.shape for k, v in outputs["reconstructions"].items()} == {
            "signal": (BATCH_SIZE, SIGNAL_LENGTH),
            "image": (BATCH_SIZE, *IMAGE_SHAPE),
        }
        assert set(outputs["latent_params"]) == {"z_a", "z_b"}

    def test_the_single_latent_form_builds_too(self) -> None:
        """`latent_mode: single` takes the same shared modules."""
        text = f"""
latent_mode: single
encoders:
  backbone:
    name: {SHARED_ENCODER}
    kwargs:
      input_dims: {{signal: {SIGNAL_LENGTH}, image: 1024}}
decoders:
  shared_out:
    name: {SHARED_DECODER}
    kwargs:
      output_shapes: {{signal: [{SIGNAL_LENGTH}], image: [32, 32]}}
single_latent: {{dim: {LATENT_DIM}}}
"""
        model = buildModelFromConfig(_modelConfigFromText(text))
        assert model.encoders["backbone"].latent_dim == LATENT_DIM
        assert set(model(_batches(1)[0])["reconstructions"]) == {"signal", "image"}


class TestModalityNames:
    """A batch key is a modality, and a modality has exactly one owner."""

    def test_a_model_without_shared_modules_names_its_modalities_after_its_instances(
        self,
    ) -> None:
        """The two sets equal `tuple(encoders)` and `tuple(decoders)`, as before ADR 0027."""
        classic = GlobalVae.createSingleLatent(
            modality_configs={
                "signal": {"encoder": "1d_cnn_encoder_v1", "decoder": "1d_cnn_decoder_v1"},
                "image": {"encoder": "2d_cnn_encoder_v1", "decoder": "2d_cnn_decoder_v1"},
            },
            latent_dim=LATENT_DIM,
            fusion_strategy="poe",
            encoder_kwargs={
                "signal": {"latent_dim": LATENT_DIM, "hidden_channels": (8, 16)},
                "image": {"latent_dim": LATENT_DIM, "hidden_channels": (8, 16)},
            },
            decoder_kwargs={
                "signal": {
                    "latent_dim": LATENT_DIM,
                    "output_length": SIGNAL_LENGTH,
                    "upsample_modes": "conv_transpose",
                    "hidden_channels": (16, 8),
                },
                "image": {
                    "latent_dim": LATENT_DIM,
                    "output_shape": IMAGE_SHAPE,
                    "seed_shape": (8, 8),
                    "upsample_modes": "conv_transpose",
                    "hidden_channels": (16, 8),
                },
            },
        )
        assert classic.input_modalities == tuple(classic.encoders)
        assert classic.output_modalities == tuple(classic.decoders)

    def test_select_encoder_inputs_keeps_the_modalities_of_a_shared_encoder(
        self, model: GlobalVae
    ) -> None:
        """A decoder-only target is dropped, the two modalities the encoder reads are kept."""
        batch = {**_batches(1)[0], "extra_target": torch.randn(BATCH_SIZE, 3)}
        assert set(model.selectEncoderInputs(batch)) == {"signal", "image"}

    def test_select_encoder_inputs_refuses_a_batch_with_no_input_modality(
        self, model: GlobalVae
    ) -> None:
        """The instance name of the shared encoder is not a modality, so it does not count."""
        with pytest.raises(ValueError, match="input modalities"):
            model.selectEncoderInputs({"backbone": torch.randn(BATCH_SIZE, 3)})

    def test_forward_refuses_the_instance_name_of_a_shared_encoder(self, model: GlobalVae) -> None:
        """The batch key is the modality (`signal`), not the encoder (`backbone`)."""
        with pytest.raises(KeyError, match="backbone"):
            model({"backbone": torch.randn(BATCH_SIZE, SIGNAL_LENGTH)})

    def test_the_shared_encoder_receives_the_present_modalities_in_declared_order(self) -> None:
        """Absent modalities are omitted, and the dict follows `input_modalities`."""
        torch.manual_seed(0)
        model = _buildModel()
        calls = _recordEncoderCalls(model)
        batch = _batches(1)[0]
        model({"image": batch["image"], "signal": batch["signal"]})
        model({"image": batch["image"]})
        model({"signal": batch["signal"]})
        assert calls == [("signal", "image"), ("image",), ("signal",)]

    def test_a_shared_decoder_is_flattened_into_the_reconstructions_by_modality(self) -> None:
        """The reconstructions are keyed by modality, not by the decoder's instance name."""
        torch.manual_seed(0)
        model = _buildModel(decoders="D1")
        reconstructions = model(_batches(1)[0])["reconstructions"]
        assert set(reconstructions) == {"signal", "image"}
        assert "shared_out" not in reconstructions

    def test_a_one_modality_encoder_and_a_shared_one_can_share_a_latent_space(self) -> None:
        """Two encoders, one of them shared, are fused like any two; absent ones do not run."""
        torch.manual_seed(0)
        model = buildModelFromConfig(
            ModelConfig(
                latent_mode="single",
                encoders={
                    "audio": EncoderConfig(
                        name="1d_cnn_encoder_v1", kwargs={"hidden_channels": (8, 16)}
                    ),
                    "backbone": sharedEncoderConfig(SHAPES),
                },
                decoders={"shared_out": sharedDecoderConfig(SHAPES)},
                single_latent=SingleLatentConfig(
                    dim=LATENT_DIM, fusion=FusionConfig(strategy="poe")
                ),
            )
        )
        assert model.input_modalities == ("audio", "signal", "image")
        calls = _recordEncoderCalls(model)
        batch = _batches(1)[0]
        audio = torch.randn(BATCH_SIZE, SIGNAL_LENGTH)

        model({"audio": audio})
        assert calls == []
        model({"audio": audio, "signal": batch["signal"]})
        assert calls == [("signal",)]
        outputs = model({"audio": audio, "signal": batch["signal"], "image": batch["image"]})
        assert calls[-1] == ("signal", "image")
        assert set(outputs["reconstructions"]) == {"signal", "image"}
        assert set(model.fusions) == {"z_fused"}


class TestConstructionGuards:
    """The model refuses an ambiguous or malformed declaration at construction."""

    @staticmethod
    def _config(
        encoders: dict[str, EncoderConfig], decoders: dict[str, DecoderConfig]
    ) -> ModelConfig:
        return ModelConfig(
            latent_mode="single",
            encoders=encoders,
            decoders=decoders,
            single_latent=SingleLatentConfig(
                dim=LATENT_DIM,
                fusion=FusionConfig(strategy="poe") if len(encoders) > 1 else None,
            ),
        )

    def test_two_encoders_cannot_ingest_the_same_modality(self) -> None:
        """A one-modality encoder named `signal` and a shared one that reads `signal` clash."""
        config = self._config(
            {
                "signal": EncoderConfig(
                    name="1d_cnn_encoder_v1", kwargs={"hidden_channels": (8, 16)}
                ),
                "backbone": sharedEncoderConfig(SHAPES),
            },
            {"shared_out": sharedDecoderConfig(SHAPES)},
        )
        with pytest.raises(ValueError, match="Modality 'signal' is ingested by both encoder"):
            buildModelFromConfig(config)

    def test_two_decoders_cannot_reconstruct_the_same_modality(self) -> None:
        """A decoder named `signal` and a shared decoder that outputs `signal` clash."""
        decoders = {
            "signal": _oneModalityDecoders()["signal"],
            "shared_out": sharedDecoderConfig(SHAPES),
        }
        config = ModelConfig(
            latent_mode="single",
            encoders={"backbone": sharedEncoderConfig(SHAPES)},
            decoders=decoders,
            single_latent=SingleLatentConfig(dim=LATENT_DIM),
        )
        with pytest.raises(ValueError, match="Modality 'signal' is reconstructed by both decoder"):
            buildModelFromConfig(config)

    @pytest.mark.parametrize(
        "declared",
        [(), "signal", ("signal", "signal"), ("signal", ""), ("signal", 3)],
        ids=["empty", "bare-string", "repeated", "empty-name", "not-a-string"],
    )
    def test_a_shared_encoder_must_declare_its_modalities_properly(self, declared: Any) -> None:
        """No modality, a bare string, a repeated name or a non-string name is refused."""
        config = self._config(
            {"backbone": EncoderConfig(name=MISDECLARED_ENCODER, kwargs={"declared": declared})},
            {"shared_out": sharedDecoderConfig(SHAPES)},
        )
        with pytest.raises(ValueError, match="Shared encoder 'backbone' must declare"):
            buildModelFromConfig(config)

    @pytest.mark.parametrize(
        "declared",
        [(), "signal", ("signal", "signal"), ("signal", ""), ("signal", 3)],
        ids=["empty", "bare-string", "repeated", "empty-name", "not-a-string"],
    )
    def test_a_shared_decoder_must_declare_its_modalities_properly(self, declared: Any) -> None:
        """The same rules hold for `output_modalities`."""
        config = self._config(
            {"backbone": sharedEncoderConfig(SHAPES)},
            {"shared_out": DecoderConfig(name=MISDECLARED_DECODER, kwargs={"declared": declared})},
        )
        with pytest.raises(ValueError, match="Shared decoder 'shared_out' must declare"):
            buildModelFromConfig(config)

    def test_a_shared_decoder_returning_other_modalities_than_declared_is_refused(self) -> None:
        """Exactly the declared set must come back: a missing key fails the forward pass."""
        config = self._config(
            {"backbone": sharedEncoderConfig(SHAPES)},
            {
                "shared_out": DecoderConfig(
                    name=MISDECLARED_DECODER,
                    kwargs={"declared": ("signal", "image"), "returned": ("signal",)},
                )
            },
        )
        model = buildModelFromConfig(config)
        with pytest.raises(ValueError, match="Shared decoder 'shared_out' returned"):
            model(_batches(1)[0])

    def test_a_shared_decoder_in_a_multi_latent_graph_gets_its_assembled_width(self) -> None:
        """The decoder's `latent_dim` is the assembler's output, filled in like any decoder's."""
        config = ModelConfig(
            latent_mode="several",
            encoders={"backbone": sharedEncoderConfig(SHAPES)},
            decoders={"shared_out": sharedDecoderConfig(SHAPES)},
            latent=LatentConfig(
                spaces={
                    "z_a": LatentSpaceConfig(dim=5, fed_by=["backbone"]),
                    "z_b": LatentSpaceConfig(dim=7, fed_by={"backbone": {"head": "linear"}}),
                },
                decoders_consume={
                    "shared_out": DecoderConsumptionConfig(
                        spaces=["z_a", "z_b"], assembler="concat"
                    )
                },
            ),
        )
        model = buildModelFromConfig(config)
        assert _projectionWidth(model, "signal") == 12


class TestModalityDropout:
    """Modality dropout hides modalities inside the shared stream (spec §5)."""

    def test_trainer_feeds_the_shared_encoder_non_empty_subsets_of_its_modalities(self) -> None:
        """With dropout the encoder sees full and partial dicts, and never an empty one."""
        torch.manual_seed(0)
        model = _buildModel()
        calls = _recordEncoderCalls(model)
        Trainer(model, device="cpu", modality_dropout_p=0.5).fit(_batches(4), num_epochs=6)
        seen = set(calls)
        assert seen <= {("signal", "image"), ("signal",), ("image",)}
        assert ("signal", "image") in seen
        assert seen & {("signal",), ("image",)}

    def test_every_modality_is_reconstructed_when_only_one_is_fed(self) -> None:
        """A shared decoder reconstructs both modalities from the signal alone."""
        torch.manual_seed(0)
        model = _buildModel(decoders="D1")
        batch = _batches(1)[0]
        outputs = model({"signal": batch["signal"]})
        assert set(outputs["reconstructions"]) == {"signal", "image"}
        loss = _lossOf(model, batch, inputs={"signal": batch["signal"]})
        assert torch.isfinite(loss)

    def test_a_hidden_modality_gets_no_gradient(self) -> None:
        """Only the signal tokenizer is updated by a signal-only pass, the image one is not."""
        torch.manual_seed(0)
        model = _buildModel(decoders="D1")
        batch = _batches(1)[0]
        backward(_lossOf(model, batch, inputs={"signal": batch["signal"]}))
        for name, parameter in model.named_parameters():
            if name.startswith("encoders.backbone.tokenizers.image."):
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None, name

    def test_every_parameter_receives_a_gradient_when_nothing_is_hidden(
        self, model: GlobalVae
    ) -> None:
        """Without dropout the whole model, shared encoder and decoders included, is trained."""
        backward(_lossOf(model, _batches(1)[0]))
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name

    @pytest.mark.parametrize("p", [0.3, 1.0])
    def test_the_loss_stays_finite_with_modality_dropout(self, model: GlobalVae, p: float) -> None:
        """Even with `p=1.0`, which still keeps one modality, training does not break."""
        torch.manual_seed(0)
        history = Trainer(model, device="cpu", modality_dropout_p=p).fit(_batches(3), num_epochs=4)
        assert all(torch.isfinite(torch.tensor(entry["train/loss/total"])) for entry in history)

    def test_the_loss_falls_with_modality_dropout(self, model: GlobalVae) -> None:
        """The mean loss of the last epochs is below the first epoch's.

        The learning rate is small on purpose: the linear dummy encoder has no normalisation, and
        with `lr=5e-3` the change of input statistics between a full and a partial batch makes the
        loss spike for a few epochs before it comes back (without dropout it falls steadily).
        """
        torch.manual_seed(0)
        trainer = Trainer(
            model, device="cpu", optimizer_kwargs={"lr": 1e-3}, beta=0.1, modality_dropout_p=0.3
        )
        history = trainer.fit(_batches(4), num_epochs=12)
        last = sum(entry["train/loss/total"] for entry in history[-3:]) / 3
        assert last < history[0]["train/loss/total"]

    def test_reconstruction_weights_are_keyed_by_modality(self) -> None:
        """Per-modality weights apply to the reconstructions a shared decoder returns."""
        torch.manual_seed(0)
        model = _buildModel(decoders="D1")
        batch = _batches(1)[0]
        reconstructions = model(batch, use_mean=True)["reconstructions"]
        plain = computeTotalReconstructionLoss(reconstructions, batch)
        weighted = computeTotalReconstructionLoss(
            reconstructions, batch, weights={"signal": 2.0, "image": 0.0}
        )
        signal_only = computeTotalReconstructionLoss(
            {"signal": reconstructions["signal"]}, {"signal": batch["signal"]}
        )
        assert torch.allclose(weighted, 2.0 * signal_only)
        assert not torch.allclose(weighted, plain)


class TestCrossModalReports:
    """Cross-modal reports query a shared encoder with each of its modalities (spec §5)."""

    def test_the_default_subsets_are_the_modalities_the_shared_encoder_ingests(
        self, model: GlobalVae
    ) -> None:
        """One singleton per modality, then both: the modalities, not the instance name."""
        assert resolveDefaultInputSubsets(model) == [
            frozenset({"signal"}),
            frozenset({"image"}),
            frozenset({"signal", "image"}),
        ]

    def test_every_cell_holds_every_modality_the_decoders_reconstruct(
        self, model: GlobalVae
    ) -> None:
        """Both the shared decoder and the per-modality decoders fill the whole matrix."""
        model.eval()
        collected = collectCrossModalReconstructions(model, _batches(2), device="cpu")
        assert set(collected) == set(resolveDefaultInputSubsets(model))
        for subset, per_decoder in collected.items():
            assert set(per_decoder) == {"signal", "image"}, subset
            originals, reconstructions = per_decoder["image"]
            assert originals.shape == reconstructions.shape == (2 * BATCH_SIZE, *IMAGE_SHAPE)

    def test_a_subset_reaches_inside_the_shared_stream(self) -> None:
        """Reconstructing from the signal alone differs from reconstructing from both."""
        torch.manual_seed(0)
        model = _buildModel()
        model.eval()
        collected = collectCrossModalReconstructions(model, _batches(1), device="cpu")
        alone = collected[frozenset({"signal"})]["image"][1]
        both = collected[frozenset({"signal", "image"})]["image"][1]
        assert not torch.allclose(alone, both)

    def test_an_explicit_subset_cannot_name_the_encoder_instance(self, model: GlobalVae) -> None:
        """`backbone` is not a modality, so a subset naming it is refused."""
        with pytest.raises(ValueError, match="unknown modality name"):
            collectCrossModalReconstructions(
                model, _batches(1), input_subsets=[{"backbone"}], device="cpu"
            )

    def test_the_cross_modal_metrics_run_on_the_whole_matrix(self, model: GlobalVae) -> None:
        """A metric is finite for every (subset, modality) cell."""
        model.eval()
        collected = collectCrossModalReconstructions(model, _batches(2), device="cpu")
        metrics = computeCrossModalReconstructionMetrics(collected)
        assert set(metrics) == set(collected)
        for per_decoder in metrics.values():
            assert set(per_decoder) == {"signal", "image"}
            for per_metric in per_decoder.values():
                assert all(torch.isfinite(torch.tensor(value)) for value in per_metric.values())

    def test_the_figure_is_written_for_a_shared_encoder_over_two_1d_modalities(
        self, tmp_path: Path
    ) -> None:
        """The matrix figure counts the two modalities, not the one encoder."""
        torch.manual_seed(0)
        model = _buildModel(shapes=PAIR_SHAPES)
        data = _batches(2, shapes=PAIR_SHAPES)
        paths = exportCrossModalFigures(model, data, tmp_path, device="cpu")
        assert [path.name for path in paths] == ["cross_modal_reconstructions.png"]
        assert paths[0].is_file()

    def test_there_is_nothing_cross_modal_for_a_shared_encoder_over_one_modality(
        self, tmp_path: Path
    ) -> None:
        """A single input modality has one possible subset: the export is a no-op."""
        torch.manual_seed(0)
        shapes = {"signal_a": (SIGNAL_LENGTH,)}
        model = _buildModel(shapes=shapes)
        assert exportCrossModalFigures(model, _batches(1, shapes=shapes), tmp_path) == []


class TestEvaluationAndFigures:
    """Evaluation, the latent collectors and the figure export need no change."""

    def test_evaluate_reports_every_reconstructed_modality(self, model: GlobalVae) -> None:
        """Metrics are keyed by modality, whichever decoders produced them."""
        results = evaluate(model, _batches(2), device="cpu")
        assert set(results.reconstruction_metrics) == {"signal", "image"}
        assert set(results.regularization_metrics) == {"z_fused"}
        assert results.num_samples == 2 * BATCH_SIZE
        assert torch.isfinite(torch.tensor(results.total_reconstruction_loss))

    def test_the_latent_collector_runs_over_a_batch_with_a_decoder_only_target(
        self, model: GlobalVae
    ) -> None:
        """Targets that are not input modalities are filtered out before the encoder."""
        data = [{**batch, "extra_target": torch.randn(BATCH_SIZE, 3)} for batch in _batches(2)]
        mu, logvar = collectLatentParams(model, data, "z_fused", device="cpu")
        assert mu.shape == logvar.shape == (2 * BATCH_SIZE, LATENT_DIM)

    def test_the_evaluation_figures_follow_the_reconstructed_modalities(
        self, tmp_path: Path
    ) -> None:
        """One reconstruction grid per modality, even though there is one shared decoder."""
        torch.manual_seed(0)
        model = _buildModel(shapes=PAIR_SHAPES)
        paths = exportEvaluationFigures(
            model,
            _batches(2, shapes=PAIR_SHAPES),
            tmp_path,
            device="cpu",
            latent_projection_method="pca",
        )
        names = {path.name for path in paths}
        assert {"reconstructions_signal_a.png", "reconstructions_signal_b.png"} <= names
        assert "latent_z_fused.png" in names
        assert not any("shared_out" in name or "backbone" in name for name in names)


class TestCheckpoint:
    """The shared modules are part of the model state, named by instance."""

    def test_state_dict_names_the_shared_modules_by_instance_name(self) -> None:
        """Keys start with `encoders.backbone` and `decoders.shared_out`."""
        keys = set(_buildModel().state_dict())
        assert "encoders.backbone.tokenizers.signal.weight" in keys
        assert "encoders.backbone.to_logvar.bias" in keys
        assert "decoders.shared_out.projections.image.weight" in keys

    def test_a_checkpoint_round_trip_reproduces_the_posteriors_and_reconstructions(
        self, tmp_path: Path
    ) -> None:
        """A restored model gives the saved model's outputs."""
        torch.manual_seed(0)
        model = _buildModel()
        batch = _batches(1)[0]
        Trainer(model, device="cpu").fit(_batches(2), num_epochs=1)
        path = tmp_path / "shared.pt"
        saveCheckpoint(path, model)

        torch.manual_seed(123)
        restored = _buildModel()
        loadCheckpoint(path, restored)
        model.eval()
        restored.eval()
        with torch.no_grad():
            expected = model(batch, use_mean=True)
            actual = restored(batch, use_mean=True)
        for name, (mu, logvar) in expected["latent_params"].items():
            assert torch.allclose(actual["latent_params"][name][0], mu)
            assert torch.allclose(actual["latent_params"][name][1], logvar)
        for name, reconstruction in expected["reconstructions"].items():
            assert torch.allclose(actual["reconstructions"][name], reconstruction)
