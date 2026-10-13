"""The learned assemblers `weighted_sum` and `attention` in an `EN-LN-DN` model (roadmap P1-4).

The model is the `EN-LN-DN` row of spec §2.1 with the real modules of the two modalities the
framework ships: a `1d_cnn` signal and a `2d_cnn` image, each encoded into a latent space of its
own, and a signal and an image decoder that each read both latent spaces through the assembler
under test. The decoders carry no `latent_dim`: it is taken from the assembler's `output_dim`.

The checks follow the "done when" of roadmap P1-4:

* both assemblers work end to end: the model builds, trains a few steps with a falling loss, and
  every parameter, the assemblers' own included, receives a gradient,
* subsets of the latent spaces: with a modality left out (spec §5) both decoders still reconstruct,
  from the latent space that is present, through the same assembler,
* the model can be written in YAML, `kwargs` included, and the YAML and Python routes agree,
* a checkpoint carries the learned parameters.

`tests/integration/test_configuration_matrix.py` runs the same row once per assembler for shapes
and gradients, and `tests/unit/test_learned_assemblers.py` holds the assemblers' own values.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.assemblers.attention import AttentionAssembler
from global_vae.assemblers.weighted_sum import WeightedSumAssembler
from global_vae.config.model import ModelConfig, buildModelFromConfig, resolveModelConfig
from global_vae.latent.base import LatentSpace, RoutingGraph
from global_vae.losses.reconstruction import computeTotalReconstructionLoss
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint, saveCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward

BATCH_SIZE = 8
SIGNAL_LENGTH = 32
IMAGE_SHAPE = (32, 32)
LATENT_DIM = 8

SIGNAL_ENCODER = "1d_cnn_encoder_v1"
SIGNAL_DECODER = "1d_cnn_decoder_v1"
IMAGE_ENCODER = "2d_cnn_encoder_v1"
IMAGE_DECODER = "2d_cnn_decoder_v1"

Batch = dict[str, torch.Tensor]

# One case per assembler, with arguments that change what it builds: `attention` with a width
# unrelated to the latent spaces and several heads, `weighted_sum` without the softmax.
CASES: list[Any] = [
    pytest.param("weighted_sum", {}, {}, LATENT_DIM, id="weighted_sum"),
    pytest.param("weighted_sum", {"normalize": False}, {}, LATENT_DIM, id="weighted_sum-raw"),
    pytest.param("attention", {}, {}, 12, id="attention-default-width"),
    pytest.param("attention", {"output_dim": 12, "num_heads": 3}, {}, 12, id="attention-heads"),
]


def _encoderKwargs(latent_dim: int) -> dict[str, Any]:
    """Small encoder arguments, shared by the 1D and 2D encoders."""
    return {"latent_dim": latent_dim, "hidden_channels": (8, 16)}


def _decoderKwargs(registry_name: str) -> dict[str, Any]:
    """Small decoder arguments that reach the modality's size exactly, with no `latent_dim`."""
    common: dict[str, Any] = {"upsample_modes": "conv_transpose", "hidden_channels": (16, 8)}
    if registry_name == SIGNAL_DECODER:
        return {**common, "output_length": SIGNAL_LENGTH}
    return {**common, "output_shape": IMAGE_SHAPE, "seed_shape": (8, 8)}


def _buildModel(
    assembler: str, assembler_kwargs: dict[str, Any], signal_dim: int, image_dim: int
) -> GlobalVae:
    """`EN-LN-DN`: one latent space per encoder, a signal and an image decoder over both.

    Args:
        assembler: Assembler every decoder uses to combine the two latent spaces.
        assembler_kwargs: Its constructor arguments next to `input_dims`.
        signal_dim: Dimensionality of `z_signal`.
        image_dim: Dimensionality of `z_image`.

    Returns:
        The model. The decoders are built without `latent_dim`.
    """
    graph = RoutingGraph(
        latent_specs={
            "z_signal": LatentSpace("z_signal", signal_dim),
            "z_image": LatentSpace("z_image", image_dim),
        },
        encoder_to_latents={"signal": ["z_signal"], "image": ["z_image"]},
        latent_to_decoders={"z_signal": ["signal", "image"], "z_image": ["signal", "image"]},
        decoder_assemblers={"signal": assembler, "image": assembler},
    )
    return GlobalVae(
        encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
        decoder_configs={"signal": SIGNAL_DECODER, "image": IMAGE_DECODER},
        routing_graph=graph,
        encoder_kwargs={
            "signal": _encoderKwargs(signal_dim),
            "image": _encoderKwargs(image_dim),
        },
        decoder_kwargs={
            "signal": _decoderKwargs(SIGNAL_DECODER),
            "image": _decoderKwargs(IMAGE_DECODER),
        },
        assembler_kwargs={"signal": assembler_kwargs, "image": assembler_kwargs},
    )


def _modelFor(assembler: str, assembler_kwargs: dict[str, Any]) -> GlobalVae:
    """A seeded `EN-LN-DN` model; `weighted_sum` needs equal widths, `attention` does not."""
    torch.manual_seed(0)
    image_dim = LATENT_DIM if assembler == "weighted_sum" else 12
    return _buildModel(assembler, assembler_kwargs, LATENT_DIM, image_dim)


def _batches(num_batches: int = 4, seed: int = 0) -> list[Batch]:
    """A small fixed dataset whose signal and image share a few latent factors.

    Low-rank structure keeps the reconstruction loss learnable in a handful of steps.
    """
    generator = torch.Generator().manual_seed(seed)
    signal_basis = torch.randn(3, SIGNAL_LENGTH, generator=generator)
    image_basis = torch.randn(3, *IMAGE_SHAPE, generator=generator)
    batches: list[Batch] = []
    for _ in range(num_batches):
        factors = torch.randn(BATCH_SIZE, 3, generator=generator)
        batches.append(
            {
                "signal": factors @ signal_basis,
                "image": torch.einsum("bk,khw->bhw", factors, image_basis),
            }
        )
    return batches


def _lossOf(model: GlobalVae, batch: Batch, inputs: Batch | None = None) -> torch.Tensor:
    """Total loss of `batch` (reconstruction plus regularization), encoding only `inputs`."""
    outputs = model(inputs if inputs is not None else batch)
    reconstruction = computeTotalReconstructionLoss(outputs["reconstructions"], batch)
    return reconstruction + model.computeRegularizationLoss(outputs["latent_params"])


@pytest.mark.parametrize(("assembler", "assembler_kwargs", "unused", "width"), CASES)
class TestEnLnDnWithALearnedAssembler:
    """`EN-LN-DN` with `weighted_sum` or `attention` between the latent spaces and the decoders."""

    def test_the_decoders_are_sized_from_the_assembler(
        self, assembler: str, assembler_kwargs: dict[str, Any], unused: dict[str, Any], width: int
    ) -> None:
        """No `latent_dim` was given: both decoders take the assembler's `output_dim`."""
        model = _modelFor(assembler, assembler_kwargs)
        assert set(model.assemblers) == {"signal", "image"}
        for name in ("signal", "image"):
            assert model.assemblers[name].output_dim == width
        outputs = model(_batches(1)[0])
        assert outputs["reconstructions"]["signal"].shape == (BATCH_SIZE, SIGNAL_LENGTH)
        assert outputs["reconstructions"]["image"].shape == (BATCH_SIZE, *IMAGE_SHAPE)

    def test_each_decoder_has_an_assembler_of_its_own(
        self, assembler: str, assembler_kwargs: dict[str, Any], unused: dict[str, Any], width: int
    ) -> None:
        """Two decoders over the same spaces do not share weights: one module per decoder."""
        model = _modelFor(assembler, assembler_kwargs)
        assert model.assemblers["signal"] is not model.assemblers["image"]
        names = [name for name, _ in model.named_parameters() if name.startswith("assemblers.")]
        assert any(name.startswith("assemblers.signal.") for name in names)
        assert any(name.startswith("assemblers.image.") for name in names)

    def test_every_parameter_receives_a_finite_gradient(
        self, assembler: str, assembler_kwargs: dict[str, Any], unused: dict[str, Any], width: int
    ) -> None:
        """One backward pass leaves no parameter without a gradient, the assemblers' included."""
        model = _modelFor(assembler, assembler_kwargs)
        backward(_lossOf(model, _batches(1)[0]))
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        for name, parameter in model.named_parameters():
            if name.startswith("assemblers."):
                assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name

    def test_the_loss_falls_over_a_few_epochs(
        self, assembler: str, assembler_kwargs: dict[str, Any], unused: dict[str, Any], width: int
    ) -> None:
        """Mean total loss over the last epochs is below the first epoch's."""
        model = _modelFor(assembler, assembler_kwargs)
        trainer = Trainer(model, device="cpu", optimizer_kwargs={"lr": 5e-3}, beta=0.1)
        history = trainer.fit(_batches(4), num_epochs=12)
        first = history[0]["train/loss/total"]
        last = sum(entry["train/loss/total"] for entry in history[-3:]) / 3
        assert last < first
        assert all(torch.isfinite(torch.tensor(entry["train/loss/total"])) for entry in history)

    def test_training_updates_the_assembler_parameters(
        self, assembler: str, assembler_kwargs: dict[str, Any], unused: dict[str, Any], width: int
    ) -> None:
        """The optimizer sees the assemblers' parameters: they move during training."""
        model = _modelFor(assembler, assembler_kwargs)
        before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.startswith("assemblers.")
        }
        assert before
        Trainer(model, device="cpu", optimizer_kwargs={"lr": 5e-3}).fit(_batches(2), num_epochs=2)
        moved = [
            name
            for name, p in model.named_parameters()
            if name.startswith("assemblers.") and not torch.equal(p, before[name])
        ]
        assert moved

    @pytest.mark.parametrize("present", ["signal", "image"])
    def test_a_modality_left_out_still_decodes_through_the_assembler(
        self,
        assembler: str,
        assembler_kwargs: dict[str, Any],
        unused: dict[str, Any],
        width: int,
        present: str,
    ) -> None:
        """With one encoder, both decoders run on the one latent space available (spec §5).

        Unlike `concat`, these assemblers accept a subset, and a decoder sized for the
        assembler's `output_dim` still gets that width, even from a single latent space.
        """
        model = _modelFor(assembler, assembler_kwargs)
        outputs = model({present: _batches(1)[0][present]})
        assert set(outputs["latent_samples"]) == {f"z_{present}"}
        assert set(outputs["reconstructions"]) == {"signal", "image"}
        assert outputs["reconstructions"]["signal"].shape == (BATCH_SIZE, SIGNAL_LENGTH)
        assert outputs["reconstructions"]["image"].shape == (BATCH_SIZE, *IMAGE_SHAPE)

    def test_the_assembler_is_told_which_latent_space_remains(
        self, assembler: str, assembler_kwargs: dict[str, Any], unused: dict[str, Any], width: int
    ) -> None:
        """The model passes `present`: the position of each latent given, in `input_dims`."""
        model = _modelFor(assembler, assembler_kwargs)
        seen: list[Any] = []
        model.assemblers["signal"].register_forward_hook(
            lambda _module, args, _output: seen.append(args[1])
        )
        model(_batches(1)[0])
        model({"image": _batches(1)[0]["image"]})
        model({"signal": _batches(1)[0]["signal"]})
        assert seen == [None, [1], [0]]

    def test_a_dropped_modality_leaves_the_other_modalitys_assembler_parameters_alone(
        self, assembler: str, assembler_kwargs: dict[str, Any], unused: dict[str, Any], width: int
    ) -> None:
        """With `image` only, the loss trains the image side and leaves the signal encoder alone."""
        model = _modelFor(assembler, assembler_kwargs)
        batch = _batches(1)[0]
        backward(_lossOf(model, batch, inputs={"image": batch["image"]}))
        for name, parameter in model.named_parameters():
            if name.startswith("encoders.signal."):
                assert parameter.grad is None, name
            elif name.startswith(("encoders.image.", "decoders.", "assemblers.")):
                # Some assembler parameters belong to the absent space (its projection) and are
                # not part of this pass; every encoder and decoder parameter present is.
                if name.startswith("assemblers.") and ".projections.0." in name:
                    assert parameter.grad is None, name
                else:
                    assert parameter.grad is not None, name

    def test_a_checkpoint_carries_the_learned_parameters(
        self,
        assembler: str,
        assembler_kwargs: dict[str, Any],
        unused: dict[str, Any],
        width: int,
        tmp_path: Path,
    ) -> None:
        """A model loaded from a checkpoint reproduces the saved model's reconstructions."""
        model = _modelFor(assembler, assembler_kwargs)
        Trainer(model, device="cpu").fit(_batches(2), num_epochs=1)
        path = tmp_path / "learned_assembler.pt"
        saveCheckpoint(path, model)
        assembler_keys = [key for key in model.state_dict() if key.startswith("assemblers.")]
        assert assembler_keys

        restored = _buildModel(
            assembler,
            assembler_kwargs,
            LATENT_DIM,
            LATENT_DIM if assembler == "weighted_sum" else 12,
        )
        loadCheckpoint(path, restored)
        model.eval()
        restored.eval()
        batch = _batches(1)[0]
        with torch.no_grad():
            expected = model(batch, use_mean=True)["reconstructions"]
            actual = restored(batch, use_mean=True)["reconstructions"]
        for name, reconstruction in expected.items():
            assert torch.allclose(actual[name], reconstruction, atol=1e-5), name


YAML_TEMPLATE = """
name: global_vae_learned_assembler
latent_mode: several
encoders:
  signal: {{name: 1d_cnn_encoder_v1, kwargs: {{hidden_channels: [8, 16]}}}}
  image: {{name: 2d_cnn_encoder_v1, kwargs: {{hidden_channels: [8, 16]}}}}
decoders:
  image_out:
    name: 2d_cnn_decoder_v1
    kwargs:
      output_shape: [32, 32]
      seed_shape: [8, 8]
      hidden_channels: [16, 8]
      upsample_modes: conv_transpose
latent:
  spaces:
    z_signal: {{dim: {signal_dim}, fed_by: [signal]}}
    z_image: {{dim: {image_dim}, fed_by: [image]}}
  decoders_consume:
    image_out:
      spaces: [z_signal, z_image]
      assembler: {assembler}
      kwargs: {kwargs}
"""


def _yamlConfig(assembler: str, kwargs: str, signal_dim: int, image_dim: int) -> ModelConfig:
    """Parse a model config written in YAML, with the assembler and its kwargs filled in."""
    text = YAML_TEMPLATE.format(
        assembler=assembler, kwargs=kwargs, signal_dim=signal_dim, image_dim=image_dim
    )
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(text))
    )
    assert isinstance(config, ModelConfig)
    return config


class TestFromYaml:
    """The assemblers' kwargs reach them from a YAML model config (roadmap P1-4, needs P1-3)."""

    def test_weighted_sum_with_its_kwarg(self) -> None:
        """`normalize: false` written in YAML is the assembler's `normalize`."""
        config = _yamlConfig("weighted_sum", "{normalize: false}", 16, 16)
        model = buildModelFromConfig(config)
        assembler = model.assemblers["image_out"]
        assert isinstance(assembler, WeightedSumAssembler)
        assert assembler.normalize is False
        assert assembler.input_dims == (16, 16)

    def test_attention_with_its_kwargs_and_unequal_latent_widths(self) -> None:
        """`output_dim` and `num_heads` from YAML, latent spaces of 6 and 10, no `latent_dim`."""
        config = _yamlConfig("attention", "{output_dim: 12, num_heads: 4}", 6, 10)
        resolved = resolveModelConfig(config)
        assert resolved.assembler_kwargs == {"image_out": {"output_dim": 12, "num_heads": 4}}
        assert resolved.decoder_kwargs["image_out"]["latent_dim"] == 12
        model = buildModelFromConfig(config)
        assembler = model.assemblers["image_out"]
        assert isinstance(assembler, AttentionAssembler)
        assert (assembler.output_dim, assembler.num_heads) == (12, 4)
        assert assembler.input_dims == (6, 10)
        batch = {"signal": torch.randn(2, SIGNAL_LENGTH), "image": torch.randn(2, *IMAGE_SHAPE)}
        reconstruction = model(batch)["reconstructions"]["image_out"]
        assert reconstruction.shape == (2, *IMAGE_SHAPE)

    def test_attention_default_width_is_the_widest_latent_space(self) -> None:
        """With no kwargs the decoder is sized by the default `output_dim = max(input_dims)`."""
        config = _yamlConfig("attention", "{}", 6, 10)
        assert resolveModelConfig(config).decoder_kwargs["image_out"]["latent_dim"] == 10

    def test_the_resolution_does_not_move_the_random_state(self) -> None:
        """Sizing the decoder builds a throwaway assembler that must not shift later init."""
        config = _yamlConfig("attention", "{output_dim: 12}", 6, 10)
        torch.manual_seed(7)
        expected = torch.rand(3)
        torch.manual_seed(7)
        resolveModelConfig(config)
        assert torch.equal(torch.rand(3), expected)

    def test_weighted_sum_over_different_widths_is_reported_under_the_latent_section(self) -> None:
        """The dimension rule of `sum` applies, and the message starts with the config path."""
        config = _yamlConfig("weighted_sum", "{}", 6, 10)
        with pytest.raises(ValueError, match=r"^model.latent: .*requires matching dimensionality"):
            resolveModelConfig(config)

    def test_invalid_attention_kwargs_are_reported_with_the_field_path(self) -> None:
        """A head count that does not divide the width names the `decoders_consume` entry."""
        config = _yamlConfig("attention", "{output_dim: 10, num_heads: 4}", 6, 10)
        with pytest.raises(
            ValueError,
            match=r"model.latent.decoders_consume.image_out: assembler 'attention'.*multiple of",
        ):
            resolveModelConfig(config)

    def test_the_yaml_and_the_python_route_build_the_same_structure(self) -> None:
        """Same module tree and parameter shapes whether the model is written in YAML or Python."""
        torch.manual_seed(0)
        from_yaml = buildModelFromConfig(
            _yamlConfig("attention", "{output_dim: 12, num_heads: 3}", 8, 12)
        )
        graph = RoutingGraph(
            latent_specs={
                "z_signal": LatentSpace("z_signal", 8),
                "z_image": LatentSpace("z_image", 12),
            },
            encoder_to_latents={"signal": ["z_signal"], "image": ["z_image"]},
            latent_to_decoders={"z_signal": ["image_out"], "z_image": ["image_out"]},
            decoder_assemblers={"image_out": "attention"},
        )
        from_python = GlobalVae(
            encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
            decoder_configs={"image_out": IMAGE_DECODER},
            routing_graph=graph,
            encoder_kwargs={"signal": _encoderKwargs(8), "image": _encoderKwargs(12)},
            decoder_kwargs={"image_out": _decoderKwargs(IMAGE_DECODER)},
            assembler_kwargs={"image_out": {"output_dim": 12, "num_heads": 3}},
        )
        yaml_shapes = {name: tuple(t.shape) for name, t in from_yaml.state_dict().items()}
        python_shapes = {name: tuple(t.shape) for name, t in from_python.state_dict().items()}
        assert yaml_shapes == python_shapes
