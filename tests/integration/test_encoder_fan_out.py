"""Encoder fan-out through latent heads, end to end (spec §2.2, roadmap P1-2, ADR 0024).

The model under test is the shared plus private topology of spec §9, with the real modules of the
two modalities the framework ships: a `1d_cnn` signal and a `2d_cnn` image. `z_shared` is fused
from both encoders with PoE, each modality also has a private latent space, and every decoder
consumes `{z_shared, z_private_m}` through an assembler. The signal encoder therefore feeds two
latent spaces, and so does the image encoder: the first through an identity edge, the second
through a `linear` head.

The checks follow the "done when" of roadmap P1-2:

* the model builds and trains a few steps with a falling loss,
* every parameter, heads included, receives a gradient,
* the three latent spaces differ,
* modality dropout still removes an absent encoder's edges (its heads do not run, its latent spaces
  and, with `concat`, its decoder are absent from the pass),
* evaluation and cross-modal reports run unchanged.

Two further groups cover what the fan-out topology adds: the assembler rule for a decoder whose
private latent is missing (`sum` and `average` still decode from what is available, `concat` skips
the decoder, ADR 0024), and a checkpoint round trip that includes the heads.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
import global_vae.fusion  # noqa: F401  (registers the built-in fusion strategies)
import global_vae.heads  # noqa: F401  (registers the built-in latent heads)
from global_vae.evaluation.cross_modal import (
    computeCrossModalReconstructionMetrics,
    exportCrossModalFigures,
)
from global_vae.evaluation.evaluate import evaluate
from global_vae.latent.base import HeadSpec, LatentSpace, RoutingGraph
from global_vae.latent.routing_graph_builders.shared_private import buildSharedPrivateRoutingGraph
from global_vae.losses.reconstruction import computeTotalReconstructionLoss
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint, saveCheckpoint
from global_vae.training.trainer import Trainer
from global_vae.utils.autograd import backward
from global_vae.visualization.reconstruction_plot import collectCrossModalReconstructions

BATCH_SIZE = 8
SIGNAL_LENGTH = 32
IMAGE_SHAPE = (32, 32)
SHARED_DIM = 8
PRIVATE_DIM = 4

SIGNAL_ENCODER = "1d_cnn_encoder_v1"
SIGNAL_DECODER = "1d_cnn_decoder_v1"
IMAGE_ENCODER = "2d_cnn_encoder_v1"
IMAGE_DECODER = "2d_cnn_decoder_v1"

Batch = dict[str, torch.Tensor]


def _decoderKwargs(registry_name: str, latent_dim: int) -> dict[str, Any]:
    """Small decoder arguments that reach the modality's size exactly."""
    common: dict[str, Any] = {
        "latent_dim": latent_dim,
        "upsample_modes": "conv_transpose",
        "hidden_channels": (16, 8),
    }
    if registry_name == SIGNAL_DECODER:
        return {**common, "output_length": SIGNAL_LENGTH}
    return {**common, "output_shape": IMAGE_SHAPE, "seed_shape": (8, 8)}


def _buildModel(
    assembler: str = "concat", private_dim: int = PRIVATE_DIM, fusion: str = "poe"
) -> GlobalVae:
    """Signal plus image, `z_shared` fused, one private latent per modality behind a linear head.

    Args:
        assembler: Assembler every decoder uses to join `{z_shared, z_private_m}`.
        private_dim: Dimensionality of each private latent space (`sum` and `average` need it to
            equal `SHARED_DIM`).
        fusion: Fusion strategy of `z_shared`.

    Returns:
        The model.
    """
    graph = buildSharedPrivateRoutingGraph(
        ["signal", "image"], SHARED_DIM, private_dim, assembler=assembler
    )
    decoder_dim = SHARED_DIM + private_dim if assembler == "concat" else SHARED_DIM
    encoder_kwargs = {"latent_dim": SHARED_DIM, "hidden_channels": (8, 16)}
    return GlobalVae(
        encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
        decoder_configs={"signal": SIGNAL_DECODER, "image": IMAGE_DECODER},
        routing_graph=graph,
        fusion_strategies={"z_shared": fusion},
        encoder_kwargs={"signal": encoder_kwargs, "image": encoder_kwargs},
        decoder_kwargs={
            "signal": _decoderKwargs(SIGNAL_DECODER, decoder_dim),
            "image": _decoderKwargs(IMAGE_DECODER, decoder_dim),
        },
    )


def _buildSignalPair() -> GlobalVae:
    """Two 1D modalities with a shared plus private topology, for the 1D-only figure code."""
    graph = buildSharedPrivateRoutingGraph(["signal_a", "signal_b"], SHARED_DIM, PRIVATE_DIM)
    encoder_kwargs = {"latent_dim": SHARED_DIM, "hidden_channels": (8, 16)}
    decoder_kwargs = _decoderKwargs(SIGNAL_DECODER, SHARED_DIM + PRIVATE_DIM)
    return GlobalVae(
        encoder_configs={"signal_a": SIGNAL_ENCODER, "signal_b": SIGNAL_ENCODER},
        decoder_configs={"signal_a": SIGNAL_DECODER, "signal_b": SIGNAL_DECODER},
        routing_graph=graph,
        fusion_strategies={"z_shared": "poe"},
        encoder_kwargs={"signal_a": encoder_kwargs, "signal_b": encoder_kwargs},
        decoder_kwargs={"signal_a": decoder_kwargs, "signal_b": decoder_kwargs},
    )


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


@pytest.fixture
def model() -> GlobalVae:
    """A freshly built, seeded shared plus private model."""
    torch.manual_seed(0)
    return _buildModel()


class TestForward:
    """Shapes of what the fan-out model produces."""

    def test_three_latent_spaces_have_their_own_widths(self, model: GlobalVae) -> None:
        """`z_shared` is `SHARED_DIM` wide, each private space `PRIVATE_DIM`."""
        outputs = model(_batches(1)[0])
        widths = {name: mu.shape for name, (mu, _) in outputs["latent_params"].items()}
        assert widths == {
            "z_shared": (BATCH_SIZE, SHARED_DIM),
            "z_private_signal": (BATCH_SIZE, PRIVATE_DIM),
            "z_private_image": (BATCH_SIZE, PRIVATE_DIM),
        }
        for name, (mu, logvar) in outputs["latent_params"].items():
            assert logvar.shape == mu.shape, name

    def test_every_decoder_reconstructs_its_modality(self, model: GlobalVae) -> None:
        """Each decoder receives `{z_shared, z_private_m}` joined by `concat`."""
        outputs = model(_batches(1)[0])
        shapes = {name: tensor.shape for name, tensor in outputs["reconstructions"].items()}
        assert shapes == {
            "signal": (BATCH_SIZE, SIGNAL_LENGTH),
            "image": (BATCH_SIZE, *IMAGE_SHAPE),
        }

    def test_the_three_latent_spaces_differ(self, model: GlobalVae) -> None:
        """No two latent spaces carry the same posterior, the point of the heads."""
        torch.manual_seed(1)
        mus = {
            name: mu.detach() for name, (mu, _) in model(_batches(1)[0])["latent_params"].items()
        }
        assert not torch.allclose(mus["z_private_signal"], mus["z_private_image"])
        # `z_shared` is wider, so compare it with each private space on the widths they share.
        for private in ("z_private_signal", "z_private_image"):
            assert not torch.allclose(mus["z_shared"][:, :PRIVATE_DIM], mus[private])

    def test_the_shared_posterior_is_the_poe_of_the_two_identity_edges(
        self, model: GlobalVae
    ) -> None:
        """With identity on the shared edges, `z_shared` fuses the raw encoder outputs."""
        model.eval()
        batch = _batches(1)[0]
        with torch.no_grad():
            signal_params = model.encoders["signal"](batch["signal"])
            image_params = model.encoders["image"](batch["image"])
            expected = model.fusions["z_shared"]({"signal": signal_params, "image": image_params})
            actual = model(batch)["latent_params"]["z_shared"]
        assert torch.allclose(actual[0], expected[0]) and torch.allclose(actual[1], expected[1])

    def test_a_private_posterior_is_the_head_of_its_encoder_output(self, model: GlobalVae) -> None:
        """`z_private_signal` is exactly the linear head applied to the signal encoder's output."""
        model.eval()
        batch = _batches(1)[0]
        head = model.getHead("signal", "z_private_signal")
        assert head is not None
        with torch.no_grad():
            expected = head(*model.encoders["signal"](batch["signal"]))
            actual = model(batch)["latent_params"]["z_private_signal"]
        assert torch.allclose(actual[0], expected[0]) and torch.allclose(actual[1], expected[1])

    def test_graph_order_decides_the_concat_order(self, model: GlobalVae) -> None:
        """A decoder receives `[z_shared, z_private_m]`, shared first, as the graph lists them."""
        received: list[torch.Tensor] = []
        model.assemblers["signal"].register_forward_hook(
            lambda _module, inputs, _output: received.extend(inputs[0])
        )
        outputs = model(_batches(1)[0], use_mean=True)
        assert len(received) == 2
        assert torch.equal(received[0], outputs["latent_samples"]["z_shared"])
        assert torch.equal(received[1], outputs["latent_samples"]["z_private_signal"])


class TestGradients:
    """Every parameter takes part in the loss, heads included."""

    def test_every_parameter_receives_a_finite_gradient(self, model: GlobalVae) -> None:
        """One backward pass over both modalities leaves no parameter without a gradient."""
        torch.manual_seed(0)
        backward(_lossOf(model, _batches(1)[0]))
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"parameters without a gradient: {missing}"
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all(), name

    def test_head_parameters_receive_a_nonzero_gradient(self, model: GlobalVae) -> None:
        """The linear heads are trained: both of each head's projections get a signal."""
        torch.manual_seed(0)
        backward(_lossOf(model, _batches(1)[0]))
        head_parameters = {
            name: p for name, p in model.named_parameters() if name.startswith("heads.")
        }
        assert set(head_parameters) == {
            f"heads.{modality}.z_private_{modality}.{layer}.{kind}"
            for modality in ("signal", "image")
            for layer in ("to_mu", "to_logvar")
            for kind in ("weight", "bias")
        }
        for name, parameter in head_parameters.items():
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name

    def test_each_encoder_gets_gradient_from_both_of_its_latent_spaces(
        self, model: GlobalVae
    ) -> None:
        """A loss on `z_private_signal` alone still reaches the signal encoder, through its head."""
        torch.manual_seed(0)
        outputs = model(_batches(1)[0])
        mu, logvar = outputs["latent_params"]["z_private_signal"]
        backward(mu.pow(2).sum() + logvar.pow(2).sum())
        signal = [p for n, p in model.named_parameters() if n.startswith("encoders.signal.")]
        image = [p for n, p in model.named_parameters() if n.startswith("encoders.image.")]
        assert all(p.grad is not None for p in signal)
        assert all(p.grad is None for p in image)


class TestTraining:
    """The fan-out model trains with the existing `Trainer`."""

    def test_loss_falls_over_a_few_epochs(self, model: GlobalVae) -> None:
        """Mean total loss over the last epochs is below the first epoch's."""
        trainer = Trainer(model, device="cpu", optimizer_kwargs={"lr": 5e-3}, beta=0.1)
        history = trainer.fit(_batches(4), num_epochs=12)
        first = history[0]["train/loss/total"]
        last = sum(entry["train/loss/total"] for entry in history[-3:]) / 3
        assert last < first
        assert all(torch.isfinite(torch.tensor(entry["train/loss/total"])) for entry in history)

    def test_training_updates_the_heads(self, model: GlobalVae) -> None:
        """After training, the head weights have moved from their initial values."""
        head = model.getHead("signal", "z_private_signal")
        assert head is not None
        before = head.state_dict()["to_mu.weight"].clone()
        Trainer(model, device="cpu", optimizer_kwargs={"lr": 5e-3}).fit(_batches(2), num_epochs=2)
        assert not torch.equal(head.state_dict()["to_mu.weight"], before)

    def test_every_latent_space_is_regularized_and_beta_applies_per_space(
        self, model: GlobalVae
    ) -> None:
        """The three latent spaces have their own regularizer, and a beta of 0 silences one."""
        assert set(model.regularizers) == {"z_shared", "z_private_signal", "z_private_image"}
        params = model(_batches(1)[0])["latent_params"]
        total = model.computeRegularizationLoss(params)
        without_private = model.computeRegularizationLoss(
            params, beta={"z_shared": 1.0, "z_private_signal": 0.0, "z_private_image": 0.0}
        )
        assert without_private < total


class TestModalityDropout:
    """An absent encoder's edges are removed with it (spec §5)."""

    @staticmethod
    def _countCalls(model: GlobalVae) -> dict[str, int]:
        """Register a call counter on every head, keyed by `encoder:latent`."""
        counts: dict[str, int] = {}
        for encoder in ("signal", "image"):
            for latent in ("z_shared", f"z_private_{encoder}"):
                head = model.getHead(encoder, latent)
                assert head is not None
                key = f"{encoder}:{latent}"
                counts[key] = 0

                def hook(_module: nn.Module, _inputs: Any, _output: Any, key: str = key) -> None:
                    counts[key] += 1

                head.register_forward_hook(hook)
        return counts

    def test_an_absent_encoders_heads_do_not_run(self, model: GlobalVae) -> None:
        """With only `image` encoded, the signal edges are not evaluated at all."""
        counts = self._countCalls(model)
        model({"image": _batches(1)[0]["image"]})
        assert counts == {
            "signal:z_shared": 0,
            "signal:z_private_signal": 0,
            "image:z_shared": 1,
            "image:z_private_image": 1,
        }

    def test_an_absent_encoders_private_latent_is_absent(self, model: GlobalVae) -> None:
        """Only the latent spaces some present encoder feeds appear in the pass."""
        outputs = model({"image": _batches(1)[0]["image"]})
        assert set(outputs["latent_params"]) == {"z_shared", "z_private_image"}
        assert set(outputs["latent_samples"]) == {"z_shared", "z_private_image"}

    def test_the_shared_latent_of_one_encoder_is_that_encoders_own_output(
        self, model: GlobalVae
    ) -> None:
        """With one active encoder, no fusion runs and `z_shared` is its (head-adapted) output."""
        model.eval()
        image = _batches(1)[0]["image"]
        with torch.no_grad():
            expected = model.encoders["image"](image)
            actual = model({"image": image})["latent_params"]["z_shared"]
        assert torch.allclose(actual[0], expected[0]) and torch.allclose(actual[1], expected[1])

    def test_a_decoder_with_an_incomplete_concat_input_is_skipped(self, model: GlobalVae) -> None:
        """`concat` cannot decode from `z_shared` alone, so the signal decoder leaves the pass."""
        outputs = model({"image": _batches(1)[0]["image"]})
        assert set(outputs["reconstructions"]) == {"image"}

    def test_dropped_modality_gets_no_gradient(self, model: GlobalVae) -> None:
        """A loss on an `image`-only pass leaves the signal encoder and its heads untouched."""
        batch = _batches(1)[0]
        backward(_lossOf(model, batch, inputs={"image": batch["image"]}))
        for name, parameter in model.named_parameters():
            if name.startswith(("encoders.signal.", "heads.signal.", "decoders.signal.")):
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None, name

    @pytest.mark.parametrize("p", [0.5, 1.0])
    def test_trainer_with_modality_dropout_trains_without_error(
        self, model: GlobalVae, p: float
    ) -> None:
        """The training loop copes with encoders, latent spaces and decoders coming and going."""
        torch.manual_seed(0)
        trainer = Trainer(model, device="cpu", modality_dropout_p=p)
        history = trainer.fit(_batches(3), num_epochs=4)
        assert all(torch.isfinite(torch.tensor(entry["train/loss/total"])) for entry in history)

    @pytest.mark.parametrize("assembler", ["sum", "average"])
    def test_tolerant_assemblers_still_decode_from_the_available_latents(
        self, assembler: str
    ) -> None:
        """With `sum` or `average`, the signal decoder runs on `z_shared` alone."""
        torch.manual_seed(0)
        model = _buildModel(assembler=assembler, private_dim=SHARED_DIM)
        outputs = model({"image": _batches(1)[0]["image"]})
        assert set(outputs["reconstructions"]) == {"signal", "image"}
        assert outputs["reconstructions"]["signal"].shape == (BATCH_SIZE, SIGNAL_LENGTH)

    @pytest.mark.parametrize("assembler", ["sum", "average"])
    def test_tolerant_assemblers_pass_the_shared_latent_through_unchanged(
        self, assembler: str
    ) -> None:
        """A decoder with one available latent receives it as it is, as before this change."""
        torch.manual_seed(0)
        model = _buildModel(assembler=assembler, private_dim=SHARED_DIM)
        received: list[torch.Tensor] = []
        model.decoders["signal"].register_forward_hook(
            lambda _module, inputs, _output: received.append(inputs[0])
        )
        outputs = model({"image": _batches(1)[0]["image"]}, use_mean=True)
        assert torch.equal(received[0], outputs["latent_samples"]["z_shared"])


class TestEvaluationAndCrossModalReports:
    """Evaluation and the cross-modal reports run on the fan-out model without changes."""

    def test_evaluate_reports_every_decoder_and_every_latent_space(self, model: GlobalVae) -> None:
        """Reconstruction metrics per decoder, a regularization value per latent space."""
        results = evaluate(model, _batches(2), device="cpu")
        assert set(results.reconstruction_metrics) == {"signal", "image"}
        assert set(results.regularization_metrics) == {
            "z_shared",
            "z_private_signal",
            "z_private_image",
        }
        assert results.num_samples == 2 * BATCH_SIZE
        assert torch.isfinite(torch.tensor(results.total_reconstruction_loss))
        assert torch.isfinite(torch.tensor(results.total_regularization_loss))

    def test_cross_modal_collection_covers_the_default_subsets(self, model: GlobalVae) -> None:
        """Each single modality and the pair are collected; absent decoders are absent cells."""
        collected = _collect(model)
        signal, image = frozenset({"signal"}), frozenset({"image"})
        both = signal | image
        assert set(collected) == {signal, image, both}
        assert set(collected[both]) == {"signal", "image"}
        # `concat` cannot decode a modality from `z_shared` alone, so each single-modality
        # subset only has the decoder of the modality that was fed in.
        assert set(collected[signal]) == {"signal"}
        assert set(collected[image]) == {"image"}

    def test_cross_modal_metrics_run_on_the_partial_matrix(self, model: GlobalVae) -> None:
        """The metric matrix is computed for every cell that exists."""
        collected = _collect(model)
        metrics = computeCrossModalReconstructionMetrics(collected)
        assert set(metrics) == set(collected)
        assert {k: set(v) for k, v in metrics.items()} == {k: set(v) for k, v in collected.items()}
        for per_decoder in metrics.values():
            for per_metric in per_decoder.values():
                assert all(torch.isfinite(torch.tensor(value)) for value in per_metric.values())

    def test_cross_modal_figure_is_written_with_cells_missing(self, tmp_path: Path) -> None:
        """The matrix figure copes with decoders absent from a subset.

        The overlay plots are for 1D series (image plots are not implemented, roadmap section
        2.3), so this uses two signal modalities.
        """
        torch.manual_seed(0)
        model = _buildSignalPair()
        data = [{"signal_a": b["signal"], "signal_b": b["signal"].flip(-1)} for b in _batches(2)]
        paths = exportCrossModalFigures(model, data, tmp_path, device="cpu")
        assert [path.name for path in paths] == ["cross_modal_reconstructions.png"]
        assert paths[0].is_file()

    def test_tolerant_assembler_fills_the_whole_cross_modal_matrix(self) -> None:
        """With `sum`, every decoder reconstructs from every subset."""
        torch.manual_seed(0)
        model = _buildModel(assembler="sum", private_dim=SHARED_DIM)
        collected = _collect(model)
        assert all(set(per_decoder) == {"signal", "image"} for per_decoder in collected.values())


def _collect(
    model: GlobalVae,
) -> dict[frozenset[str], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Run the default cross-modal collection over two batches, in eval mode."""
    model.eval()
    return collectCrossModalReconstructions(model, _batches(2), device="cpu")


class TestCheckpoint:
    """Heads are part of the model state, and old checkpoints without heads still load."""

    def test_state_dict_names_the_heads_by_encoder_and_latent_space(self, model: GlobalVae) -> None:
        """A head's weights are saved as `heads.<encoder>.<latent>.<layer>.<name>`."""
        keys = set(model.state_dict())
        assert "heads.signal.z_private_signal.to_mu.weight" in keys
        assert "heads.image.z_private_image.to_logvar.bias" in keys
        # The identity edges build modules too (an explicit `HeadSpec`), with nothing to save.
        assert not [key for key in keys if key.startswith("heads.signal.z_shared.")]

    def test_round_trip_restores_the_outputs(self, model: GlobalVae, tmp_path: Path) -> None:
        """A model loaded from a checkpoint reproduces the saved model's latent posteriors."""
        batch = _batches(1)[0]
        Trainer(model, device="cpu").fit(_batches(2), num_epochs=1)
        path = tmp_path / "fan_out.pt"
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
            assert torch.allclose(actual["latent_params"][name][0], mu), name
            assert torch.allclose(actual["latent_params"][name][1], logvar), name

    def test_a_checkpoint_without_heads_loads_into_a_model_without_heads(
        self, tmp_path: Path
    ) -> None:
        """Graphs without heads keep the state they always had (nothing under `heads.`)."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", SHARED_DIM)},
            encoder_to_latents={"signal": ["z"]},
            latent_to_decoders={"z": ["signal"]},
        )

        def build() -> GlobalVae:
            return GlobalVae(
                encoder_configs={"signal": SIGNAL_ENCODER},
                decoder_configs={"signal": SIGNAL_DECODER},
                routing_graph=graph,
                encoder_kwargs={"signal": {"latent_dim": SHARED_DIM, "hidden_channels": (8, 16)}},
                decoder_kwargs={"signal": _decoderKwargs(SIGNAL_DECODER, SHARED_DIM)},
            )

        source = build()
        assert not [key for key in source.state_dict() if key.startswith("heads.")]
        path = tmp_path / "plain.pt"
        saveCheckpoint(path, source)
        loadCheckpoint(path, build())


class TestOtherFusionStrategies:
    """The heads feed whichever fusion `z_shared` uses."""

    @pytest.mark.parametrize("fusion", ["poe", "moe"])
    def test_fusion_receives_the_head_adapted_parameters(self, fusion: str) -> None:
        """A fusion sees the output of each edge's head, not the raw encoder output."""
        torch.manual_seed(0)
        graph = buildSharedPrivateRoutingGraph(
            ["signal", "image"], SHARED_DIM, PRIVATE_DIM, shared_head="linear"
        )
        encoder_kwargs = {"latent_dim": 6, "hidden_channels": (8, 16)}
        model = GlobalVae(
            encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
            decoder_configs={"signal": SIGNAL_DECODER, "image": IMAGE_DECODER},
            routing_graph=graph,
            fusion_strategies={"z_shared": fusion},
            encoder_kwargs={"signal": encoder_kwargs, "image": encoder_kwargs},
            decoder_kwargs={
                "signal": _decoderKwargs(SIGNAL_DECODER, SHARED_DIM + PRIVATE_DIM),
                "image": _decoderKwargs(IMAGE_DECODER, SHARED_DIM + PRIVATE_DIM),
            },
        )
        seen: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        model.fusions["z_shared"].register_forward_hook(
            lambda _module, inputs, _output: seen.update(inputs[0])
        )
        model(_batches(1)[0])
        # The encoders are 6 wide and `z_shared` is 8 wide: only a head can bridge them.
        assert {name: mu.shape[-1] for name, (mu, _) in seen.items()} == {
            "signal": SHARED_DIM,
            "image": SHARED_DIM,
        }


class TestTwoLinearHeadsOnOneEncoder:
    """The independence the heads exist for: one encoder, two latent spaces, two posteriors."""

    @staticmethod
    def _trunkModel(heads: dict[str, HeadSpec]) -> GlobalVae:
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 4), "z_b": LatentSpace("z_b", 4)},
            encoder_to_latents={"signal": ["z_a", "z_b"]},
            latent_to_decoders={"z_a": ["signal"], "z_b": ["signal"]},
            decoder_assemblers={"signal": "concat"},
            encoder_heads={"signal": heads},
        )
        return GlobalVae(
            encoder_configs={"signal": SIGNAL_ENCODER},
            decoder_configs={"signal": SIGNAL_DECODER},
            routing_graph=graph,
            encoder_kwargs={"signal": {"latent_dim": 4, "hidden_channels": (8, 16)}},
            decoder_kwargs={"signal": _decoderKwargs(SIGNAL_DECODER, 8)},
        )

    @pytest.mark.parametrize(
        "heads",
        [
            {"z_a": HeadSpec("linear"), "z_b": HeadSpec("linear")},
            {"z_b": HeadSpec("linear")},
        ],
        ids=["two-linear", "identity-and-linear"],
    )
    def test_the_two_posteriors_differ(self, heads: dict[str, HeadSpec]) -> None:
        """Both spaces have the same width as the encoder, and still different parameters."""
        torch.manual_seed(0)
        model = self._trunkModel(heads)
        params = model({"signal": _batches(1)[0]["signal"]})["latent_params"]
        assert not torch.allclose(params["z_a"][0], params["z_b"][0])
        assert not torch.allclose(params["z_a"][1], params["z_b"][1])

    def test_without_heads_the_graph_is_refused_rather_than_built_with_equal_posteriors(
        self,
    ) -> None:
        """The failure the old guard prevented is still impossible to build."""
        with pytest.raises(ValueError, match="identical posteriors"):
            self._trunkModel({})

    def test_a_single_encoder_trains_two_latent_spaces(self) -> None:
        """One encoder, two latent spaces, one decoder over both: loss falls, all params learn."""
        torch.manual_seed(0)
        model = self._trunkModel({"z_b": HeadSpec("linear")})
        data: list[Batch] = [{"signal": b["signal"]} for b in _batches(4)]
        history = Trainer(model, device="cpu", optimizer_kwargs={"lr": 5e-3}, beta=0.1).fit(
            data, num_epochs=12
        )
        assert history[-1]["train/loss/total"] < history[0]["train/loss/total"]
