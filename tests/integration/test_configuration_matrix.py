"""Configuration matrix, stage 1: the four `EN-*` rows without fan-out (spec §2.1, §10).

Spec §10 asks for an integration test that builds each of the 8 architecture combinations on dummy
tensors and checks shapes and gradients. Roadmap P1-6 delivers it in three stages; this file is
stage 1, delivered by P0-4(d). It covers the rows that `GlobalVae` can build today, with the real
modules of the two modalities the framework ships (`1d_cnn_*` for a signal, `2d_cnn_*` for an
image) instead of the linear dummies of `test_en_l1_dn_default.py`:

| Row | Encoders | Latent spaces | Decoders |
|---|---|---|---|
| `EN-L1-DN` | signal, image | one, fused with PoE | signal, image |
| `EN-L1-D1` | signal, image | one, fused with PoE | one (`image_out`) |
| `EN-LN-DN` | signal, image | two (`z_signal`, `z_image`) | signal, image, each over both latents |
| `EN-LN-D1` | signal, image | two (`z_signal`, `z_image`) | one (`image_out`) over both latents |

The two `LN` rows run once per assembler (`concat`, `sum`, `average`), which is what puts the
assembler call in `GlobalVae.forward` under test. In none of the rows does an encoder feed more
than one latent space, so no fan-out is involved (that is stage 2, after roadmap P1-2). The `E1-*`
rows (stage 3) need a shared encoder and decoder, which waits on roadmap P1-5.

A single shared decoder (`D1`) produces one tensor, so with two modalities of different shapes the
`D1` rows reconstruct one of them: the image, from both encoders, as `examples/03` does. Its name
(`image_out`) differs from every encoder name, so the batch also carries a decoder-only target,
which `GlobalVae.selectEncoderInputs` filters out of the encoder inputs (ADR 0019).

For each case the tests check: the graph has no fan-out, the model has the expected latent spaces,
fusions and assemblers, the reconstruction and latent shapes, that each assembler received the
latents in graph order and returned the right combination, a finite loss, and a finite gradient on
every parameter.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
import global_vae.fusion  # noqa: F401  (registers the built-in fusion strategies)
from global_vae.assemblers.registry import listRegisteredAssemblers
from global_vae.latent.base import LatentSpace, RoutingGraph
from global_vae.latent.routing_graph_builders.single import buildSingleLatentRoutingGraph
from global_vae.losses.reconstruction import computeTotalReconstructionLoss
from global_vae.models.global_vae import GlobalVae

BATCH_SIZE = 4
SIGNAL_LENGTH = 32
IMAGE_SHAPE = (32, 32)
FUSED_LATENT_DIM = 8

SIGNAL_ENCODER = "1d_cnn_encoder_v1"
SIGNAL_DECODER = "1d_cnn_decoder_v1"
IMAGE_ENCODER = "2d_cnn_encoder_v1"
IMAGE_DECODER = "2d_cnn_decoder_v1"

ASSEMBLERS = ("concat", "sum", "average")
STAGE_1_ROWS = frozenset({"EN-L1-DN", "EN-L1-D1", "EN-LN-DN", "EN-LN-D1"})


def _encoderKwargs(latent_dim: int) -> dict[str, Any]:
    """Small encoder arguments, shared by the 1D and 2D encoders, to keep the test fast."""
    return {"latent_dim": latent_dim, "hidden_channels": (8, 16)}


def _decoderKwargs(registry_name: str, latent_dim: int) -> dict[str, Any]:
    """Small decoder arguments that reach the modality's size exactly.

    Args:
        registry_name: `SIGNAL_DECODER` or `IMAGE_DECODER`.
        latent_dim: Dimensionality of the tensor the decoder receives.

    Returns:
        Constructor keyword arguments for that decoder.
    """
    common: dict[str, Any] = {
        "latent_dim": latent_dim,
        "upsample_modes": "conv_transpose",
        "hidden_channels": (16, 8),
    }
    if registry_name == SIGNAL_DECODER:
        return {**common, "output_length": SIGNAL_LENGTH}
    return {**common, "output_shape": IMAGE_SHAPE, "seed_shape": (8, 8)}


def _latentDims(assembler: str) -> tuple[int, int, int]:
    """Dimensions for the two latent spaces of an `LN` row, and for the assembled tensor.

    `concat` has no dimension restriction, so it gets two different sizes, which makes a wrong
    input order or a wrong decoder width visible. `sum` and `average` need equal sizes.

    Args:
        assembler: Assembler name.

    Returns:
        `(signal latent dim, image latent dim, decoder input dim)`.
    """
    if assembler == "concat":
        return 6, 10, 16
    return 8, 8, 8


def _buildEnL1Dn(assembler: str | None) -> GlobalVae:
    """`EN-L1-DN`: per-modality encoders and decoders around one fused latent space."""
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {"encoder": SIGNAL_ENCODER, "decoder": SIGNAL_DECODER},
            "image": {"encoder": IMAGE_ENCODER, "decoder": IMAGE_DECODER},
        },
        latent_dim=FUSED_LATENT_DIM,
        fusion_strategy="poe",
        encoder_kwargs={
            "signal": _encoderKwargs(FUSED_LATENT_DIM),
            "image": _encoderKwargs(FUSED_LATENT_DIM),
        },
        decoder_kwargs={
            "signal": _decoderKwargs(SIGNAL_DECODER, FUSED_LATENT_DIM),
            "image": _decoderKwargs(IMAGE_DECODER, FUSED_LATENT_DIM),
        },
    )


def _buildEnL1D1(assembler: str | None) -> GlobalVae:
    """`EN-L1-D1`: both encoders fused into one latent space, one shared image decoder."""
    graph = buildSingleLatentRoutingGraph(
        encoder_names=["signal", "image"],
        decoder_names=["image_out"],
        latent_dim=FUSED_LATENT_DIM,
        latent_name="z_fused",
    )
    return GlobalVae(
        encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
        decoder_configs={"image_out": IMAGE_DECODER},
        routing_graph=graph,
        fusion_strategies={"z_fused": "poe"},
        encoder_kwargs={
            "signal": _encoderKwargs(FUSED_LATENT_DIM),
            "image": _encoderKwargs(FUSED_LATENT_DIM),
        },
        decoder_kwargs={"image_out": _decoderKwargs(IMAGE_DECODER, FUSED_LATENT_DIM)},
    )


def _buildMultiLatent(decoders: dict[str, str], assembler: str) -> GlobalVae:
    """Two encoders, each feeding its own latent space, and decoders over both.

    Args:
        decoders: Decoder name -> registry name. Every decoder consumes both latent spaces.
        assembler: Assembler every decoder uses to combine the two latents.

    Returns:
        The model, with no fan-out and no fusion.
    """
    signal_dim, image_dim, decoder_dim = _latentDims(assembler)
    decoder_names = list(decoders)
    graph = RoutingGraph(
        latent_specs={
            "z_signal": LatentSpace("z_signal", signal_dim),
            "z_image": LatentSpace("z_image", image_dim),
        },
        encoder_to_latents={"signal": ["z_signal"], "image": ["z_image"]},
        latent_to_decoders={"z_signal": list(decoder_names), "z_image": list(decoder_names)},
        decoder_assemblers=dict.fromkeys(decoder_names, assembler),
    )
    return GlobalVae(
        encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
        decoder_configs=decoders,
        routing_graph=graph,
        encoder_kwargs={
            "signal": _encoderKwargs(signal_dim),
            "image": _encoderKwargs(image_dim),
        },
        decoder_kwargs={
            name: _decoderKwargs(registry_name, decoder_dim)
            for name, registry_name in decoders.items()
        },
    )


def _buildEnLnDn(assembler: str | None) -> GlobalVae:
    """`EN-LN-DN`: one latent space per encoder, a signal and an image decoder over both."""
    assert assembler is not None
    return _buildMultiLatent({"signal": SIGNAL_DECODER, "image": IMAGE_DECODER}, assembler)


def _buildEnLnD1(assembler: str | None) -> GlobalVae:
    """`EN-LN-D1`: one latent space per encoder, one shared image decoder over both."""
    assert assembler is not None
    return _buildMultiLatent({"image_out": IMAGE_DECODER}, assembler)


@dataclass(frozen=True)
class _Row:
    """What differs between the rows of the matrix.

    Attributes:
        build: Builds the model; the argument is the assembler name, or `None` for an `L1` row.
        reconstructions: Decoder name -> reconstruction shape without the batch axis.
        latents: Latent space names the model must have.
        fused: Latent spaces that must carry a fusion module.
        assembled_decoders: Decoders that must carry an assembler.
    """

    build: Callable[[str | None], GlobalVae]
    reconstructions: dict[str, tuple[int, ...]]
    latents: frozenset[str]
    fused: frozenset[str]
    assembled_decoders: frozenset[str]


_ROWS: dict[str, _Row] = {
    "EN-L1-DN": _Row(
        build=_buildEnL1Dn,
        reconstructions={"signal": (SIGNAL_LENGTH,), "image": IMAGE_SHAPE},
        latents=frozenset({"z_fused"}),
        fused=frozenset({"z_fused"}),
        assembled_decoders=frozenset(),
    ),
    "EN-L1-D1": _Row(
        build=_buildEnL1D1,
        reconstructions={"image_out": IMAGE_SHAPE},
        latents=frozenset({"z_fused"}),
        fused=frozenset({"z_fused"}),
        assembled_decoders=frozenset(),
    ),
    "EN-LN-DN": _Row(
        build=_buildEnLnDn,
        reconstructions={"signal": (SIGNAL_LENGTH,), "image": IMAGE_SHAPE},
        latents=frozenset({"z_signal", "z_image"}),
        fused=frozenset(),
        assembled_decoders=frozenset({"signal", "image"}),
    ),
    "EN-LN-D1": _Row(
        build=_buildEnLnD1,
        reconstructions={"image_out": IMAGE_SHAPE},
        latents=frozenset({"z_signal", "z_image"}),
        fused=frozenset(),
        assembled_decoders=frozenset({"image_out"}),
    ),
}

STAGE_1_CASES = [
    pytest.param("EN-L1-DN", None, id="EN-L1-DN"),
    pytest.param("EN-L1-D1", None, id="EN-L1-D1"),
    *(pytest.param("EN-LN-DN", name, id=f"EN-LN-DN-{name}") for name in ASSEMBLERS),
    *(pytest.param("EN-LN-D1", name, id=f"EN-LN-D1-{name}") for name in ASSEMBLERS),
]


@dataclass
class _Run:
    """One built model, one forward pass, and what the assemblers saw during it.

    Attributes:
        row: The row under test.
        assembler: The assembler name, or `None` for an `L1` row.
        model: The model.
        batch: Per-modality batch, including any decoder-only target.
        outputs: What `GlobalVae.forward` returned.
        assembler_calls: Decoder name -> `(latents in, tensor out)` of each assembler call.
    """

    row: _Row
    assembler: str | None
    model: GlobalVae
    batch: dict[str, torch.Tensor]
    outputs: dict[str, Any]
    assembler_calls: dict[str, list[tuple[list[torch.Tensor], torch.Tensor]]]


def _combine(assembler: str, latents: list[torch.Tensor]) -> torch.Tensor:
    """What each assembler is specified to return, written out without the assembler classes."""
    if assembler == "concat":
        return torch.cat(latents, dim=-1)
    stacked = torch.stack(latents, dim=0)
    return stacked.sum(dim=0) if assembler == "sum" else stacked.mean(dim=0)


@pytest.fixture
def run(row_code: str, assembler: str | None) -> _Run:
    """Build the model for one case and run one training-mode forward pass over a fixed batch.

    Args:
        row_code: Key of `_ROWS`.
        assembler: Assembler name, or `None` for an `L1` row.

    Returns:
        The model, batch, outputs and recorded assembler calls.
    """
    torch.manual_seed(0)
    row = _ROWS[row_code]
    model = row.build(assembler)

    batch = {
        "signal": torch.randn(BATCH_SIZE, SIGNAL_LENGTH),
        "image": torch.randn(BATCH_SIZE, *IMAGE_SHAPE),
    }
    for decoder_name, shape in row.reconstructions.items():
        batch.setdefault(decoder_name, torch.randn(BATCH_SIZE, *shape))

    calls: dict[str, list[tuple[list[torch.Tensor], torch.Tensor]]] = {}
    for decoder_name, module in model.assemblers.items():
        calls[decoder_name] = []

        def record(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            output: torch.Tensor,
            name: str = decoder_name,
        ) -> None:
            calls[name].append((list(args[0]), output))

        module.register_forward_hook(record)

    outputs = model(model.selectEncoderInputs(batch))
    return _Run(row, assembler, model, batch, outputs, calls)


def _totalLoss(run: _Run) -> torch.Tensor:
    """Reconstruction plus regularization loss for the forward pass held by `run`."""
    reconstruction = computeTotalReconstructionLoss(run.outputs["reconstructions"], run.batch)
    regularization = run.model.computeRegularizationLoss(run.outputs["latent_params"])
    return reconstruction + regularization


def test_stage_1_lists_every_en_row_without_fan_out() -> None:
    """Dropping a row from the matrix, or an assembler from an `LN` row, fails this test."""
    assert set(_ROWS) == STAGE_1_ROWS
    assert {case.values[0] for case in STAGE_1_CASES} == STAGE_1_ROWS
    for code in ("EN-LN-DN", "EN-LN-D1"):
        covered = {case.values[1] for case in STAGE_1_CASES if case.values[0] == code}
        assert covered == set(ASSEMBLERS), code
    for code in ("EN-L1-DN", "EN-L1-D1"):
        covered = {case.values[1] for case in STAGE_1_CASES if case.values[0] == code}
        assert covered == {None}, code
    assert set(ASSEMBLERS) <= set(listRegisteredAssemblers())


@pytest.mark.parametrize(("row_code", "assembler"), STAGE_1_CASES)
class TestStage1Rows:
    """The checks every stage-1 case must pass."""

    def test_no_encoder_feeds_more_than_one_latent_space(self, run: _Run) -> None:
        """Stage 1 is the fan-out-free half of the matrix."""
        for encoder, latents in run.model.routing_graph.encoder_to_latents.items():
            assert len(latents) == 1, encoder

    def test_model_has_the_expected_latents_fusions_and_assemblers(self, run: _Run) -> None:
        """The row's topology shows up in the built model."""
        assert set(run.model.latent_spaces) == run.row.latents
        assert set(run.model.fusions) == run.row.fused
        assert set(run.model.assemblers) == run.row.assembled_decoders
        assert set(run.model.encoders) == {"signal", "image"}
        assert set(run.model.decoders) == set(run.row.reconstructions)

    def test_reconstruction_shapes(self, run: _Run) -> None:
        """Each decoder returns a batch of its modality, and no decoder is missing."""
        reconstructions = run.outputs["reconstructions"]
        assert set(reconstructions) == set(run.row.reconstructions)
        for name, shape in run.row.reconstructions.items():
            assert reconstructions[name].shape == (BATCH_SIZE, *shape), name

    def test_latent_parameter_and_sample_shapes(self, run: _Run) -> None:
        """Every latent space reports `(mu, logvar)` and a sample of its own dimension."""
        params, samples = run.outputs["latent_params"], run.outputs["latent_samples"]
        assert set(params) == run.row.latents
        assert set(samples) == run.row.latents
        for name, latent in run.model.latent_spaces.items():
            mu, logvar = params[name]
            assert mu.shape == (BATCH_SIZE, latent.dim), name
            assert logvar.shape == (BATCH_SIZE, latent.dim), name
            assert samples[name].shape == (BATCH_SIZE, latent.dim), name

    def test_assemblers_receive_the_latents_in_graph_order(self, run: _Run) -> None:
        """Each assembler is called once with the sampled latents and returns their combination."""
        if run.assembler is None:
            assert run.assembler_calls == {}
            return
        samples = run.outputs["latent_samples"]
        _, _, decoder_dim = _latentDims(run.assembler)
        for decoder_name in run.row.assembled_decoders:
            calls = run.assembler_calls[decoder_name]
            assert len(calls) == 1, decoder_name
            latents_in, assembled = calls[0]
            assert len(latents_in) == 2
            assert torch.equal(latents_in[0], samples["z_signal"])
            assert torch.equal(latents_in[1], samples["z_image"])
            assert assembled.shape == (BATCH_SIZE, decoder_dim)
            assert torch.equal(assembled, _combine(run.assembler, latents_in))

    def test_loss_is_a_finite_scalar(self, run: _Run) -> None:
        """Reconstruction plus regularization is one finite number."""
        loss = _totalLoss(run)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_every_parameter_receives_a_finite_gradient(self, run: _Run) -> None:
        """Backward through the loss reaches every parameter of every module."""
        run.model.zero_grad()
        _totalLoss(run).backward()
        parameters = dict(run.model.named_parameters())
        assert parameters
        for name, parameter in parameters.items():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
