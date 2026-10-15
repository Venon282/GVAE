"""Configuration matrix: all eight rows of spec §2.1 in one test (spec §10, roadmap P1-6).

Spec §10 asks for an integration test that builds each of the 8 architecture combinations on dummy
tensors and checks shapes and gradients. Roadmap P1-6 delivers it in three stages, all in this
file, and `MATRIX` (at the end) is the explicit table of the eight row codes with a builder for
each. `TestEveryRow` is the one parametrized test over it: for every row, and for every registered
assembler on an `LN` row, it checks the reconstruction shapes, the shapes of every latent space, a
finite loss, a finite gradient on every parameter, and that a few optimizer steps lower the loss.
`test_the_eight_rows_are_all_in_the_matrix` fails when a row, or an assembler of an `LN` row, is
dropped. The per-stage classes below keep what is specific to a topology.

Stage 1, delivered by P0-4(d), covers the four `EN-*` rows, with the real modules of the two
modalities the framework ships (`1d_cnn_*` for a signal, `2d_cnn_*` for an image) instead of the
linear dummies of `test_en_l1_dn_default.py`. Stage 2 adds the shared plus private variants of the
`EN-LN-*` rows (encoder fan-out, ADR 0024). Stage 3, delivered by P1-5 (ADR 0027), covers the four
`E1-*` rows, which are built from a `ModelConfig` and use the dummy shared encoder and decoder of
`_shared_modules.py`. The stage 1 rows:

| Row | Encoders | Latent spaces | Decoders |
|---|---|---|---|
| `EN-L1-DN` | signal, image | one, fused with PoE | signal, image |
| `EN-L1-D1` | signal, image | one, fused with PoE | one (`image_out`) |
| `EN-LN-DN` | signal, image | two (`z_signal`, `z_image`) | signal, image, each over both latents |
| `EN-LN-D1` | signal, image | two (`z_signal`, `z_image`) | one (`image_out`) over both latents |

The two `LN` rows run once per assembler (`concat`, `sum`, `average`, `weighted_sum`,
`attention`: every registered one, roadmap P1-4), which is what puts the assembler call in
`GlobalVae.forward` under test. In none of the stage 1 rows does an encoder feed more than one
latent space, so no fan-out is involved.

A single shared decoder (`D1`) produces one tensor, so with two modalities of different shapes the
`D1` rows of stages 1 and 2 reconstruct one of them: the image, from both encoders, as `examples/03`
does. Its name (`image_out`) differs from every encoder name, so the batch also carries a
decoder-only target, which `GlobalVae.selectEncoderInputs` filters out of the encoder inputs
(ADR 0019).

The stage 1 topology tests check that the graph has no fan-out, that the model has the expected
latent spaces, fusions and assemblers, and that each assembler received the latents in graph order
and returned the right combination (the value of the learned assemblers is checked in
`tests/unit/test_learned_assemblers.py`; here `weighted_sum` must start as the average, and
`attention` must return the width its decoder was built for).

The stage 2 variants keep the two `EN-LN-*` rows and let each encoder feed two latent spaces: a
`z_shared` that both encoders feed through an identity edge and fuse with PoE, and a
`z_private_<modality>` behind a `linear` head. Their decoders are sized from their assembler:

* `EN-LN-DN+shared-private`: `signal` over `z_shared` and `z_private_signal`, `image` over
  `z_shared` and `z_private_image`.
* `EN-LN-D1+shared-private`: `image_out` over `z_shared`, `z_private_signal` and
  `z_private_image`.

The stage 3 rows have one encoder, named `backbone`, that ingests both modalities, so none of them
has a fusion. A shared decoder (`shared_out`) reconstructs both modalities, so the `D1` rows here
reconstruct the signal and the image, unlike the `D1` rows of stages 1 and 2:

| Row | Encoders | Latent spaces | Decoders |
|---|---|---|---|
| `E1-L1-DN` | `backbone` (signal, image) | one (`z_fused`) | signal, image |
| `E1-L1-D1` | `backbone` | one (`z_fused`) | `shared_out` (signal, image) |
| `E1-LN-DN` | `backbone` | two (`z_a`, `z_b`), `z_b` behind a head | signal, image, both latents |
| `E1-LN-D1` | `backbone` | two (`z_a`, `z_b`) | `shared_out` over both |

The stage 3 topology tests also check that the shared encoder receives both modalities in a single
call, that the modality names of the model are those of the shared modules, and that the fan-out of
the `LN` rows goes through a head (the behaviour of modality dropout, the reports and the guards is
in `test_shared_encoder_decoder.py`).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
import global_vae.fusion  # noqa: F401  (registers the built-in fusion strategies)
from global_vae.assemblers.registry import listRegisteredAssemblers
from global_vae.config.model import (
    DecoderConfig,
    DecoderConsumptionConfig,
    LatentConfig,
    LatentSpaceConfig,
    ModelConfig,
    SingleLatentConfig,
    buildModelFromConfig,
)
from global_vae.latent.base import HeadSpec, LatentSpace, RoutingGraph
from global_vae.latent.routing_graph_builders.shared_private import (
    buildSharedPrivateRoutingGraph,
)
from global_vae.latent.routing_graph_builders.single import buildSingleLatentRoutingGraph
from global_vae.losses.reconstruction import computeTotalReconstructionLoss
from global_vae.models.global_vae import GlobalVae
from global_vae.utils.autograd import backward
from tests.integration._shared_modules import sharedDecoderConfig, sharedEncoderConfig

BATCH_SIZE = 4
SIGNAL_LENGTH = 32
IMAGE_SHAPE = (32, 32)
FUSED_LATENT_DIM = 8

SIGNAL_ENCODER = "1d_cnn_encoder_v1"
SIGNAL_DECODER = "1d_cnn_decoder_v1"
IMAGE_ENCODER = "2d_cnn_encoder_v1"
IMAGE_DECODER = "2d_cnn_decoder_v1"

ASSEMBLERS = ("concat", "sum", "average", "weighted_sum", "attention")
STAGE_1_ROWS = frozenset({"EN-L1-DN", "EN-L1-D1", "EN-LN-DN", "EN-LN-D1"})
STAGE_3_ROWS = frozenset({"E1-L1-DN", "E1-L1-D1", "E1-LN-DN", "E1-LN-D1"})
EIGHT_ROWS = frozenset(
    {
        "E1-L1-D1",
        "E1-L1-DN",
        "E1-LN-D1",
        "E1-LN-DN",
        "EN-L1-D1",
        "EN-L1-DN",
        "EN-LN-D1",
        "EN-LN-DN",
    }
)
BACKBONE = "backbone"
SHARED_OUT = "shared_out"


def _encoderKwargs(latent_dim: int) -> dict[str, Any]:
    """Small encoder arguments, shared by the 1D and 2D encoders, to keep the test fast."""
    return {"latent_dim": latent_dim, "hidden_channels": (8, 16)}


def _decoderKwargs(registry_name: str, latent_dim: int | None) -> dict[str, Any]:
    """Small decoder arguments that reach the modality's size exactly.

    Args:
        registry_name: `SIGNAL_DECODER` or `IMAGE_DECODER`.
        latent_dim: Dimensionality of the tensor the decoder receives, or `None` to leave it
            out, in which case the model sizes the decoder from its assembler.

    Returns:
        Constructor keyword arguments for that decoder.
    """
    common: dict[str, Any] = {
        "upsample_modes": "conv_transpose",
        "hidden_channels": (16, 8),
    }
    if latent_dim is not None:
        common["latent_dim"] = latent_dim
    if registry_name == SIGNAL_DECODER:
        return {**common, "output_length": SIGNAL_LENGTH}
    return {**common, "output_shape": IMAGE_SHAPE, "seed_shape": (8, 8)}


def _latentDims(assembler: str) -> tuple[int, int, int]:
    """Dimensions for the two latent spaces of an `LN` row, and for the assembled tensor.

    `concat` and `attention` have no dimension restriction, so they get two different sizes, which
    makes a wrong input order or a wrong decoder width visible. `attention` is given no `output_dim`
    here, so it takes the default, the widest input. `sum`, `average` and `weighted_sum` need equal
    sizes.

    Args:
        assembler: Assembler name.

    Returns:
        `(signal latent dim, image latent dim, decoder input dim)`.
    """
    if assembler == "concat":
        return 6, 10, 16
    if assembler == "attention":
        return 6, 10, 10
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
        assembler_inputs: Decoder name -> the latent spaces its assembler must receive, in
            order. Empty for the rows whose own tests spell the order out.
    """

    build: Callable[[str | None], GlobalVae]
    reconstructions: dict[str, tuple[int, ...]]
    latents: frozenset[str]
    fused: frozenset[str]
    assembled_decoders: frozenset[str]
    assembler_inputs: dict[str, tuple[str, ...]] = field(default_factory=dict)


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


def _combine(assembler: str, latents: list[torch.Tensor]) -> torch.Tensor | None:
    """What each assembler is specified to return, written out without the assembler classes.

    Args:
        assembler: Assembler name.
        latents: The latents the assembler received.

    Returns:
        The expected output, or `None` for `attention`, whose value depends on random parameters
        (its output width is checked instead). `weighted_sum` starts with equal weights, so it
        returns the mean until it is trained.
    """
    if assembler == "attention":
        return None
    if assembler == "concat":
        return torch.cat(latents, dim=-1)
    stacked = torch.stack(latents, dim=0)
    return stacked.sum(dim=0) if assembler == "sum" else stacked.mean(dim=0)


@pytest.fixture
def run(row_code: str, assembler: str | None) -> _Run:
    """Build the model for one case and run one training-mode forward pass over a fixed batch.

    Args:
        row_code: Key of `_ROWS` or `_STAGE_2_ROWS`.
        assembler: Assembler name, or `None` for an `L1` row.

    Returns:
        The model, batch, outputs and recorded assembler calls.
    """
    torch.manual_seed(0)
    row = {**_ROWS, **_STAGE_2_ROWS}[row_code]
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
    assert set(ASSEMBLERS) == set(listRegisteredAssemblers())


@pytest.mark.parametrize(("row_code", "assembler"), STAGE_1_CASES)
class TestStage1Rows:
    """The topology checks of the stage 1 cases (shapes, loss and gradients: `TestEveryRow`)."""

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
            assert assembled.shape[-1] == run.model.assemblers[decoder_name].output_dim
            expected = _combine(run.assembler, latents_in)
            if expected is not None:
                assert torch.allclose(assembled, expected, atol=1e-6)


# ----------------------------------------------------------------------------------------------
# Stage 2: the shared plus private variants of the `EN-LN-*` rows (roadmap P1-6, ADR 0024).
# ----------------------------------------------------------------------------------------------

SHARED_PRIVATE = "+shared-private"
PRIVATE_LATENTS = ("z_private_signal", "z_private_image")


def _sharedPrivateDims(assembler: str) -> tuple[int, int]:
    """Widths of `z_shared` and of each private latent space for one assembler.

    `concat` and `attention` get two different widths, so a wrong input order or a decoder of the
    wrong width is visible. `sum`, `average` and `weighted_sum` need equal widths.

    Args:
        assembler: Assembler name.

    Returns:
        `(shared dim, private dim)`.
    """
    if assembler in ("concat", "attention"):
        return 6, 10
    return 8, 8


def _buildEnLnDnSharedPrivate(assembler: str | None) -> GlobalVae:
    """`EN-LN-DN` with fan-out: a shared and a private latent space, a decoder per modality.

    Built from the `buildSharedPrivateRoutingGraph` preset. Each decoder consumes `z_shared` and
    the private latent space of its own modality. The decoders are not given a `latent_dim`: the
    model sizes them from their assembler.
    """
    assert assembler is not None
    shared_dim, private_dim = _sharedPrivateDims(assembler)
    graph = buildSharedPrivateRoutingGraph(
        modality_names=["signal", "image"],
        shared_dim=shared_dim,
        private_dim=private_dim,
        assembler=assembler,
    )
    return GlobalVae(
        encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
        decoder_configs={"signal": SIGNAL_DECODER, "image": IMAGE_DECODER},
        routing_graph=graph,
        fusion_strategies={"z_shared": "poe"},
        encoder_kwargs={
            "signal": _encoderKwargs(shared_dim),
            "image": _encoderKwargs(shared_dim),
        },
        decoder_kwargs={
            "signal": _decoderKwargs(SIGNAL_DECODER, None),
            "image": _decoderKwargs(IMAGE_DECODER, None),
        },
    )


def _buildEnLnD1SharedPrivate(assembler: str | None) -> GlobalVae:
    """`EN-LN-D1` with fan-out: one image decoder over the shared and both private spaces.

    No preset builds this graph, so it is written out. The decoder is not given a `latent_dim`:
    the model sizes it from its assembler.
    """
    assert assembler is not None
    shared_dim, private_dim = _sharedPrivateDims(assembler)
    graph = RoutingGraph(
        latent_specs={
            "z_shared": LatentSpace("z_shared", shared_dim),
            "z_private_signal": LatentSpace("z_private_signal", private_dim),
            "z_private_image": LatentSpace("z_private_image", private_dim),
        },
        encoder_to_latents={
            "signal": ["z_shared", "z_private_signal"],
            "image": ["z_shared", "z_private_image"],
        },
        latent_to_decoders={
            "z_shared": ["image_out"],
            "z_private_signal": ["image_out"],
            "z_private_image": ["image_out"],
        },
        decoder_assemblers={"image_out": assembler},
        encoder_heads={
            "signal": {"z_shared": HeadSpec(), "z_private_signal": HeadSpec("linear")},
            "image": {"z_shared": HeadSpec(), "z_private_image": HeadSpec("linear")},
        },
    )
    return GlobalVae(
        encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
        decoder_configs={"image_out": IMAGE_DECODER},
        routing_graph=graph,
        fusion_strategies={"z_shared": "poe"},
        encoder_kwargs={
            "signal": _encoderKwargs(shared_dim),
            "image": _encoderKwargs(shared_dim),
        },
        decoder_kwargs={"image_out": _decoderKwargs(IMAGE_DECODER, None)},
    )


_SHARED_PRIVATE_LATENTS = frozenset({"z_shared", *PRIVATE_LATENTS})

_STAGE_2_ROWS: dict[str, _Row] = {
    f"EN-LN-DN{SHARED_PRIVATE}": _Row(
        build=_buildEnLnDnSharedPrivate,
        reconstructions={"signal": (SIGNAL_LENGTH,), "image": IMAGE_SHAPE},
        latents=_SHARED_PRIVATE_LATENTS,
        fused=frozenset({"z_shared"}),
        assembled_decoders=frozenset({"signal", "image"}),
        assembler_inputs={
            "signal": ("z_shared", "z_private_signal"),
            "image": ("z_shared", "z_private_image"),
        },
    ),
    f"EN-LN-D1{SHARED_PRIVATE}": _Row(
        build=_buildEnLnD1SharedPrivate,
        reconstructions={"image_out": IMAGE_SHAPE},
        latents=_SHARED_PRIVATE_LATENTS,
        fused=frozenset({"z_shared"}),
        assembled_decoders=frozenset({"image_out"}),
        assembler_inputs={"image_out": ("z_shared", *PRIVATE_LATENTS)},
    ),
}

STAGE_2_CASES = [
    pytest.param(code, name, id=f"{code}-{name}") for code in _STAGE_2_ROWS for name in ASSEMBLERS
]


def test_stage_2_lists_both_shared_private_variants() -> None:
    """Dropping a variant from the matrix, or an assembler from one, fails this test."""
    assert set(_STAGE_2_ROWS) == {f"EN-LN-DN{SHARED_PRIVATE}", f"EN-LN-D1{SHARED_PRIVATE}"}
    assert {case.values[0] for case in STAGE_2_CASES} == set(_STAGE_2_ROWS)
    for code in _STAGE_2_ROWS:
        covered = {case.values[1] for case in STAGE_2_CASES if case.values[0] == code}
        assert covered == set(ASSEMBLERS), code


@pytest.mark.parametrize(("row_code", "assembler"), STAGE_2_CASES)
class TestStage2Rows:
    """The topology checks of the shared plus private variants (the rest: `TestEveryRow`)."""

    def test_the_model_has_a_shared_space_and_a_private_space_per_modality(self, run: _Run) -> None:
        """Three latent spaces, only `z_shared` fused, and the decoders the row names."""
        model = run.model
        assert set(model.latent_spaces) == run.row.latents
        assert set(model.fusions) == run.row.fused
        assert set(model.assemblers) == run.row.assembled_decoders
        assert set(model.encoders) == {"signal", "image"}
        assert set(model.decoders) == set(run.row.reconstructions)

    def test_each_encoder_feeds_two_latent_spaces_through_one_head(self, run: _Run) -> None:
        """The shared edge is an identity edge, the private edge a `linear` head (ADR 0024)."""
        model = run.model
        for modality in ("signal", "image"):
            private = f"z_private_{modality}"
            edges = model.routing_graph.encoder_to_latents[modality]
            assert set(edges) == {"z_shared", private}, modality
            heads = model.routing_graph.encoder_heads[modality]
            assert heads["z_shared"].name == "identity", modality
            assert heads[private].name == "linear", modality
            assert model.getHead(modality, private) is not None, modality

    def test_a_private_posterior_differs_from_the_shared_one(self, run: _Run) -> None:
        """One encoder output, two posteriors: the head keeps the private one apart."""
        params = run.outputs["latent_params"]
        shared_mu = params["z_shared"][0]
        for private in PRIVATE_LATENTS:
            private_mu = params[private][0]
            width = min(shared_mu.shape[-1], private_mu.shape[-1])
            assert not torch.allclose(shared_mu[:, :width], private_mu[:, :width]), private

    def test_assemblers_receive_the_latents_in_graph_order(self, run: _Run) -> None:
        """Each assembler is called once, with its latent spaces in order, and combines them."""
        assert run.assembler is not None
        samples = run.outputs["latent_samples"]
        assert set(run.row.assembler_inputs) == run.row.assembled_decoders
        for decoder_name, latent_names in run.row.assembler_inputs.items():
            calls = run.assembler_calls[decoder_name]
            assert len(calls) == 1, decoder_name
            latents_in, assembled = calls[0]
            assert len(latents_in) == len(latent_names), decoder_name
            for received, latent_name in zip(latents_in, latent_names, strict=True):
                assert torch.equal(received, samples[latent_name]), (decoder_name, latent_name)
            assert assembled.shape[-1] == run.model.assemblers[decoder_name].output_dim
            expected = _combine(run.assembler, latents_in)
            if expected is not None:
                assert torch.allclose(assembled, expected, atol=1e-6), decoder_name


# ----------------------------------------------------------------------------------------------
# Stage 3: the four `E1-*` rows (roadmap P1-5, ADR 0027), built from a `ModelConfig`.
# ----------------------------------------------------------------------------------------------


def _oneModalityDecoders() -> dict[str, DecoderConfig]:
    """A signal decoder and an image decoder, without `latent_dim` (the config layer fills it)."""
    common: dict[str, Any] = {"upsample_modes": "conv_transpose", "hidden_channels": (16, 8)}
    return {
        "signal": DecoderConfig(
            name=SIGNAL_DECODER, kwargs={**common, "output_length": SIGNAL_LENGTH}
        ),
        "image": DecoderConfig(
            name=IMAGE_DECODER,
            kwargs={**common, "output_shape": IMAGE_SHAPE, "seed_shape": (8, 8)},
        ),
    }


def _singleLatentModel(decoders: dict[str, DecoderConfig]) -> GlobalVae:
    """One shared encoder, one latent space, and the given decoders."""
    return buildModelFromConfig(
        ModelConfig(
            latent_mode="single",
            encoders={BACKBONE: sharedEncoderConfig(_SHARED_RECONSTRUCTIONS)},
            decoders=decoders,
            single_latent=SingleLatentConfig(dim=FUSED_LATENT_DIM),
        )
    )


def _fanOutModel(decoders: dict[str, DecoderConfig], assembler: str) -> GlobalVae:
    """One shared encoder feeding two latent spaces, every decoder consuming both.

    `z_a` takes the encoder's output as it is, `z_b` goes through a linear head, so the two
    posteriors differ (ADR 0024).

    Args:
        decoders: Decoder name -> config.
        assembler: Assembler every decoder uses to combine the two latent spaces.

    Returns:
        The model.
    """
    a_dim, b_dim, _ = _latentDims(assembler)
    return buildModelFromConfig(
        ModelConfig(
            latent_mode="several",
            encoders={BACKBONE: sharedEncoderConfig(_SHARED_RECONSTRUCTIONS)},
            decoders=decoders,
            latent=LatentConfig(
                spaces={
                    "z_a": LatentSpaceConfig(dim=a_dim, fed_by=[BACKBONE]),
                    "z_b": LatentSpaceConfig(dim=b_dim, fed_by={BACKBONE: {"head": "linear"}}),
                },
                decoders_consume={
                    name: DecoderConsumptionConfig(spaces=["z_a", "z_b"], assembler=assembler)
                    for name in decoders
                },
            ),
        )
    )


def _buildE1L1Dn(assembler: str | None) -> GlobalVae:
    """`E1-L1-DN`: a shared trunk, one latent space, a decoder per modality."""
    return _singleLatentModel(_oneModalityDecoders())


def _buildE1L1D1(assembler: str | None) -> GlobalVae:
    """`E1-L1-D1`: a shared encoder and a shared decoder around one latent space."""
    return _singleLatentModel({SHARED_OUT: sharedDecoderConfig(_SHARED_RECONSTRUCTIONS)})


def _buildE1LnDn(assembler: str | None) -> GlobalVae:
    """`E1-LN-DN`: one backbone fanning out to two latent spaces, a decoder per modality."""
    assert assembler is not None
    return _fanOutModel(_oneModalityDecoders(), assembler)


def _buildE1LnD1(assembler: str | None) -> GlobalVae:
    """`E1-LN-D1`: one backbone fanning out to two latent spaces, one shared decoder."""
    assert assembler is not None
    return _fanOutModel({SHARED_OUT: sharedDecoderConfig(_SHARED_RECONSTRUCTIONS)}, assembler)


@dataclass(frozen=True)
class _SharedRow:
    """What differs between the `E1-*` rows.

    Attributes:
        build: Builds the model; the argument is the assembler name, or `None` for an `L1` row.
        decoders: Decoder instance names the model must have.
        latents: Latent space names the model must have.
        assembled_decoders: Decoders that must carry an assembler.
    """

    build: Callable[[str | None], GlobalVae]
    decoders: frozenset[str]
    latents: frozenset[str]
    assembled_decoders: frozenset[str]


_SHARED_RECONSTRUCTIONS: dict[str, tuple[int, ...]] = {
    "signal": (SIGNAL_LENGTH,),
    "image": IMAGE_SHAPE,
}

_SHARED_ROWS: dict[str, _SharedRow] = {
    "E1-L1-DN": _SharedRow(
        build=_buildE1L1Dn,
        decoders=frozenset({"signal", "image"}),
        latents=frozenset({"z_fused"}),
        assembled_decoders=frozenset(),
    ),
    "E1-L1-D1": _SharedRow(
        build=_buildE1L1D1,
        decoders=frozenset({SHARED_OUT}),
        latents=frozenset({"z_fused"}),
        assembled_decoders=frozenset(),
    ),
    "E1-LN-DN": _SharedRow(
        build=_buildE1LnDn,
        decoders=frozenset({"signal", "image"}),
        latents=frozenset({"z_a", "z_b"}),
        assembled_decoders=frozenset({"signal", "image"}),
    ),
    "E1-LN-D1": _SharedRow(
        build=_buildE1LnD1,
        decoders=frozenset({SHARED_OUT}),
        latents=frozenset({"z_a", "z_b"}),
        assembled_decoders=frozenset({SHARED_OUT}),
    ),
}

STAGE_3_CASES = [
    pytest.param("E1-L1-DN", None, id="E1-L1-DN"),
    pytest.param("E1-L1-D1", None, id="E1-L1-D1"),
    *(pytest.param("E1-LN-DN", name, id=f"E1-LN-DN-{name}") for name in ASSEMBLERS),
    *(pytest.param("E1-LN-D1", name, id=f"E1-LN-D1-{name}") for name in ASSEMBLERS),
]


@dataclass
class _SharedRun:
    """One built `E1-*` model, one forward pass, and what the modules saw during it.

    Attributes:
        row: The row under test.
        assembler: The assembler name, or `None` for an `L1` row.
        model: The model.
        batch: Per-modality batch.
        outputs: What `GlobalVae.forward` returned.
        assembler_calls: Decoder name -> `(latents in, tensor out)` of each assembler call.
        encoder_calls: The modality names the shared encoder received, one tuple per call.
    """

    row: _SharedRow
    assembler: str | None
    model: GlobalVae
    batch: dict[str, torch.Tensor]
    outputs: dict[str, Any]
    assembler_calls: dict[str, list[tuple[list[torch.Tensor], torch.Tensor]]]
    encoder_calls: list[tuple[str, ...]]


@pytest.fixture
def shared_run(row_code: str, assembler: str | None) -> _SharedRun:
    """Build the model for one `E1-*` case and run one training-mode forward pass.

    Args:
        row_code: Key of `_SHARED_ROWS`.
        assembler: Assembler name, or `None` for an `L1` row.

    Returns:
        The model, batch, outputs and recorded calls.
    """
    torch.manual_seed(0)
    row = _SHARED_ROWS[row_code]
    model = row.build(assembler)
    batch = {
        "signal": torch.randn(BATCH_SIZE, SIGNAL_LENGTH),
        "image": torch.randn(BATCH_SIZE, *IMAGE_SHAPE),
    }

    assembler_calls: dict[str, list[tuple[list[torch.Tensor], torch.Tensor]]] = {}
    for decoder_name, module in model.assemblers.items():
        assembler_calls[decoder_name] = []

        def record(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            output: torch.Tensor,
            name: str = decoder_name,
        ) -> None:
            assembler_calls[name].append((list(args[0]), output))

        module.register_forward_hook(record)

    encoder_calls: list[tuple[str, ...]] = []

    def recordEncoder(
        _module: torch.nn.Module, args: tuple[Any, ...], _output: tuple[torch.Tensor, ...]
    ) -> None:
        encoder_calls.append(tuple(args[0]))

    model.encoders[BACKBONE].register_forward_hook(recordEncoder)

    outputs = model(model.selectEncoderInputs(batch))
    return _SharedRun(row, assembler, model, batch, outputs, assembler_calls, encoder_calls)


def test_stage_3_lists_every_e1_row() -> None:
    """Dropping a row from the matrix, or an assembler from an `LN` row, fails this test."""
    assert set(_SHARED_ROWS) == STAGE_3_ROWS
    assert {case.values[0] for case in STAGE_3_CASES} == STAGE_3_ROWS
    for code in ("E1-LN-DN", "E1-LN-D1"):
        covered = {case.values[1] for case in STAGE_3_CASES if case.values[0] == code}
        assert covered == set(ASSEMBLERS), code
    for code in ("E1-L1-DN", "E1-L1-D1"):
        covered = {case.values[1] for case in STAGE_3_CASES if case.values[0] == code}
        assert covered == {None}, code


@pytest.mark.parametrize(("row_code", "assembler"), STAGE_3_CASES)
class TestStage3Rows:
    """The topology checks of the `E1-*` cases (shapes, loss and gradients: `TestEveryRow`)."""

    def test_one_shared_encoder_and_the_rows_other_modules(self, shared_run: _SharedRun) -> None:
        """The row's topology shows up in the built model, with no fusion anywhere."""
        model = shared_run.model
        assert set(model.encoders) == {BACKBONE}
        assert set(model.decoders) == shared_run.row.decoders
        assert set(model.latent_spaces) == shared_run.row.latents
        assert set(model.assemblers) == shared_run.row.assembled_decoders
        assert not model.fusions

    def test_the_model_names_its_modalities_after_the_shared_modules(
        self, shared_run: _SharedRun
    ) -> None:
        """The modalities are those the modules declare, not the instance names."""
        assert shared_run.model.input_modalities == ("signal", "image")
        assert shared_run.model.output_modalities == ("signal", "image")

    def test_the_shared_encoder_receives_both_modalities_in_one_call(
        self, shared_run: _SharedRun
    ) -> None:
        """The stream is built once, from the two modalities, in the order the encoder declares."""
        assert shared_run.encoder_calls == [("signal", "image")]

    def test_fan_out_goes_through_a_head_and_l1_has_a_single_edge(
        self, shared_run: _SharedRun
    ) -> None:
        """One backbone, one edge per latent space, and a head on all but one of them."""
        model = shared_run.model
        edges = model.routing_graph.encoder_to_latents[BACKBONE]
        assert set(edges) == shared_run.row.latents
        if len(edges) == 1:
            assert model.getHead(BACKBONE, edges[0]) is None
            return
        assert model.getHead(BACKBONE, "z_a") is None
        assert model.getHead(BACKBONE, "z_b") is not None

    def test_the_two_latent_spaces_of_the_fan_out_rows_differ(self, shared_run: _SharedRun) -> None:
        """One backbone output, two posteriors: the head is what keeps them apart."""
        if len(shared_run.row.latents) == 1:
            return
        params = shared_run.outputs["latent_params"]
        width = min(params["z_a"][0].shape[-1], params["z_b"][0].shape[-1])
        assert not torch.allclose(params["z_a"][0][:, :width], params["z_b"][0][:, :width])

    def test_assemblers_receive_the_latents_in_graph_order(self, shared_run: _SharedRun) -> None:
        """Each assembler is called once with the sampled latents and returns their combination."""
        if shared_run.assembler is None:
            assert shared_run.assembler_calls == {}
            return
        samples = shared_run.outputs["latent_samples"]
        _, _, decoder_dim = _latentDims(shared_run.assembler)
        for decoder_name in shared_run.row.assembled_decoders:
            calls = shared_run.assembler_calls[decoder_name]
            assert len(calls) == 1, decoder_name
            latents_in, assembled = calls[0]
            assert len(latents_in) == 2
            assert torch.equal(latents_in[0], samples["z_a"])
            assert torch.equal(latents_in[1], samples["z_b"])
            assert assembled.shape == (BATCH_SIZE, decoder_dim)
            assert assembled.shape[-1] == shared_run.model.assemblers[decoder_name].output_dim
            expected = _combine(shared_run.assembler, latents_in)
            if expected is not None:
                assert torch.allclose(assembled, expected, atol=1e-6)


# ----------------------------------------------------------------------------------------------
# The matrix: every row, one parametrized test (roadmap P1-6).
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _MatrixRow:
    """A row of the matrix, as `TestEveryRow` needs it.

    Attributes:
        build: Builds the model; the argument is the assembler name, or `None` for an `L1` row.
        reconstructions: Key of `"reconstructions"` -> shape of one sample, without the batch axis.
    """

    build: Callable[[str | None], GlobalVae]
    reconstructions: dict[str, tuple[int, ...]]


_IMAGE_OUT: dict[str, tuple[int, ...]] = {"image_out": IMAGE_SHAPE}

# Spec §2.1 order. A row that is missing here is a row that is no longer tested. The two last
# entries are the stage 2 variants: they share the reconstructions of the `EN-LN-*` row they
# extend, and a `+` in the code marks them.
MATRIX: dict[str, _MatrixRow] = {
    "E1-L1-D1": _MatrixRow(_buildE1L1D1, _SHARED_RECONSTRUCTIONS),
    "E1-L1-DN": _MatrixRow(_buildE1L1Dn, _SHARED_RECONSTRUCTIONS),
    "E1-LN-D1": _MatrixRow(_buildE1LnD1, _SHARED_RECONSTRUCTIONS),
    "E1-LN-DN": _MatrixRow(_buildE1LnDn, _SHARED_RECONSTRUCTIONS),
    "EN-L1-D1": _MatrixRow(_buildEnL1D1, _IMAGE_OUT),
    "EN-L1-DN": _MatrixRow(_buildEnL1Dn, _SHARED_RECONSTRUCTIONS),
    "EN-LN-D1": _MatrixRow(_buildEnLnD1, _IMAGE_OUT),
    "EN-LN-DN": _MatrixRow(_buildEnLnDn, _SHARED_RECONSTRUCTIONS),
    f"EN-LN-D1{SHARED_PRIVATE}": _MatrixRow(_buildEnLnD1SharedPrivate, _IMAGE_OUT),
    f"EN-LN-DN{SHARED_PRIVATE}": _MatrixRow(_buildEnLnDnSharedPrivate, _SHARED_RECONSTRUCTIONS),
}


def _assemblersFor(row_code: str) -> tuple[str | None, ...]:
    """Every registered assembler for an `LN` row, none for an `L1` row."""
    return ASSEMBLERS if "-LN-" in row_code else (None,)


MATRIX_CASES = [
    pytest.param(code, name, id=code if name is None else f"{code}-{name}")
    for code in MATRIX
    for name in _assemblersFor(code)
]


def test_the_eight_rows_are_all_in_the_matrix() -> None:
    """Dropping a row of spec §2.1, a stage 2 variant or an assembler of an `LN` row fails here."""
    assert {code for code in MATRIX if "+" not in code} == EIGHT_ROWS
    assert {code.split("+")[0] for code in MATRIX} == EIGHT_ROWS
    assert set(_ROWS) | set(_STAGE_2_ROWS) | set(_SHARED_ROWS) == set(MATRIX)

    stage_rows: dict[str, _Row | _SharedRow] = {**_ROWS, **_STAGE_2_ROWS, **_SHARED_ROWS}
    for code, row in MATRIX.items():
        assert row.build is stage_rows[code].build, code
    for code, en_row in {**_ROWS, **_STAGE_2_ROWS}.items():
        assert MATRIX[code].reconstructions == en_row.reconstructions, code

    stage_cases = STAGE_1_CASES + STAGE_2_CASES + STAGE_3_CASES
    assert {case.values[0] for case in stage_cases} == set(MATRIX)
    assert {case.values[0] for case in MATRIX_CASES} == set(MATRIX)
    for code in MATRIX:
        covered = {case.values[1] for case in MATRIX_CASES if case.values[0] == code}
        assert covered == (set(ASSEMBLERS) if "-LN-" in code else {None}), code
    assert set(ASSEMBLERS) == set(listRegisteredAssemblers())


@dataclass
class _Built:
    """One built model and one training-mode forward pass over a fixed batch.

    Attributes:
        model: The model.
        batch: Per-modality batch, including any decoder-only target.
        reconstructions: Key of `"reconstructions"` -> expected shape without the batch axis.
        outputs: What `GlobalVae.forward` returned.
    """

    model: GlobalVae
    batch: dict[str, torch.Tensor]
    reconstructions: dict[str, tuple[int, ...]]
    outputs: dict[str, Any]


@pytest.fixture
def built(row_code: str, assembler: str | None) -> _Built:
    """Build the model of one case and run one forward pass over a fixed batch.

    Args:
        row_code: Key of `MATRIX`.
        assembler: Assembler name, or `None` for an `L1` row.

    Returns:
        The model, batch, expected reconstruction shapes and outputs.
    """
    torch.manual_seed(0)
    row = MATRIX[row_code]
    model = row.build(assembler)
    batch = {
        "signal": torch.randn(BATCH_SIZE, SIGNAL_LENGTH),
        "image": torch.randn(BATCH_SIZE, *IMAGE_SHAPE),
    }
    for name, shape in row.reconstructions.items():
        batch.setdefault(name, torch.randn(BATCH_SIZE, *shape))
    outputs = model(model.selectEncoderInputs(batch))
    return _Built(model, batch, row.reconstructions, outputs)


def _loss(model: GlobalVae, batch: dict[str, torch.Tensor], outputs: dict[str, Any]) -> Any:
    """Reconstruction plus regularization loss of one forward pass over `batch`."""
    reconstruction = computeTotalReconstructionLoss(outputs["reconstructions"], batch)
    return reconstruction + model.computeRegularizationLoss(outputs["latent_params"])


@pytest.mark.parametrize(("row_code", "assembler"), MATRIX_CASES)
class TestEveryRow:
    """The checks every row of the matrix must pass, whatever its topology."""

    def test_reconstruction_shapes(self, built: _Built) -> None:
        """Each reconstruction is a batch of its modality, and none is missing or extra."""
        reconstructions = built.outputs["reconstructions"]
        assert set(reconstructions) == set(built.reconstructions)
        for name, shape in built.reconstructions.items():
            assert reconstructions[name].shape == (BATCH_SIZE, *shape), name

    def test_latent_parameter_and_sample_shapes(self, built: _Built) -> None:
        """Every latent space reports `(mu, logvar)` and a sample of its own dimension."""
        params, samples = built.outputs["latent_params"], built.outputs["latent_samples"]
        assert set(params) == set(built.model.latent_spaces)
        assert set(samples) == set(built.model.latent_spaces)
        for name, latent in built.model.latent_spaces.items():
            mu, logvar = params[name]
            assert mu.shape == (BATCH_SIZE, latent.dim), name
            assert logvar.shape == (BATCH_SIZE, latent.dim), name
            assert samples[name].shape == (BATCH_SIZE, latent.dim), name

    def test_loss_is_a_finite_scalar(self, built: _Built) -> None:
        """Reconstruction plus regularization is one finite number."""
        loss = _loss(built.model, built.batch, built.outputs)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_every_parameter_receives_a_finite_gradient(self, built: _Built) -> None:
        """Backward through the loss reaches every parameter of every module."""
        built.model.zero_grad()
        backward(_loss(built.model, built.batch, built.outputs))
        parameters = dict(built.model.named_parameters())
        assert parameters
        for name, parameter in parameters.items():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    def test_a_few_optimizer_steps_lower_the_loss(self, built: _Built) -> None:
        """Training on a fixed batch fits it: the last losses are below the first ones."""
        model, batch = built.model, built.batch
        inputs = model.selectEncoderInputs(batch)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses: list[float] = []
        for _ in range(25):
            optimizer.zero_grad()
            loss = _loss(model, batch, model(inputs))
            backward(loss)
            optimizer.step()
            losses.append(loss.item())
        assert all(math.isfinite(value) for value in losses)
        assert sum(losses[-5:]) < sum(losses[:5])
