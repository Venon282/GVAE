"""`GlobalVae` refuses what it cannot build (spec §2.2, §10, roadmap P0-4(a) and (f)).

Two guards run in `GlobalVae.__init__`, before any module is created:

* `validateRoutingGraph` runs for every configuration, not only the Phase-1 default (spec §10).
  `tests/unit/test_routing_graph_validation.py` tests the function; the class below checks that
  the model constructor actually calls it.
* An encoder that feeds two latent spaces raises `NotImplementedError`. `AbstractEncoder.forward`
  returns a single `(mu, logvar)` pair, so reusing it for two independent latent spaces would
  give them identical posteriors (ADR 0002). This is the test that roadmap item P1-2 inverts when
  encoder fan-out through latent heads lands: it then asserts that the same graphs build, instead
  of that they raise.
"""

from typing import Any

import pytest

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
import global_vae.fusion  # noqa: F401  (registers the built-in fusion strategies)
from global_vae.latent.base import LatentSpace, RoutingGraph
from global_vae.latent.routing_graph_builders.shared_private import buildSharedPrivateRoutingGraph
from global_vae.models.global_vae import GlobalVae

ENCODER = "1d_cnn_encoder_v1"
DECODER = "1d_cnn_decoder_v1"
SIGNAL_LENGTH = 32


def _encoderKwargs(latent_dim: int = 8) -> dict[str, Any]:
    """Small 1D encoder arguments, so a model that does build stays cheap."""
    return {"latent_dim": latent_dim, "hidden_channels": (8, 16)}


def _decoderKwargs(latent_dim: int) -> dict[str, Any]:
    """Small 1D decoder arguments that reach `SIGNAL_LENGTH` exactly."""
    return {
        "latent_dim": latent_dim,
        "output_length": SIGNAL_LENGTH,
        "upsample_modes": "conv_transpose",
        "hidden_channels": (16, 8),
    }


def _buildModel(graph: RoutingGraph, **kwargs: Any) -> GlobalVae:
    """Build a `GlobalVae` of 1D modules for every encoder and decoder named in `graph`.

    Args:
        graph: The routing graph to build from.
        **kwargs: Extra `GlobalVae` arguments, such as `fusion_strategies`.

    Returns:
        The model.
    """
    decoder_names = sorted({d for ds in graph.latent_to_decoders.values() for d in ds})
    return GlobalVae(
        encoder_configs=dict.fromkeys(graph.encoder_to_latents, ENCODER),
        decoder_configs=dict.fromkeys(decoder_names, DECODER),
        routing_graph=graph,
        **kwargs,
    )


class TestRoutingGraphIsValidatedAtConstruction:
    """The constructor rejects an invalid graph with the `ValueError` of `validateRoutingGraph`."""

    def test_latent_space_without_an_encoder(self) -> None:
        """An unfed latent space is refused."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 8)},
            encoder_to_latents={"signal": []},
            latent_to_decoders={"z": ["signal"]},
        )
        with pytest.raises(ValueError, match="'z' has no encoder feeding it"):
            _buildModel(graph)

    def test_latent_space_without_a_decoder(self) -> None:
        """A latent space nobody consumes is refused."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 8)},
            encoder_to_latents={"signal": ["z"]},
        )
        with pytest.raises(ValueError, match="'z' has no decoder consuming it"):
            _buildModel(graph)

    def test_decoder_with_several_latents_and_no_assembler(self) -> None:
        """Two latent spaces into one decoder, with no assembler, is refused."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 8)},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"]},
        )
        with pytest.raises(ValueError, match="no assembler assigned"):
            _buildModel(graph)

    @pytest.mark.parametrize("assembler", ["sum", "average"])
    def test_dimension_locked_assembler_over_different_dims(self, assembler: str) -> None:
        """`sum` and `average` over latent spaces of dims 8 and 4 are refused."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 4)},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"]},
            decoder_assemblers={"dec": assembler},
        )
        with pytest.raises(ValueError, match="requires matching dimensionality"):
            _buildModel(graph)


class TestEncoderFanOutGuard:
    """An encoder feeding two latent spaces raises `NotImplementedError` (ADR 0002)."""

    def test_shared_private_preset_is_refused(self) -> None:
        """The shared plus private topology is valid as a graph but cannot be built yet."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], shared_dim=8, private_dim=4)
        with pytest.raises(NotImplementedError, match="fan-out"):
            _buildModel(graph, fusion_strategies={"z_shared": "poe"})

    def test_message_names_the_encoder_and_its_latent_spaces(self) -> None:
        """The message says which encoder fans out, to what, and where to read more."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], shared_dim=8, private_dim=4)
        with pytest.raises(NotImplementedError) as excinfo:
            _buildModel(graph, fusion_strategies={"z_shared": "poe"})
        message = str(excinfo.value)
        assert "Encoder 'signal' is assigned to 2 latent spaces" in message
        assert "z_shared" in message and "z_private_signal" in message
        assert "0002-generalize-global-vae-to-routing-graph" in message

    def test_single_shared_encoder_feeding_two_latents_is_refused(self) -> None:
        """The `E1-LN-*` shape: one trunk, two latent spaces, one decoder over both."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 8)},
            encoder_to_latents={"trunk": ["z_a", "z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"]},
            decoder_assemblers={"dec": "concat"},
        )
        with pytest.raises(NotImplementedError, match="Encoder 'trunk'"):
            _buildModel(graph)

    def test_only_the_encoder_that_fans_out_is_named(self) -> None:
        """With one fan-out encoder and one ordinary encoder, the message names the first."""
        graph = RoutingGraph(
            latent_specs={
                "z_a": LatentSpace("z_a", 8),
                "z_b": LatentSpace("z_b", 8),
                "z_c": LatentSpace("z_c", 8),
            },
            encoder_to_latents={"plain": ["z_c"], "wide": ["z_a", "z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"], "z_c": ["dec"]},
            decoder_assemblers={"dec": "concat"},
        )
        with pytest.raises(NotImplementedError) as excinfo:
            _buildModel(graph)
        message = str(excinfo.value)
        assert "Encoder 'wide'" in message
        assert "Encoder 'plain'" not in message

    def test_graph_without_fan_out_still_builds(self) -> None:
        """Control: two encoders, two latents, one each, builds with real modules."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 8)},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"]},
            decoder_assemblers={"dec": "sum"},
        )
        model = GlobalVae(
            encoder_configs={"enc_a": ENCODER, "enc_b": ENCODER},
            decoder_configs={"dec": DECODER},
            routing_graph=graph,
            encoder_kwargs={"enc_a": _encoderKwargs(), "enc_b": _encoderKwargs()},
            decoder_kwargs={"dec": _decoderKwargs(latent_dim=8)},
        )
        assert set(model.latent_spaces) == {"z_a", "z_b"}
        assert set(model.assemblers) == {"dec"}
