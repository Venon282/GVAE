"""`GlobalVae` refuses what it cannot build, and builds encoder fan-out (spec §2.2, §10).

Roadmap P0-4 added the refusals; roadmap P1-2 turned the fan-out refusal into a working build.

Two things are checked in `GlobalVae.__init__`, before any module is created:

* `validateRoutingGraph` runs for every configuration, not only the Phase-1 default (spec §10).
  `tests/unit/test_routing_graph_validation.py` tests the function; the first class below checks
  that the model constructor actually calls it.
* Encoder fan-out. Before roadmap P1-2 an encoder that fed two latent spaces raised
  `NotImplementedError`, because `AbstractEncoder.forward` returns a single `(mu, logvar)` pair and
  reusing it would give two independent latent spaces identical posteriors (ADR 0002). That test
  was inverted when fan-out through latent heads landed (ADR 0024): the same graphs now build, each
  encoder -> latent edge with its head, and the graph that would still give identical posteriors
  (two identity edges) is refused by `validateRoutingGraph` with a `ValueError`. The next class
  covers the errors a head can raise while the model is being built.
* Decoder width. Roadmap P1-4: a decoder's `latent_dim` is checked against what its assembler (or
  its single latent space) delivers when the model is built, and filled in when omitted, instead of
  failing at the first forward pass with `mat1 and mat2 shapes cannot be multiplied`. The last
  class covers that, and the assemblers the model builds from the routing graph.
"""

from typing import Any, cast

import pytest
import torch

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.decoders.registry as decoder_registry_module
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
import global_vae.fusion  # noqa: F401  (registers the built-in fusion strategies)
import global_vae.heads  # noqa: F401  (registers the built-in latent heads)
import global_vae.heads.registry as registry_module
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.heads.identity import IdentityHead
from global_vae.heads.linear import LinearHead
from global_vae.heads.registry import registerHead
from global_vae.latent.base import HeadSpec, LatentSpace, RoutingGraph
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


def _buildModel(
    graph: RoutingGraph, encoder_dim: int = 8, decoder_dim: int = 8, **kwargs: Any
) -> GlobalVae:
    """Build a `GlobalVae` of small 1D modules for every encoder and decoder named in `graph`.

    Args:
        graph: The routing graph to build from.
        encoder_dim: `latent_dim` of every encoder.
        decoder_dim: `latent_dim` (input width) of every decoder.
        **kwargs: Extra `GlobalVae` arguments, such as `fusion_strategies`.

    Returns:
        The model.
    """
    decoder_names = sorted({d for ds in graph.latent_to_decoders.values() for d in ds})
    return GlobalVae(
        encoder_configs=dict.fromkeys(graph.encoder_to_latents, ENCODER),
        decoder_configs=dict.fromkeys(decoder_names, DECODER),
        routing_graph=graph,
        encoder_kwargs=dict.fromkeys(graph.encoder_to_latents, _encoderKwargs(encoder_dim)),
        decoder_kwargs=dict.fromkeys(decoder_names, _decoderKwargs(decoder_dim)),
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


class TestEncoderFanOut:
    """An encoder feeding several latent spaces builds, with one head per edge (ADR 0024)."""

    def test_shared_private_preset_builds(self) -> None:
        """The topology that used to raise `NotImplementedError` is now a working model."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], shared_dim=8, private_dim=4)
        model = _buildModel(graph, decoder_dim=12, fusion_strategies={"z_shared": "poe"})
        assert set(model.latent_spaces) == {"z_shared", "z_private_signal", "z_private_image"}
        assert set(model.fusions) == {"z_shared"}
        assert set(model.assemblers) == {"signal", "image"}

    def test_one_head_module_is_built_per_edge(self) -> None:
        """`getHead(encoder, latent)` is the head the preset asked for on that edge."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], shared_dim=8, private_dim=4)
        model = _buildModel(graph, decoder_dim=12, fusion_strategies={"z_shared": "poe"})
        assert set(model.heads) == {"signal", "image"}
        for modality in ("signal", "image"):
            assert isinstance(model.getHead(modality, "z_shared"), IdentityHead)
            assert isinstance(model.getHead(modality, f"z_private_{modality}"), LinearHead)

    def test_head_widths_are_the_encoder_width_and_the_latent_width(self) -> None:
        """`in_dim` is the encoder's `latent_dim`, `out_dim` the target latent space's `dim`."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], shared_dim=8, private_dim=4)
        model = _buildModel(graph, decoder_dim=12, fusion_strategies={"z_shared": "poe"})
        shared = model.getHead("signal", "z_shared")
        private = model.getHead("signal", "z_private_signal")
        assert shared is not None and private is not None
        assert (shared.in_dim, shared.out_dim) == (8, 8)
        assert (private.in_dim, private.out_dim) == (8, 4)

    def test_head_kwargs_reach_the_head(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`HeadSpec.kwargs` are passed to the head constructor next to the two widths."""
        monkeypatch.setattr(registry_module, "_HEAD_REGISTRY", dict(registry_module._HEAD_REGISTRY))
        seen: dict[str, Any] = {}

        @registerHead("recording_head")
        class _RecordingHead(LinearHead):
            def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0) -> None:
                super().__init__(in_dim, out_dim)
                seen.update(in_dim=in_dim, out_dim=out_dim, scale=scale)

        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 4)},
            encoder_to_latents={"signal": ["z"]},
            latent_to_decoders={"z": ["signal"]},
            encoder_heads={"signal": {"z": HeadSpec("recording_head", kwargs={"scale": 2.5})}},
        )
        _buildModel(graph, encoder_dim=8, decoder_dim=4)
        assert seen == {"in_dim": 8, "out_dim": 4, "scale": 2.5}

    def test_single_shared_encoder_feeding_two_latents_builds_with_a_head(self) -> None:
        """The `E1-LN-*` shape of one trunk over two latent spaces: identity plus a linear head."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 8)},
            encoder_to_latents={"trunk": ["z_a", "z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"]},
            decoder_assemblers={"dec": "concat"},
            encoder_heads={"trunk": {"z_b": HeadSpec("linear")}},
        )
        model = _buildModel(graph, decoder_dim=16)
        assert model.getHead("trunk", "z_a") is None
        assert isinstance(model.getHead("trunk", "z_b"), LinearHead)

    def test_two_identity_edges_are_refused_as_identical_posteriors(self) -> None:
        """No heads on a fan-out is the old failure: now a `ValueError` from the validator."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 8)},
            encoder_to_latents={"trunk": ["z_a", "z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"]},
            decoder_assemblers={"dec": "concat"},
        )
        with pytest.raises(ValueError, match="identical posteriors"):
            _buildModel(graph, decoder_dim=16)

    def test_message_names_the_encoder_and_its_latent_spaces(self) -> None:
        """The message says which encoder fans out through identity edges, and to what."""
        graph = buildSharedPrivateRoutingGraph(
            ["signal", "image"], shared_dim=8, private_dim=8, private_head="identity"
        )
        with pytest.raises(ValueError) as excinfo:
            _buildModel(graph, fusion_strategies={"z_shared": "poe"})
        message = str(excinfo.value)
        assert "Encoder 'signal' feeds 2 latent spaces" in message
        assert "z_shared" in message and "z_private_signal" in message

    def test_only_the_encoder_with_two_identity_edges_is_named(self) -> None:
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
        with pytest.raises(ValueError) as excinfo:
            _buildModel(graph)
        message = str(excinfo.value)
        assert "Encoder 'wide'" in message
        assert "Encoder 'plain'" not in message

    def test_graph_without_heads_builds_no_head_module(self) -> None:
        """A graph without fan-out is built as before: no head module, no `heads.*` state."""
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
        assert len(model.heads) == 0
        assert not [key for key in model.state_dict() if key.startswith("heads.")]

    def test_an_edge_with_no_entry_is_not_width_checked(self) -> None:
        """An implicit identity edge builds without a check, as every graph did before heads."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 4)},
            encoder_to_latents={"signal": ["z"]},
            latent_to_decoders={"z": ["signal"]},
        )
        model = _buildModel(graph, encoder_dim=8, decoder_dim=4)
        assert len(model.heads) == 0


class TestHeadsThatCannotBeBuilt:
    """Errors raised while the model builds its heads name the edge and the widths."""

    @staticmethod
    def _graph(heads: dict[str, dict[str, HeadSpec]], latent_dim: int = 4) -> RoutingGraph:
        return RoutingGraph(
            latent_specs={"z": LatentSpace("z", latent_dim)},
            encoder_to_latents={"signal": ["z"]},
            latent_to_decoders={"z": ["signal"]},
            encoder_heads=heads,
        )

    def test_explicit_identity_head_checks_the_encoder_width(self) -> None:
        """`identity` between an 8-wide encoder and a 4-wide latent space is refused."""
        graph = self._graph({"signal": {"z": HeadSpec("identity")}})
        with pytest.raises(ValueError) as excinfo:
            _buildModel(graph, encoder_dim=8, decoder_dim=4)
        message = str(excinfo.value)
        assert "Cannot build head 'identity' on the edge 'signal' -> 'z'" in message
        assert "encoder latent_dim=8" in message and "dim=4" in message
        assert "in_dim == out_dim" in message

    def test_explicit_identity_head_is_fine_when_the_widths_match(self) -> None:
        """The same head with a 4-wide encoder builds."""
        graph = self._graph({"signal": {"z": HeadSpec("identity")}})
        model = _buildModel(graph, encoder_dim=4, decoder_dim=4)
        assert isinstance(model.getHead("signal", "z"), IdentityHead)

    def test_a_linear_head_bridges_different_widths(self) -> None:
        """The same graph with `linear` builds, whatever the encoder width."""
        graph = self._graph({"signal": {"z": HeadSpec("linear")}})
        model = _buildModel(graph, encoder_dim=8, decoder_dim=4)
        head = model.getHead("signal", "z")
        assert head is not None
        assert (head.in_dim, head.out_dim) == (8, 4)

    def test_unknown_head_name_raises_key_error_listing_the_registered_heads(self) -> None:
        """An unregistered head name fails like every other registry lookup."""
        graph = self._graph({"signal": {"z": HeadSpec("does_not_exist")}})
        with pytest.raises(KeyError, match="does_not_exist.*identity, linear"):
            _buildModel(graph, encoder_dim=4, decoder_dim=4)

    def test_head_on_an_encoder_missing_from_encoder_configs(self) -> None:
        """A head for an encoder the model does not build is refused with a clear message."""
        graph = self._graph({"signal": {"z": HeadSpec("linear")}})
        with pytest.raises(
            ValueError, match="encoder 'signal', but that encoder is not in `encoder_configs`"
        ):
            GlobalVae(
                encoder_configs={},
                decoder_configs={"signal": DECODER},
                routing_graph=graph,
                decoder_kwargs={"signal": _decoderKwargs(latent_dim=4)},
            )

    def test_declared_out_dim_that_differs_from_the_latent_dim_is_refused(self) -> None:
        """The validator's width rule (spec §10) is enforced by the constructor too."""
        graph = self._graph({"signal": {"z": HeadSpec("linear", out_dim=16)}})
        with pytest.raises(ValueError, match="declares out_dim=16"):
            _buildModel(graph, encoder_dim=8, decoder_dim=4)


class TestDecoderWidthFollowsTheRoutingGraph:
    """A decoder's `latent_dim` is what its assembler delivers: checked, or filled in (P1-4)."""

    @staticmethod
    def _graph(assembler: str, dim_a: int = 8, dim_b: int = 4) -> RoutingGraph:
        """Two encoders, one latent space each, one decoder over both through `assembler`."""
        return RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", dim_a), "z_b": LatentSpace("z_b", dim_b)},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"]},
            decoder_assemblers={"dec": assembler},
        )

    @staticmethod
    def _model(
        graph: RoutingGraph,
        decoder_kwargs: dict[str, Any],
        assembler_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> GlobalVae:
        """Build the model of `_graph`, with explicit decoder kwargs (`latent_dim` or not).

        Each encoder is as wide as the one latent space it feeds, so the model also runs.
        """
        return GlobalVae(
            encoder_configs={"enc_a": ENCODER, "enc_b": ENCODER},
            decoder_configs={"dec": DECODER},
            routing_graph=graph,
            encoder_kwargs={
                encoder: _encoderKwargs(graph.latent_specs[spaces[0]].dim)
                for encoder, spaces in graph.encoder_to_latents.items()
            },
            decoder_kwargs={"dec": decoder_kwargs},
            assembler_kwargs=assembler_kwargs,
        )

    @staticmethod
    def _reconstruct(model: GlobalVae) -> torch.Tensor:
        """One forward pass over both encoders; returns the decoder's reconstruction."""
        inputs = {name: torch.randn(2, SIGNAL_LENGTH) for name in ("enc_a", "enc_b")}
        return cast(torch.Tensor, model(inputs)["reconstructions"]["dec"])

    def test_a_wrong_latent_dim_behind_concat_is_refused_when_the_model_is_built(self) -> None:
        """The audit's case: dims 8 and 4 need a 12-wide decoder, and 8 used to fail at forward."""
        with pytest.raises(ValueError) as excinfo:
            self._model(self._graph("concat"), _decoderKwargs(latent_dim=8))
        message = str(excinfo.value)
        assert "Decoder 'dec' is built with latent_dim=8, but it receives 12 features" in message
        assert "ConcatAssembler" in message and "[8, 4]" in message
        assert "Remove `latent_dim`" in message

    def test_the_right_latent_dim_behind_concat_builds_and_runs(self) -> None:
        """The sum of the widths, written by hand as before, is still accepted."""
        model = self._model(self._graph("concat"), _decoderKwargs(latent_dim=12))
        assert self._reconstruct(model).shape == (2, SIGNAL_LENGTH)

    def test_a_wrong_latent_dim_behind_a_single_latent_space_is_refused(self) -> None:
        """No assembler: the width is the latent space's `dim`, and a mismatch is refused too."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 4)},
            encoder_to_latents={"signal": ["z"]},
            latent_to_decoders={"z": ["signal"]},
        )
        with pytest.raises(
            ValueError, match=r"latent_dim=8, but it receives 4 features from latent space 'z'"
        ):
            _buildModel(graph, encoder_dim=4, decoder_dim=8)

    @pytest.mark.parametrize(
        ("assembler", "dims", "assembler_kwargs", "width"),
        [
            ("concat", (8, 4), {}, 12),
            ("sum", (8, 8), {}, 8),
            ("average", (8, 8), {}, 8),
            ("weighted_sum", (8, 8), {}, 8),
            ("weighted_sum", (8, 8), {"normalize": False}, 8),
            ("attention", (8, 4), {}, 8),
            ("attention", (8, 4), {"output_dim": 6, "num_heads": 2}, 6),
        ],
        ids=[
            "concat",
            "sum",
            "average",
            "weighted_sum",
            "weighted_sum-raw",
            "attention",
            "attention-sized",
        ],
    )
    def test_latent_dim_is_filled_in_from_the_assembler_when_omitted(
        self,
        assembler: str,
        dims: tuple[int, int],
        assembler_kwargs: dict[str, Any],
        width: int,
    ) -> None:
        """With no `latent_dim`, every assembler sizes its decoder, and the model runs."""
        decoder_kwargs = {k: v for k, v in _decoderKwargs(0).items() if k != "latent_dim"}
        model = self._model(
            self._graph(assembler, *dims),
            decoder_kwargs,
            assembler_kwargs={"dec": assembler_kwargs},
        )
        assert model.assemblers["dec"].output_dim == width
        assert self._reconstruct(model).shape == (2, SIGNAL_LENGTH)

    @pytest.mark.parametrize("assembler", ["weighted_sum", "attention"])
    def test_a_wrong_latent_dim_behind_a_learned_assembler_is_refused(self, assembler: str) -> None:
        """The learned assemblers announce their width like the others, so they are checked too."""
        with pytest.raises(ValueError, match=r"latent_dim=3, but it receives 8 features"):
            self._model(self._graph(assembler, 8, 8), _decoderKwargs(latent_dim=3))

    def test_an_explicit_latent_dim_that_matches_an_attention_output_dim_is_accepted(self) -> None:
        """`output_dim` in the assembler kwargs is the width to give the decoder."""
        model = self._model(
            self._graph("attention"), _decoderKwargs(latent_dim=6), {"dec": {"output_dim": 6}}
        )
        assert self._reconstruct(model).shape == (2, SIGNAL_LENGTH)

    def test_assembler_kwargs_and_input_dims_reach_the_assembler(self) -> None:
        """The model passes the widths of the consumed latent spaces, in graph order."""
        model = self._model(
            self._graph("attention", 8, 4),
            _decoderKwargs(latent_dim=6),
            {"dec": {"output_dim": 6, "num_heads": 3}},
        )
        assembler = model.assemblers["dec"]
        assert assembler.input_dims == (8, 4)
        assert assembler.num_heads == 3

    def test_input_dims_follow_the_order_of_the_routing_graph(self) -> None:
        """The first latent space of `latent_to_decoders` is the first input, whatever its name."""
        graph = RoutingGraph(
            latent_specs={"z_b": LatentSpace("z_b", 4), "z_a": LatentSpace("z_a", 8)},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
            latent_to_decoders={"z_b": ["dec"], "z_a": ["dec"]},
            decoder_assemblers={"dec": "concat"},
        )
        model = self._model(graph, _decoderKwargs(latent_dim=12))
        assert model.assemblers["dec"].input_dims == (4, 8)
        assert graph.decoderInputs() == {"dec": ["z_b", "z_a"]}

    def test_an_assembler_that_cannot_be_built_names_the_decoder_and_the_widths(self) -> None:
        """A bad kwarg is reported with the decoder, the latent spaces and their widths."""
        with pytest.raises(ValueError) as excinfo:
            self._model(self._graph("concat"), _decoderKwargs(12), {"dec": {"nonsense": 1}})
        message = str(excinfo.value)
        assert "Cannot build assembler 'concat' for decoder 'dec'" in message
        assert "['z_a', 'z_b']" in message and "[8, 4]" in message and "nonsense" in message

    def test_invalid_attention_arguments_are_reported_with_the_decoder(self) -> None:
        """A `ValueError` from the assembler's own check keeps its message."""
        with pytest.raises(ValueError, match=r"decoder 'dec'.*must be a multiple of `num_heads`"):
            self._model(
                self._graph("attention"),
                _decoderKwargs(latent_dim=10),
                {"dec": {"output_dim": 10, "num_heads": 4}},
            )

    def test_unknown_assembler_name_raises_key_error_listing_the_registered_ones(self) -> None:
        """An unregistered assembler fails like every other registry lookup."""
        with pytest.raises(
            KeyError, match="does_not_exist.*attention, average, concat, sum, weighted_sum"
        ):
            self._model(self._graph("does_not_exist"), _decoderKwargs(latent_dim=12))

    def test_an_assembler_entry_for_a_decoder_that_consumes_nothing_builds_no_module(self) -> None:
        """There is nothing to size it from and it would never run, so none is built."""
        graph = self._graph("concat")
        graph.decoder_assemblers["ghost"] = "concat"
        model = self._model(graph, _decoderKwargs(latent_dim=12))
        assert set(model.assemblers) == {"dec"}

    def test_a_decoder_with_no_latent_dim_parameter_is_built_from_its_kwargs_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing is injected into a decoder whose constructor takes no `latent_dim`."""
        monkeypatch.setattr(
            decoder_registry_module,
            "_DECODER_REGISTRY",
            dict(decoder_registry_module._DECODER_REGISTRY),
        )

        @registerDecoder("fixed_width_decoder_for_test")
        class _FixedWidthDecoder(AbstractDecoder):
            def __init__(self, scale: float = 1.0) -> None:
                super().__init__()
                self.scale = scale

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return z * self.scale

            @property
            def modality_name(self) -> str:
                return "vector"

        graph = self._graph("concat")
        model = GlobalVae(
            encoder_configs={"enc_a": ENCODER, "enc_b": ENCODER},
            decoder_configs={"dec": "fixed_width_decoder_for_test"},
            routing_graph=graph,
            encoder_kwargs={"enc_a": _encoderKwargs(8), "enc_b": _encoderKwargs(4)},
            decoder_kwargs={"dec": {"scale": 2.0}},
        )
        assert model.decoders["dec"].scale == 2.0
        reconstruction = self._reconstruct(model)
        assert reconstruction.shape == (2, 12)
