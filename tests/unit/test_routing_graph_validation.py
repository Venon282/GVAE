"""Unit tests for `latent.base.validateRoutingGraph` (spec §2.2, §10, roadmap P0-4(a)).

`validateRoutingGraph` is the construction-time gate every `GlobalVae` goes through, so each of
its rejection paths gets its own test, next to the accepting cases that show the gate is not
simply rejecting everything:

1. a latent space with no encoder feeding it,
2. a latent space with no decoder consuming it,
3. a decoder consuming several latent spaces with no assembler,
4. an assembler whose class sets `requires_matching_dims` (`sum`, `average`, `weighted_sum`) over
   latent spaces of different dimensionality. The flag is read through the assembler registry, so
   `validateRoutingGraph` names no assembler (roadmap P1-4, spec §10 and §12).

Three further rejections concern latent heads (spec §2.2, §10, roadmap P1-2, ADR 0024):

5. a head declared on an edge that does not exist, or on an unknown latent space,
6. a head whose declared `out_dim` differs from its target latent space's `dim`, or whose
   `kwargs` set `in_dim` or `out_dim`,
7. an encoder that feeds several latent spaces through more than one identity edge, which would
   give those spaces identical posteriors (the reason ADR 0002 added its guard).
"""

import ast
import inspect
import textwrap
from collections.abc import Sequence

import pytest
import torch

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
import global_vae.assemblers.registry as assembler_registry
from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.registry import listRegisteredAssemblers
from global_vae.latent.base import HeadSpec, LatentSpace, RoutingGraph, validateRoutingGraph


def _singleLatentGraph() -> RoutingGraph:
    """One encoder, one latent space, one decoder: the smallest valid graph."""
    return RoutingGraph(
        latent_specs={"z": LatentSpace("z", 8)},
        encoder_to_latents={"enc": ["z"]},
        latent_to_decoders={"z": ["dec"]},
    )


def _twoLatentGraph(
    assembler: str | None, dim_a: int = 8, dim_b: int = 8, decoder: str = "dec"
) -> RoutingGraph:
    """Two latent spaces, one encoder each, both consumed by a single decoder.

    Args:
        assembler: Assembler assigned to `decoder`, or `None` for no entry at all.
        dim_a: Dimensionality of `z_a`.
        dim_b: Dimensionality of `z_b`.
        decoder: Name of the decoder consuming both latent spaces.

    Returns:
        The routing graph, with no fan-out (each encoder feeds exactly one latent space).
    """
    return RoutingGraph(
        latent_specs={"z_a": LatentSpace("z_a", dim_a), "z_b": LatentSpace("z_b", dim_b)},
        encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
        latent_to_decoders={"z_a": [decoder], "z_b": [decoder]},
        decoder_assemblers={} if assembler is None else {decoder: assembler},
    )


class TestRejectsLatentWithNoEncoder:
    """Rejection 1: every latent space needs at least one encoder feeding it."""

    def test_no_encoder_at_all(self) -> None:
        """A graph with no `encoder_to_latents` entry leaves its latent space orphaned."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 8)},
            latent_to_decoders={"z": ["dec"]},
        )
        with pytest.raises(ValueError, match=r"Latent space 'z' has no encoder feeding it"):
            validateRoutingGraph(graph)

    def test_encoder_present_but_feeds_another_latent(self) -> None:
        """An encoder feeding some other latent space does not rescue an unfed one."""
        graph = RoutingGraph(
            latent_specs={"z_fed": LatentSpace("z_fed", 8), "z_unfed": LatentSpace("z_unfed", 8)},
            encoder_to_latents={"enc": ["z_fed"]},
            latent_to_decoders={"z_fed": ["dec"], "z_unfed": ["dec"]},
            decoder_assemblers={"dec": "concat"},
        )
        with pytest.raises(ValueError, match=r"'z_unfed' has no encoder feeding it"):
            validateRoutingGraph(graph)

    def test_encoder_with_an_empty_latent_list(self) -> None:
        """An encoder that is listed but feeds nothing is the same as no encoder."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 8)},
            encoder_to_latents={"enc": []},
            latent_to_decoders={"z": ["dec"]},
        )
        with pytest.raises(ValueError, match="no encoder feeding it"):
            validateRoutingGraph(graph)


class TestRejectsLatentWithNoDecoder:
    """Rejection 2: every latent space needs at least one decoder consuming it."""

    def test_latent_missing_from_latent_to_decoders(self) -> None:
        """A latent space with no `latent_to_decoders` key has no consumer."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 8)},
            encoder_to_latents={"enc": ["z"]},
        )
        with pytest.raises(ValueError, match=r"Latent space 'z' has no decoder consuming it"):
            validateRoutingGraph(graph)

    def test_latent_with_an_empty_decoder_list(self) -> None:
        """A latent space mapped to an empty decoder list has no consumer either."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 8)},
            encoder_to_latents={"enc": ["z"]},
            latent_to_decoders={"z": []},
        )
        with pytest.raises(ValueError, match="no decoder consuming it"):
            validateRoutingGraph(graph)

    def test_only_the_unconsumed_latent_is_named(self) -> None:
        """With two latent spaces, the message names the one that has no decoder."""
        graph = RoutingGraph(
            latent_specs={"z_used": LatentSpace("z_used", 8), "z_dead": LatentSpace("z_dead", 8)},
            encoder_to_latents={"enc_a": ["z_used"], "enc_b": ["z_dead"]},
            latent_to_decoders={"z_used": ["dec"]},
        )
        with pytest.raises(ValueError, match=r"'z_dead' has no decoder consuming it"):
            validateRoutingGraph(graph)


class TestRejectsMultiLatentDecoderWithoutAssembler:
    """Rejection 3: a decoder consuming several latent spaces needs an assembler."""

    def test_no_assembler_entry_at_all(self) -> None:
        """Two latents into one decoder, with an empty `decoder_assemblers`."""
        with pytest.raises(ValueError, match=r"Decoder 'dec' consumes 2 latent spaces"):
            validateRoutingGraph(_twoLatentGraph(assembler=None))

    def test_message_names_the_latent_spaces(self) -> None:
        """The message lists the latent spaces so the missing wiring is easy to find."""
        with pytest.raises(ValueError, match=r"z_a.*z_b.*no assembler assigned"):
            validateRoutingGraph(_twoLatentGraph(assembler=None))

    def test_assembler_assigned_to_another_decoder_does_not_count(self) -> None:
        """An assembler entry is per decoder: `other` having one does not help `dec`."""
        graph = _twoLatentGraph(assembler=None)
        graph.decoder_assemblers["other"] = "concat"
        with pytest.raises(ValueError, match="Decoder 'dec'"):
            validateRoutingGraph(graph)

    def test_three_latents_into_one_decoder(self) -> None:
        """The count in the message is the number of latent spaces the decoder consumes."""
        graph = RoutingGraph(
            latent_specs={name: LatentSpace(name, 4) for name in ("z_a", "z_b", "z_c")},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"], "enc_c": ["z_c"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"], "z_c": ["dec"]},
        )
        with pytest.raises(ValueError, match=r"Decoder 'dec' consumes 3 latent spaces"):
            validateRoutingGraph(graph)

    def test_only_the_decoder_without_an_assembler_is_rejected(self) -> None:
        """A second decoder that does have an assembler does not mask the first."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 8)},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
            latent_to_decoders={"z_a": ["dec_ok", "dec_bad"], "z_b": ["dec_ok", "dec_bad"]},
            decoder_assemblers={"dec_ok": "concat"},
        )
        with pytest.raises(ValueError, match="Decoder 'dec_bad'"):
            validateRoutingGraph(graph)


class _LockedAssembler(AbstractAssembler):
    """Test strategy that sets the class-level flag, to prove the validator reads it."""

    requires_matching_dims = True

    @property
    def output_dim(self) -> int:
        return self.input_dims[0]

    def forward(
        self, latents: list[torch.Tensor], present: Sequence[int] | None = None
    ) -> torch.Tensor:
        return latents[0]


class _FreeAssembler(_LockedAssembler):
    """Test strategy that leaves the flag at its default."""

    requires_matching_dims = False


class TestRejectsDimensionLockedAssemblerOverDifferentDims:
    """Rejection 4: a class with `requires_matching_dims` needs one dimension across its inputs."""

    @pytest.mark.parametrize("assembler", ["sum", "average", "weighted_sum"])
    def test_differing_dims_are_rejected(self, assembler: str) -> None:
        """The three dimension-locked built-ins reject latent spaces of dims 8 and 4."""
        graph = _twoLatentGraph(assembler=assembler, dim_a=8, dim_b=4)
        with pytest.raises(ValueError, match="requires matching dimensionality"):
            validateRoutingGraph(graph)

    def test_message_names_decoder_assembler_and_dims(self) -> None:
        """The message carries the decoder, the assembler and the sorted differing dims."""
        graph = _twoLatentGraph(assembler="sum", dim_a=8, dim_b=4, decoder="signal_decoder")
        with pytest.raises(
            ValueError,
            match=r"Decoder 'signal_decoder' uses assembler 'sum'.*differing dims \[4, 8\]",
        ):
            validateRoutingGraph(graph)

    def test_one_odd_dim_among_three_is_enough(self) -> None:
        """Two matching dims do not make up for a third that differs."""
        graph = RoutingGraph(
            latent_specs={
                "z_a": LatentSpace("z_a", 8),
                "z_b": LatentSpace("z_b", 8),
                "z_c": LatentSpace("z_c", 16),
            },
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"], "enc_c": ["z_c"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"], "z_c": ["dec"]},
            decoder_assemblers={"dec": "average"},
        )
        with pytest.raises(ValueError, match=r"differing dims \[8, 16\]"):
            validateRoutingGraph(graph)

    def test_a_strategy_that_sets_the_flag_gets_the_same_check_without_any_change_here(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A registered class with `requires_matching_dims = True` is rejected over dims 8, 4."""
        monkeypatch.setitem(
            assembler_registry._ASSEMBLER_REGISTRY, "locked_for_test", _LockedAssembler
        )
        graph = _twoLatentGraph(assembler="locked_for_test", dim_a=8, dim_b=4)
        with pytest.raises(
            ValueError,
            match=r"uses assembler 'locked_for_test', which requires matching dimensionality",
        ):
            validateRoutingGraph(graph)

    def test_a_strategy_that_leaves_the_flag_unset_is_not_checked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default is `False`: any widths pass, as for `concat` and `attention`."""
        monkeypatch.setitem(assembler_registry._ASSEMBLER_REGISTRY, "free_for_test", _FreeAssembler)
        validateRoutingGraph(_twoLatentGraph(assembler="free_for_test", dim_a=8, dim_b=4))

    def test_the_validator_names_no_assembler(self) -> None:
        """Roadmap P1-4: no registered assembler name appears as a literal in the function."""
        source = textwrap.dedent(inspect.getsource(validateRoutingGraph))
        literals = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        # Docstrings and messages are prose; a name used as data would be a whole literal.
        assert not literals & set(listRegisteredAssemblers())
        signature = inspect.signature(validateRoutingGraph)
        assert "dimension_locked_assemblers" not in signature.parameters

    def test_an_unknown_assembler_on_a_multi_latent_decoder_raises_key_error(self) -> None:
        """The flag is read from the registry, so an unregistered name is a registry `KeyError`."""
        graph = _twoLatentGraph(assembler="does_not_exist")
        with pytest.raises(KeyError, match="does_not_exist.*Available:.*concat"):
            validateRoutingGraph(graph)


class TestAcceptsValidGraphs:
    """The gate lets every well-formed topology through."""

    def test_single_latent_one_encoder_one_decoder(self) -> None:
        """The smallest graph, `signal -> z -> signal`, is accepted."""
        validateRoutingGraph(_singleLatentGraph())

    def test_single_latent_with_several_encoders_and_decoders(self) -> None:
        """Several encoders and decoders around one latent need no assembler."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 8)},
            encoder_to_latents={"enc_a": ["z"], "enc_b": ["z"]},
            latent_to_decoders={"z": ["dec_a", "dec_b"]},
        )
        validateRoutingGraph(graph)

    @pytest.mark.parametrize(
        ("assembler", "dim_a", "dim_b"),
        [
            ("concat", 8, 8),
            ("concat", 8, 4),
            ("sum", 8, 8),
            ("average", 4, 4),
            ("weighted_sum", 8, 8),
            ("attention", 8, 8),
            ("attention", 8, 4),
        ],
        ids=[
            "concat-equal",
            "concat-different",
            "sum-equal",
            "average-equal",
            "weighted_sum-equal",
            "attention-equal",
            "attention-different",
        ],
    )
    def test_multi_latent_decoder_with_a_compatible_assembler(
        self, assembler: str, dim_a: int, dim_b: int
    ) -> None:
        """`concat` and `attention` accept any dims; the dimension-locked ones accept equal dims."""
        validateRoutingGraph(_twoLatentGraph(assembler=assembler, dim_a=dim_a, dim_b=dim_b))

    def test_assembler_on_a_single_input_decoder_is_not_checked(self) -> None:
        """A dimension-locked assembler on a decoder with one input has nothing to compare."""
        graph = _singleLatentGraph()
        graph.decoder_assemblers["dec"] = "sum"
        validateRoutingGraph(graph)

    def test_dimension_check_is_per_decoder(self) -> None:
        """Latent spaces of different dims are fine as long as no `sum` joins them."""
        graph = RoutingGraph(
            latent_specs={
                "z_a": LatentSpace("z_a", 8),
                "z_b": LatentSpace("z_b", 8),
                "z_c": LatentSpace("z_c", 4),
            },
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"], "enc_c": ["z_c"]},
            latent_to_decoders={"z_a": ["dec_sum"], "z_b": ["dec_sum"], "z_c": ["dec_solo"]},
            decoder_assemblers={"dec_sum": "sum"},
        )
        validateRoutingGraph(graph)

    def test_an_unregistered_assembler_on_a_single_input_decoder_is_not_looked_up(self) -> None:
        """Nothing is assembled for one input, so the name is not resolved by the gate."""
        graph = _singleLatentGraph()
        graph.decoder_assemblers["dec"] = "does_not_exist"
        validateRoutingGraph(graph)

    def test_encoder_fan_out_with_a_head_on_one_edge_is_valid(self) -> None:
        """Fan-out is legal when at most one of the encoder's edges is an identity edge."""
        graph = _fanOutGraph(heads={"z_b": HeadSpec("linear")})
        validateRoutingGraph(graph)

    def test_graph_without_heads_still_validates(self) -> None:
        """`encoder_heads` defaults to empty, and an encoder with one edge needs no head."""
        assert _singleLatentGraph().encoder_heads == {}
        validateRoutingGraph(_singleLatentGraph())


def _fanOutGraph(
    heads: dict[str, HeadSpec] | None = None,
    dim_a: int = 8,
    dim_b: int = 4,
    extra_latent: bool = False,
) -> RoutingGraph:
    """One encoder feeding `z_a` and `z_b` (and `z_c`), one decoder over all of them.

    Args:
        heads: Latent space name -> head on that edge of `enc`; `None` for no heads at all.
        dim_a: Dimensionality of `z_a`.
        dim_b: Dimensionality of `z_b`.
        extra_latent: Also feed a third latent space `z_c` (dim 2) from `enc`.

    Returns:
        The routing graph, with `enc` fanning out to every latent space.
    """
    names = ["z_a", "z_b"] + (["z_c"] if extra_latent else [])
    specs = {"z_a": LatentSpace("z_a", dim_a), "z_b": LatentSpace("z_b", dim_b)}
    if extra_latent:
        specs["z_c"] = LatentSpace("z_c", 2)
    return RoutingGraph(
        latent_specs=specs,
        encoder_to_latents={"enc": names},
        latent_to_decoders={name: ["dec"] for name in names},
        decoder_assemblers={"dec": "concat"},
        encoder_heads={} if heads is None else {"enc": heads},
    )


class TestHeadSpec:
    """`HeadSpec` and the `encoder_heads` field of `RoutingGraph`."""

    def test_defaults_to_the_identity_head_with_no_declared_width(self) -> None:
        """No argument means `identity`, no `out_dim`, and no extra `kwargs`."""
        head = HeadSpec()
        assert (head.name, head.out_dim, head.kwargs) == ("identity", None, {})

    def test_kwargs_are_not_shared_between_instances(self) -> None:
        """The default `kwargs` dict is created per instance."""
        first, second = HeadSpec("linear"), HeadSpec("linear")
        first.kwargs["x"] = 1
        assert second.kwargs == {}

    def test_encoder_heads_is_the_last_field_so_positional_construction_keeps_working(self) -> None:
        """Code that passes the first four fields positionally is unaffected by heads."""
        graph = RoutingGraph(
            {"z": LatentSpace("z", 8)}, {"enc": ["z"]}, {"z": ["dec"]}, {"dec": "concat"}
        )
        assert graph.encoder_heads == {}


class TestRejectsHeadOnAnUnknownEdge:
    """Rejection 5: a head must sit on an encoder -> latent edge the graph really has."""

    def test_encoder_that_feeds_nothing(self) -> None:
        """A head for an encoder that is not in `encoder_to_latents` is refused."""
        graph = _singleLatentGraph()
        graph.encoder_heads = {"ghost": {"z": HeadSpec("linear")}}
        with pytest.raises(ValueError, match=r"'ghost' -> 'z'.*does not feed"):
            validateRoutingGraph(graph)

    def test_latent_the_encoder_does_not_feed(self) -> None:
        """A head on an edge to a real latent space that this encoder does not feed is refused."""
        graph = _twoLatentGraph(assembler="concat")
        graph.encoder_heads = {"enc_a": {"z_b": HeadSpec("linear")}}
        with pytest.raises(ValueError, match=r"'enc_a' -> 'z_b'.*does not feed latent space"):
            validateRoutingGraph(graph)

    def test_message_lists_what_the_encoder_does_feed(self) -> None:
        """The message names the latent spaces the encoder feeds, to point at the typo."""
        graph = _twoLatentGraph(assembler="concat")
        graph.encoder_heads = {"enc_a": {"z_b": HeadSpec("linear")}}
        with pytest.raises(ValueError, match=r"it feeds \['z_a'\]"):
            validateRoutingGraph(graph)

    def test_latent_space_missing_from_the_graph(self) -> None:
        """An edge to a name that is in `encoder_to_latents` but not a latent space is refused."""
        graph = RoutingGraph(
            latent_specs={"z": LatentSpace("z", 8)},
            encoder_to_latents={"enc": ["z", "missing"]},
            latent_to_decoders={"z": ["dec"]},
            encoder_heads={"enc": {"missing": HeadSpec("linear")}},
        )
        with pytest.raises(ValueError, match=r"'missing' is not a latent space of this graph"):
            validateRoutingGraph(graph)


class TestRejectsHeadWidthMismatch:
    """Rejection 6: a head's output width is the target latent space's `dim` (spec §10)."""

    def test_declared_out_dim_differs_from_the_latent_dim(self) -> None:
        """`out_dim=16` into a latent space of `dim=4` is refused."""
        graph = _fanOutGraph(heads={"z_b": HeadSpec("linear", out_dim=16)})
        with pytest.raises(ValueError, match=r"out_dim=16.*'z_b' has dim=4"):
            validateRoutingGraph(graph)

    def test_message_names_the_edge_and_the_head(self) -> None:
        """The message says which head, on which edge."""
        graph = _fanOutGraph(heads={"z_b": HeadSpec("linear", out_dim=16)})
        with pytest.raises(ValueError, match=r"Head 'linear' on the edge 'enc' -> 'z_b'"):
            validateRoutingGraph(graph)

    def test_declared_out_dim_equal_to_the_latent_dim_is_accepted(self) -> None:
        """The declaration is an assertion: when it matches, nothing happens."""
        graph = _fanOutGraph(heads={"z_b": HeadSpec("linear", out_dim=4)})
        validateRoutingGraph(graph)

    def test_undeclared_out_dim_is_never_a_mismatch(self) -> None:
        """With `out_dim=None` the width is taken from the latent space, so nothing can differ."""
        validateRoutingGraph(_fanOutGraph(heads={"z_b": HeadSpec("linear")}))

    @pytest.mark.parametrize("reserved", ["in_dim", "out_dim"])
    def test_widths_cannot_be_overridden_through_kwargs(self, reserved: str) -> None:
        """The model derives `in_dim` and `out_dim`, so `kwargs` may not carry them."""
        graph = _fanOutGraph(heads={"z_b": HeadSpec("linear", kwargs={reserved: 3})})
        with pytest.raises(ValueError, match=rf"sets \['{reserved}'\] in `kwargs`"):
            validateRoutingGraph(graph)

    def test_other_kwargs_are_left_alone(self) -> None:
        """Validation does not look inside the head's own arguments."""
        graph = _fanOutGraph(heads={"z_b": HeadSpec("linear", kwargs={"anything": 3})})
        validateRoutingGraph(graph)


class TestRejectsTwoIdentityEdgesFromOneEncoder:
    """Rejection 7: two identity edges would give two latent spaces identical posteriors."""

    def test_no_heads_at_all_on_a_fan_out(self) -> None:
        """Both edges default to identity, which is the ADR 0002 failure."""
        with pytest.raises(ValueError, match="identical posteriors"):
            validateRoutingGraph(_fanOutGraph())

    def test_message_names_the_encoder_and_its_identity_edges(self) -> None:
        """The message says which encoder, which latent spaces, and what to do."""
        with pytest.raises(ValueError) as excinfo:
            validateRoutingGraph(_fanOutGraph())
        message = str(excinfo.value)
        assert "Encoder 'enc' feeds 2 latent spaces" in message
        assert "['z_a', 'z_b']" in message
        assert "'linear'" in message

    def test_explicit_identity_heads_count_as_identity(self) -> None:
        """Writing `HeadSpec("identity")` on both edges is the same as writing nothing."""
        graph = _fanOutGraph(heads={"z_a": HeadSpec("identity"), "z_b": HeadSpec("identity")})
        with pytest.raises(ValueError, match="identical posteriors"):
            validateRoutingGraph(graph)

    def test_one_explicit_identity_and_one_missing_entry_still_count_twice(self) -> None:
        """An edge with no entry and an explicit identity edge are both identity edges."""
        graph = _fanOutGraph(heads={"z_a": HeadSpec("identity")})
        with pytest.raises(ValueError, match="identical posteriors"):
            validateRoutingGraph(graph)

    def test_one_identity_edge_and_one_linear_edge_is_accepted(self) -> None:
        """The shape of spec §9: the shared edge is the identity, the private edge projects."""
        validateRoutingGraph(_fanOutGraph(heads={"z_b": HeadSpec("linear")}))
        validateRoutingGraph(_fanOutGraph(heads={"z_a": HeadSpec("linear")}))

    def test_two_linear_edges_are_accepted(self) -> None:
        """Heads with learned parameters start different, so any number of them is fine."""
        graph = _fanOutGraph(heads={"z_a": HeadSpec("linear"), "z_b": HeadSpec("linear")})
        validateRoutingGraph(graph)

    def test_three_edges_allow_one_identity_and_reject_two(self) -> None:
        """With three latent spaces, one identity plus two linear passes; two identity does not."""
        validateRoutingGraph(
            _fanOutGraph(
                heads={"z_b": HeadSpec("linear"), "z_c": HeadSpec("linear")}, extra_latent=True
            )
        )
        with pytest.raises(ValueError, match=r"2 of those edges are identity edges"):
            validateRoutingGraph(_fanOutGraph(heads={"z_c": HeadSpec("linear")}, extra_latent=True))

    def test_an_encoder_with_one_edge_may_use_the_identity(self) -> None:
        """The rule is about fan-out: a single identity edge is the ordinary case."""
        validateRoutingGraph(_singleLatentGraph())

    def test_identity_head_names_is_configurable(self) -> None:
        """A custom pass-through head can be declared an identity head, and then counts as one."""
        graph = _fanOutGraph(heads={"z_b": HeadSpec("my_passthrough")})
        validateRoutingGraph(graph)  # by default only 'identity' counts as an identity head
        with pytest.raises(ValueError, match="identical posteriors"):
            validateRoutingGraph(
                graph, identity_head_names=frozenset({"identity", "my_passthrough"})
            )


class TestDecoderInputs:
    """`RoutingGraph.decoderInputs` says which latents a decoder gets, and in which order.

    `GlobalVae` sizes each assembler with it and hands the latent vectors over in this order, and
    the config resolution reads it to compute the width a decoder receives (roadmap P1-4).
    """

    def test_inverts_latent_to_decoders(self) -> None:
        """Latent space -> decoders becomes decoder -> latent spaces."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 4)},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
            latent_to_decoders={"z_a": ["dec_1", "dec_2"], "z_b": ["dec_1"]},
            decoder_assemblers={"dec_1": "concat"},
        )
        assert graph.decoderInputs() == {"dec_1": ["z_a", "z_b"], "dec_2": ["z_a"]}

    def test_order_is_the_order_of_latent_to_decoders_not_alphabetical(self) -> None:
        """A latent space declared first is the first input, even if its name sorts last."""
        graph = RoutingGraph(
            latent_specs={"z_b": LatentSpace("z_b", 4), "z_a": LatentSpace("z_a", 8)},
            encoder_to_latents={"enc_a": ["z_a"], "enc_b": ["z_b"]},
            latent_to_decoders={"z_b": ["dec"], "z_a": ["dec"]},
            decoder_assemblers={"dec": "concat"},
        )
        assert graph.decoderInputs()["dec"] == ["z_b", "z_a"]

    def test_a_graph_with_no_consumer_has_no_entries(self) -> None:
        """A decoder that consumes nothing has no entry, and an empty graph gives `{}`."""
        assert RoutingGraph(latent_specs={}).decoderInputs() == {}
        graph = _singleLatentGraph()
        graph.latent_to_decoders = {"z": []}
        assert graph.decoderInputs() == {}

    def test_the_validator_and_the_inverse_agree_on_a_multi_latent_decoder(self) -> None:
        """The decoder the validator checks for an assembler is the one `decoderInputs` lists."""
        graph = _twoLatentGraph(assembler=None)
        assert graph.decoderInputs() == {"dec": ["z_a", "z_b"]}
        with pytest.raises(ValueError, match=r"consumes 2 latent spaces \['z_a', 'z_b'\]"):
            validateRoutingGraph(graph)
