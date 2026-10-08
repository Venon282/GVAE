"""Unit tests for `latent.base.validateRoutingGraph` (spec §2.2, §10, roadmap P0-4(a)).

`validateRoutingGraph` is the construction-time gate every `GlobalVae` goes through, so each of
its rejection paths gets its own test, next to the accepting cases that show the gate is not
simply rejecting everything:

1. a latent space with no encoder feeding it,
2. a latent space with no decoder consuming it,
3. a decoder consuming several latent spaces with no assembler,
4. a `sum` or `average` assembler over latent spaces of different dimensionality.

The rejection that spec §10 adds for latent heads (a head whose output dimension differs from
its target latent) cannot exist yet: heads arrive with roadmap P1-1 and P1-2, which extend this
file.
"""

import pytest

from global_vae.latent.base import LatentSpace, RoutingGraph, validateRoutingGraph


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


class TestRejectsDimensionLockedAssemblerOverDifferentDims:
    """Rejection 4: `sum` and `average` need every input latent space to share one dimension."""

    @pytest.mark.parametrize("assembler", ["sum", "average"])
    def test_differing_dims_are_rejected(self, assembler: str) -> None:
        """Both dimension-locked built-ins reject latent spaces of dims 8 and 4."""
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

    def test_custom_dimension_locked_set_is_honoured(self) -> None:
        """A strategy added to `dimension_locked_assemblers` gets the same check."""
        graph = _twoLatentGraph(assembler="weighted_sum", dim_a=8, dim_b=4)
        with pytest.raises(ValueError, match="requires matching dimensionality"):
            validateRoutingGraph(graph, dimension_locked_assemblers=frozenset({"weighted_sum"}))


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
        [("concat", 8, 8), ("concat", 8, 4), ("sum", 8, 8), ("average", 4, 4)],
        ids=["concat-equal", "concat-different", "sum-equal", "average-equal"],
    )
    def test_multi_latent_decoder_with_a_compatible_assembler(
        self, assembler: str, dim_a: int, dim_b: int
    ) -> None:
        """`concat` accepts any dims; `sum` and `average` accept equal ones."""
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

    def test_empty_dimension_locked_set_disables_the_dimension_check(self) -> None:
        """With no locked assemblers, `sum` over different dims passes this gate."""
        graph = _twoLatentGraph(assembler="sum", dim_a=8, dim_b=4)
        validateRoutingGraph(graph, dimension_locked_assemblers=frozenset())

    def test_encoder_fan_out_is_a_valid_graph(self) -> None:
        """Fan-out is legal in a routing graph; only `GlobalVae` does not support it yet."""
        graph = RoutingGraph(
            latent_specs={"z_a": LatentSpace("z_a", 8), "z_b": LatentSpace("z_b", 8)},
            encoder_to_latents={"enc": ["z_a", "z_b"]},
            latent_to_decoders={"z_a": ["dec"], "z_b": ["dec"]},
            decoder_assemblers={"dec": "concat"},
        )
        validateRoutingGraph(graph)
