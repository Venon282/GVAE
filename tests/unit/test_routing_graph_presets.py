"""Unit tests for the routing-graph presets (spec §2.2, roadmap P0-4(e)).

`buildSingleLatentRoutingGraph` and `buildSharedPrivateRoutingGraph` are conveniences that build a
plain `RoutingGraph`; they are not special cases inside the framework. These tests check the graph
each one returns, and that `validateRoutingGraph` accepts it, or rejects it for the documented
reason (a dimension-locked assembler over latent spaces of different sizes).

Building a `GlobalVae` from the shared plus private graph is a separate matter: it is refused for
now because of encoder fan-out, which `tests/integration/test_global_vae_construction_guards.py`
covers.
"""

import pytest

from global_vae.latent.base import RoutingGraph, validateRoutingGraph
from global_vae.latent.routing_graph_builders.shared_private import buildSharedPrivateRoutingGraph
from global_vae.latent.routing_graph_builders.single import buildSingleLatentRoutingGraph


class TestBuildSingleLatentRoutingGraph:
    """One latent space feeding every decoder, fed by every encoder."""

    def test_has_exactly_one_latent_space(self) -> None:
        """The graph holds one `LatentSpace` with the requested name and dimension."""
        graph = buildSingleLatentRoutingGraph(["signal"], ["signal"], latent_dim=12)
        assert list(graph.latent_specs) == ["z"]
        assert graph.latent_specs["z"].name == "z"
        assert graph.latent_specs["z"].dim == 12

    def test_custom_latent_name(self) -> None:
        """`latent_name` renames the latent space everywhere it appears."""
        graph = buildSingleLatentRoutingGraph(["signal"], ["signal"], 8, latent_name="z_fused")
        assert list(graph.latent_specs) == ["z_fused"]
        assert graph.latent_specs["z_fused"].name == "z_fused"
        assert graph.encoder_to_latents == {"signal": ["z_fused"]}
        assert graph.latent_to_decoders == {"z_fused": ["signal"]}

    def test_every_encoder_feeds_the_latent_space(self) -> None:
        """Each encoder maps to the one latent space, and only to it (no fan-out)."""
        graph = buildSingleLatentRoutingGraph(["signal", "image"], ["signal", "image"], 8)
        assert graph.encoder_to_latents == {"signal": ["z"], "image": ["z"]}

    def test_every_decoder_consumes_the_latent_space(self) -> None:
        """The latent space lists every decoder, in the order given."""
        graph = buildSingleLatentRoutingGraph(["signal", "image"], ["image", "signal"], 8)
        assert graph.latent_to_decoders == {"z": ["image", "signal"]}

    def test_needs_no_assemblers(self) -> None:
        """Every decoder has exactly one input, so `decoder_assemblers` stays empty."""
        graph = buildSingleLatentRoutingGraph(["signal", "image"], ["signal", "image"], 8)
        assert graph.decoder_assemblers == {}

    def test_encoder_and_decoder_names_are_independent(self) -> None:
        """A shared decoder (`*-D1`) or a decoder-only target name needs no matching encoder."""
        graph = buildSingleLatentRoutingGraph(["signal", "image_in"], ["image_out"], 8)
        assert set(graph.encoder_to_latents) == {"signal", "image_in"}
        assert graph.latent_to_decoders == {"z": ["image_out"]}

    def test_decoder_list_is_copied(self) -> None:
        """Editing the caller's list afterwards does not change the graph."""
        decoders = ["signal"]
        graph = buildSingleLatentRoutingGraph(["signal"], decoders, 8)
        decoders.append("image")
        assert graph.latent_to_decoders == {"z": ["signal"]}

    def test_returns_a_routing_graph(self) -> None:
        """The preset returns the general `RoutingGraph`, not a special type."""
        graph = buildSingleLatentRoutingGraph(["signal"], ["signal"], 8)
        assert type(graph) is RoutingGraph

    @pytest.mark.parametrize(
        ("encoders", "decoders"),
        [
            (["signal"], ["signal"]),
            (["signal", "image"], ["signal", "image"]),
            (["signal", "image"], ["shared"]),
        ],
        ids=["one-modality", "per-modality", "shared-decoder"],
    )
    def test_passes_validation(self, encoders: list[str], decoders: list[str]) -> None:
        """The graph is accepted for one or several encoders and decoders."""
        validateRoutingGraph(buildSingleLatentRoutingGraph(encoders, decoders, 8))


class TestBuildSharedPrivateRoutingGraph:
    """`z_shared` plus one `z_private_{modality}` per modality."""

    def test_latent_spaces_and_dimensions(self) -> None:
        """One shared space and one private space per modality, each with its own dimension."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], shared_dim=16, private_dim=4)
        assert list(graph.latent_specs) == ["z_shared", "z_private_signal", "z_private_image"]
        assert graph.latent_specs["z_shared"].dim == 16
        assert graph.latent_specs["z_private_signal"].dim == 4
        assert graph.latent_specs["z_private_image"].dim == 4

    def test_latent_space_names_match_their_keys(self) -> None:
        """Each `LatentSpace.name` equals the key it is stored under."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], 16, 4)
        for key, latent in graph.latent_specs.items():
            assert latent.name == key

    def test_every_encoder_feeds_shared_and_its_own_private_space(self) -> None:
        """Each encoder maps to `z_shared` and its private space: the fan-out topology."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], 16, 4)
        assert graph.encoder_to_latents == {
            "signal": ["z_shared", "z_private_signal"],
            "image": ["z_shared", "z_private_image"],
        }

    def test_shared_space_feeds_every_decoder_and_private_only_its_own(self) -> None:
        """`z_shared` goes to all decoders; `z_private_m` goes to decoder `m` alone."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], 16, 4)
        assert graph.latent_to_decoders == {
            "z_shared": ["signal", "image"],
            "z_private_signal": ["signal"],
            "z_private_image": ["image"],
        }

    def test_default_assembler_is_concat_for_every_decoder(self) -> None:
        """Each decoder joins `{z_shared, z_private_m}` with `concat` unless told otherwise."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], 16, 4)
        assert graph.decoder_assemblers == {"signal": "concat", "image": "concat"}

    @pytest.mark.parametrize("assembler", ["concat", "sum", "average"])
    def test_assembler_argument_is_applied_to_every_decoder(self, assembler: str) -> None:
        """The `assembler` argument is the one stored for each modality's decoder."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], 8, 8, assembler=assembler)
        assert graph.decoder_assemblers == {"signal": assembler, "image": assembler}

    def test_single_modality(self) -> None:
        """One modality gives `z_shared` plus one private space, and is valid."""
        graph = buildSharedPrivateRoutingGraph(["signal"], 8, 4)
        assert list(graph.latent_specs) == ["z_shared", "z_private_signal"]
        assert graph.latent_to_decoders["z_shared"] == ["signal"]
        validateRoutingGraph(graph)

    def test_three_modalities(self) -> None:
        """The topology scales with the number of modalities."""
        modalities = ["signal", "image", "spectrum"]
        graph = buildSharedPrivateRoutingGraph(modalities, 8, 4)
        assert len(graph.latent_specs) == 1 + len(modalities)
        assert graph.latent_to_decoders["z_shared"] == modalities
        assert all(len(latents) == 2 for latents in graph.encoder_to_latents.values())
        validateRoutingGraph(graph)

    def test_returns_a_routing_graph(self) -> None:
        """The preset returns the general `RoutingGraph`, not a special type."""
        graph = buildSharedPrivateRoutingGraph(["signal"], 8, 4)
        assert type(graph) is RoutingGraph

    @pytest.mark.parametrize(
        ("assembler", "shared_dim", "private_dim"),
        [("concat", 16, 4), ("concat", 8, 8), ("sum", 8, 8), ("average", 6, 6)],
    )
    def test_passes_validation_with_a_compatible_assembler(
        self, assembler: str, shared_dim: int, private_dim: int
    ) -> None:
        """`concat` accepts any dims; `sum` and `average` need `private_dim == shared_dim`."""
        graph = buildSharedPrivateRoutingGraph(
            ["signal", "image"], shared_dim, private_dim, assembler=assembler
        )
        validateRoutingGraph(graph)

    @pytest.mark.parametrize("assembler", ["sum", "average"])
    def test_building_succeeds_but_validation_rejects_mismatched_dims(self, assembler: str) -> None:
        """The builder does not check dims itself; `validateRoutingGraph` does."""
        graph = buildSharedPrivateRoutingGraph(["signal", "image"], 16, 4, assembler=assembler)
        with pytest.raises(ValueError, match="requires matching dimensionality"):
            validateRoutingGraph(graph)

    def test_no_modalities_is_rejected_by_validation(self) -> None:
        """With no modality, `z_shared` has no encoder, which validation refuses."""
        graph = buildSharedPrivateRoutingGraph([], 8, 4)
        assert list(graph.latent_specs) == ["z_shared"]
        with pytest.raises(ValueError, match="'z_shared' has no encoder feeding it"):
            validateRoutingGraph(graph)
