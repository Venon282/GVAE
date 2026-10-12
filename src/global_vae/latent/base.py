"""Latent-space representation and routing-graph validation (spec §2.2).

A `LatentSpace` is one independent Gaussian latent with its own
posterior, its own prior, and its own regularization term, not
necessarily KL divergence (spec §2.3; see `losses/regularizers/` for
the pluggable strategies). "Several latent spaces" in the spec is not
a fixed shared/private split. It is an arbitrary number of these,
wired to encoders and decoders through a configurable routing graph.
`routing_graph_builders/single.py` (one latent space) and
`routing_graph_builders/shared_private.py` (shared plus private, one
example topology) are convenience presets built on top of the general
`RoutingGraph` defined here; the framework does not treat either as a
hardcoded special case.

When one encoder feeds several latent spaces, each encoder -> latent
edge carries a latent head (`HeadSpec`, `heads/`) that adapts the
encoder's `(mu, logvar)` to that space, so the spaces stay independent
(spec §2.2, `docs/adr/0024-encoder-fan-out-through-latent-heads.md`).
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LatentSpace:
    """A single independent latent space.

    Attributes:
        name: Unique identifier (e.g. `"z_shared"`, `"z_private_signal"`).
        dim: Dimensionality of `z` in this latent space.
    """

    name: str
    dim: int

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample `z` via the reparameterization trick.

        Args:
            mu: Mean of the approximate posterior, shape `(batch, dim)`.
            logvar: Log-variance of the approximate posterior, shape
                `(batch, dim)`.

        Returns:
            A sampled latent tensor of shape `(batch, dim)`.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


@dataclass
class HeadSpec:
    """The latent head on one encoder -> latent edge (spec §2.2).

    The model builds the head as `getHeadClass(name)(in_dim=..., out_dim=...,
    **kwargs)`: `in_dim` is the encoder's own `latent_dim` and `out_dim` is the
    target latent space's `dim`, so neither belongs in `kwargs`.

    Attributes:
        name: Head registry name (`heads/`), `"identity"` by default.
        out_dim: Output width the author intends for this head, or `None` to
            leave it to the target latent space. When given,
            `validateRoutingGraph` requires it to equal that latent space's
            `dim` (spec §10).
        kwargs: Extra constructor arguments for the head.
    """

    name: str = "identity"
    out_dim: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingGraph:
    """Encoder, latent, and decoder wiring for one model instance (spec §2.2).

    Attributes:
        latent_specs: Latent space name -> `LatentSpace`.
        encoder_to_latents: Encoder name -> list of latent space names
            it feeds. An encoder feeding several latents does so through
            one latent head per edge (`encoder_heads`).
        latent_to_decoders: Latent space name -> list of decoder names
            that consume it.
        decoder_assemblers: Decoder name -> assembler strategy name.
            Required for every decoder consuming more than one latent
            space (see `assemblers/`); not needed for decoders
            consuming exactly one.
        encoder_heads: Encoder name -> latent space name -> `HeadSpec`,
            the latent head on that encoder -> latent edge. An edge with
            no entry is an identity edge: the encoder's `(mu, logvar)`
            is used as it is, and no module is built for it. Declared
            last so that code building a `RoutingGraph` positionally
            keeps working.
    """

    latent_specs: dict[str, LatentSpace]
    encoder_to_latents: dict[str, list[str]] = field(default_factory=dict)
    latent_to_decoders: dict[str, list[str]] = field(default_factory=dict)
    decoder_assemblers: dict[str, str] = field(default_factory=dict)
    encoder_heads: dict[str, dict[str, HeadSpec]] = field(default_factory=dict)


def validateRoutingGraph(
    graph: RoutingGraph,
    dimension_locked_assemblers: frozenset[str] = frozenset({"sum", "average"}),
    identity_head_names: frozenset[str] = frozenset({"identity"}),
) -> None:
    """Validate a routing graph against the constraints in spec §2.2 and §10.

    Six constraints are enforced:
      1. Every latent space must have at least one encoder feeding it
         and at least one decoder consuming it (no orphan latent
         spaces).
      2. Every decoder that consumes more than one latent space must
         have an assembler assigned in `decoder_assemblers`. A decoder
         consuming several latent spaces with no assembler was
         previously accepted silently; this is now rejected.
      3. `sum`/`average` assemblers require all of their input latent
         spaces to share the same dimensionality (`concat` has no such
         restriction).
      4. Every entry of `encoder_heads` is on an edge that exists in
         `encoder_to_latents`.
      5. A head's declared `out_dim` equals the target latent space's
         `dim`, and its `kwargs` do not set `in_dim` or `out_dim` (the
         model derives both).
      6. An encoder feeding several latent spaces has at most one
         identity edge (an edge with no head entry, or whose head is
         one of `identity_head_names`). Two identity edges would give
         two latent spaces the same posterior (ADR 0002, ADR 0024).

    Args:
        graph: The routing graph to validate.
        dimension_locked_assemblers: Assembler names that require
            matching dimensionality across their inputs.
        identity_head_names: Head registry names that pass `(mu,
            logvar)` through unchanged.

    Raises:
        ValueError: If any constraint is violated.
    """
    fed_latents = {latent for latents in graph.encoder_to_latents.values() for latent in latents}

    for name in graph.latent_specs:
        if name not in fed_latents:
            raise ValueError(f"Latent space '{name}' has no encoder feeding it (orphan latent).")
        if not graph.latent_to_decoders.get(name):
            raise ValueError(f"Latent space '{name}' has no decoder consuming it (orphan latent).")

    decoder_to_latents: dict[str, list[str]] = {}
    for latent_name, decoder_names in graph.latent_to_decoders.items():
        for decoder_name in decoder_names:
            decoder_to_latents.setdefault(decoder_name, []).append(latent_name)

    for decoder_name, latent_names in decoder_to_latents.items():
        if len(latent_names) <= 1:
            continue

        assembler_name = graph.decoder_assemblers.get(decoder_name)
        if assembler_name is None:
            raise ValueError(
                f"Decoder '{decoder_name}' consumes {len(latent_names)} latent spaces "
                f"{latent_names} but has no assembler assigned in `decoder_assemblers`."
            )

        if assembler_name in dimension_locked_assemblers:
            dims = {graph.latent_specs[name].dim for name in latent_names}
            if len(dims) > 1:
                raise ValueError(
                    f"Decoder '{decoder_name}' uses assembler '{assembler_name}', which "
                    f"requires matching dimensionality, but its input latent spaces "
                    f"{latent_names} have differing dims {sorted(dims)}."
                )

    _validateHeads(graph, identity_head_names)


def _validateHeads(graph: RoutingGraph, identity_head_names: frozenset[str]) -> None:
    """Check the latent heads of a routing graph (constraints 4 to 6 above).

    Args:
        graph: The routing graph to validate.
        identity_head_names: Head registry names that pass `(mu,
            logvar)` through unchanged.

    Raises:
        ValueError: If a head is on an edge or a latent space that does not
            exist, declares an `out_dim` that differs from its latent
            space's `dim`, sets `in_dim` or `out_dim` in its `kwargs`, or
            if an encoder feeding several latent spaces has more than one
            identity edge.
    """
    for encoder_name, edge_heads in graph.encoder_heads.items():
        fed = graph.encoder_to_latents.get(encoder_name, [])
        for latent_name, head in edge_heads.items():
            if latent_name not in fed:
                raise ValueError(
                    f"Head '{head.name}' is declared on the edge '{encoder_name}' -> "
                    f"'{latent_name}', but encoder '{encoder_name}' does not feed latent "
                    f"space '{latent_name}' (it feeds {fed})."
                )
            reserved = sorted({"in_dim", "out_dim"} & set(head.kwargs))
            if reserved:
                raise ValueError(
                    f"Head '{head.name}' on the edge '{encoder_name}' -> '{latent_name}' sets "
                    f"{reserved} in `kwargs`. The model derives both widths: `in_dim` from the "
                    f"encoder's `latent_dim`, `out_dim` from the latent space's `dim`."
                )
            if latent_name not in graph.latent_specs:
                raise ValueError(
                    f"Head '{head.name}' is declared on the edge '{encoder_name}' -> "
                    f"'{latent_name}', but '{latent_name}' is not a latent space of this graph "
                    f"(latent spaces: {sorted(graph.latent_specs)})."
                )
            latent_dim = graph.latent_specs[latent_name].dim
            if head.out_dim is not None and head.out_dim != latent_dim:
                raise ValueError(
                    f"Head '{head.name}' on the edge '{encoder_name}' -> '{latent_name}' "
                    f"declares out_dim={head.out_dim}, but latent space '{latent_name}' has "
                    f"dim={latent_dim}."
                )

    for encoder_name, latent_names in graph.encoder_to_latents.items():
        if len(latent_names) < 2:
            continue
        edge_heads = graph.encoder_heads.get(encoder_name, {})
        identity_edges = [
            latent_name
            for latent_name in latent_names
            if latent_name not in edge_heads or edge_heads[latent_name].name in identity_head_names
        ]
        if len(identity_edges) > 1:
            raise ValueError(
                f"Encoder '{encoder_name}' feeds {len(latent_names)} latent spaces "
                f"{latent_names}, and {len(identity_edges)} of those edges are identity edges "
                f"{identity_edges}: they would receive the same `(mu, logvar)`, so those latent "
                f"spaces would have identical posteriors. Give all but one of them a "
                f"non-identity head (for example 'linear') in `encoder_heads`."
            )
