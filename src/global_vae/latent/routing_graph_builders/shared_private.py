"""Convenience preset: shared plus private latent spaces (spec §2.2).

This is one example `RoutingGraph` topology among many, not the
definition of "several latent spaces" (see `latent/base.py`). It builds
a graph where every encoder feeds a `z_shared` latent space through
Fusion, each encoder also feeds its own untouched `z_private_{modality}`,
and each modality's decoder consumes `{z_shared, z_private_{modality}}`
via the `concat` assembler.

Current limitation: each modality's encoder is assigned to two latent
spaces here (`z_shared` and its own `z_private_{modality}`), which is
exactly the encoder fan-out case `GlobalVae` does not support yet (see
`docs/adr/0002-generalize-global-vae-to-routing-graph.md`). The
`RoutingGraph` this function returns is valid and passes
`validateRoutingGraph`, but building a `GlobalVae` from it will raise
`NotImplementedError` until encoders can expose one `(mu, logvar)` pair
per latent space they feed, rather than one pair total.
"""

from global_vae.latent.base import LatentSpace, RoutingGraph


def buildSharedPrivateRoutingGraph(
    modality_names: list[str],
    shared_dim: int,
    private_dim: int,
    assembler: str = "concat",
) -> RoutingGraph:
    """Build a shared plus private `RoutingGraph`.

    Args:
        modality_names: One encoder and one decoder per modality name;
            each gets its own private latent space.
        shared_dim: Dimensionality of `z_shared`.
        private_dim: Dimensionality of each `z_private_{modality}`.
            Must equal `shared_dim` if `assembler` is `"sum"` or
            `"average"` (validated by `validateRoutingGraph`).
        assembler: Assembler strategy each decoder uses to combine
            `{z_shared, z_private_{modality}}`. Defaults to `"concat"`,
            which has no dimensionality restriction.

    Returns:
        A `RoutingGraph` with one shared and `len(modality_names)`
        private latent spaces. The caller still needs to pass a fusion
        strategy for `z_shared` to `GlobalVae` (it is fed by more than
        one encoder); the private spaces need none, since each is fed
        by exactly one encoder.
    """
    shared_name = "z_shared"
    latent_specs: dict[str, LatentSpace] = {shared_name: LatentSpace(shared_name, shared_dim)}
    encoder_to_latents: dict[str, list[str]] = {}
    latent_to_decoders: dict[str, list[str]] = {shared_name: []}
    decoder_assemblers: dict[str, str] = {}

    for modality in modality_names:
        private_name = f"z_private_{modality}"
        latent_specs[private_name] = LatentSpace(private_name, private_dim)
        # Each modality's encoder feeds both the shared space (via Fusion,
        # combined with the other modalities' encoders) and its own
        # untouched private space.
        encoder_to_latents[modality] = [shared_name, private_name]
        latent_to_decoders[shared_name].append(modality)
        latent_to_decoders[private_name] = [modality]
        decoder_assemblers[modality] = assembler

    return RoutingGraph(
        latent_specs=latent_specs,
        encoder_to_latents=encoder_to_latents,
        latent_to_decoders=latent_to_decoders,
        decoder_assemblers=decoder_assemblers,
    )
