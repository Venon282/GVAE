"""Convenience preset: shared plus private latent spaces (spec §2.2).

This is one example `RoutingGraph` topology among many, not the
definition of "several latent spaces" (see `latent/base.py`). It builds
a graph where every encoder feeds a `z_shared` latent space through
Fusion, each encoder also feeds its own `z_private_{modality}`, and each
modality's decoder consumes `{z_shared, z_private_{modality}}` via the
`concat` assembler.

Each modality's encoder is assigned to two latent spaces here
(`z_shared` and its own `z_private_{modality}`): the encoder fan-out
case. It goes through latent heads (spec §2.2), one per edge, so the two
spaces get their own posteriors instead of the same `(mu, logvar)` twice
(`docs/adr/0024-encoder-fan-out-through-latent-heads.md`). By default the
shared edge is the identity and the private edge a `linear` head, which
is the arrangement spec §9 shows. A graph whose two edges were both
identity edges would be refused by `validateRoutingGraph`.
"""

from global_vae.latent.base import HeadSpec, LatentSpace, RoutingGraph


def buildSharedPrivateRoutingGraph(
    modality_names: list[str],
    shared_dim: int,
    private_dim: int,
    assembler: str = "concat",
    shared_head: str = "identity",
    private_head: str = "linear",
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
        shared_head: Head registry name on every encoder -> `z_shared`
            edge. Defaults to `"identity"`, which needs each encoder's
            `latent_dim` to equal `shared_dim` (the model checks it).
            Use `"linear"` to project encoders of another width.
        private_head: Head registry name on every encoder ->
            `z_private_{modality}` edge. Defaults to `"linear"`: the
            private space is a projection of the same encoder output,
            and a head that is not the identity is what keeps it
            independent of `z_shared`. Passing `"identity"` for both
            heads builds a graph that `validateRoutingGraph` refuses.

    Returns:
        A `RoutingGraph` with one shared and `len(modality_names)`
        private latent spaces, and a `HeadSpec` on every encoder ->
        latent edge. The caller still needs to pass a fusion strategy
        for `z_shared` to `GlobalVae` (it is fed by more than one
        encoder when there are several modalities); the private spaces
        need none, since each is fed by exactly one encoder.
    """
    shared_name = "z_shared"
    latent_specs: dict[str, LatentSpace] = {shared_name: LatentSpace(shared_name, shared_dim)}
    encoder_to_latents: dict[str, list[str]] = {}
    encoder_heads: dict[str, dict[str, HeadSpec]] = {}
    latent_to_decoders: dict[str, list[str]] = {shared_name: []}
    decoder_assemblers: dict[str, str] = {}

    for modality in modality_names:
        private_name = f"z_private_{modality}"
        latent_specs[private_name] = LatentSpace(private_name, private_dim)
        # Each modality's encoder feeds both the shared space (via Fusion,
        # combined with the other modalities' encoders) and its own
        # private space, each through its own latent head.
        encoder_to_latents[modality] = [shared_name, private_name]
        encoder_heads[modality] = {
            shared_name: HeadSpec(name=shared_head),
            private_name: HeadSpec(name=private_head),
        }
        latent_to_decoders[shared_name].append(modality)
        latent_to_decoders[private_name] = [modality]
        decoder_assemblers[modality] = assembler

    return RoutingGraph(
        latent_specs=latent_specs,
        encoder_to_latents=encoder_to_latents,
        latent_to_decoders=latent_to_decoders,
        decoder_assemblers=decoder_assemblers,
        encoder_heads=encoder_heads,
    )
