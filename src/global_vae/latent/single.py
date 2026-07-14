"""Convenience preset: a single shared latent space feeding every decoder.

This is one specific `RoutingGraph` topology among others (spec §2.2),
corresponding to the `*-L1-*` rows of the configuration matrix (spec
§2.1). It is not a separate mechanism from `RoutingGraph` — just a
constructor that builds one for the common case.
"""

from global_vae.latent.base import LatentSpace, RoutingGraph


def buildSingleLatentRoutingGraph(
    encoder_names: list[str], decoder_names: list[str], latent_dim: int, latent_name: str = "z"
) -> RoutingGraph:
    """Build a `RoutingGraph` with one latent space feeding every decoder.

    Args:
        encoder_names: Names of every encoder in the model; all of
            them feed the single latent space (directly, or via
            Fusion if there is more than one).
        decoder_names: Names of every decoder in the model; all of
            them consume the single latent space directly (no
            Assembler needed, since there is only one input).
        latent_dim: Dimensionality of the shared latent space.
        latent_name: Identifier for the latent space.

    Returns:
        A `RoutingGraph` with exactly one `LatentSpace`.
    """
    latent = LatentSpace(name=latent_name, dim=latent_dim)
    return RoutingGraph(
        latent_specs={latent_name: latent},
        encoder_to_latents={encoder: [latent_name] for encoder in encoder_names},
        latent_to_decoders={latent_name: list(decoder_names)},
    )
