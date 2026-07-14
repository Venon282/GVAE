"""Latent-space representation and routing-graph validation (spec §2.2).

A `LatentSpace` is one independent Gaussian latent with its own
posterior, its own prior, and its own KL term. "Several latent spaces"
in the spec is *not* a fixed shared/private split — it is an arbitrary
number of these, wired to encoders and decoders through a configurable
routing graph. `single.py` (one latent space) and `factorized.py`
(shared + private) are just two convenience presets built on top of
the general `RoutingGraph` defined here; the framework does not treat
either as a hardcoded special case.
"""

from dataclasses import dataclass, field

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

    def klDivergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-sample KL divergence to a standard normal prior.

        Args:
            mu: Posterior mean, shape `(batch, dim)`.
            logvar: Posterior log-variance, shape `(batch, dim)`.

        Returns:
            Per-sample KL divergence, shape `(batch,)`. Callers sum
            across latent spaces and average across the batch.
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

@dataclass
class RoutingGraph:
    """Encoder <-> latent <-> decoder wiring for one model instance (spec §2.2).

    Attributes:
        latent_specs: Latent space name -> `LatentSpace`.
        encoder_to_latents: Encoder name -> list of latent space names
            it feeds. An encoder feeding several latents does so via
            several independent projection heads.
        latent_to_decoders: Latent space name -> list of decoder names
            that consume it.
        decoder_assemblers: Decoder name -> assembler strategy name.
            Only meaningful for decoders consuming more than one
            latent space (see `latent/assembler.py`).
    """

    latent_specs: dict[str, LatentSpace]
    encoder_to_latents: dict[str, list[str]] = field(default_factory=dict)
    latent_to_decoders: dict[str, list[str]] = field(default_factory=dict)
    decoder_assemblers: dict[str, str] = field(default_factory=dict)

def validateRoutingGraph(
    graph: RoutingGraph,
    dimension_locked_assemblers: frozenset[str] = frozenset({"sum", "average"}),
) -> None:
    """Validate a routing graph against the constraints in spec §2.2.

    Two constraints are enforced:
      1. Every latent space must have at least one encoder feeding it
         and at least one decoder consuming it (no orphan latent
         spaces).
      2. `sum`/`average` assemblers require all of their input latent
         spaces to share the same dimensionality (`concat` has no such
         restriction).

    Args:
        graph: The routing graph to validate.
        dimension_locked_assemblers: Assembler names that require
            matching dimensionality across their inputs.

    Raises:
        ValueError: If either constraint is violated.
    """
    fed_latents = {latent for latents in graph.encoder_to_latents.values() for latent in latents}

    # check if every latent space have at least one encoder and one decoder
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
        if assembler_name in dimension_locked_assemblers:
            dims = {graph.latent_specs[name].dim for name in latent_names}
            if len(dims) > 1:
                raise ValueError(
                    f"Decoder '{decoder_name}' uses assembler '{assembler_name}', which "
                    f"requires matching dimensionality, but its input latent spaces "
                    f"{latent_names} have differing dims {sorted(dims)}."
                )
